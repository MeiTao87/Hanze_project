from kafka import KafkaConsumer
import json
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy.ma as ma
from shapely.geometry import Polygon
from math import sin, cos, sqrt, atan2, radians
import h5py
# anomally detection libray
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from pyod.models.cblof import CBLOF  # Cluster-based Local Outlier Factor
from pyod.models.knn import KNN
from pyod.models.lof import LOF

#Set this to earliest to look at all the messages. Latest for only new messages
consumer_setting = 'latest'
# estimate R for the earth
R = 6373.0
#Consumer object
consumer = KafkaConsumer('adress_topic',
                         bootstrap_servers=['localhost:9092'],
                         auto_offset_reset=consumer_setting) # earliest latest

# Mask using a concentration threshold
def filter_ch4_qa(file_ch4, qa_threashold=0.5):
    groups = file_ch4.groups['PRODUCT']
    qa_value = groups.variables['qa_value'][0]
    ch4 = groups.variables['methane_mixing_ratio'][0]
    ch4_mask = ma.masked_less(qa_value, qa_threashold).mask
    output_ch4 = ma.masked_array(ch4, mask=ch4_mask)
    return output_ch4

# function to calculate the distance between two coordinates
def calc_distance(a, b):
    dlat = a[0] - b[0]
    dlon = a[1] - b[1]
    a = sin(dlat / 2)**2 + cos(a[0]) * cos(b[0]) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# funtion to build the list contain the latitude, longitude and CH4 concentration within polygen
def remove_masked_data(lat_constraints, lon_constraints, adress_list):
    #Looking at one file at a time
    values = []
    for adress in adress_list:
        #Reading the file
        file = Dataset(adress, 'r')  # Dataset is a netCDF4 method used to open .nc files
        # Extracting the concentrations, longitude and latitude
        ch4 = file.groups['PRODUCT'].variables['methane_mixing_ratio'][0]
        lat = file.groups['PRODUCT'].variables['latitude'][0]  # Gets latitude array
        lon = file.groups['PRODUCT'].variables['longitude'][0]  # Gets longitude array
        #Looping over all pixels to find the valid data
        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                #Check if the pixel is masked, meaning that there is no data there
                if(ch4[i, j] is not ma.masked):
                    # Check if the latitude falls within the boundries
                    if(lat[i,j] <= lat_constraints[0] and lat[i,j] >= lat_constraints[1]): # 40, 35
                        # Check if the longitude falls within the boundries
                        if(lon[i,j] <= lon_constraints[0] and lon[i,j] >= lon_constraints[1]): # 115, 110
                            # Appending all valid values to a list
                            values.append(np.array([lat[i,j], lon[i,j], ch4[i, j]]))
    return values

# Function for applying spatial allignment on data from a cluster of days
# The function looks for pixels that allign and takes the average
def overlap_cluster_of_days(values):
    # The blacklist is used to ensure that an instance that's been used once is not used again
    blacklist = []
    # The threshold is the maximum distance between pixels that is considered aligned
    threshold = 3.0
    print('Before overlapping',len(values))
    # The weights list is kept to keep track of the number of instances used to calculate the average
    weights = np.ones(len(values))
    # Two for loops for comparing all pixels with each other once
    for i in range(len(values)-1):
        print(i)
        # Do not compare if already been combined with another pixel
        if(i not in blacklist):
            for j in range(i+1, len(values)):
                # Do not compare if already been combined with another pixel
                if(j not in blacklist):
                    #Calculate distance between two pixels
                    distance = calc_distance((values[i][0], values[i][1]), (values[j][0], values[j][1]))
                    # If distance is below the threshold, combine pixels by taking the average
                    if(distance < threshold):
                        print(distance)
                        # Taking average of the concentration and location
                        average_lat = (values[i][0]*weights[i] + values[j][0]) / (weights[i]+1)
                        average_lon = (values[i][1]*weights[i] + values[j][1]) / (weights[i]+1)
                        average_ch4 = (values[i][2]*weights[i] + values[j][2]) / (weights[i]+1)
                        # Updating the weights to ensure fair average calculation
                        weights[i]+=1
                        # Store the combined pixels into the index of the first pixel
                        values[i] = [average_lat, average_lon, average_ch4]
                        # Blacklist the second pixel
                        blacklist.append(j)
    # The following part removes all the blacklisted pixels from the list
    blacklist.sort()
    # Loop over all blacklist indices
    for i in range(len(blacklist)):
        #Remove the index from the list
        #Whenever a value is removed, the index of all the following numbers change by -1
        #We account for that by subtracting i
        values.pop(blacklist[i]-i)
    print('After overlapping',len(values))
    return values

# Function for extracting features based on the concentration of a historical sample and a younger sample
# The function finds spatially aligning pixels and extracts the absolute and relative change.
# The concentration of the youngest data is stored as well.
# Spatially aligning the data is done by looping over the younger samples and
# comparing it with all the historical samples, looking for the shortest distance.
def compare_different_days(values_now, values_prev):
    # The threshold is the maximum distance between pixels that is considered aligned
    threshold = 3.0 # maybe need to change
    # List of features
    features = []
    # Loop over all the instances of the younger data
    for i in range(len(values_now)):
        # Initialize the smallest distance found with inf
        smallest_distance = float("Inf")
        print(i)
        # Loop over all instances of the historical data
        for j in range(len(values_prev)):
            # Calculating the distance
            distance = calc_distance((values_now[i][0], values_now[i][1]), (values_prev[j][0], values_prev[j][1]))
            if(distance < smallest_distance):
                smallest_distance = distance
                coordinate = j
        # When done finding the smallest distance, check if it's smaller than the threshold
        if(smallest_distance <= threshold):
            # Take the delta
            delta_ch4 = values_now[i][2] - values_prev[coordinate][2]
            # We discriminate all instances where the concentration is reduced, these are not leakages
            # Therefore, if delta is smaller than 0, make it 0. In that case, also make the ratio 1
            if(delta_ch4 < 0):
                delta_ch4 = 0
                ratio_ch4 = 1
            # Else, calculate the actual ratio
            else:
                ratio_ch4 = values_now[i][2] / values_prev[coordinate][2]
            # Calculate the average location of the two samples
            average_lat = (values_now[i][0] + values_prev[coordinate][0]) / 2
            average_lon = (values_now[i][1] + values_prev[coordinate][1]) / 2
            # Store the features
            features.append([average_lat, average_lon, delta_ch4, ratio_ch4, values_now[i][2]])
    return features

if __name__ =='__main__':
    # Infinite for loop waiting for new messages
    for message in consumer:
        print("Receiving a message")
        # Decode incoming message
        whole_message = message.value.decode('ascii')
        # Convert from json to dict.
        data = json.loads(whole_message)
        # Unpacking the longitude and latitude bounding box
        lat_constraints = data['lat_constraints']
        lon_constraints = data['lon_constraints']
        # Unpacking all the adresses corresponding to the bounding box
        adress_list_now = data['adress_list_now']
        adress_list_prev = data['adress_list_prev']
        print("Extracting data within boundries for current day")
        values_now = remove_masked_data(lat_constraints, lon_constraints, adress_list_now)
        print("Extracting data within boundries for previous day")
        values_prev = remove_masked_data(lat_constraints, lon_constraints, adress_list_prev)
        print('len(values_now', len(values_now))
        print('len(values_prev)', len(values_prev))
        # This is removing the overlap in data from the same day
        print("Overlapping data from current day")
        values_now = overlap_cluster_of_days(values_now)
        print("Overlapping data from previous day")
        values_prev = overlap_cluster_of_days(values_prev)
        # This code is checking the delta in ch4 concentrations between two days
        print("Extracting ch4 changes for the two days")
        features = compare_different_days(values_now, values_prev)
        print('Length of the features', len(features))
        with h5py.File('/home/mt/Hanze/Tropomi/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s/data/extracted_features.hdf5', 'w') as f:
            dset = f.create_dataset("data", data=features)
        print('Feature extraction completed!')

        #   anormaly detection
        # building the feature_array
        # features.append([average_lat, average_lon, delta_ch4, ratio_ch4, values_now[i][2]])
        lat = features[:,0].reshape(-1, 1)
        lon = features[:,1].reshape(-1, 1)
        delta_ch4 = features[:,2].reshape(-1, 1)
        ratio_ch4 = features[:,3].reshape(-1, 1)
        concentration = features[:,4].reshape(-1, 1)
        min_max_scaler = MinMaxScaler()
        nor_delta = min_max_scaler.fit_transform(delta_ch4)
        nor_ratio = min_max_scaler.fit_transform(ratio_ch4)
        nor_concentration = min_max_scaler.fit_transform(concentration)
        # nonamlized features 
        feature_array = np.hstack((np.hstack((nor_delta, nor_ratio)), nor_concentration))

        
        # density-based clustering and distance-based K nearest neighbor
        dbscan = DBSCAN(eps=0.3, min_samples=10).fit(feature_array) 
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        label = dbscan.labels_
        indices_dbscan = [i for i, x in enumerate(label) if x == -1] # getting the index of the outliers

        # distance-based K nearest neighbor
        contamination = 0.01 #  the proportion of outliers
        n_neighbors = 5 # number of NN
        method = 'mean' #largest': use the distance to the kth neighbor as the outlier score
                        # 'mean': use the average of all k neighbors as the outlier score
                        # 'median': use the median of the distance to k neighbors as the outlier score
        algorithm = 'auto' # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}            
        clf = KNN(contamination=contamination, n_neighbors=n_neighbors, method=method, algorithm=algorithm)
        clf.fit(feature_array) # (n_samples, n_features) (5783, 3)
        y_pred = clf.predict(feature_array) # 0 is inlier, 1 is outliers
        indices_knn = [i for i, x in enumerate(y_pred) if x == 1] # getting the index of the outliers

        # outliers' coordinates are: lat,lon: (features[indices_knn][0],features[indices_knn][1]) 
        # lat, lon: (features[indices_dbscan][0],features[indices_dbscan][1])
        outlier_knn_coordinates = (features[indices_knn][0],features[indices_knn][1]) 
        outliers_dbscan_coordinates = (features[indices_dbscan][0],features[indices_dbscan][1])

        # maybe plot on basemap

        #     m = Basemap(width=100000, height=100000, resolution='i', projection='stere', lat_ts=40, lat_0=38.6,
        #                 lon_0=54.2)
        #     m.drawcoastlines(linewidth=0.5)
        #     m.drawstates()
        #     m.drawcountries()
        #     m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
        #     m.drawmeridians(np.arange(-180, 181., 10.), labels=[0, 0, 0, 1], fontsize=10)
        #
        #     vmin1 = np.min(data)
        #     vmax1 = np.max(data)
        #     # draw CH4 on the map
        #     m.pcolor(lon, lat, data, latlon=True, vmin=vmin1, vmax=vmax1, cmap='jet')
        #     cb = m.colorbar()
        #     fig_title = file_adress[-66:-58]
        # #     print(fig_title)
        #     plt.title(fig_title)
        #
        #     fig = plt.gcf()
        #     # saves as a png
        #     pngfile = '{0}.png'.format(file_adress[:-3])
        #     fig.savefig(pngfile, dpi=750)
        #     # Show the plot window.
        #     # plt.show()
        #     # close the file
        #     plt.clf()
        #     plt.cla()
        #     file.close()
        #     print('Figure saved.')