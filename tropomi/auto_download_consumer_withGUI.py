from kafka import KafkaConsumer
import json
import numpy as np
from mpl_toolkits.basemap import Basemap
from collections import namedtuple
import matplotlib.pyplot as plt
import sys
from netCDF4 import Dataset
import numpy.ma as ma
from shapely.geometry import Polygon
import cv2
from mpl_toolkits.basemap import Basemap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
# import geopy.distance
from math import sin, cos, sqrt, atan2, radians
R = 6373.0
import h5py
from tkinter import *
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from pyod.models.cblof import CBLOF  # Cluster-based Local Outlier Factor
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from scipy.stats import norm

window=Tk()

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
    threshold = 4.9
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
    threshold = 4.9
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
            # features are in following format:
            #             [ [ [lat] [lon] [delta ch4] [ratio ch4] [ch4 value] ] 
            #               [ [lat] [lon] [delta ch4] [ratio ch4] [ch4 value] ]
            # features =    [ [lat] [lon] [delta ch4] [ratio ch4] [ch4 value] ]
            #               [ [lat] [lon] [delta ch4] [ratio ch4] [ch4 value] ]
            #               [ [lat] [lon] [delta ch4] [ratio ch4] [ch4 value] ] ]
    return features

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
    with h5py.File("C:/Users/Samir/Documents/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s-master/extracted_features.hdf5", 'w') as f:
        dset = f.create_dataset("data", data=features)
    print('Feature extraction completed!')

    # anormaly detection
    # building the feature_array
    # features.append([average_lat, average_lon, delta_ch4, ratio_ch4, values_now[i][2]])
    features = np.array(features)
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
    # DBSCAN: density-based spacial clustering anomaly detection
    dbscan = DBSCAN(eps=0.3, min_samples=10).fit(feature_array) 
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    label = dbscan.labels_
    indices_dbscan = [i for i, x in enumerate(label) if x == -1] # getting the index of the outliers

    def calculate_WSS(points, kmax):
        sse = []
        for k in range(1, kmax+1):
            kmeans = KMeans(n_clusters = k).fit(points)
            centroids = kmeans.cluster_centers_
            pred_clusters = kmeans.predict(points)
            curr_sse = 0
            
            # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
            for i in range(len(points)):
                curr_center = centroids[pred_clusters[i]]
                curr_sse += (points[i, 0] - curr_center[0]) * 2 + (points[i, 1] - curr_center[1]) * 2
            sse.append(curr_sse)
        return sse

    with h5py.File("C:/Users/Samir/Documents/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s-master/extracted_features.hdf5", 'r') as f:
        data = f['data']
        print(len(data))
        print(type(data))
        print((data[0]).shape)
        print(data)
        # 3D plot of the features
        '''fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs = data[:,2] # delta
        ys = data[:,3] # ratio
        zs = data[:,4] # concentrations
        ax.scatter(xs, ys, zs)

        ax.set_xlabel('Delta')
        ax.set_ylabel('Ratio')
        ax.set_zlabel('Concentration')
        plt.title('3D plot of three features')
        plt.show()
        
        # # delta_ch4
        plt.subplot(3, 2, 1)
        plt.title('Histogram of CH4 difference', fontsize=10)
        plt.xlabel('CH4 difference', fontsize=8)
        plt.ylabel('Numer of data in the bin', fontsize=8)
        bin_delta = np.arange(0, np.max(data[:,2]), 5)
        plt.hist(data[:,2], bins=bin_delta)

        plt.subplot(3, 2, 2)
        plt.title('Histogram of CH4 difference', fontsize=10)
        plt.xlabel('CH4 difference', fontsize=8)
        # plt.ylabel('Numer of data in the bin', fontsize=8)
        plt.hist(data[:,2], bins=bin_delta[5:])
        # plt.yscale('log')

        # concentrations
        plt.subplot(3, 2, 3)
        plt.title('Histogram of CH4 concentration', fontsize=10)
        plt.xlabel('CH4 concentration', fontsize=8)
        plt.ylabel('Numer of data in the bin', fontsize=8)
        bin_concentration = np.arange(np.min(data[:,-1]), np.max(data[:,-1]), 20)
        plt.hist(data[:,-1], bins=bin_concentration)

        plt.subplot(3, 2, 4)
        plt.title('Histogram of CH4 concentration', fontsize=10)
        plt.xlabel('CH4 concentration', fontsize=8)
        # plt.ylabel('Numer of data in the bin', fontsize=8)
        bin_concentration = np.arange(np.min(data[:,-1]), np.max(data[:,-1]), 20)
        plt.hist(data[:,-1], bins=bin_concentration[-7:])

        # ratio
        plt.subplot(3, 2, 5)
        plt.title('Histogram of CH4 ratio', fontsize=10)
        plt.xlabel('CH4 ratio', fontsize=8)
        plt.ylabel('Numer of data in the bin', fontsize=8)
        bin_ratio = np.arange(np.min(data[:,-2]), np.max(data[:,-2]), 0.008)
        plt.hist(data[:,-2], bins=bin_ratio)
        # plt.yscale('log')
        
        plt.subplot(3, 2, 6)
        plt.title('Histogram of CH4 ratio', fontsize=10)
        plt.xlabel('CH4 ratio', fontsize=8)
        # plt.ylabel('Numer of data in the bin', fontsize=8)
        bin_ratio = np.arange(np.min(data[:,-2]), np.max(data[:,-2]), 0.008)
        plt.hist(data[:,-2], bins=bin_ratio[1:])
        
        plt.tight_layout()
        plt.show()
        '''
        lon = data[:,1].reshape(-1, 1)
        lat = data[:,0].reshape(-1, 1)
        delta_ch4 = data[:,2].reshape(-1, 1)
        ratio_ch4 = data[:,3].reshape(-1, 1)
        concentration = data[:,4].reshape(-1, 1)
        # print(delta_ch4.shape) # (5783,)
        # print(np.argmax(delta_ch4))
        # print(np.argmax(ratio_ch4))
        # print(np.argmax(concentration))
        
        min_max_scaler = MinMaxScaler()
        nor_delta = min_max_scaler.fit_transform(delta_ch4)
        nor_ratio = min_max_scaler.fit_transform(ratio_ch4)
        nor_concentration = min_max_scaler.fit_transform(concentration)
        # nonamlized features 
        feature_array = np.hstack((np.hstack((nor_delta, nor_ratio)), nor_concentration))

        # choose the optimal K
        # sse = calculate_WSS(feature_array,10)
        # print(sse)
        # plt.plot(sse)
        # plt.show()
        print('KNN')
        # # distance-based K nearest neighbor
        contamination = 0.1 #  the proportion of outliers
        n_neighbors = 5 # number of NN
        method = 'mean' #largest': use the distance to the kth neighbor as the outlier score
                        # 'mean': use the average of all k neighbors as the outlier score
                        # 'median': use the median of the distance to k neighbors as the outlier score
        algorithm = 'auto' # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}            
        clf = KNN(contamination=contamination, n_neighbors=n_neighbors, method=method, algorithm=algorithm)
        clf.fit(feature_array) # (n_samples, n_features) (5783, 3)
        y_pred = clf.predict(feature_array) # 0 is inlier, 1 is outliers
        indices_knn = [i for i, x in enumerate(y_pred) if x == 1] # getting the index of the outliers
        sum_delta, sum_ratio, sum_concentration = 0, 0, 0
        if len(indices_knn) != 0:
            for i in indices_knn:
                sum_delta += delta_ch4[i]
                sum_ratio += ratio_ch4[i]
                sum_concentration += concentration[i]
                print('LAT, LON', lat[i], lon[i])
            aver_delta = sum_delta / len(indices_knn)
            aver_ratio = sum_ratio / len(indices_knn)
            aver_concentration = sum_concentration / len(indices_knn)
            print(aver_delta, aver_ratio, aver_concentration)
            print('len_indices', len(indices_knn))    
            #print(indices_knn)
        else:
            print("no outliers detected by KNN method")
        outlier_knn_lats = []
        outlier_knn_lons = []
        outlier_knn_concentration = []
        for i in range(len(indices_knn)):
            outlier_knn_lats.append(features[i][0])
            outlier_knn_lons.append(features[i][1])
            outlier_knn_concentration.append(features[i][-1])
        outlier_knn_lats = np.asarray(outlier_knn_lats)
        outlier_knn_lons = np.asarray(outlier_knn_lons)
        # outlier_knn_concentration = np.asarray(outlier_knn_concentration)

        # outlier_knn_lat, outlier_knn_lon = np.meshgrid(outlier_knn_lats, outlier_knn_lons)
        # outlier_knn_concentrations = np.zeros(outlier_knn_lat.shape)
        # for i in range(len(outlier_knn_lats)):
        #     outlier_knn_concentrations[i][i] = outlier_knn_concentration[i]
    
        print('DBSCAN')
        db = DBSCAN(eps=0.3, min_samples=10).fit(feature_array) 
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        label = db.labels_
        print(label)
        indices = [i for i, x in enumerate(label) if x == -1] # getting the index of the outliers
        sum_delta, sum_ratio, sum_concentration = 0, 0, 0
        for i in indices:
            sum_delta += delta_ch4[i]
            sum_ratio += ratio_ch4[i]
            sum_concentration += concentration[i]
            print('LAT, LON', lat[i], lon[i])
        aver_delta = sum_delta / len(indices)
        aver_ratio = sum_ratio / len(indices)
        aver_concentration = sum_concentration / len(indices)
        print(aver_delta, aver_ratio, aver_concentration)
        print('len_indices', len(indices))    
        print(indices)

        outlier_dbscan_lats = []
        outlier_dbscan_lons = []
        outlier_dbscan_concentration = []
        for i in range(len(indices)):
            outlier_dbscan_lats.append(data[indices[i]][0])
            outlier_dbscan_lons.append(data[indices[i]][1])
            outlier_dbscan_concentration.append(data[indices[i]][-1])
        outlier_dbscan_lats = np.asarray(outlier_dbscan_lats)
        outlier_dbscan_lons = np.asarray(outlier_dbscan_lons)
        #m = Basemap(resolution='i', projection='cyl', lat_ts=40, lat_0=37, lon_0=113)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1 = Basemap(resolution='i', projection='cyl', lat_ts=40, lat_0=37, lon_0=113)
        x_kn, y_kn = ax1(outlier_knn_lons, outlier_knn_lats)
        x_db, y_db = ax1(outlier_dbscan_lons, outlier_dbscan_lats)
        ax1.drawcoastlines(linewidth=0.5)
        ax1.drawstates()
        ax1.drawcountries()
        ax1.scatter(x_kn, y_kn, marker='o',color='b', s=7, label='possible leakage detected using KNN')
        ax1.scatter(x_db, y_db, marker='*',color='r', s=7, label='possible leakage detected using DBSCAN')
        plt.legend(loc='lower center', shadow=False, fontsize='medium')
        plt.show()
