import h5py
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from pyod.models.cblof import CBLOF  # Cluster-based Local Outlier Factor
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from mpl_toolkits.mplot3d import Axes3D

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
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
    sse.append(curr_sse)
  return sse

with h5py.File('/home/mt/Hanze/Tropomi/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s/data/extracted_features.hdf5', 'r') as f:
    data = f['data']
    print(len(data))

    # 3D plot of the features
    fig = plt.figure()
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
    feature_array = np.hstack((np.hstack((nor_delta, nor_ratio)), nor_concentration))
    # print(feature_array.shape) # (5783, 3)

    # choose the optimal K
    # sse = calculate_WSS(feature_array,10)
    # print(sse)
    # plt.plot(sse)
    # plt.show()
    
    print('KMeans')
    for K in range(5, 10):
        print(K)
        kmeans = KMeans(n_clusters=K, random_state=0).fit(feature_array)
        label = kmeans.labels_
        bins, hist = np.histogram(label, bins=np.arange(0, K+1, 1))
        smallest_cluster = np.argmin(bins)
        indices = [i for i, x in enumerate(label) if x == smallest_cluster]
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
        print()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = ['red', 'green', 'blue', 'magenta', 'cyan', 'black', 'yellow', 'pink', 'orange', 'lime', 'teal', 'purple', 'gold']
            
        xs = data[:,2] # delta
        ys = data[:,3] # ratio
        zs = data[:,4] # concentrations
        for i in range(len(label)):
            ax.scatter(xs[i], ys[i], zs[i], c=cmap[label[i]])

        ax.set_xlabel('Delta')
        ax.set_ylabel('Ratio')
        ax.set_zlabel('Concentration')
        title = 'Cluster using KMeans(K=' + str(K) +')'
        plt.title(title)
        plt.show()


    # Distance-based K Nearest Neighbor
    print()
    print('Distance-based K Nearest Neighbor')
    contamination = 0.01 #  the proportion of outliers
    n_neighbors = 5 # number of NN
    method = 'mean' #largest': use the distance to the kth neighbor as the outlier score
                    # 'mean': use the average of all k neighbors as the outlier score
                    # 'median': use the median of the distance to k neighbors as the outlier score
    algorithm = 'auto' # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}            
    clf = KNN(contamination=contamination, n_neighbors=n_neighbors, method=method, algorithm=algorithm)
    clf.fit(feature_array) # (n_samples, n_features) (5783, 3)
    scores_pred = clf.decision_function(feature_array) * -1
    y_pred = clf.predict(feature_array) # 0 is inlier, 1 is outliers
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    indices = [i for i, x in enumerate(y_pred) if x == 1] # getting the index of the outliers
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
    print()
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = ['red', 'green', 'blue', 'magenta', 'cyan', 'black', 'yellow', 'pink', 'orange', 'lime', 'teal', 'purple', 'gold']
        
    xs = data[:,2] # delta
    ys = data[:,3] # ratio
    zs = data[:,4] # concentrations
    for i in range(len(y_pred)):
        ax.scatter(xs[i], ys[i], zs[i], c=cmap[y_pred[i]])

    ax.set_xlabel('Delta')
    ax.set_ylabel('Ratio')
    ax.set_zlabel('Concentration')
    title = 'Distance-based K Nearest Neighbor'
    plt.title(title)
    plt.show()


    # Density-based K Nearest Neighbor -- LOF
    # Breunig et al. [1999, 2000] assign an anomaly score to a given data instance, known as Local Outlier Factor (LOF).
    print()
    print('Density-based K Nearest Neighbor')
    algorithm = 'auto' # percentage of outliers
    n_neighbors = 10 # number of clusters
    clf = LOF(n_neighbors=n_neighbors, algorithm=algorithm)
    clf.fit(feature_array) # (n_samples, n_features) (5783, 3)
    scores_pred = clf.decision_function(feature_array) * -1
    y_pred = clf.predict(feature_array) # 0 is inlier, 1 is outliers
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    indices = [i for i, x in enumerate(y_pred) if x == 1] # getting the index of the outliers
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
    print()
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = ['red', 'green', 'blue', 'magenta', 'cyan', 'black', 'yellow', 'pink', 'orange', 'lime', 'teal', 'purple', 'gold']
        
    xs = data[:,2] # delta
    ys = data[:,3] # ratio
    zs = data[:,4] # concentrations
    for i in range(len(y_pred)):
        ax.scatter(xs[i], ys[i], zs[i], c=cmap[y_pred[i]])
    ax.set_xlabel('Delta')
    ax.set_ylabel('Ratio')
    ax.set_zlabel('Concentration')
    title = 'Density-based K Nearest Neighbor'
    plt.title(title)
    plt.show()


    
    # CBLOF (Cluster-based Local Outlier Factor)
    print()
    print('CBLOF')
    outliers_fraction = 0.01 # percentage of outliers
    n_clusters = 5 # number of clusters
    clf = CBLOF(n_clusters=n_clusters, contamination=outliers_fraction, check_estimator=False, random_state=0)
    clf.fit(feature_array) # (n_samples, n_features) (5783, 3)
    scores_pred = clf.decision_function(feature_array) * -1
    y_pred = clf.predict(feature_array) # 0 is inlier, 1 is outliers
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    indices = [i for i, x in enumerate(y_pred) if x == 1] # getting the index of the outliers
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
    print()
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = ['red', 'green', 'blue', 'magenta', 'cyan', 'black', 'yellow', 'pink', 'orange', 'lime', 'teal', 'purple', 'gold']
        
    xs = data[:,2] # delta
    ys = data[:,3] # ratio
    zs = data[:,4] # concentrations
    for i in range(len(y_pred)):
        ax.scatter(xs[i], ys[i], zs[i], c=cmap[y_pred[i]])
    ax.set_xlabel('Delta')
    ax.set_ylabel('Ratio')
    ax.set_zlabel('Concentration')
    title = 'Cluster-based Local Outlier Factor'
    plt.title(title)
    plt.show()