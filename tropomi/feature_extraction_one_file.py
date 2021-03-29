import numpy as np
from mpl_toolkits.basemap import Basemap
from collections import namedtuple
import matplotlib.pyplot as plt
import sys
from netCDF4 import Dataset
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D


def filter_ch4_qa(file_ch4, qa_threashold=0.5):
    groups = file_ch4.groups['PRODUCT']
    qa_value = groups.variables['qa_value'][0]
    ch4 = groups.variables['methane_mixing_ratio'][0]
    ch4_mask = ma.masked_less(qa_value, qa_threashold).mask
    output_ch4 = ma.masked_array(ch4, mask=ch4_mask)
    return output_ch4

FILE_NAME = '/home/mt/Hanze/Tropomi/tropomi-group-repo-s-group-legend-ab-c-d-k-m-s/data/adress/S5P_OFFL_L2__CH4____20200606T034446_20200606T052616_13716_01_010302_20200607T203727.nc'
file = Dataset(FILE_NAME, 'r')  # Dataset is a netCDF4 method used to open .nc files
ds = file
lat = ds.groups['PRODUCT'].variables['latitude'][0][:][:]  # Gets latitude array
lon = ds.groups['PRODUCT'].variables['longitude'][0][:][:]  # Gets longitude array
data = filter_ch4_qa(ds)  # data will be the array of pollutants concentrations

# if mask is True change the "data" to 0
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if data.mask[row, col]:
            data.data[row, col] = 0

# data is numpy masked array
# print(data.shape) # 4172, 215

#We are going to use windows of the following size
size = 5
window_size = (size, size)
#Horizontally it fits the following number of times:
h_slices = data.shape[0]//window_size[0]
#This is what's left horizontally:
h_rest = data.shape[0]%window_size[0]
#How much the horizontal windows are offset
h_offset = h_rest//2
#Vertically it fits the following number of times:
v_slices = data.shape[1]//window_size[1]
#This is what's left vertically:
v_rest = data.shape[1]%window_size[1]
#How much the vertical windows are offset
v_offset = v_rest//2

sliced_data = np.zeros((h_slices, v_slices, window_size[0], window_size[1]))
for i in range(h_slices):
    for j in range(v_slices):
        #Take a slice and store it in the sliced dataset
        sliced_data[i, j] = data[i*window_size[0]+h_offset:(i+1)*window_size[0]+h_offset, j*window_size[1]+v_offset:(j+1)*window_size[1]+v_offset]


res_max = []
res_sum = []
for i in range(sliced_data.shape[0]):
    for j in range(sliced_data.shape[1]):
        res_max.append(np.max(sliced_data[i, j, :, :]))
        res_sum.append(np.sum(sliced_data[i, j, :, :]))

res_max_array = np.asarray(res_max).reshape(sliced_data.shape[0],sliced_data.shape[1])
res_sum_array = np.asarray(res_sum).reshape(sliced_data.shape[0],sliced_data.shape[1])

plt.subplot(1,2,1)
plt.imshow(res_max_array)

plt.subplot(1,2,2)
plt.imshow(res_sum_array)

plt.show()