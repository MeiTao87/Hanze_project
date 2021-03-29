# TROPOMI Project

## Reading data

### Reading from google earthengine

* note that the "Sentinel-5P OFFL CH4: Offline Methane" data is Level 3.
  * https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_CH4

* Libray needed: https://developers.google.com/earth-engine/python_install



## Downloading Tropomi data:

* Libray to download TROPOMI data: pytropomi https://github.com/bugsuse/pytropomi







### Reading TROPOMI data using netCDF4

* Libray needed: 1.netCDF4; 2. mpl_toolkits.basemap
* " XXXX_CH4_file.groups['PRODUCT'].variables['qa_value'] [0] " is the quality assurance value, it is recommended to use TROPOMI CH4 data associated with a quality assurance value qa_value > 0.5. [1]
* example function can do that can be found in validation_of_the_paper "filter_ch4_qa()"

### Visualize CH4 data

* Software: PanoplyJ https://www.giss.nasa.gov/tools/panoply/

* ![Screenshot from 2020-05-14 18-09-20](/Screenshot from 2020-05-14 18-09-20.png)


### Research data: Methods to read satellite images

* https://www.researchgate.net/publication/220413797_An_Unsupervised_Artificial_Neural_Network_Method_for_Satellite_Image_Segmentation

* https://www.academia.edu/4061334/Processing_of_Satellite_Image_using_Digital_Image_Processing  


### Ref

[1] https://sentinel.esa.int/documents/247904/3541451/Sentinel-5P-Methane-Product-Readme-File Part 3.1

* video about Sentinel-5P https://www.youtube.com/watch?v=gMJ4fSgDE1g 
