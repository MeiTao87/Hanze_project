import ee
import ee.mapclient
ee.Authenticate()
ee.Initialize()

# load the image collection
collection = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4").select('CH4_column_volume_mixing_ratio_dry_air').filterDate('2019-06-01', '2019-07-16')

# visualization parameters
band_viz = {
  'min': 1750,
  'max': 1900,
  'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red']
}

ee.mapclient.centerMap(0.0, 0.0, 2) # set center
ee.mapclient.addToMap(collection.mean(), band_viz, 'S5P CH4') # add CH4 to map

