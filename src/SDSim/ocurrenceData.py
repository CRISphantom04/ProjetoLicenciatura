import rasterio as rio
from rasterio.plot import show
import numpy as np
import pandas as pd

# Function to get the occurence values of the species
def ocurrence_data(species_name, filename, limit):

    df = pd.DataFrame()

    # Open the raster file    
    raster_file = rio.open(filename)

    #Read  the first band  into a numpy array
    raster_array = raster_file.read(1)


    final_csv = filename + ".csv"
    f = open(final_csv, 'w')
    
    species_list = []
    latitude_list = []
    longitude_list = []

    for row, col in np.ndindex(raster_array.shape):
        if raster_array[row, col] / 100 >= limit:
            latitude, longitude = raster_file.xy(row, col)
            species_list.append(species_name)
            latitude_list.append(latitude)
            longitude_list.append(longitude)

    df['Species_name'] = species_list
    df['Latitude'] = latitude_list
    df['Longitude'] = longitude_list

    df.to_csv(final_csv, index=False)
    