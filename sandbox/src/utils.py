from osgeo import gdal, gdalconst
import numpy as np


def open_image(path):
    src = gdal.Open(path, gdalconst.GA_ReadOnly)
    h1 = np.asarray(src.GetRasterBand(1).ReadAsArray())
    h2 = np.asarray(src.GetRasterBand(2).ReadAsArray())
    h3 = np.asarray(src.GetRasterBand(3).ReadAsArray())
    img = [h1, h2, h3]
    return img