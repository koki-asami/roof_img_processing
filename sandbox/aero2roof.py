#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import X
import csv
from numba import jit
import numpy as np
from osgeo import gdal, gdalconst #, gdal_array, osr
from PIL import Image
import pandas as pd
from pathlib import Path
import os
from glob import glob


ORTHOGONAL_COORD_SYSTEM_ID = 2 # Kyoshu are

ORIGIN_LON_LAT = {  #https://www.gsi.go.jp/LAW/heimencho.html
    1: [129.30, 33.0],
    2: [131.0, 33.0],
}

IMG_WIDTH = 10000
IMG_HEIGHT = 7500

OUT_IMG_SIZE = 512

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / 'data'

COORD_FILE = 'c20160401建築物の外周線_FeatureVert_xy.csv'  # 'c20160401建築物の外周線_FeatureVert_XYTableToPoint.csv'  # 'c20160401建築物の外周線_FeatureVert_xy.csv'  # c02KD67120220401建築物_Merged_xy.csv'
COORD_FILE_PATH = DATA_DIR / COORD_FILE

AERO_IMG_DIR = DATA_DIR / '20cm高解像度_A_地区'  # '20cm高解像度_A_地区'  # '10cm高解像度'

SAVE_DIR = ROOT_DIR / '20cm高解像度_A_地区_512_new'  # '20cm高解像度_A_地区'  # 'kumamoto_fault_area+_A(C1-c3)_10cm'
SAVE_DIR.mkdir(exist_ok=True)

def get_location_point(path):
    with open(path, 'r') as f:
        '''
        Args:
            data (list): parameters to apply affine transform for raster coordinares.
                        see http://cse.naro.affrc.go.jp/takahasi/gisdata/avg/worldraster.htm
                        data[0]: Length of a pixel along the x-axis.
                        data[1]: Rotation parameter of rows.
                        data[2]: Rotation parameter of colunms.
                        data[3]: Length of a pixel along the y-axis.
                        data[4]: Center of x coordimate of top-left pixel.
                        data[5]: Center of y coordimate of top-left pixel.
                        
        '''
        data = [float(line.split('\n')[0]) for line in f.readlines()]
    
    scale = data[0]
    x0 = data[4]
    y1 = data[5]
    x1 = x0 + IMG_WIDTH*scale
    y0 = y1 + IMG_HEIGHT*(-scale)
    location_point = [x0, x1, y0, y1]
    return location_point

def open_image(path):
    src = gdal.Open(path, gdalconst.GA_ReadOnly)
    h1 = np.asarray(src.GetRasterBand(1).ReadAsArray())
    h2 = np.asarray(src.GetRasterBand(2).ReadAsArray())
    h3 = np.asarray(src.GetRasterBand(3).ReadAsArray())
    img = [h1, h2, h3]
    return img

@jit(nopython=True, cache=True)
def intersection(apx, apy, bpx, bpy, cpx, cpy, dpx, dpy):
    s1 = ((apx - cpx) * (bpy - cpy)) + ((bpx - cpx) * (cpy - apy))
    s2 = ((apx - dpx) * (bpy - dpy)) + ((bpx - dpx) * (dpy - apy))
    return s1 * s2

@jit(nopython=True, cache=True)
def lat2pix(lat, lat_up, lat_down):
    return (lat_up - lat) / (lat_up - lat_down) * IMG_HEIGHT

@jit(nopython=True, cache=True)
def lon2pix(lon, lon_l, lon_r):
    return (lon - lon_l) / (lon_r - lon_l) * IMG_WIDTH

def image_judge(location_points, polygon):
    flag=False
    for lon, lat in polygon:
        if location_points[0] < lon < location_points[1] and location_points[2] < lat < location_points[3]:
            flag = True
    return flag

def make_circle_list(polygon_x, polygon_y, circle_list=[], c=[], k_list=[], k=[], e_list=[], e=[]):
    circle_list = []
    c = []
    k_list = []
    k = []
    e_list = []
    e = []
    for i in range(len(polygon_x)):
        if c == []:
            c.append(i)
            k.append(polygon_x[i])
            e.append(polygon_y[i])
        else:
            if not (polygon_x[i] == polygon_x[c[0]] and polygon_y[i] == polygon_y[c[0]]):
                c.append(i)
                k.append(polygon_x[i])
                e.append(polygon_y[i])
            else:
                c.append(i)
                k.append(polygon_x[i])
                e.append(polygon_y[i])
                circle_list.append(c)
                k_list.append(k)
                e_list.append(e)
                c = []
                k = []
                e = []
    return circle_list

def imaging(id, img, polygon_x, polygon_y, save_dir=None):
    lon_min = min(polygon_x)
    lon_max = max(polygon_x)
    lat_min = min(polygon_y)
    lat_max = max(polygon_y)
    imagelistr = []
    imagelistg = []
    imagelistb = []
    img_w = lon_max - lon_min + 1
    img_h = lat_max - lat_min + 1
    circle_lisit = make_circle_list(polygon_x, polygon_y)
    for x3 in range(lat_min, lat_max + 1):
        for x4 in range(lon_min, lon_max + 1):
            x3 = x3 + 0.5
            intersection_num = 0
            for c in circle_lisit:
                for n in range(0, len(c) - 1):
                    if (
                        intersection(x4, x3, lon_max, x3, polygon_x[c[n]], polygon_y[c[n]], polygon_x[c[n + 1]], polygon_y[c[n + 1]]) <= 0
                            ) and (
                        intersection(polygon_x[c[n]], polygon_y[c[n]], polygon_x[c[n + 1]], polygon_y[c[n + 1]], x4, x3, lon_max, x3) <= 0):
                        intersection_num += 1
            x3 = int(x3 - 0.5)
            if intersection_num % 2 == 1 and x3 < IMG_HEIGHT and x4 < IMG_WIDTH:
                imagelistr.append(img[0][x3, x4])
                imagelistg.append(img[1][x3, x4])
                imagelistb.append(img[2][x3, x4])
            else:
                imagelistr.append(0)
                imagelistg.append(0)
                imagelistb.append(0)

    if sum(imagelistr) / len(imagelistr) == 0:  # skip if image s blanc
        print(f'roof_{id} cannot be found...')
    else:
        imagelistr11 = np.asarray(imagelistr)
        imagelistg11 = np.asarray(imagelistg)
        imagelistb11 = np.asarray(imagelistb)
        imagelistr2 = np.reshape(imagelistr11, (img_h, img_w))
        imagelistg2 = np.reshape(imagelistg11, (img_h, img_w))
        imagelistb2 = np.reshape(imagelistb11, (img_h, img_w))
        img = Image.new('RGB', (img_w, img_h), (255, 255, 255))
        for y in range(0, img_w):
            for x in range(0, img_h):
                img.putpixel((y, x), (imagelistr2[x, y], imagelistg2[x, y], imagelistb2[x, y]))
        img_resized = img.resize((OUT_IMG_SIZE, OUT_IMG_SIZE), resample=Image.BICUBIC)
        img_tmp = imagelistr2 + imagelistg2 + imagelistb2
        try:
            middle = img_tmp[int(img_h/2)-1][int(img_w/2)] \
                    + img_tmp[int(img_h/2)][int(img_w/2)] \
                    + img_tmp[int(img_h/2)-1][int(img_w/2)-1] \
                    + img_tmp[int(img_h/2)][int(img_w/2)] \
                    + img_tmp[int(img_h/2)+1][int(img_w/2)] \
                    + img_tmp[int(img_h/2)][int(img_w/2)+1] \
                    + img_tmp[int(img_h/2)+1][int(img_w/2)+1] \
                    + img_tmp[int(img_h/2)+1][int(img_w/2)-1] \
                    + img_tmp[int(img_h/2)-1][int(img_w/2)+1]
        except:
            middle = 0
        if save_dir is not None and middle/9 not in (0, 765):
            print(f'Saving roof_{id}...')
            img_resized.save(str(save_dir) + '/' + str(id) + '.bmp')

def main():
    # Load polygon coordinates as DATAFRAME
    df = pd.read_csv(COORD_FILE_PATH, usecols=[2,3,4]) # 1,2,3 #without id
    # Exstract unique building_id from df
    ids = df.drop_duplicates(subset = ['ORIG_FID'])['ORIG_FID'].to_list()
    for img_path in sorted(AERO_IMG_DIR.glob('*.jpg')):
        print('\n', img_path)
        jgw_path = Path(img_path).with_suffix('.jgw')
        location_points = get_location_point(jgw_path)
        img = open_image(str(img_path))
        for id in ids:
            selected_df = df.loc[df['ORIG_FID']==id]
            polygon = [[float(row['POINT_X']), float(row['POINT_Y'])] for idx, row in selected_df.iterrows()]

            if image_judge(location_points, polygon):  # judge whether the selected polygon is in the target image
                # convert coordinates of polygon from lonlat to pixel
                polygon_x = [int(lon2pix(p[0], location_points[0], location_points[1])) for p in polygon]
                polygon_y = [int(lat2pix(p[1], location_points[3], location_points[2])) for p in polygon]
                
                # crop roof image from aerial photo
                imaging(id, img, polygon_x, polygon_y, save_dir=SAVE_DIR)

                # remove cropped roof
                ids.remove(id)
            

if __name__ == '__main__':
    main()