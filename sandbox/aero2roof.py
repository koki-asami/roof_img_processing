#!/usr/bin/env python
# -*- coding: utf-8 -*-

from re import X
import csv
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
IMG_HIGHT = 7500

DATA_DIR = Path('./')

COORD_FILE = 'Export_output.csv'  # "c02KD67120220401建築物_Merged_xy.csv"
COORD_FILE_PATH = DATA_DIR / COORD_FILE

AERO_IMG_DIR = DATA_DIR / "10cm高解像度"

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
    y0 = y1 + IMG_HIGHT*(-scale)
    location_point = [x0, x1, y0, y1]
    return location_point

def open_image(path):
    src = gdal.Open(path, gdalconst.GA_ReadOnly)
    h1 = np.asarray(src.GetRasterBand(1).ReadAsArray())
    h2 = np.asarray(src.GetRasterBand(2).ReadAsArray())
    h3 = np.asarray(src.GetRasterBand(3).ReadAsArray())
    img = [h1, h2, h3]
    return img

# print("start")
# ar, ag, ab = open_image('20160417-02KD661-a20.jpg')
# br, bg, bb = open_image('20160417-02KD662-a20.jpg')
# cr, cg, cb = open_image('20160415-02KD663-a20.jpg')
# dr, dg, db = open_image('20160417-02KD664-a20.jpg')
# color_list = np.asarray([[ar, ag, ab], [br, bg, bb],[cr, cg, cb], [dr,dg,db]])
# ar, ag, ab, br, bg, bb, cr, cg, cb, dr, dg, db = [], [], [], [], [], [], [], [], [],[],[],[]
# print("finish")

def intersection(apx, apy, bpx, bpy, cpx, cpy, dpx, dpy): # intersection
    s1 = ((apx - cpx) * (bpy - cpy)) + ((bpx - cpx) * (cpy - apy))
    s2 = ((apx - dpx) * (bpy - dpy)) + ((bpx - dpx) * (dpy - apy))
    return s1 * s2

def lat2pix(lat, lat_up, lat_down):
    return (lat_up - lat) / (lat_up - lat_down) * IMG_HIGHT

def lon2pix(lon, lon_l, lon_r):
    return (lon - lon_l) / (lon_r - lon_l) * IMG_WIDTH


# def koten(apx, apy, bpx, bpy, cpx, cpy, dpx, dpy): # intersection
#     s1 = ((apx - cpx) * (bpy - cpy)) + ((bpx - cpx) * (cpy - apy))
#     s2 = ((apx - dpx) * (bpy - dpy)) + ((bpx - dpx) * (dpy - apy))
#     return s1 * s2
# def idotopix(ido,upido,downido):
#     return (upido-ido)/(upido-downido)*7500
# def keidotopix(keido,leftkeido,rightkeido):
#     return (keido-leftkeido)/(rightkeido-leftkeido)*10000

# #西経度、東経度、南緯度、北緯度
# location_point_of_image1 = [-15999.9, -13999.9, -19500.1, -18000.1]
# location_point_of_image2 = [-13999.9, -11999.9, -19500.1, -18000.1]
# location_point_of_image3 = [-15999.9, -13999.9, -21000.1, -19500.1]
# location_point_of_image4 = [-13999.9, -11999.9, -21000.1, -19500.1]
# location_point_list = np.asarray([location_point_of_image1, location_point_of_image2, location_point_of_image3, location_point_of_image4])


def image_judge(location_points, polygon):
    flag = False
    for lon, lat in polygon:
        if location_points[0] < lon < location_points[1] and location_points[2] < lat < location_points[3]:
            flag = True
    return flag

def make_circle_list(listpk, listpi):
    circle_list = []
    c = []
    k_list = []
    k = []
    e_list = []
    e = []
    for i in range(len(listpk)):
        if c == []:
            c.append(i)
            k.append(listpk[i])
            e.append(listpi[i])
        else:
            if not (listpk[i] == listpk[c[0]] and listpi[i] == listpi[c[0]]):
                c.append(i)
                k.append(listpk[i])
                e.append(listpi[i])
            else:
                c.append(i)
                k.append(listpk[i])
                e.append(listpi[i])
                circle_list.append(c)
                k_list.append(k)
                e_list.append(e)
                c = []
                k = []
                e = []
    return circle_list

def imaging(id, img, polygon_x, polygon_y):
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
            if intersection_num % 2 == 1 :
                imagelistr.append(img[0][x3, x4])
                imagelistg.append(img[1][x3, x4])
                imagelistb.append(img[2][x3, x4])
            else:
                imagelistr.append(0)
                imagelistg.append(0)
                imagelistb.append(0)
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
    img_resize = img.resize((227, 227), resample=Image.BICUBIC)
    img_resize.save('F5/' + str(id) + "_" + str(n) + '.bmp')

def main():

    # Load polygon coordinates as DATAFRAME
    df = pd.read_csv(COORD_FILE_PATH, usecols=[0,1,2])  # [1,2,3])
    # Exstract unique building_id from df
    ids = df.drop_duplicates(subset = ['id'])['id'].to_list()
    imgs = ['20160417-02KD664-a20.jpg']
    for img_path in imgs: # sorted(AERO_IMG_DIR.glob('*.jpg')):
        print(img_path)
        jgw_path = Path(img_path).with_suffix('.jgw')
        location_points = get_location_point(jgw_path)
        img = open_image(str(img_path))
        for id in ids:
            selected_df = df.loc[df['id']==id]
            polygon = [[float(row['POINT_X']), float(row['POINT_Y'])] for idx, row in selected_df.iterrows()]

            if image_judge(location_points, polygon):  # judge whether the selected polygon is in the target image
                # convert coordinates of polygon from lonlat to pixel
                polygon_x = [int(lon2pix(p[0], location_points[0], location_points[1])) for p in polygon]
                polygon_y = [int(lat2pix(p[1], location_points[3], location_points[2])) for p in polygon]
                
                # crop roof image from aerial photo
                imaging(id, img, polygon_x, polygon_y)
            

if __name__ == '__main__':
    main()