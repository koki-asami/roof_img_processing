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


DATA_DIR = Path('./')

COORD_FILE = "c02KD67120220401建築物_Merged_xy.csv"
COORD_FILE_PATH = DATA_DIR / COORD_FILE

AERO_IMG_DIR = DATA_DIR / "10cm高解像度"

def get_location_point(path):
    with open(path, 'r') as f:
        data = [float(line.split('\n')[0]) for line in f.readlines()]
    print(data)
    return 0

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

def koten(apx, apy, bpx, bpy, cpx, cpy, dpx, dpy):
    s1 = ((apx - cpx) * (bpy - cpy)) + ((bpx - cpx) * (cpy - apy))
    s2 = ((apx - dpx) * (bpy - dpy)) + ((bpx - dpx) * (dpy - apy))
    return s1 * s2

def idotopix(ido,upido,downido):
    return (upido-ido)/(upido-downido)*7500
def keidotopix(keido,leftkeido,rightkeido):
    return (keido-leftkeido)/(rightkeido-leftkeido)*10000

#西経度、東経度、南緯度、北緯度
location_point_of_image1 = [-15999.9, -13999.9, -19500.1, -18000.1]
location_point_of_image2 = [-13999.9, -11999.9, -19500.1, -18000.1]
location_point_of_image3 = [-15999.9, -13999.9, -21000.1, -19500.1]
location_point_of_image4 = [-13999.9, -11999.9, -21000.1, -19500.1]
location_point_list = np.asarray([location_point_of_image1, location_point_of_image2, location_point_of_image3, location_point_of_image4])

def image_judge(location_point_list, pi, pk):
    judge_array = np.ones((5))
    for n in range(0,len(pi)):
        point_judge = 4
        for m in range(0, len(location_point_list)):
            if location_point_list[m, 0] < pk[n] < location_point_list[m, 1] and location_point_list[m, 2] < pi[n] < location_point_list[m, 3]:
                point_judge = m
        judge_array[point_judge] = judge_array[point_judge] * 0
    judge = 4
    if (judge_array == np.asarray([0,1,1,1,1])).all():
        judge = 0
    elif (judge_array == np.asarray([1,0,1,1,1])).all():
        judge = 1
    elif (judge_array == np.asarray([1,1,0,1,1])).all():
        judge = 2
    elif (judge_array == np.asarray([1,1,1,0,1])).all():
        judge = 3
    return judge

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

def imaging(hr, hg, hb, listpk, listpi, x1, n):
    minkeidop = min(listpk)
    maxkeidop = max(listpk)
    minidop = min(listpi)
    maxidop = max(listpi)
    imagelistr = []
    imagelistg = []
    imagelistb = []
    imgyoko = maxkeidop - minkeidop + 1
    imgtate = maxidop - minidop + 1
    circle_lisit = make_circle_list(listpk, listpi)
    for x3 in range(minidop, maxidop + 1):
        for x4 in range(minkeidop, maxkeidop + 1):
            x3 = x3 + 0.5
            majiwari = 0
            for c in circle_lisit:
                for n in range(0, len(c) - 1):
                    if koten(x4, x3, maxkeidop, x3, listpk[c[n]], listpi[c[n]], listpk[c[n + 1]], listpi[c[n + 1]]) <= 0 and koten(
                            listpk[c[n]], listpi[c[n]], listpk[c[n + 1]], listpi[c[n + 1]], x4, x3, maxkeidop, x3) <= 0:
                        majiwari = majiwari + 1
            x3 = int(x3 - 0.5)
            if majiwari % 2 == 1 :
                imagelistr.append(hr[x3, x4])
                imagelistg.append(hg[x3, x4])
                imagelistb.append(hb[x3, x4])
            else:
                imagelistr.append(0)
                imagelistg.append(0)
                imagelistb.append(0)
    imagelistr11 = np.asarray(imagelistr)
    imagelistg11 = np.asarray(imagelistg)
    imagelistb11 = np.asarray(imagelistb)
    imagelistr2 = np.reshape(imagelistr11, (imgtate, imgyoko))
    imagelistg2 = np.reshape(imagelistg11, (imgtate, imgyoko))
    imagelistb2 = np.reshape(imagelistb11, (imgtate, imgyoko))
    img = Image.new('RGB', (imgyoko, imgtate), (255, 255, 255))
    for y in range(0, imgyoko):
        for x in range(0, imgtate):
            img.putpixel((y, x), (imagelistr2[x, y], imagelistg2[x, y], imagelistb2[x, y]))
    img_resize = img.resize((227, 227), resample=Image.BICUBIC)
    img_resize.save('F5/mashiki_kiban' + str(x1) + "_" + str(n) + '.bmp')

def delete_list(h, del_axis):
    for i in range(0,len(del_axis)):
        h = np.delete(h, 0, 0)
    return h

# polygonnub=183530
# iii=0
# area = []
# for x1 in range(0, polygonnub+1):
#     print(x1)
#     listpk = []
#     listpi = []
#     del_axis = []
#     for x2 in range(0, int(zz2.shape[0])):
#         if int(zz2[x2, 0]) == x1:
#             listpk.append(zz2[x2, 1])
#             listpi.append(zz2[x2, 2])
#             del_axis.append(x2)
#         else:
#             break
#     if not listpk==[]:
#         n = image_judge(location_point_list, listpi, listpk)
#         if not n == 4:
#             for i in range(0, len(listpk)):
#                 listpk[i] = int(keidotopix(listpk[i], location_point_list[n, 0], location_point_list[n, 1]))
#                 listpi[i] = int(idotopix(listpi[i], location_point_list[n, 3], location_point_list[n, 2]))
#             imaging(color_list[n, 0], color_list[n, 1], color_list[n, 2], listpk, listpi, x1, n)
#     zz2 = delete_list(zz2, del_axis)

def main():

    # Load polygon coordinates as DATAFRAME
    df = pd.read_csv(COORD_FILE_PATH, usecols=[1,2,3])
    # Exstract unique building_id from df
    ids = df.drop_duplicates(subset = ['id'])['id'].to_list()

    for id in ids[:2]:
        selected_df = df.loc[df['id']==id]
        polygon = [[float(row['POINT_X']), float(row['POINT_Y'])] for idx, row in selected_df.iterrows()]
        print(polygon)
    exit()

    print(ids[1], df.loc[df['id']==ids[1]])
    exit()
    for img_path in sorted(AERO_IMG_DIR.glob('*.jpg')):
        print(img_path)
        jgw_path = img_path.with_suffix('.jgw')
        get_location_point(jgw_path)
        img = open_image(str(img_path))

if __name__ == '__main__':
    main()