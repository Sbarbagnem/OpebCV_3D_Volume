import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os
import time
from functions import *

path_left = 'Immagini/finale/Calibrate/left_path/' # path delle immagini per calibrare
path_right = 'Immagini/finale/Calibrate/right_path/'
res = np.asarray([[1,1,1,1]])

for i in range(1,7):

    directory = "output/"+str(i)+"/" # directory dove salvo gli output

    if not os.path.exists(directory):
        os.makedirs(directory)

    path = i

    fileSinistro = 'Immagini/finale/Test/Left'+str(i)+'.png' # le due immagini dalle camere
    fileDestro = 'Immagini/finale/Test/Right'+str(i)+'.png'

    imgLeft = cv2.cvtColor(np.uint8(edge(cv2.imread(fileSinistro))),
                            cv2.COLOR_BGR2GRAY)
    imgRight = cv2.cvtColor(np.uint8(edge(cv2.imread(fileDestro))),
                            cv2.COLOR_BGR2GRAY)

    im1 = np.uint8(edge(cv2.imread(fileSinistro))) # per proiezione 3d, rettifica RGB

    image_size = imgLeft.shape[::-1]
    numero_im = 19

    if(i==0): # da fare solo per una nuova calibrazione, la prima volta

        ### FIND INTRINSIC PARAMETER ###
        #print 'findIntrinsic Parameter'
        mtxL, distL, mtxR, distR, obj_points, img_left_points, img_right_points = calibrareCamere(path_left,path_right,numero_im)

        ### UNDISTORT IMAGE
        #imgLeft, imgRight = undistortImage(imgLeft,imgRight,mtxL,distL,mtxR,distR)

        ### STEREO CALIBRATION ###
        #print 'stereo Calibrate'
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        stereocalib_retval, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(obj_points,img_left_points,img_right_points,
                                                                                      mtxL,distL,mtxR,distR,image_size,
                                                                                      flags = cv2.CALIB_FIX_INTRINSIC,
                                                                                      criteria = stereocalib_criteria)
        save(mtxL,distL,mtxR,distR,R,T)

    mtxL,distL,mtxR,distR,R,T = read()

    ### STEREO RECTIFICATION ###
    #print 'rettifica Immagini'
    imgLeftRemap, imgRightRemap, Q = rettificaImmagini(imgLeft,imgRight,mtxL, distL, mtxR, distR, image_size,

                                                  R, T, path)
    ### STEREO CORRESPONDENCE ###
    #print 'Stereo Correspondence'
    disp = stereoMatch(imgLeftRemap,imgRightRemap, path);

    ### POINT CLOUD ###
    #print 'compute PointCloud'
    punti, texture, angle = pointCloud(disp, im1, imgRight, mtxL, distL, mtxR, distR, image_size, R, T, Q, path)

    ###TROVO PARAMETRI ROTAZIONE###
    #print 'trovo parametri rotazione'
    resTemp = findParametri(punti,'bianco')
    res = np.vstack([res,resTemp])

    ###RUOTO POINT CLOUD E TRASLO IN ORIGINE###
    #print 'ruoto point cloud'
    fileName = 'pointCloudGiusto'+str(i)+'.ply'
    punti = rotateTraslate(resTemp[0:3],punti,texture,fileName,i)

    ###TROVO VOLUME FITTANDO PIANI###
    #print 'trovo volume'
    #volume = findVolume(punti)

    #print 'Il volume stimato della scatola e ' + str(volume)
print res





'''DA FARE - sostituire while con for, costruire prima range con np.arange OK
           - sistemare bins histogram OK
           - sistemare center hist OK
           - capire perche ci mette tanto a fare la moltiplicazioni, nella prova non e cosi -----> RISOLTO, CASTO LE MATRICI A FLOAT
'''
