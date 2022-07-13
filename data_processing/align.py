import numpy as np
import time
import glob
import cv2
import collections
import threading
import os
import operator


def undistort(img,k1,k6):

    src    = img
    width  = src.shape[1]
    height = src.shape[0]
    
    distCoeff = np.zeros((8,1),np.float64)
    k1 = k1
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0#4.0e-7
    k4 = 0.0
    k5 = 0.0#-5.0e-5#0
    k6 = k6
    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2
    distCoeff[4,0] = k3
    distCoeff[5,0] = k4
    distCoeff[6,0] = k5
    distCoeff[7,0] = k6
    
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 22.0        # define focal length x
    cam[1,1] = 22.0        # define focal length y
    # here the undistortion will be computed
    dst = cv2.undistort(src,cam,distCoeff)
    #cv2.imshow('src',cv2.resize(src,(width*2,height*2),interpolation=cv2.INTER_NEAREST))
    #cv2.imshow('dst',cv2.resize(dst,(width*2,height*2),interpolation=cv2.INTER_NEAREST))
    #cv2.imshow('ol',cv2.addWeighted(cv2.resize(dst,(width*2,height*2),interpolation=cv2.INTER_NEAREST),0.2,cv2.resize(src,(width*2,height*2),interpolation=cv2.INTER_NEAREST),0.8,0))
    #cv2.waitKey(0)
    return dst

#undistort(cv2.imread('/Users/shengyang/Desktop/截圖 2022-05-02 16.45.33.png'),0.0,-2.0e-5)
#dd

temperatures_path = './data3/temperatures'
rgb_path = './data3/rgb'
mask_path = './data3/mask'

t_max = 30.0
t_min = 19.0
init_interpolation_size = 128

'''
hor_shifts = [0]
ver_shifts = [0,6,12]
scales = [-18,-12,-6,0]
t_thrs = [24.0,24.5,25.0]
k1s = [2.0e-2,1.0e-2,4.0e-3,0]
k6s = [-8.0e-5,-6.0e-5,-4.0e-5,0]
'''

'''
hor_shifts = [0]
ver_shifts = [0,6,12,18]
scales = [0,6,12,18]
t_thrs = [24.5]
k1s = [2.0e-2,1.0e-2,4.0e-3,0]
k6s = [-8.0e-5,-4.0e-5,-2.0e-5,0]
'''
hor_shifts = [-2]
ver_shifts = [0]
scales = [-24,-26,-28]
t_thrs = [24.0,24.5,25.0]
k1s = [0]
k6s = [0]

display = False

thermals = []
rgbs = []
masks = []

#files = [f for f in os.listdir(temperatures_path) if os.path.isfile(os.path.join(temperatures_path, f))]
files = ['1643047291402.npy','1643047415126.npy','1643047752420.npy',
                #'1643048046246.npy','1643048430560.npy','1643048628931.npy',
                '1643049012476.npy','1643049223069.npy','1643049592147.npy',
                #'1643049798574.npy','1643050090169.npy','1643050150929.npy',
                '1643050280749.npy','1643050347295.npy','1643050495512.npy',
                #'1643050561171.npy','1643050678773.npy','1643050735278.npy',
                '1643050876906.npy','1643050939285.npy']

for filename in files:
    if filename[-4:] == '.npy':
        print('LOADING '+ filename)

        all_temperatures = np.load(os.path.join(temperatures_path,filename))
        all_temperatures = np.maximum(all_temperatures,t_min)
        all_temperatures = np.minimum(all_temperatures,t_max)
        all_temperatures = (all_temperatures-t_min)/(t_max-t_min)

        for i, temperatures in enumerate(all_temperatures):

            thermal_image = np.zeros((8,8,1), np.uint8) 
            for y in range(8):
                for x in range(8):
                    cv2.rectangle(thermal_image, (int(y), int(x)), (int((y+1)), int((x+1))), temperatures[(7-y)*8+(x)]*255, -1)
                    
            #rgb_image = cv2.imread(os.path.join(rgb_path,filename[:-4],str(i)+'.png'))
            mask_image = cv2.imread(os.path.join(mask_path,filename[:-4],str(i)+'.png'))
            
            #kernel = np.ones((3,3), np.uint8)
            #mask_image = cv2.dilate(mask_image, kernel, iterations = 8)
            #mask_image = cv2.erode(mask_image, kernel, iterations = 20)

            thermals.append(thermal_image)
            #rgbs.append(rgb_image)
            masks.append(mask_image)


print('start')
all_scores = {}
for hor_shift in hor_shifts:
    for ver_shift in ver_shifts:
        for scale in scales:
            for t_thr in t_thrs:
                for k1 in k1s:
                    for k6 in k6s:
                        scores = []
                        for i in range(len(thermals)):
            
                            thermal_image = thermals[i]
                            resized_thermal_image = cv2.resize(thermal_image, (init_interpolation_size, init_interpolation_size), interpolation=cv2.INTER_CUBIC)
                            resized_thermal_image = undistort(resized_thermal_image,k1,k6)
                            resized_thermal_image = cv2.resize(resized_thermal_image, (init_interpolation_size+scale, init_interpolation_size+scale), interpolation=cv2.INTER_AREA)
                            resized_thermal_image = resized_thermal_image > (t_thr-t_min)/(t_max-t_min)*255
                            resized_thermal_image = resized_thermal_image.astype(np.float32)*2-1
                            resized_thermal_image = np.pad(resized_thermal_image,((52+ver_shift-scale//2,52-ver_shift-scale//2),(80+hor_shift-scale//2,80-hor_shift-scale//2)),constant_values=-1)
        
                            mask_image = masks[i][:,2:-2,0]
                            mask_image = cv2.resize(mask_image, (128,72), interpolation=cv2.INTER_AREA)
                            mask_image = mask_image > 127
                            mask_image = mask_image.astype(np.float32)*2-1
                            mask_image = np.pad(mask_image,((80,80),(80,80)),constant_values=((0,0),(0,0)))
        
                            scores.append(np.sum(mask_image*resized_thermal_image))

                            if display:
                                overlay = cv2.addWeighted(mask_image,0.5,resized_thermal_image,0.5,0)
                                #cv2.imshow('rgb_image', rgb_image)
                                cv2.imshow('mask_image', mask_image*0.5+0.5)
                                cv2.imshow('thr', resized_thermal_image*0.5+0.5)
                                cv2.imshow('overlay', overlay*0.5+0.5)
                                cv2.waitKey(50)
            
                        mean_score = np.mean(np.array(scores)).astype(int)
                        print('SCORE of HS %i VS %i S %i K1 %.4f K6 %.5f T %.1f: %i' % (hor_shift, ver_shift, scale, k1, k6, t_thr, mean_score))
                        all_scores['HS %i VS %i S %i K1 %.4f K6 %.5f T %.1f' % (hor_shift, ver_shift, scale, k1, k6, t_thr)] = mean_score

#print(sorted(all_scores.items()))
sorted_scores = sorted(all_scores.items(), key=operator.itemgetter(1), reverse=True)
print('SORTED RESULTS:')
for item in sorted_scores:
    print(item)



