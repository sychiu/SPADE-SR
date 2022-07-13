import numpy as np
import time
import glob
import cv2
import collections
import threading
import os




def undistort(img,k1,k6):

    src    = img
    width  = src.shape[1]
    height = src.shape[0]
    
    distCoeff = np.zeros((8,1),np.float64)
    k1 = 2.0e-2
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0
    k3 = 0.0#4.0e-7
    k4 = 0.0
    k5 = 0.0#-5.0e-5#0
    k6 = 0.0
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

    return dst

def moving_average(x, ws=3):
    ret = np.cumsum(x, axis=-1)
    ret[:,ws:] = ret[:,ws:] - ret[:,:-ws]
    return ret[:,ws - 1:] / ws

temperatures_path = 'data/raw_temperature'
rgb_path = './data/raw_rgb'
#mask_path = './data2/mask'

rgb_out_path = 'data/processed_temperature'
thermal_out_path = './data/processed_rgb'
#cal = np.load('cal.npy')

pre_t_max = 30.0 #30.0
pre_t_min = 19.0 #19.0

post_t_max = 30.0 #30.0
post_t_min = 19.0 #24.0

noise_scale = 0
interpolation_size = 128

scale = -26 #-26
hor_shift = -2 #-2
ma_windowsize = 1

save = True
display = not save


if save:
    os.mkdir(rgb_out_path)
    os.mkdir(thermal_out_path)


files = [f for f in os.listdir(temperatures_path) if os.path.isfile(os.path.join(temperatures_path, f))]


for filename in files:
    if filename[-4:] == '.npy':
        print('PROCESSING '+ filename)
        if save:
            rgb_save_path = os.path.join(rgb_out_path,filename[:-4])
            os.mkdir(rgb_save_path)
            thermal_save_path = os.path.join(thermal_out_path,filename[:-4])
            os.mkdir(thermal_save_path)

        all_temperatures = np.load(os.path.join(temperatures_path,filename))[:,:-1] #- cal 

        if ma_windowsize > 1:
            all_temperatures = np.transpose(moving_average(np.transpose(all_temperatures), ws=ma_windowsize))

        if noise_scale>0:
            all_temperatures = all_temperatures + np.random.normal(scale=noise_scale, size=all_temperatures.shape)

        all_temperatures = np.maximum(all_temperatures,pre_t_min)
        all_temperatures = np.minimum(all_temperatures,pre_t_max)
        all_temperatures = (all_temperatures-pre_t_min)/(pre_t_max-pre_t_min)

        for i, temperatures in enumerate(all_temperatures):

            thermal_image = np.zeros((8,8,1), np.uint8) 
            for y in range(8):
                for x in range(8):
                    cv2.rectangle(thermal_image, (int(y), int(x)), (int((y+1)), int((x+1))), temperatures[(7-y)*8+(x)]*255, -1)


            resized_thermal_image = cv2.resize(thermal_image, (interpolation_size+scale, interpolation_size+scale), interpolation=cv2.INTER_CUBIC)
            resized_thermal_image[resized_thermal_image<(post_t_min-pre_t_min)/(pre_t_max-pre_t_min)*255] = 0
            resized_thermal_image[resized_thermal_image>(post_t_max-pre_t_min)/(pre_t_max-pre_t_min)*255] = 255

            wh_diff = interpolation_size+scale-72
            resized_thermal_image = resized_thermal_image[wh_diff//2:-wh_diff//2,:]

            large_thermal_image = cv2.resize(resized_thermal_image, (128, 96), interpolation=cv2.INTER_CUBIC)
            small_thermal_image = cv2.resize(resized_thermal_image, (8, 5), interpolation=cv2.INTER_AREA)

            rgb_image = cv2.imread(os.path.join(rgb_path,filename[:-4],str(i+ma_windowsize-1)+'.png'))[:,2:-2]
            rgb_image = cv2.resize(rgb_image, (128,72), interpolation=cv2.INTER_AREA)
            rgb_image = rgb_image[:,-scale//2+hor_shift:scale//2+hor_shift]

            large_rgb_image = cv2.resize(rgb_image, (128,96), interpolation=cv2.INTER_CUBIC)
            small_rgb_image = cv2.resize(rgb_image, (64,40), interpolation=cv2.INTER_AREA)

            

            if save:
                cv2.imwrite(os.path.join(thermal_save_path,str(i)+'.png'), small_thermal_image)
                cv2.imwrite(os.path.join(rgb_save_path,str(i)+'.png'), small_rgb_image)


            if display:
                
                #overlay = cv2.addWeighted(large_rgb_image,0.3,np.repeat((np.expand_dims(large_thermal_image,-1)),3,axis=-1),1.0,0)
                #cv2.imshow('rgb_image', large_rgb_image)
                #cv2.imshow('thr', large_thermal_image)
                #cv2.imshow('raw',cv2.resize(thermal_image,(120,120),interpolation=cv2.INTER_LINEAR))
                #cv2.imshow('overlay', cv2.resize(overlay,(overlay.shape[1]*2, overlay.shape[0]*2), interpolation=cv2.INTER_NEAREST))
                
                overlay = cv2.addWeighted(cv2.resize(small_rgb_image,(128,80),interpolation=cv2.INTER_NEAREST),0.4,np.repeat((np.expand_dims(cv2.resize(small_thermal_image,(128,80),interpolation=cv2.INTER_NEAREST),-1)),3,axis=-1),0.8,0)
                cv2.imshow('small_rgb_image', cv2.resize(small_rgb_image,(128,80),interpolation=cv2.INTER_NEAREST))
                cv2.imshow('small_thr', cv2.resize(small_thermal_image,(128,80),interpolation=cv2.INTER_NEAREST))
                #cv2.imshow('cubic_thr', cv2.resize(cubic,(128*2,96*2),interpolation=cv2.INTER_NEAREST))
                #cv2.imshow('area_thr', cv2.resize(area,(128*2,96*2),interpolation=cv2.INTER_NEAREST))
                #cv2.imshow('concat', cv2.resize(concat,(128*2,96*2),interpolation=cv2.INTER_NEAREST))
                cv2.imshow('overlay', cv2.resize(overlay, (overlay.shape[1]*2,overlay.shape[0]*2), interpolation=cv2.INTER_NEAREST))
                cv2.waitKey(1)



#if save:
    #np.save(rgb_out_path+'.npy',np.array())



