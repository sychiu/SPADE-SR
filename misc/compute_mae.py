import os
import cv2
import sys
import time
#import yaml
#import argparse
import numpy as np
#import tensorflow as tf
#tf.enable_eager_execution()
#from tensorflow import keras
#from sklearn.utils import shuffle
#from utils import load_dataset, draw_border
import natsort
#from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

font = {'size'   : 18}
matplotlib.rc('font', **font)
src_img_idxes = {'1643049093723':235,'1643049592147':802,'1643050561171':300,'1643049798574':343,'1643048310719':430,'1643050876906':373,
            '1643050347295':112, '1643047149874':226, '1643050280749':254, '1643049666649':309, '1643047291402':1007, '1643050678773':201,
            '1643048046246':619,'1643048508528':147,'1643049466235':232,'1643047415126':517,'1643047895815':92,'1643048430560':965,
            '1643047752420':70,'1643050090169':79,'1643047614384':419,'1643049012476':1028,'1643049223069':885,'1643050150929':126,
            '1643050735278':162,'1643048628931':279,'1643050495512':397,'1643047076555':315,'1643050939285':100,'1643048889330':220}

#GPU USAGES
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])

'''
thermal_path = './data3/processed_small_thermal'
rgb_path = './data3/processed_rgb'

out_thermal_path = './data3_trim/processed_small_thermal'
out_rgb_path = './data3_trim/processed_rgb'
os.mkdir(out_thermal_path)
os.mkdir(out_rgb_path)

folders = [f for f in os.listdir(thermal_path) if not os.path.isfile(os.path.join(thermal_path, f))]
for folder in folders:
    print('Loading: ',folder)
    out_thermal_path_folder = os.path.join(out_thermal_path,folder)
    os.mkdir(out_thermal_path_folder)
    out_rgb_path_folder = os.path.join(out_rgb_path,folder)
    os.mkdir(out_rgb_path_folder)

    folder_len = len(os.listdir(os.path.join(thermal_path, folder)))
    for idx, filename in enumerate(os.listdir(os.path.join(thermal_path, folder))):
        if filename[-4:] == '.png' and idx>10 and idx<(folder_len-50):
            thermal_img = cv2.imread(os.path.join(thermal_path, folder, filename))
            rgb_img = cv2.imread(os.path.join(rgb_path, folder, filename))
            cv2.imwrite(os.path.join(out_thermal_path_folder,'%i.png'%idx))
            

'''
'''
#LOAD DATASET
thermal_images = []
rgb_images = []

thermal_path = './data3/processed_small_thermal'
rgb_path = './data3/processed_rgb'
out_path = '../random_lrt_human'

folders = [f for f in os.listdir(thermal_path) if not os.path.isfile(os.path.join(thermal_path, f))]
for folder in folders:
    print('Loading: ',folder)
    for filename in os.listdir(os.path.join(thermal_path, folder)):
        if filename[-4:] == '.png':
            thermal_img = cv2.imread(os.path.join(thermal_path, folder, filename))[:,:,0:1]
            rgb_img = cv2.imread(os.path.join(rgb_path, folder, filename))
            thermal_images.append(thermal_img)
            rgb_images.append(rgb_img)

thermal_images = np.array(thermal_images)
rgb_images = np.array(rgb_images)
print(rgb_images.shape[0])
dd

rand_idxes = np.random.randint(0, high=rgb_images.shape[0], size=100)
print(rand_idxes)
selected_thermals = thermal_images[rand_idxes]
selected_rgbs = rgb_images[rand_idxes]

for i in range(selected_thermals.shape[0]):
    cv2.imwrite(os.path.join(out_path,'%i_thermal.png'%i),cv2.resize(selected_thermals[i],(320,200),interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(os.path.join(out_path,'%i_rgb.png'%i),selected_rgbs[i])
'''

thermal_folder = './data_final/thermal_wr'
true_rgb_path = './data_final/rgb'
fake_rgb_path = './mae_YESBN_E10'
out_path = './mae_result_YESBN_E10'
folders = [f for f in os.listdir(fake_rgb_path) if not os.path.isfile(os.path.join(fake_rgb_path, f))]
trim=50
all_errors = []

#0-1!!!
for folder_name in folders:
    output_folder = os.path.join(out_path,folder_name)
    os.mkdir(output_folder)
    errors = []
    display_point = []
    #clip_length = len(os.listdir(os.path.join(fake_rgb_path, folder_name)))
    name_list = natsort.natsorted([item for item in os.listdir(os.path.join(fake_rgb_path, folder_name)) if item[-4:]=='.png'])
    clip_length = len(name_list)
    for count in range(trim,clip_length-trim,1):
        true_rgb = cv2.imread(os.path.join(true_rgb_path, folder_name, name_list[count])).astype("float32")/255
        fake_rgb = cv2.imread(os.path.join(fake_rgb_path, folder_name, name_list[count])).astype("float32")/255
        error = np.mean(np.abs(true_rgb-fake_rgb))
        errors.append(error)
        #cv2.imshow('true',true_rgb)
        #cv2.imshow('fake',fake_rgb)
        #cv2.waitKey(1)
        if (count-trim)%((clip_length-trim*2)//6) == 0 and count>trim:
            display_point.append(count)
            cv2.imwrite(os.path.join(output_folder,'real_%i.png'%count),cv2.resize(true_rgb,(256,160),interpolation=cv2.INTER_NEAREST)*255)
            cv2.imwrite(os.path.join(output_folder,'fake_%i_%i.png'%(count,error*10000)),cv2.resize(fake_rgb,(256,160),interpolation=cv2.INTER_NEAREST)*255)
            thermal_image = cv2.resize(cv2.imread(os.path.join(thermal_folder,folder_name,name_list[count])),(256,160),interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_folder,'thermal_%i.png'%count),thermal_image)
        
    errors = np.array(errors)
    all_errors.append(np.mean(errors))
    '''
    max_error_idx = np.argmax(errors)
    max_true_rgb = cv2.resize(cv2.imread(os.path.join(true_rgb_path,folder_name,'%i.png'%(max_error_idx+trim))),(256,160),interpolation=cv2.INTER_NEAREST)
    max_fake_rgb = cv2.resize(cv2.imread(os.path.join(fake_rgb_path,folder_name,'%i.png'%(max_error_idx+trim))),(256,160),interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(output_folder,'max_error_real_%i.png'%(max_error_idx+trim)),max_true_rgb)
    cv2.imwrite(os.path.join(output_folder,'max_error_fake_%i_%i.png'%((max_error_idx+trim),errors[max_error_idx]*10000)),max_fake_rgb)
    thermal_image = cv2.resize(cv2.imread(os.path.join(thermal_folder,folder_name,'%i.png'%(max_error_idx+trim))),(256,160),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_folder,'max_error_thermal.png'),thermal_image)

    source_true_rgb = cv2.resize(cv2.imread(os.path.join(true_rgb_path,folder_name,'%i.png'%src_img_idxes[folder_name])),(256,160),interpolation=cv2.INTER_NEAREST)
    source_fake_rgb = cv2.resize(cv2.imread(os.path.join(fake_rgb_path,folder_name,'%i.png'%src_img_idxes[folder_name])),(256,160),interpolation=cv2.INTER_NEAREST)
    source_thermal = cv2.resize(cv2.imread(os.path.join(thermal_folder,folder_name,'%i.png'%src_img_idxes[folder_name])),(256,160),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(output_folder,'source_real.png'),source_true_rgb)
    cv2.imwrite(os.path.join(output_folder,'source_fake_%i.png'%(errors[src_img_idxes[folder_name]-trim]*10000)),source_fake_rgb)
    cv2.imwrite(os.path.join(output_folder,'source_thermal.png'),source_thermal)

    plt.rcParams["figure.figsize"] = (20,4)
    fig, axs = plt.subplots(1)
    axs.grid(False)

    #patch = matplotlib.patches.Circle((50,50), 101, alpha=0.5, transform=None)
    #fig.artists.append(patch)

    #x1, y1 = [-100, errors.shape[0]+100], [np.mean(errors), np.mean(errors)]
    #axs.plot(x1, y1,linewidth=1,ls='--', c=(0.5,0.5,0.5))
    axs.plot(np.arange(errors.shape[0]),errors,color=(222/255,29/255,1/255),label='hi',linewidth=2.5)[0]
    axs.set_ylim([0.0,0.15])
    axs.set_xlim([-25,errors.shape[0]+25])
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs.set_ylabel('error (mae)',labelpad=15)
    axs.set_xlabel('frame #',labelpad=15)
    if len(display_point)>5:
        display_point.pop(-1)
    for dp in display_point:
        #x1, y1 = [dp, dp], [-1, 1]
        #axs.plot(x1, y1,linewidth=1,ls='--', c=(0.5,0.5,0.5),alpha=0.8)
        axs.plot(dp-trim,errors[dp-trim],'.',markersize=25,mec='none',mfc=(0.5,0.5,0.5),alpha=0.8)  

    #x1, y1 = [max_error_idx, max_error_idx], [-1, 1]
    #axs.plot(x1, y1,linewidth=1,ls='--', c=(0.0,0.0,0.0),alpha=0.8)
    axs.plot(max_error_idx,errors[max_error_idx],'o',fillstyle='none',markersize=13,mec=(0,0,0),mew=3,alpha=0.8)

    x1, y1 = [src_img_idxes[folder_name]-trim, src_img_idxes[folder_name]-trim], [-1, 1]
    axs.plot(x1, y1,linewidth=1,ls='--', c=(1/255,29/255,222/255),alpha=0.9)
    axs.plot(src_img_idxes[folder_name]-trim,errors[src_img_idxes[folder_name]-trim],'x',fillstyle='none',markersize=17,mec=(1/255,29/255,222/255),mew=5,alpha=0.9)

    title = 'CLIP %s  MEAN %.3f' % (folder_name[-6:], np.mean(errors))
    #title = 'MEAN MAE %.3f' % (np.mean(errors))
    axs.set_title(title)
    axs.set_aspect(errors.shape[0]+50)
    plt.savefig(os.path.join(output_folder,'error_graph.pdf'))
    #plt.show(fig)
    plt.close()
    '''

print('FINAL_MEAN:', sum(all_errors)/len(all_errors))
    
    