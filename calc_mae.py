import os
import cv2
import sys
import time
import yaml
import argparse
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow import keras
import natsort

#GPU USAGES
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    #SET GPU MEMORY LIMIT
    #tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])
    #SET GPU GROWTH LIMIT
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU hardware devices available")


#READ CFG
parser = argparse.ArgumentParser()
parser.add_argument("cfg_path", type=str, help="path to cfg file")
args = parser.parse_args()
with open(args.cfg_path, "r") as stream:
    cfg = yaml.load(stream)

#LOAD MODELS
generator = keras.models.load_model(cfg['GENERATOR'])
encoder = keras.models.load_model(cfg['ENCODER'])

#LOAD DATASET
thermal_images = []
rgb_images = []
#thermal_images = np.load('data_final/thermal_wr_folders.npy',allow_pickle=True)
#rgb_images = np.load('data_final/rgb_folders.npy',allow_pickle=True)

folder_names = []
thermal_path = cfg['THERMAL_PATH']
rgb_path = cfg['RGB_PATH']
folders = [f for f in os.listdir(thermal_path) if not os.path.isfile(os.path.join(thermal_path, f))]
for folder in folders:
    print('Loading: ',folder)
    folder_names.append(folder)
    
    temp_thermals = []
    temp_rgbs = []
    for filename in natsort.natsorted([name for name in os.listdir(os.path.join(thermal_path, folder)) if name[-4:] == '.png']):
        thermal_img = cv2.imread(os.path.join(thermal_path, folder, filename))[:,:,0:1]
        rgb_img = cv2.imread(os.path.join(rgb_path, folder, filename))
        temp_thermals.append(thermal_img)
        temp_rgbs.append(rgb_img)
    
    
    temp_thermals = np.array(temp_thermals).astype("float32")
    temp_thermals = (temp_thermals - 127.5) / 127.5  
    temp_rgbs = np.array(temp_rgbs).astype("float32")
    temp_rgbs = (temp_rgbs - 127.5) / 127.5
    thermal_images.append(temp_thermals)
    rgb_images.append(temp_rgbs)
    
    
'''
try:
    np.save('data_final/rgb_folders.npy', np.array(rgb_images))
    np.save('data_final/thermal_wr_folders.npy', np.array(thermal_images))
except:
    print('ohoh')
'''


#MAKE LOG FOLDER
folder_name = os.path.join('./experiments',cfg['filename'])
try:
    os.mkdir(folder_name)
except:
    folder_name += str(int(time.time()))[-5:]
    os.mkdir(folder_name)
    
#DUMP CFG
with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
    yaml.dump(cfg, file, sort_keys=False)

trim = 50
bs = cfg['BATCH_SIZE']
all_errors = []
errors = []
for run in range(cfg['RUNS']):
    print('run: ', run)
    errors = []
    for folder_idx in range(len(thermal_images)):
        #print('processing: ', folder_names[folder_idx])
        source_idx = np.random.randint(trim, high=thermal_images[folder_idx].shape[0]-trim, dtype=int)
        source_rgb = rgb_images[folder_idx][source_idx:source_idx+1]
        source_code = encoder(source_rgb, training=False)
    
        #print('folder_length: ', thermal_images[folder_idx].shape[0])
        #print('trimmed_idx: ', trim, ' to ', thermal_images[folder_idx].shape[0]-trim)
        calc_len = thermal_images[folder_idx].shape[0] - trim*2
        batch_steps = calc_len//bs
        remain_start = batch_steps*bs + trim
        remain_end = thermal_images[folder_idx].shape[0] - trim
        
        for i in range(batch_steps):
            start_idx = i*bs + trim
            end_idx = (i+1)*bs + trim
            generated_rgb = generator([thermal_images[folder_idx][start_idx:end_idx],np.repeat(source_code,end_idx-start_idx,axis=0)],training=False)*0.5+0.5
            corr_rgb = rgb_images[folder_idx][start_idx:end_idx]*0.5+0.5
            error = np.mean(np.abs(generated_rgb-corr_rgb))
            errors.append(error)
            
        if batch_steps*bs != calc_len:
            generated_rgb = generator([thermal_images[folder_idx][remain_start:remain_end],np.repeat(source_code,remain_end-remain_start,axis=0)],training=False)*0.5+0.5
            corr_rgb = rgb_images[folder_idx][remain_start:remain_end]*0.5+0.5
            error = np.mean(np.abs(generated_rgb-corr_rgb))
            errors.append(error)
               
    all_errors.append(sum(errors)/len(errors))

final_error = sum(all_errors)/len(all_errors)
print('FINAL_ERROR:', final_error)
with open(os.path.join(folder_name,'log.txt'), "a") as f:
    f.write(str(final_error))