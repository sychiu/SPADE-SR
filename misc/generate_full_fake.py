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
from sklearn.utils import shuffle
from utils import make_log_folder
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

if __name__ == '__main__':

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
    folder_names = []
    thermal_path = cfg['THERMAL_PATH']
    rgb_path = cfg['RGB_PATH']
    folders = [f for f in os.listdir(thermal_path) if not os.path.isfile(os.path.join(thermal_path, f))]
    for folder in folders:
        print('Loading: ',folder)
        temp_thermals = []
        temp_rgbs = []
        for filename in natsort.natsorted(os.listdir(os.path.join(thermal_path, folder))):
            if filename[-4:] == '.png':
                thermal_img = cv2.imread(os.path.join(thermal_path, folder, filename))[:,:,0:1]
                rgb_img = cv2.imread(os.path.join(rgb_path, folder, filename))
                temp_thermals.append(thermal_img)
                temp_rgbs.append(rgb_img)
        
        folder_names.append(folder)
        temp_thermals = np.array(temp_thermals).astype("float32")
        temp_thermals = (temp_thermals - 127.5) / 127.5  
        temp_rgbs = np.array(temp_rgbs).astype("float32")
        temp_rgbs = (temp_rgbs - 127.5) / 127.5
        thermal_images.append(temp_thermals)
        rgb_images.append(temp_rgbs)
    
    
    #MAKE LOG FOLDER
    folder_name = make_log_folder(os.path.join('./experiments',cfg['filename']))
        
    #DUMP CFG
    with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
        yaml.dump(cfg, file)

    #SORRY FOR THE SLOWNESS!!
    trim = 50
    for folder_idx in range(len(thermal_images)):
        out_path = os.path.join(folder_name,folder_names[folder_idx])
        os.mkdir(out_path)
        source_idx = np.random.randint(trim, high=thermal_images[folder_idx].shape[0]-trim, dtype=int)
        print(folder_names[folder_idx], source_idx)

        with open(os.path.join(folder_name, 'source_img_idxes.log'), "a") as f:
            f.write(str(folder_names[folder_idx])+": "+str(source_idx))
            
        source_rgb = rgb_images[folder_idx][source_idx:source_idx+1]
        source_code = encoder(source_rgb, training=False)
        for i in range(thermal_images[folder_idx].shape[0]):
    
            generated_rgb = generator([thermal_images[folder_idx][i:i+1],source_code],training=False)[0]    
            cv2.imwrite(os.path.join(out_path,'%i.png' % i), (generated_rgb.numpy()*0.5+0.5)*255)  



