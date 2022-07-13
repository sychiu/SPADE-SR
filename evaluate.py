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
from utils import load_lrt_human, make_log_folder


'''
CFG AUTO-PARSING FIELD PAIR LIST
1. IF THE FIELD NAME IS SAME FOR BOTH INVERSION AND GAN TRAINING CONFIG, JUST ENTER THE LIST WITH THE FIELD NAME,
E.G, [NAME]

2. IF THE FIELD NAME YOU WISH TO COPY FROM IS DIFFERENT, ENTER THE LIST WITH INVERSION CONFIG FIRST THEN THE GAN TRAINING CONFIG
E.G, [NAME_IN_INV_CONFIG, NAME_IN_GAN_CONFIG]
'''
cfg_keypair_list = [['THERMAL_PATH'],['RGB_PATH'],['GENERATOR']]


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
    parser.add_argument("inversion_package_path", type=str, help="path to inversion package", nargs='?')
    args = parser.parse_args()
    with open(args.cfg_path, "r") as stream:
        cfg = yaml.load(stream)
    
    if args.inversion_package_path != None:
        with open(os.path.join(args.inversion_package_path,"cfg.yaml"), "r") as stream:
            inversion_cfg = yaml.load(stream)
            
        if cfg['filename'] == 'None':
            cfg['filename'] = 'eva_' + inversion_cfg['filename']
            
        cfg['ENCODER'] = os.path.join(args.inversion_package_path,'weight',cfg['ENCODER'])
        
        for keypair in cfg_keypair_list:
            cfg_key = keypair[0]
            if len(keypair) == 1:
                inversion_cfg_key = keypair[0]
            else:
                inversion_cfg_key = keypair[1]
            
            if cfg[cfg_key] == 'None':
                cfg[cfg_key] = inversion_cfg[inversion_cfg_key]
    
    #LOAD MODELS
    generator = keras.models.load_model(cfg['GENERATOR'])
    encoder = keras.models.load_model(cfg['ENCODER'])
    
    #LOAD DATASET
    thermal_images, rgb_images = load_lrt_human(cfg['THERMAL_PATH'], cfg['RGB_PATH'])
    print('thermal images shape: ', thermal_images.shape)
    print('rgb images shape: ', rgb_images.shape)
    
    #MAKE LOG FOLDER
    folder_name = make_log_folder(os.path.join('experiments', cfg['filename']))
        
    #DUMP CFG
    with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
        yaml.dump(cfg, file, sort_keys=False)
    
    for run in range(cfg['RUNS']):
        source_idxs = np.random.randint(0, high=thermal_images.shape[0], size=cfg['SOURCE_NUMS'], dtype=int)
        source_rgbs = rgb_images[source_idxs]
        source_thermals = thermal_images[source_idxs]
        source_codes = encoder(source_rgbs, training=False)
        
        target_idxs = np.random.randint(0, high=thermal_images.shape[0], size=cfg['TARGET_NUMS'], dtype=int)
        target_thermals = thermal_images[target_idxs]
        
        results = []
        for i in range(cfg['TARGET_NUMS']):
            results.append(generator([tf.repeat([target_thermals[i]],cfg['SOURCE_NUMS'],axis=0), source_codes], training=False))
        
        
        h = np.zeros((40,64,3),dtype=np.float32)
        for j in range(cfg['TARGET_NUMS']):
            h = np.concatenate((h, cv2.resize(np.repeat(target_thermals[j],3,axis=-1), (64,40), interpolation=cv2.INTER_NEAREST)),axis=0)
        v = h
                
        for i in range(cfg['SOURCE_NUMS']):
            h = source_rgbs[i]
            for j in range(cfg['TARGET_NUMS']):
                h = np.concatenate((h, results[j][i].numpy()),axis=0)
            v = np.concatenate((v,h),axis=1)
        
        for i in range(cfg['TARGET_NUMS']+1):
            cv2.line(v, (0, i*40), ((cfg['SOURCE_NUMS']+1)*64, i*40), (-1, -1, -1), 1)
            
        for i in range(cfg['SOURCE_NUMS']+1):
            cv2.line(v, (i*64, 0), (i*64, (cfg['TARGET_NUMS']+1)*40), (-1, -1, -1), 1)
        
        #cv2.rectangle(v, (64, 1), ((cfg['SOURCE_NUMS']+1)*64-2, 40), (-1, 0, -1), 2)
        #cv2.rectangle(v, (1, 40), (64, (cfg['TARGET_NUMS']+1)*40-2), (-1, -1, 0), 2)
            
        cv2.imwrite(os.path.join(folder_name,'result_%i.png' % run), (v*0.5+0.5)*255) 



