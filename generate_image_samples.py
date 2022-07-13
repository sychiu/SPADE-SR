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
from utils import load_lrt_human
#from tensorflow.keras import layers

#GPU USAGES
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    #SET GPU MEMORY LIMIT
    #tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])
    #SET GPU GROWTH LIMIT
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU hardware devices available")


'''
CFG AUTO-PARSING FIELD PAIR LIST
1. IF THE FIELD NAME IS SAME FOR BOTH INVERSION AND GAN TRAINING CONFIG, JUST ENTER THE LIST WITH THE FIELD NAME,
E.G, [NAME]

2. IF THE FIELD NAME YOU WISH TO COPY FROM IS DIFFERENT, ENTER THE LIST WITH INVERSION CONFIG FIRST THEN THE GAN TRAINING CONFIG
E.G, [NAME_IN_INV_CONFIG, NAME_IN_GAN_CONFIG]
'''
cfg_keypair_list = [['THERMAL_PATH'],['RGB_PATH'],['GENERATOR']]
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
        cfg['filename'] = 'samples_' + inversion_cfg['filename']
        
    #cfg['ENCODER'] = os.path.join(args.inversion_package_path,'weight',cfg['ENCODER'])
    
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

#LOAD DATASET
thermal_images, rgb_images = load_lrt_human(cfg['THERMAL_PATH'], cfg['RGB_PATH'])
print('thermal images shape: ', thermal_images.shape)
print('rgb images shape: ', rgb_images.shape)


#MAKE LOG FOLDER
folder_name = os.path.join('./experiments',cfg['filename'])
try:
    os.mkdir(folder_name)
except:
    folder_name += str(int(time.time()))[-5:]
    os.mkdir(folder_name)

real_folder = os.path.join(folder_name,'real_samples')
fake_folder = os.path.join(folder_name,'fake_samples')
os.mkdir(real_folder)
os.mkdir(fake_folder)
#os.mkdir(thermal_folder)
    

#DUMP CFG
with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
    yaml.dump(cfg, file, sort_keys=False)
    
def get_single_real_sample():
    rand_idx = np.random.randint(0, high=rgb_images.shape[0])
    rgb_img = cv2.resize(rgb_images[rand_idx], (cfg['RGB_X'],cfg['RGB_Y']), interpolation=cv2.INTER_NEAREST)
    thm_img = cv2.resize(thermal_images[rand_idx], (cfg['THM_X'],cfg['THM_Y']), interpolation=cv2.INTER_NEAREST)
    rgb_img[-cfg['THM_Y']:,-cfg['THM_X']:] = np.repeat(np.expand_dims(thm_img,-1),3,axis=-1)
    return rgb_img


def get_single_fake_sample():
    rand_idx = np.random.randint(0, high=rgb_images.shape[0])
    generated_img = generator([thermal_images[rand_idx:rand_idx+1], tf.random.normal(shape=(1, 128))], training=False)[0]
    rgb_img = cv2.resize(generated_img.numpy(), (cfg['RGB_X'],cfg['RGB_Y']), interpolation=cv2.INTER_NEAREST)
    thm_img = cv2.resize(thermal_images[rand_idx], (cfg['THM_X'],cfg['THM_Y']), interpolation=cv2.INTER_NEAREST)
    rgb_img[-cfg['THM_Y']:,-cfg['THM_X']:] = np.repeat(np.expand_dims(thm_img,-1),3,axis=-1)
    return rgb_img
    
    
for run in range(cfg['RUNS']):
    h = get_single_real_sample()
    for row in range(cfg['ROW_NUMS']-1):
        h = np.concatenate((h, get_single_real_sample()),axis=0)
    
    v = h
    for col in range(cfg['COL_NUMS']-1):
        h = get_single_real_sample()
        for row in range(cfg['ROW_NUMS']-1):
            h = np.concatenate((h, get_single_real_sample()),axis=0)
            
        v = np.concatenate((v, h),axis=1)
    
    cv2.imwrite(os.path.join(real_folder,'%i.png' % run),v*127.5+127.5)

    
for run in range(cfg['RUNS']):
    h = get_single_fake_sample()
    for row in range(cfg['ROW_NUMS']-1):
        h = np.concatenate((h, get_single_fake_sample()),axis=0)
    
    v = h
    for col in range(cfg['COL_NUMS']-1):
        h = get_single_fake_sample()
        for row in range(cfg['ROW_NUMS']-1):
            h = np.concatenate((h, get_single_fake_sample()),axis=0)
            
        v = np.concatenate((v, h),axis=1)
    
    cv2.imwrite(os.path.join(fake_folder,'%i.png' % run),v*127.5+127.5)
    