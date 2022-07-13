import os
import cv2
import time
import importlib
import numpy as np
from sklearn.utils import shuffle
#import tensorflow as tf
#import natsort

def load_lrt_human(thermal_path, rgb_path, use_shuffle=True, rev_rgb=False):

    if thermal_path[-4:] == '.npy' and rgb_path[-4:] == '.npy':
        print('Loading:', rgb_path)
        rgb_images = np.load(rgb_path)
        print('Loading:', thermal_path)
        thermal_images = np.load(thermal_path)
        
    else:
        thermal_images = []
        rgb_images = []
    
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
        thermal_images = thermal_images.astype("float32",copy=False)
        thermal_images = (thermal_images - 127.5) / 127.5
    
        rgb_images = np.array(rgb_images)
        rgb_images = rgb_images.astype("float32",copy=False)
        rgb_images = (rgb_images - 127.5) / 127.5
    
    if use_shuffle:
        thermal_images, rgb_images = shuffle(thermal_images, rgb_images)

    if rev_rgb:
        train_images = train_images[:,:,:,::-1]
        
    return thermal_images, rgb_images
    

def load_model(name):
    module = importlib.import_module(name)
    return module


class Logger:
    def __init__(self, filename, buffer_length=10):
        self.filename = filename
        self.buffer_length = buffer_length
        self.buffer_str = ''
        self.count=0
        self.first_called = False
    
    def write_atts(self, loss_dict):
        atr = ''
        for key in loss_dict:
            atr += key+','
        with open(self.filename, "a") as f:
            f.write(atr+'\n')
    
    def __call__(self, loss_dict, head_str):
        
        if self.first_called == False:
            self.write_atts(loss_dict)
            self.first_called = True
        
        log_str = head_str+','
        prt_str = head_str+': '
        
        for key in loss_dict:
            log_str += '{:.2e},'.format(loss_dict[key])
            prt_str += '%s:%.4f, ' % (key,loss_dict[key])
            
        print(prt_str, end='\r')
        self.buffer_str+=log_str
        self.buffer_str+='\n'
        self.count+=1
        
        if self.count >= self.buffer_length:
            with open(self.filename, "a") as f:
                f.write(self.buffer_str)
                self.buffer_str = ''
                self.count=0

def make_log_folder(path):
    try:
        os.mkdir(path)
    except:
        path += str(int(time.time()))[-5:]
        os.mkdir(path)

    return path

def draw_border(img, width=(2,2,2,2), color=(-1,-1,-1)):#TBLR
    return cv2.copyMakeBorder(img, width[0], width[1], width[2], width[3], cv2.BORDER_CONSTANT, value=color)
            