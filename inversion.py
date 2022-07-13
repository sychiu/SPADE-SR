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
from utils import load_model, Logger, load_lrt_human, make_log_folder, draw_border
from loss import d_source_loss, g_source_loss, calc_error, gradient_penalty, r1_penalty


'''
CFG AUTO-PARSING FIELD PAIR LIST
1. IF THE FIELD NAME IS SAME FOR BOTH INVERSION AND GAN TRAINING CONFIG, JUST ENTER THE LIST WITH THE FIELD NAME,
E.G, [NAME]

2. IF THE FIELD NAME YOU WISH TO COPY FROM IS DIFFERENT, ENTER THE LIST WITH INVERSION CONFIG FIRST THEN THE GAN TRAINING CONFIG
E.G, [NAME_IN_INV_CONFIG, NAME_IN_GAN_CONFIG]
'''
cfg_keypair_list = [['THERMAL_PATH'],['RGB_PATH'],['BATCH_SIZE'],['SOURCE_LOSS'],['RECONSTRUCTION_LOSS'],['HINGE_REC_THR'],['GP_METHOD'],['D_THERMAL_REC_WEIGHT','D_REC_WEIGHT'],
                    ['D_GP_WEIGHT'],['D_USE_FAKE_RL'],['D_USE_TRUE_RL']]


#GPU USAGES
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    #SET GPU MEMORY LIMIT
    #tf.config.experimental.set_virtual_device_configuration(physical_devices[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=16000)])
    #SET GPU GROWTH LIMIT
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU hardware devices available")
        

@tf.function
def train_step(thermal_images, rgb_images):
    with tf.GradientTape(persistent=True) as tape:
        inv_rec = encoder(rgb_images, training=True)
        rec_images = generator([thermal_images, inv_rec], training=False) #!!!!

        fake_labels, fake_rec_thermal = discriminator(rec_images, training=True)
        real_labels, real_rec_thermal = discriminator(rgb_images, training=True)

        e_rgb_rl = calc_error(rec_images, rgb_images, mode=cfg['E_RGB_REC_MODE'])
        e_thermal_rl = calc_error(thermal_images,fake_rec_thermal, mode=cfg['RECONSTRUCTION_LOSS'], hinge_thr=cfg['HINGE_REC_THR'])
        e_sl = g_source_loss(fake_labels)

        e_loss = cfg['E_RGB_REC_WEIGHT']*e_rgb_rl + cfg['E_THERMAL_REC_WEIGHT']*e_thermal_rl + cfg['E_SL_WEIGHT']*e_sl

        d_sl = d_source_loss(real_labels, fake_labels, mode=cfg['SOURCE_LOSS'])

        if cfg['D_USE_TRUE_RL']:
            d_rl_real = calc_error(thermal_images,real_rec_thermal, mode=cfg['RECONSTRUCTION_LOSS'], hinge_thr=cfg['HINGE_REC_THR'])
        else:
            d_rl_real = 0.0
            
        if cfg['D_USE_FAKE_RL']:
            d_rl_fake = calc_error(thermal_images,fake_rec_thermal, mode=cfg['RECONSTRUCTION_LOSS'], hinge_thr=cfg['HINGE_REC_THR'])
        else:
            d_rl_fake = 0.0
        
        if cfg['D_USE_TRUE_RL'] and cfg['D_USE_FAKE_RL']:
            d_rl_factor = 0.5
        else:
            d_rl_factor = 1.0

        if cfg['GP_METHOD'] == 'wgan':
            d_gp = gradient_penalty(rgb_images, rec_images, discriminator)
        elif cfg['GP_METHOD'] == 'R1':
            d_gp = r1_penalty(rgb_images, discriminator)

        d_loss = cfg['D_SL_WEIGHT']*d_sl + cfg['D_THERMAL_REC_WEIGHT']*(d_rl_real+d_rl_fake)*d_rl_factor + cfg['D_GP_WEIGHT']*d_gp

    e_gradient = tape.gradient(e_loss, encoder.trainable_variables)
    e_optimizer.apply_gradients(zip(e_gradient, encoder.trainable_variables))
    e_ema.apply(encoder.trainable_variables)
    d_gradient = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradient, discriminator.trainable_variables))

    losses = {'E_LOSS': e_loss, 'E_RGB_RL': e_rgb_rl, 'E_THERMAL_RL': e_thermal_rl, 'E_SL': e_sl, 
               'D_LOSS': d_loss, 'D_SL': d_sl, 'D_RL_REAL': d_rl_real, 'D_RL_FAKE': d_rl_fake, 'D_GP': d_gp}
    
    return losses


def validation(thermal_images, rgb_images, encoder):
    inv_rec = encoder(rgb_images, training=False)
    rec_images = generator([thermal_images, inv_rec], training=False)
    
    fake_labels, fake_rec_thermal = discriminator(rec_images, training=False)
    real_labels, real_rec_thermal = discriminator(rgb_images, training=False)
    
    return {'THERMAL_IMGS': thermal_images, 'FAKE_IMGS': rec_images, 'FAKE_REC_THERMALS': fake_rec_thermal, 
            'REAL_IMGS': rgb_images, 'REAL_REC_THERMALS': real_rec_thermal}


if __name__ == '__main__':
    #READ CFG
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="path to cfg file")
    parser.add_argument("train_package_path", type=str, help="path to training package", nargs='?')
    args = parser.parse_args()
    with open(args.cfg_path, "r") as stream:
        cfg = yaml.load(stream)


    if args.train_package_path != None:
        with open(os.path.join(args.train_package_path,"cfg.yaml"), "r") as stream:
            train_cfg = yaml.load(stream)
        
    if cfg['filename'] == 'None':
        cfg['filename'] = 'inv_' + train_cfg['filename']
        
    cfg['GENERATOR'] = os.path.join(args.train_package_path,'weight',cfg['GENERATOR'])
    cfg['DISCRIMINATOR'] = os.path.join(args.train_package_path,'weight',cfg['DISCRIMINATOR'])
    
    for keypair in cfg_keypair_list:
        cfg_key = keypair[0]
        if len(keypair) == 1:
            train_cfg_key = keypair[0]
        else:
            train_cfg_key = keypair[1]
        
        if cfg[cfg_key] == 'None':
            cfg[cfg_key] = train_cfg[train_cfg_key]


    #MODEL AND OPTIMIZERS
    generator = keras.models.load_model(cfg['GENERATOR'])#load_model(cfg['GENERATOR']).get_model(cfg)
    discriminator = keras.models.load_model(cfg['DISCRIMINATOR'])#load_model(cfg['DISCRIMINATOR']).get_model(cfg)
    encoder = load_model(cfg['ENCODER']).get_model(cfg)
    generator.trainable=False
    
    e_optimizer = keras.optimizers.Adam(learning_rate=cfg['E_LR'], beta_1=cfg['E_B1'], beta_2=cfg['E_B2'])
    d_optimizer = keras.optimizers.Adam(learning_rate=cfg['D_LR'], beta_1=cfg['D_B1'], beta_2=cfg['D_B2'])
    e_ema = tf.train.ExponentialMovingAverage(decay=cfg['EMA_DECAY'])
    
    discriminator.summary()
    generator.summary()
    encoder.summary()
    encoder_ema_model = keras.models.clone_model(encoder)
    
    
    #MAKE LOG FOLDER
    folder_name = make_log_folder(os.path.join('experiments', cfg['filename']))
    sample_folder = os.path.join(folder_name,'sample')
    os.mkdir(sample_folder)
    weight_folder = os.path.join(folder_name,'weight')
    os.mkdir(weight_folder)
    
    #RECORD CFG
    with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
        yaml.dump(cfg, file, sort_keys=False)
    
    #LOAD DATASET
    thermal_images, rgb_images = load_lrt_human(cfg['THERMAL_PATH'], cfg['RGB_PATH'])
    v_thermal_images = thermal_images[:cfg['DISPLAY_NUM']]
    v_rgb_images = rgb_images[:cfg['DISPLAY_NUM']]
    print('thermal images shape: ', thermal_images.shape)
    print('rgb images shape: ', rgb_images.shape)
    
    logger = Logger(os.path.join(folder_name,"log.txt"))
    
    #TRAIN
    for epoch in range(cfg['EPOCHS']):
        thermal_images, rgb_images = shuffle(thermal_images, rgb_images) #!!!!
        for step in range(rgb_images.shape[0]//cfg['BATCH_SIZE']):
            thermal_batch = thermal_images[step*cfg['BATCH_SIZE']:(step+1)*cfg['BATCH_SIZE']]
            rgb_batch = rgb_images[step*cfg['BATCH_SIZE']:(step+1)*cfg['BATCH_SIZE']]
            loss = train_step(thermal_batch, rgb_batch)
            logger(loss, head_str="[E%iS%i]"%(epoch,step))
            
        encoder_ema_model.set_weights(encoder.get_weights())
        for var_copy, var in zip(encoder_ema_model.trainable_variables, encoder.trainable_variables):
            var_copy.assign(e_ema.average(var))
            
        info = validation(v_thermal_images, v_rgb_images, encoder_ema_model)
    
        vis_h1 = cv2.resize(tf.repeat(info['THERMAL_IMGS'][0],3,axis=-1).numpy(), (64,40), interpolation=cv2.INTER_NEAREST)
        vis_h2 = info['REAL_IMGS'][0]
        vis_h3 = info['FAKE_IMGS'][0].numpy()
        vis_h4 = cv2.resize(tf.repeat(info['FAKE_REC_THERMALS'][0],3,axis=-1).numpy(), (64,40), interpolation=cv2.INTER_NEAREST)
    
        for i in range(1,cfg['DISPLAY_NUM'],1):
            h1_img = cv2.resize(tf.repeat(info['THERMAL_IMGS'][i],3,axis=-1).numpy(), (64,40), interpolation=cv2.INTER_NEAREST)
            h2_img = info['REAL_IMGS'][i]
            h3_img = info['FAKE_IMGS'][i].numpy()
            h4_img = cv2.resize(tf.repeat(info['FAKE_REC_THERMALS'][i],3,axis=-1).numpy(), (64,40), interpolation=cv2.INTER_NEAREST)
        
            vis_h1 = np.concatenate((vis_h1,h1_img),axis=1)
            vis_h2 = np.concatenate((vis_h2,h2_img),axis=1)
            vis_h3 = np.concatenate((vis_h3,h3_img),axis=1)
            vis_h4 = np.concatenate((vis_h4,h4_img),axis=1)
        
        vis = np.concatenate((draw_border(vis_h1,color=(-1,0,-1)), draw_border(vis_h2,color=(-1,0,-1)), draw_border(vis_h3,color=(-1,-1,0)), draw_border(vis_h4,color=(-1,-1,0))), axis=0)
        cv2.imwrite(os.path.join(sample_folder,'%i_sample.png' % (epoch)), (vis*0.5+0.5)*255)
        
        if cfg['SAVE_INTERVAL'] != 0:
            if (epoch+1) % cfg['SAVE_INTERVAL'] == 0:
                encoder_ema_model.save(os.path.join(weight_folder, 'E_%i.h5' % (epoch+1)))



