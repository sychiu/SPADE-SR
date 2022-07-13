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
from fid import calc_fid
from utils import load_model, Logger, load_lrt_human, make_log_folder
from loss import d_source_loss, g_source_loss, calc_error, gradient_penalty, r1_penalty


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
def train_discriminator(thermal_images, rgb_images):
    random_latent_vectors = tf.random.normal(shape=(cfg['BATCH_SIZE'], cfg['G_Z_DIM']))

    with tf.GradientTape() as tape:
        fake_images = generator([thermal_images, random_latent_vectors], training=True)

        fake_labels, fake_rec_thermal = discriminator(fake_images, training=True)
        real_labels, real_rec_thermal = discriminator(rgb_images, training=True)
            
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
            
        if cfg['D_GP_WEIGHT'] != 0:
            if cfg['GP_METHOD'] == 'wgan':
                d_gp = gradient_penalty(rgb_images, fake_images, discriminator)
            elif cfg['GP_METHOD'] == 'R1':
                d_gp = r1_penalty(rgb_images, discriminator)
        else:
            d_gp = 0.0
            
        d_loss = d_sl + cfg['D_REC_WEIGHT']*(d_rl_fake+d_rl_real)*d_rl_factor + cfg['D_GP_WEIGHT']*d_gp

    gradient = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(gradient, discriminator.trainable_variables))
    d_ema.apply(discriminator.trainable_variables)

    loss = {'D_LOSS': d_loss, 'D_SL': d_sl, 'D_RL_REAL': d_rl_real, 'D_RL_FAKE': d_rl_fake, 'D_GP': d_gp}

    return loss


@tf.function
def train_generator(thermal_images, rgb_images):
    random_latent_vectors = tf.random.normal(shape=(cfg['BATCH_SIZE'], cfg['G_Z_DIM']))
        
    with tf.GradientTape() as tape:
        fake_images = generator([thermal_images, random_latent_vectors], training=True)
        fake_labels, fake_rec_thermal = discriminator(fake_images, training=True)

        g_sl = g_source_loss(fake_labels)
        g_rl = calc_error(thermal_images, fake_rec_thermal, mode=cfg['RECONSTRUCTION_LOSS'], hinge_thr=cfg['HINGE_REC_THR'])
        g_loss = g_sl + cfg['G_REC_WEIGHT']*g_rl

    gradient = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(gradient, generator.trainable_variables))
    g_ema.apply(generator.trainable_variables)

    return {'G_LOSS': g_loss, 'G_SL': g_sl, 'G_RL': g_rl}


def train_step(thermal_images, rgb_images):
    if cfg['BATCH_SIZE']!=thermal_images.shape[0] or cfg['BATCH_SIZE']!=rgb_images.shape[0]:
        raise ValueError

    for i in range(cfg['D_UPDATES']):
        d_loss = train_discriminator(thermal_images, rgb_images)

    g_loss = train_generator(thermal_images, rgb_images)
    
    return {**d_loss,**g_loss}


def validation(thermal_images, rgb_images, generator):
    random_latent_vectors = tf.random.normal(shape=(tf.shape(thermal_images)[0], cfg['G_Z_DIM']))
    fake_images = generator([thermal_images, random_latent_vectors], training=False)
    
    real_labels, real_rec_thermal = discriminator(rgb_images, training=False)
    fake_labels, fake_rec_thermal = discriminator(fake_images, training=False)
    
    return {'THERMAL_IMGS': thermal_images, 'FAKE_IMGS': fake_images, 'FAKE_REC_THERMALS': fake_rec_thermal, 
            'REAL_IMGS': rgb_images, 'REAL_REC_THERMALS': real_rec_thermal}


if __name__ == '__main__':
    #READ CFG
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="path to cfg file")
    args = parser.parse_args()
    with open(args.cfg_path, "r") as stream:
        cfg = yaml.load(stream)

    #MAKE LOG FOLDER
    folder_name = make_log_folder(os.path.join('experiments',cfg['filename']))
    sample_folder = os.path.join(folder_name,'sample')
    os.mkdir(sample_folder)
    weight_folder = os.path.join(folder_name,'weight')
    os.mkdir(weight_folder)
    
    #RECORD CFG
    with open(os.path.join(folder_name,'cfg.yaml'), 'a') as file:
        yaml.dump(cfg, file, sort_keys=False)
        
    #LOAD MODEL AND INIT OPTIMIZER
    if cfg['GENERATOR'][-3:]=='.h5':
        generator = keras.models.load_model(cfg['GENERATOR'])
    else:
        generator = load_model(cfg['GENERATOR']).get_model(cfg)
        
    if cfg['DISCRIMINATOR'][-3:]=='.h5':  
        discriminator = keras.models.load_model(cfg['DISCRIMINATOR'])
    else:
        discriminator = load_model(cfg['DISCRIMINATOR']).get_model(cfg)
    
    d_optimizer = keras.optimizers.Adam(learning_rate=cfg['D_LR'], beta_1=cfg['D_B1'], beta_2=cfg['D_B2'])
    g_optimizer = keras.optimizers.Adam(learning_rate=cfg['G_LR'], beta_1=cfg['G_B1'], beta_2=cfg['G_B2'])
    d_ema = tf.train.ExponentialMovingAverage(decay=cfg['EMA_DECAY'])
    g_ema = tf.train.ExponentialMovingAverage(decay=cfg['EMA_DECAY'])
    generator_ema_model = keras.models.clone_model(generator)
    discriminator_ema_model = keras.models.clone_model(discriminator)
    discriminator.summary()
    generator.summary()
    
    #LOAD DATASET
    thermal_images, rgb_images = load_lrt_human(cfg['THERMAL_PATH'], cfg['RGB_PATH'])
    v_thermal_images = thermal_images[:cfg['DISPLAY_NUM']]
    v_rgb_images = rgb_images[:cfg['DISPLAY_NUM']]
    print('thermal images shape: ', thermal_images.shape)
    print('rgb images shape: ', rgb_images.shape)

    logger = Logger(os.path.join(folder_name,"log.txt"))

    #TRAIN
    for epoch in range(cfg['EPOCHS']):
        thermal_images, rgb_images = shuffle(thermal_images, rgb_images)

        for step in range(rgb_images.shape[0]//cfg['BATCH_SIZE']):
            thermal_batch = thermal_images[step*cfg['BATCH_SIZE']:(step+1)*cfg['BATCH_SIZE']]
            rgb_batch = rgb_images[step*cfg['BATCH_SIZE']:(step+1)*cfg['BATCH_SIZE']]
            loss = train_step(thermal_batch, rgb_batch)
            logger(loss, "[E%iS%i]"%(epoch,step))
    
        #GET GENERATOR's EXPONENTIAL MOVING AVERAGE WEIGHT
        generator_ema_model.set_weights(generator.get_weights())
        for var_copy, var in zip(generator_ema_model.trainable_variables, generator.trainable_variables):
            var_copy.assign(g_ema.average(var))
                
        info = validation(v_thermal_images, v_rgb_images, generator_ema_model)
    
        #DRAW SAMPLES, LITTLE MESSY
        vis_h1 = cv2.resize(tf.repeat(info['THERMAL_IMGS'][0],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)
        vis_h2 = info['REAL_IMGS'][0]
        vis_h3 = info['FAKE_IMGS'][0].numpy()
        vis_h4 = cv2.resize(tf.repeat(info['FAKE_REC_THERMALS'][0],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)
        vis_h5 = cv2.resize(tf.repeat(info['REAL_REC_THERMALS'][0],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)  
                
        for i in range(1,cfg['DISPLAY_NUM'],1):

            h1_img = cv2.resize(tf.repeat(info['THERMAL_IMGS'][i],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)
            h2_img = info['REAL_IMGS'][i]
            h3_img = info['FAKE_IMGS'][i].numpy()
            h4_img = cv2.resize(tf.repeat(info['FAKE_REC_THERMALS'][i],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)
            h5_img = cv2.resize(tf.repeat(info['REAL_REC_THERMALS'][i],3,axis=-1).numpy(), (cfg['IMG_SHAPE_X'],cfg['IMG_SHAPE_Y']), interpolation=cv2.INTER_NEAREST)
    
            vis_h1 = np.concatenate((vis_h1,h1_img),axis=1)
            vis_h2 = np.concatenate((vis_h2,h2_img),axis=1)
            vis_h3 = np.concatenate((vis_h3,h3_img),axis=1)
            vis_h4 = np.concatenate((vis_h4,h4_img),axis=1)
            vis_h5 = np.concatenate((vis_h5,h5_img),axis=1)
    
        vis = np.concatenate((vis_h1, vis_h2, vis_h5, vis_h3, vis_h4), axis=0)
        cv2.imwrite(os.path.join(sample_folder,'%i_sample.png' % (epoch)), (vis*0.5+0.5)*255)
        
        #SAVE MODEL AND CALC FID
        if cfg['SAVE_INTERVAL'] != 0:
            if (epoch+1) % cfg['SAVE_INTERVAL'] == 0:
                if cfg['CALC_FID']:
                    print('[CALCULATING FID]')
                    generated_rgbs = []
                    for i in range(cfg['FID_SAMPLE_SIZE']//cfg['BATCH_SIZE']):
                        rand_idx = np.random.randint(0, high=thermal_images.shape[0], size=cfg['BATCH_SIZE'], dtype=int)
                        rand_thermal_image = thermal_images[rand_idx]
                        generated_rgb = generator_ema_model([rand_thermal_image,tf.random.normal(shape=(cfg['BATCH_SIZE'], cfg['G_Z_DIM']))], training=False)
                        generated_rgbs.extend(generated_rgb.numpy().tolist())
        
                    fid_value = calc_fid(cfg['REAL_STATS_PATH'], np.array(generated_rgbs), batch_size=cfg['FID_BATCH_SIZE'], rev_rgb=True)
                    generator_name = os.path.join(weight_folder, 'G_%i_%.2f.h5' % ((epoch+1), fid_value))
                    discriminator_name = os.path.join(weight_folder, 'D_%i_%.2f.h5' % ((epoch+1), fid_value))
                else:
                    generator_name = os.path.join(weight_folder, 'G_%i.h5' % (epoch+1))
                    discriminator_name = os.path.join(weight_folder, 'D_%i.h5' % (epoch+1))

                generator_ema_model.save(generator_name)
                discriminator_ema_model.set_weights(discriminator.get_weights())
                for var_copy, var in zip(discriminator_ema_model.trainable_variables, discriminator.trainable_variables):
                    var_copy.assign(d_ema.average(var))
                discriminator_ema_model.save(discriminator_name)

