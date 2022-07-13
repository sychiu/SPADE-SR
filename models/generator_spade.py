import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.blocks_v2 import resblock_down, resblock_up_spade, spade_bn

#GENERATOR
def get_model(cfg):

    thermal_input = layers.Input(shape=(5, 8, 1))
    
    noise = layers.Input(shape=(cfg['G_Z_DIM'],))
    x = layers.Dense(5*8*cfg['G_CONV_CH']*4, use_bias=False)(noise)
    x = layers.Reshape((5, 8, cfg['G_CONV_CH']*4))(x)
    x = resblock_up_spade(x, thermal_input*0.5+0.5, cfg['G_CONV_CH']*4, cfg['G_SPADE_FILTERS'], 1, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE']) #pre_activation?
    x = resblock_up_spade(x, thermal_input*0.5+0.5, cfg['G_CONV_CH']*2, cfg['G_SPADE_FILTERS'], 2, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])
    x = resblock_up_spade(x, thermal_input*0.5+0.5, cfg['G_CONV_CH'], cfg['G_SPADE_FILTERS'], 4, in_channels=cfg['G_CONV_CH']*2, neg_slope=cfg['RELU_NEG_SLOPE'])

    #x = layers.BatchNormalization()(x)
    x = spade_bn(x, thermal_input*0.5+0.5, cfg['G_CONV_CH'], cfg['G_SPADE_FILTERS'], 8)
    x = layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(x)
    
    out_img = layers.Conv2D(3, (3,3), padding='same', use_bias=True)(x)
    out_img = layers.Activation("tanh")(out_img)

    g_model = keras.models.Model([thermal_input, noise], out_img, name="generator")

    return g_model