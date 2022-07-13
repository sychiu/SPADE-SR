import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.blocks_v2 import resblock_down, resblock_up_spade_sr, spade_bn, resblock_up

#GENERATOR
def get_model(cfg):

    thermal_input = layers.Input(shape=(5, 8, 1))
    thermal_c1 = resblock_up(thermal_input, cfg['G_CONV_THERMAL_CH'], pre_activation=False, use_bn=False, neg_slope=cfg['RELU_NEG_SLOPE'], up_size=1)
    thermal_c2 = resblock_up(thermal_c1, cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
    thermal_c3 = resblock_up(thermal_c2, cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
    thermal_c4 = resblock_up(thermal_c3, cfg['G_CONV_THERMAL_CH'], neg_slope=cfg['RELU_NEG_SLOPE'], use_bn=False)
    
    noise = layers.Input(shape=(cfg['G_Z_DIM'],))
    x = layers.Dense(5*8*cfg['G_CONV_CH']*4, use_bias=False)(noise)
    x = layers.Reshape((5, 8, cfg['G_CONV_CH']*4))(x)
    x = resblock_up_spade_sr(x, layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c1), layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c2), cfg['G_CONV_CH']*4, False, None, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE']) #pre_activation?
    x = resblock_up_spade_sr(x, layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c2), layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c3), cfg['G_CONV_CH']*2, False, None, in_channels=cfg['G_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])
    x = resblock_up_spade_sr(x, layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c3), layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c4), cfg['G_CONV_CH'], False, None, in_channels=cfg['G_CONV_CH']*2, neg_slope=cfg['RELU_NEG_SLOPE'])
    
    x = spade_bn(x, layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_c4), cfg['G_CONV_CH'], False, None)
    x = layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(x)
    
    out_img = layers.Conv2D(3, (3,3), padding='same', use_bias=True)(x)
    out_img = layers.Activation("tanh")(out_img)

    g_model = keras.models.Model([thermal_input, noise], out_img, name="generator")

    return g_model