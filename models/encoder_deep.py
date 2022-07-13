import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.blocks_v2 import resblock_down, resblock_up

#DISCRIMINATOR (BLOCKS FOLLOWING BIGGAN)
def get_model(cfg):
    
    img_input = layers.Input(shape=(cfg['IMG_SHAPE_Y'], cfg['IMG_SHAPE_X'], 3))

    x = resblock_down(img_input, cfg['E_CONV_CH'], use_bn=True, pre_activation=False, neg_slope=cfg['RELU_NEG_SLOPE'])
    x = resblock_down(x, cfg['E_CONV_CH']*2, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])
    x = resblock_down(x, cfg['E_CONV_CH']*4, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])
    x = resblock_down(x, cfg['E_CONV_CH']*8, use_bn=True, neg_slope=cfg['RELU_NEG_SLOPE'])

    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(x)

    if cfg['E_USE_GSP']:
        x = tf.reduce_sum(x, axis=[1,2])
    else:
        x = layers.Flatten()(x)

    z = layers.Dense(cfg['G_Z_DIM'], use_bias=False)(x)
    z = layers.BatchNormalization()(z)

    e_model = keras.models.Model(img_input, z, name="encoder")
    return e_model
