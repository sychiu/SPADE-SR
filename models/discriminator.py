import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.blocks_v2 import resblock_down, resblock_up

#DISCRIMINATOR (BLOCKS FOLLOWING BIGGAN)
def get_model(cfg):
    
    img_input = layers.Input(shape=(cfg['IMG_SHAPE_Y'], cfg['IMG_SHAPE_X'], 3))
    
    c1 = resblock_down(img_input, cfg['D_CONV_CH'], pre_activation=False, neg_slope=cfg['RELU_NEG_SLOPE'])
    c2 = resblock_down(c1, cfg['D_CONV_CH']*2, neg_slope=cfg['RELU_NEG_SLOPE'])
    c3 = resblock_down(c2, cfg['D_CONV_CH']*4, neg_slope=cfg['RELU_NEG_SLOPE'])
    
    feature = layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(c3)
    if cfg['D_USE_GSP']:
        feature = tf.reduce_sum(feature, axis=[1,2])
    else:
        feature = layers.Flatten()(feature)

    label = layers.Dense(1)(feature)
    
    thermal_rec = resblock_down(c3, cfg['D_CONV_THERMAL_CH'], down_size=1, neg_slope=cfg['RELU_NEG_SLOPE'])
    thermal_rec = layers.ReLU(negative_slope=cfg['RELU_NEG_SLOPE'])(thermal_rec)
    thermal_rec = layers.Conv2D(1, (3,3), padding='same')(thermal_rec)
    thermal_rec = layers.Activation("tanh")(thermal_rec)

    d_model = keras.models.Model(img_input, [label, thermal_rec], name="discriminator")
    return d_model