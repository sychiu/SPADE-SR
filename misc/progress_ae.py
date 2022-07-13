import matplotlib.pyplot as plt
import numpy as np
#import cv2
#import tensorflow as tf
#tf.enable_eager_execution()
import argparse
import yaml
import os
import time





parser = argparse.ArgumentParser()
parser.add_argument("foldername", type=str, help="path of log folder")
#parser.add_argument("savepath", type=str, help="path to save graph")
parser.add_argument("window_size", type=int, help="moving average window size")
args = parser.parse_args()

'''
x = tf.constant(0.79999)
x_head = tf.constant(0.8)
xx_head = tf.constant(0.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    error = (x_head-x)**2
    y = tf.math.abs(xx_head - error)
print(tape.gradient(y,x))
dd
'''

np.seterr(all='print')

foldername = args.foldername
log_filename = os.path.join(foldername,'log.txt')
cfg_filename = os.path.join(foldername,'cfg.yaml')



attrs = []
losses = []
with open(log_filename) as file: 
    line = file.readline()
    attrs = line.split(',')[:-1]
    for line in file.readlines():
        segs = line.split(',')[1:-1]
        losses.append(segs)

losses = np.array(losses, dtype=np.float32)
#losses = losses[64:]

losses = np.swapaxes(losses,0,1)


def moving_average_1d(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
w_size = args.window_size
ma_losses = []
for i in range(losses.shape[0]):
    ma_losses.append(moving_average_1d(losses[i],w_size))

ma_losses = np.array(ma_losses)


colors = np.random.uniform(0.0,1.0,size=(ma_losses.shape[0],3))
lw = np.ones(ma_losses.shape[0])
ls = ['solid']*ma_losses.shape[0]

'''colors[0] = [0.2,0.2,0.2]
lw[0] = 1
ls[0] = 'dashed'
colors[1] = [1.0,0.8,0.8]
lw[1] = 1
ls[1] = 'dashed'

colors[2] = [0.2,0.2,0.2]
lw[2] = 2
ls[2] = 'dashed'

colors[3] = [0.5,0.5,0.5]
lw[3] = 1
ls[3] = 'dashed'

colors[4] = [0.9,0.0,0.2]
lw[4] = 1

colors[5] = [1.0,0.8,0.8]
lw[5] = 1

colors[6] = [0.9,0.7,0.0]
lw[6] = 2

colors[7] = [0.0,0.4,0.8]
lw[7] = 2

colors[8] = [0.3,0.7,0.0]
lw[8] = 2

colors[9] = [0.5,0.5,0.5]
lw[9] = 1
'''



#ls[2] = 'dashed'

'''try:
    ls[0] = 'solid'
    ls[1] = 'solid'
    ls[2] = 'solid'
    ls[4] = 'solid'
    
    lw[0] = 2
    lw[1] = 2
    lw[4] = 2
    colors[0] = [0.9,0.1,0.1]
    colors[1] = [0.3,0.7,0.2]
    colors[2] = [0.9,0.4,0.1]
    colors[3] = [0.9,0.9,0.9]
    colors[4] = [0.1,0.1,0.6]
    colors[5] = [0.9,0.4,0.1]
    colors[6] = [0.9,0.8,0.8]
    colors[7] = [0.2,0.2,0.6]
    colors[8] = [0.7,0.7,1.0]
    colors[9] = [0.1,0.9,0.9]
except:
    pass'''


plt.rcParams["figure.figsize"] = (6,7)
fig, axs = plt.subplots(2)
axs[0].grid(True)
axs[1].grid(True)


#UPPER GRAPH
'''lines = []
for i in range(4):
    lines.append(axs[0].plot(np.arange(ma_losses.shape[1]),ma_losses[i],color=colors[i],label=attrs[i],linewidth=lw[i],ls=ls[i])[0])
axs[0].legend(handles=lines, loc='upper right', prop={'size': 8})
axs[0].set_ylim([0,0.4])
'''
#LOWER GRAPH
lines = []
for i in range(ma_losses.shape[0]):
    lines.append(axs[1].plot(np.arange(ma_losses.shape[1]),ma_losses[i],color=colors[i],label=attrs[i],linewidth=lw[i],ls=ls[i])[0])
axs[1].legend(handles=lines, loc='upper right', prop={'size': 8})
axs[1].set_ylim([0.0,1])

#TITLES
#with open(cfg_filename, "r") as stream:
    #cfg = yaml.load(stream)
#title = 'CONF%.5f  TGT%.5f' % (cfg['loss_weights']['C_CONF'],cfg['loss_weights']['C_TARGET'])
#title += '  (%s)'%os.path.basename(os.path.normpath(foldername))
#axs[0].set_title(title)

plt.tight_layout()
plt.show()


#plt.savefig(os.path.join(args.savepath,'%.1f.pdf' % time.time()))


        