import os
import glob
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle

'''
Please BE CAREFUL of the image channel order. The model excepts *RGB*. (OpenCV (cv2) reads and writes in *BGR*)
Images in LRT-Human .npy file are in BGR
注意！請留意相片的顏色通道。OpenCV (cv2) 讀取和寫入照片時，以*BGR*排列，而非模型需要的*RGB*。
儲存在.npy檔案裡的LRT-Human照片是BGR形式

The model expects a pixel value range of 0~255
照片數值要在0~255之間
'''

output_path = 'fid_data/real_stats_10k.npz' #path for where to store the statistics

images = np.load('data/rgb.npy')
images = images[:,:,:,::-1] * 127.5 + 127.5
images = shuffle(images)
images = images[:10000]


#ORIGINAL IMPLEMENTATION FROM TTUR
'''
data_path = ''
# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))
'''



########
# PATHS
########
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you

inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

print("%d images found and loaded" % len(images))
print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
