import tensorflow as tf

def d_source_loss(real_labels, fake_labels, mode='hinge'):
    if mode == 'wgan':
        return tf.reduce_mean(fake_labels)-tf.reduce_mean(real_labels)
    elif mode == 'hinge':
        return tf.reduce_mean(tf.nn.relu(1.0-real_labels))+tf.reduce_mean(tf.nn.relu(1.0+fake_labels))
    

def calc_error(arr1, arr2, mode='l1', hinge_thr=0.0):
    if mode == 'l1':
        return tf.reduce_mean(tf.math.abs(arr1-arr2))
    elif mode == 'l2':
        return tf.reduce_mean((arr1-arr2)**2)
    elif mode == 'hinge':
        return tf.reduce_mean(tf.nn.relu(tf.math.abs(arr1-arr2)-hinge_thr))


def gradient_penalty(real_images, fake_images, discriminator):
    alpha = tf.random.normal([tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        label, rec = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(label, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm) ** 2)
    
    return gp


def r1_penalty(images, discriminator):

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(images)
        label, rec = discriminator(images, training=True)
    grads = gp_tape.gradient(label, [images])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = 0.5 * tf.reduce_mean((norm) ** 2)
    
    return gp


def g_source_loss(fake_labels):
    return -tf.reduce_mean(fake_labels)