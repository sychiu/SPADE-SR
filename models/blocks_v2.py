import tensorflow as tf
from tensorflow.keras import layers

def resblock_up(
    x,
    filters,
    kernel_size=(3,3),
    up_size=2,
    use_bn=True,
    pre_activation=True,
    pre_bn=True,
    neg_slope=0.0,
    post_activation_insert=None,
    use_post_activation_insert=False
):
    
    skip = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    skip = layers.Conv2D(filters, (1,1), strides=(1,1), padding="same", use_bias=not use_bn)(skip)

    x = layers.BatchNormalization()(x) if (use_bn and pre_bn) else x
    x = layers.ReLU(negative_slope=neg_slope)(x) if pre_activation else x
    x = layers.Concatenate(axis=-1)([x, post_activation_insert]) if use_post_activation_insert else x
    x = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=neg_slope)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.Add()([x,skip])

    return x


def resblock_down(
    x,
    filters,
    kernel_size=(3,3),
    down_size=2,
    use_bn=False,
    pre_activation=True,
    neg_slope=0.0
):

    if pre_activation:
        skip = layers.Conv2D(filters, (1,1), padding="same", use_bias=not use_bn)(x)
        skip = layers.AveragePooling2D(pool_size=(down_size, down_size))(skip) if down_size > 1 else skip
    else:
        skip = layers.AveragePooling2D(pool_size=(down_size, down_size))(x) if down_size > 1 else x
        skip = layers.Conv2D(filters, (1,1), padding="same", use_bias=not use_bn)(skip)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=neg_slope)(x) if pre_activation else x
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=neg_slope)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.AveragePooling2D(pool_size=(down_size, down_size))(x) if down_size > 1 else x
    x = layers.Add()([x,skip])

    return x

def spade_bn(
    x, 
    m, 
    x_ch,
    spade_filters, 
    m_up_size,
    kernel_size=(3,3)
):
    
    x = layers.BatchNormalization(center=False, scale=False)(x)
    
    if spade_filters != False:
        m = layers.UpSampling2D((m_up_size, m_up_size), interpolation="nearest")(m)
        #m = tf.compat.v1.image.resize(m, (tf.shape(x)[1], tf.shape(x)[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        m = layers.Conv2D(spade_filters, kernel_size, padding="same")(m)
        m = layers.ReLU(negative_slope=0.0)(m)

    gamma = layers.Conv2D(x_ch, kernel_size, padding="same")(m)
    beta = layers.Conv2D(x_ch, kernel_size, padding="same")(m)
    
    out = x * (1 + gamma) + beta
    
    return out


def resblock_up_spade(
    x,
    m,
    filters,
    spade_filters,
    spade_up_size,
    in_channels,
    kernel_size=(3,3),
    up_size=2,
    pre_activation=True,
    neg_slope=0.0
):
    
    skip = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    skip = layers.Conv2D(filters, (1,1), strides=(1,1), padding="same", use_bias=False)(skip)

    x = spade_bn(x, m, in_channels, spade_filters, spade_up_size)
    x = layers.ReLU(negative_slope=neg_slope)(x) if pre_activation else x
    x = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)

    x = spade_bn(x, m, filters, spade_filters, spade_up_size*2)
    x = layers.ReLU(negative_slope=neg_slope)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)

    x = layers.Add()([x,skip])

    return x

def resblock_up_spade_sr(
    x,
    m1,
    m2,
    filters,
    spade_filters,
    spade_up_size,
    in_channels,
    kernel_size=(3,3),
    up_size=2,
    pre_activation=True,
    neg_slope=0.0
):
    
    skip = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    skip = layers.Conv2D(filters, (1,1), strides=(1,1), padding="same", use_bias=False)(skip)

    x = spade_bn(x, m1, in_channels, False, None)
    x = layers.ReLU(negative_slope=neg_slope)(x) if pre_activation else x
    x = layers.UpSampling2D((up_size, up_size), interpolation="nearest")(x) if up_size > 1 else x
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)

    x = spade_bn(x, m2, filters, False, None)
    x = layers.ReLU(negative_slope=neg_slope)(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)(x)

    x = layers.Add()([x,skip])

    return x


class Conv2D_SN(layers.Conv2D):

    def build(self, input_shape):
        super(Conv2D_SN, self).build(input_shape)
        self.u = self.add_weight(
            name='u_value',
            shape=[1,self.kernel.shape.as_list()[-1]],
            initializer=tf.truncated_normal_initializer(),
            regularizer=None,
            constraint=None,
            trainable=False,
            dtype=self.dtype)
    
    def call(self, inputs, training=None):
        if training in {1, True}:
            self.kernel.assign(self.spectral_normed_weight(self.kernel))
        return super(Conv2D_SN, self).call(inputs)
    
    def _l2normalize(self, v, eps=1e-12):
        """l2 normize the input vector."""
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def spectral_normed_weight(self, weights, num_iters=1, update_collection=None, with_sigma=False):
        
        w_shape = weights.shape.as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
        u_ = self.u
        for _ in range(num_iters):
          v_ = self._l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
          u_ = self._l2normalize(tf.matmul(v_, w_mat))
        v_ = tf.stop_gradient(v_)
        u_ = tf.stop_gradient(u_)
        #STOP GRADIENMT?
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        w_mat /= sigma
        if update_collection is None:
          #with tf.control_dependencies([u.assign(u_)]):??
          self.u.assign(u_)
          w_bar = tf.reshape(w_mat, w_shape)
        else:
          w_bar = tf.reshape(w_mat, w_shape)
          if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
        if with_sigma:
          return w_bar, sigma
        else:
          return w_bar


class Dense_SN(layers.Dense):

    def build(self, input_shape):
        super(Dense_SN, self).build(input_shape)
        self.u = self.add_weight(
            name='u_value',
            shape=[1,self.kernel.shape.as_list()[-1]],
            initializer=tf.truncated_normal_initializer(),
            regularizer=None,
            constraint=None,
            trainable=False,
            dtype=self.dtype)

    def call(self, inputs):
        self.kernel.assign(self.spectral_normed_weight(self.kernel))
        return super(Dense_SN, self).call(inputs)
    
    def _l2normalize(self, v, eps=1e-12):
        """l2 normize the input vector."""
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

    def spectral_normed_weight(self, weights, num_iters=1, update_collection=None, with_sigma=False):
        """Performs Spectral Normalization on a weight tensor.
        Specifically it divides the weight tensor by its largest singular value. This
        is intended to stabilize GAN training, by making the discriminator satisfy a
        local 1-Lipschitz constraint.
        Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]
        [sn-gan] https://openreview.net/pdf?id=B1QRgziT-
        Args:
            weights: The weight tensor which requires spectral normalization
            num_iters: Number of SN iterations.
            update_collection: The update collection for assigning persisted variable u.
                               If None, the function will update u during the forward
                               pass. Else if the update_collection equals 'NO_OPS', the
                               function will not update the u during the forward. This
                               is useful for the discriminator, since it does not update
                               u in the second pass.
                               Else, it will put the assignment in a collection
                               defined by the user. Then the user need to run the
                               assignment explicitly.
            with_sigma: For debugging purpose. If True, the fuction returns
                        the estimated singular value for the weight tensor.
        Returns:
            w_bar: The normalized weight tensor
            sigma: The estimated singular value for the weight tensor.
        """
        w_shape = weights.shape.as_list()
        w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
        u_ = self.u
        for _ in range(num_iters):
          v_ = self._l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
          u_ = self._l2normalize(tf.matmul(v_, w_mat))
        
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
        w_mat /= sigma
        if update_collection is None:
          #with tf.control_dependencies([u.assign(u_)]):
          self.u.assign(u_)
          w_bar = tf.reshape(w_mat, w_shape)
        else:
          w_bar = tf.reshape(w_mat, w_shape)
          if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
        if with_sigma:
          return w_bar, sigma
        else:
          return w_bar





def resblock_down_sn_NEW(
    x,
    filters,
    kernel_size=(3,3),
    down_size=(1,1),
    padding="same",
    use_bn=False,
    use_bias=True,
    drop_value=0
):
    
    skip = Conv2D_SN(filters, (1,1), strides=(1,1), padding=padding, use_bias=use_bias)(x)
    skip = layers.AveragePooling2D(pool_size=down_size)(skip)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=0)(x)
    x = layers.Dropout(drop_value)(x) if drop_value>0 else x
    x = Conv2D_SN(filters, kernel_size, strides=(1,1), padding=padding, use_bias=use_bias)(x)

    x = layers.BatchNormalization()(x) if use_bn is True else x
    x = layers.ReLU(negative_slope=0)(x)
    x = layers.Dropout(drop_value)(x) if drop_value>0 else x
    x = Conv2D_SN(filters, kernel_size, strides=(1,1), padding=padding, use_bias=use_bias)(x)
    x = layers.AveragePooling2D(pool_size=down_size)(x)

    x = layers.Add()([x,skip])
    return x

def resblock_down_sn(
    x,
    filters,
    kernel_size=(3,3),
    down_size=(2,2),
    use_bn=False
):
    
    skip = Conv2D_SN(filters, (1,1), padding="same", use_bias=not use_bn)(x)
    skip = layers.AveragePooling2D(pool_size=down_size)(skip)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=0)(x)
    x = Conv2D_SN(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=0)(x)
    x = Conv2D_SN(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.AveragePooling2D(pool_size=down_size)(x)
    x = layers.Add()([x,skip])

    return x

def resblock_up_sn(
    x,
    filters,
    kernel_size=(3,3),
    up_size=(2,2),
    use_bn=True
):
    
    skip = layers.UpSampling2D(up_size, interpolation="nearest")(x)
    skip = Conv2D_SN(filters, (1,1), strides=(1,1), padding="same", use_bias=not use_bn)(skip)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=0)(x)
    x = layers.UpSampling2D(up_size, interpolation="nearest")(x)
    x = Conv2D_SN(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.BatchNormalization()(x) if use_bn else x
    x = layers.ReLU(negative_slope=0)(x)
    x = Conv2D_SN(filters, kernel_size, padding="same", use_bias=not use_bn)(x)

    x = layers.Add()([x,skip])

    return x


