import tensorflow as tf
import numpy as np
from tensorflow.python.training import moving_averages

try:
    from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
except Exception:
    pass
else:
    _validate_and_load_nccl_so()
from tensorflow.contrib.nccl.ops import gen_nccl_ops


def get_sync_bn_mean_var(x, axis, num_dev):
    coef = tf.constant(np.float32(1.0 / num_dev), name="coef")
    shared_name = tf.get_variable_scope().name
    shared_name = '_'.join(shared_name.split('/')[-2:])
    with tf.device(x.device):
        batch_mean = tf.reduce_mean(x, axis=axis)
        batch_mean = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean') * coef
    with tf.device(x.device):
        batch_mean_square = tf.reduce_mean(tf.square(x), axis=axis)
        batch_mean_square = gen_nccl_ops.nccl_all_reduce(
            input=batch_mean_square,
            reduction='sum',
            num_devices=num_dev,
            shared_name=shared_name + '_NCCL_mean_square') * coef

    batch_var = batch_mean_square - tf.square(batch_mean)

    return batch_mean, batch_var


def sync_batch_norm(inputs,
                    is_training,
                    num_dev=8,
                    center=True,
                    scale=True,
                    decay=0.9,
                    epsilon=1e-05,
                    data_format="NCHW",
                    updates_collections=None,
                    scope='batch_norm'
                    ):
    if data_format not in {"NCHW", "NHWC"}:
        raise ValueError(
            "Invalid data_format {}. Allowed: NCHW, NHWC.".format(data_format))
    with tf.variable_scope(scope, 'BatchNorm', values=[inputs]):
        inputs = tf.convert_to_tensor(inputs)
        # is_training = tf.cast(is_training, tf.bool)
        original_dtype = inputs.dtype
        original_shape = inputs.get_shape()
        original_input = inputs

        num_channels = inputs.shape[1].value if data_format == 'NCHW' else inputs.shape[-1].value
        if num_channels is None:
            raise ValueError("`C` dimension must be known but is None")

        original_rank = original_shape.ndims
        if original_rank is None:
            raise ValueError("Inputs %s has undefined rank" % inputs.name)
        elif original_rank not in [2, 4]:
            raise ValueError(
                "Inputs %s has unsupported rank."
                " Expected 2 or 4 but got %d" % (inputs.name, original_rank))

        # Bring 2-D inputs into 4-D format.
        if original_rank == 2:
            new_shape = [-1, 1, 1, num_channels]
            if data_format == "NCHW":
                new_shape = [-1, num_channels, 1, 1]
            inputs = tf.reshape(inputs, new_shape)
        input_shape = inputs.get_shape()
        input_rank = input_shape.ndims

        param_shape_broadcast = None
        if data_format == "NCHW":
            param_shape_broadcast = list([1, num_channels] +
                                         [1 for _ in range(2, input_rank)])

        moving_variables_collections = [
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.MODEL_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ]
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=[num_channels],
            initializer=tf.zeros_initializer(),
            trainable=False,
            partitioner=None,
            collections=moving_variables_collections)
        moving_variance = tf.get_variable(
            "moving_variance",
            shape=[num_channels],
            initializer=tf.ones_initializer(),
            trainable=False,
            partitioner=None,
            collections=moving_variables_collections)
        moving_vars_fn = lambda: (moving_mean, moving_variance)

        if not is_training:
            mean, variance = moving_vars_fn()
        else:
            axis = 1 if data_format == "NCHW" else 3
            inputs = tf.cast(inputs, tf.float32)
            moments_axes = [i for i in range(4) if i != axis]
            mean, variance = get_sync_bn_mean_var(inputs, axis=moments_axes, num_dev=num_dev)
            mean = tf.reshape(mean, [-1])
            variance = tf.reshape(variance, [-1])
            if updates_collections is None:
                def _force_update():
                    # Update variables for mean and variance during training.
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean,
                        tf.cast(mean, moving_mean.dtype),
                        decay,
                        zero_debias=False)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance,
                        tf.cast(variance, moving_variance.dtype),
                        decay,
                        zero_debias=False)
                    with tf.control_dependencies(
                            [update_moving_mean, update_moving_variance]):
                        return tf.identity(mean), tf.identity(variance)

                mean, variance = _force_update()
            else:
                def _delay_update():
                    # Update variables for mean and variance during training.
                    update_moving_mean = moving_averages.assign_moving_average(
                        moving_mean,
                        tf.cast(mean, moving_mean.dtype),
                        decay,
                        zero_debias=False)
                    update_moving_variance = moving_averages.assign_moving_average(
                        moving_variance,
                        tf.cast(variance, moving_variance.dtype),
                        decay,
                        zero_debias=False)
                    return update_moving_mean, update_moving_variance

                update_mean, update_variance = tf.cond(is_training,
                                                       _delay_update, moving_vars_fn)
                tf.add_to_collections(updates_collections, update_mean)
                tf.add_to_collections(updates_collections, update_variance)
                vars_fn = lambda: (mean, variance)
                mean, variance = vars_fn()

        variables_collections = [
            tf.GraphKeys.MODEL_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ]
        beta, gamma = None, None
        if scale:
            gamma = tf.get_variable(
                'gamma',
                [num_channels],
                collections=variables_collections,
                initializer=tf.ones_initializer())
        if center:
            beta = tf.get_variable(
                'beta',
                [num_channels],
                collections=variables_collections,
                initializer=tf.zeros_initializer())

        if data_format == 'NCHW':
            mean = tf.reshape(mean, param_shape_broadcast)
            variance = tf.reshape(variance, param_shape_broadcast)
            if beta is not None:
                beta = tf.reshape(beta, param_shape_broadcast)
            if gamma is not None:
                gamma = tf.reshape(gamma, param_shape_broadcast)

        outputs = tf.nn.batch_normalization(
            inputs,
            mean=mean,
            variance=variance,
            offset=beta,
            scale=gamma,
            variance_epsilon=epsilon)
        outputs = tf.cast(outputs, original_dtype)

        outputs.set_shape(input_shape)
        # Bring 2-D inputs back into 2-D format.
        if original_rank == 2:
            outputs = tf.reshape(outputs, tf.shape(original_input))

        return outputs


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02)) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.random_normal_initializer(stddev=0.02))


def dense(x, fmaps, gain=1., use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)


def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.constant_initializer(0.0))
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return x + tf.reshape(b, [1, -1, 1, 1])


def leaky_relu(x, alpha=0.2):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)


def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, 1, factor, factor]
        return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]), 1, np.int32(s[3]), 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, np.int32(s[1]), np.int32(s[2]) * factor, np.int32(s[3]) * factor])
        return x


def shortcut(inputs, fout):
    x_shortcut = conv2d(inputs, fmaps=fout, kernel=1)
    return x_shortcut


def residual_block2(inputs, fin, fout, is_training, num_dev, scope):
    hidden = min(fin, fout)
    learned = (fin != fout)
    with tf.variable_scope(scope):
        with tf.variable_scope('Shortcut'):
            if learned:
                x_shortcut = shortcut(inputs, fout)
                x_shortcut = sync_batch_norm(x_shortcut, is_training=is_training, num_dev=num_dev)
            else:
                x_shortcut = inputs
        with tf.variable_scope('Conva'):
            net = apply_bias(conv2d(inputs, fmaps=hidden, kernel=3))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        with tf.variable_scope('Convb'):
            net = apply_bias(conv2d(net, fmaps=fout, kernel=3))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        net = net + x_shortcut

    return net


def residual_block3(inputs, fin, fout, is_training, num_dev, scope):
    hidden = min(fin, fout)
    learned = (fin != fout)
    with tf.variable_scope(scope):
        with tf.variable_scope('Shortcut'):
            if learned:
                x_shortcut = shortcut(inputs, fout)
                x_shortcut = sync_batch_norm(x_shortcut, is_training=is_training, num_dev=num_dev)
            else:
                x_shortcut = inputs
        with tf.variable_scope('Conva'):
            net = apply_bias(conv2d(inputs, fmaps=hidden, kernel=1))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        with tf.variable_scope('Convb'):
            net = apply_bias(conv2d(net, fmaps=hidden, kernel=3))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        with tf.variable_scope('Convc'):
            net = apply_bias(conv2d(net, fmaps=fout, kernel=1))
            net = leaky_relu(net)
            net = sync_batch_norm(net, is_training=is_training, num_dev=num_dev)
        net = net + x_shortcut

    return net


def Encoder(
            Input_img,                 # Input image: [Minibatch, Channel, Height, Width].
            size             = 256,    # Input image size.
            filter           = 64,     # Minimum number of feature maps in any layer.
            filter_max       = 512,    # Maximum number of feature maps in any layer.
            num_layers       = 14,     # Number of layers in in G_synthesis().
            is_training      = True,   # Whether or not the layer is in training mode?
            num_gpus         = 8,      # Number of gpus to use
            dlatent_size     = 512,    # Disentangled latent (W) dimensionality.
            s0               = 4,      # Base number to decide how many residual block in encoder.
            **kwargs):
    num_blocks = int(np.log2(size / s0))

    Input_img.set_shape([None, 3, size, size])
    with tf.variable_scope('FromImg'):
        net = apply_bias(conv2d(Input_img, fmaps=filter, kernel=5))
        net = leaky_relu(net)
        net = sync_batch_norm(net, is_training=is_training, num_dev=num_gpus)

    for i in range(num_blocks):
        name_scope = 'E_block_%d' % (i)
        nf1 = min(filter * 2 ** i, filter_max)
        nf2 = min(filter * 2 ** (i + 1), filter_max)
        net = downscale2d(net)
        net = residual_block2(net, fin=nf1, fout=nf2,
                              is_training=is_training, num_dev=num_gpus, scope=name_scope)

    with tf.variable_scope('Latent_out'):
        latent_w = apply_bias(dense(net, fmaps=dlatent_size * num_layers, use_wscale=False))
        latent_w = sync_batch_norm(latent_w, is_training=is_training, num_dev=num_gpus)

    return latent_w
