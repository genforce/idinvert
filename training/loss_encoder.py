"""Loss functions for training encoder."""

import numpy as np
import tensorflow as tf
from dnnlib.tflib.autosummary import autosummary


#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


#----------------------------------------------------------------------------
# Encoder loss function .
def E_loss(E, G, D, perceptual_model, reals, feature_scale=0.00005, D_scale=0.1, perceptual_img_size=256):
    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    latent_w = E.get_output_for(reals, phase=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    fake_scores_out = fp32(D.get_output_for(fake_X, None))

    with tf.variable_scope('recon_loss'):
        vgg16_input_real = tf.transpose(reals, perm=[0, 2, 3, 1])
        vgg16_input_real = tf.image.resize_images(vgg16_input_real, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_real = ((vgg16_input_real + 1) / 2) * 255
        vgg16_input_fake = tf.transpose(fake_X, perm=[0, 2, 3, 1])
        vgg16_input_fake = tf.image.resize_images(vgg16_input_fake, size=[perceptual_img_size, perceptual_img_size], method=1)
        vgg16_input_fake = ((vgg16_input_fake + 1) / 2) * 255
        vgg16_feature_real = perceptual_model(vgg16_input_real)
        vgg16_feature_fake = perceptual_model(vgg16_input_fake)
        recon_loss_feats = feature_scale * tf.reduce_mean(tf.square(vgg16_feature_real - vgg16_feature_fake))
        recon_loss_pixel = tf.reduce_mean(tf.square(fake_X - reals))
        recon_loss_feats = autosummary('Loss/scores/loss_feats', recon_loss_feats)
        recon_loss_pixel = autosummary('Loss/scores/loss_pixel', recon_loss_pixel)
        recon_loss = recon_loss_feats + recon_loss_pixel
        recon_loss = autosummary('Loss/scores/recon_loss', recon_loss)

    with tf.variable_scope('adv_loss'):
        D_scale = autosummary('Loss/scores/d_scale', D_scale)
        adv_loss = D_scale * tf.reduce_mean(tf.nn.softplus(-fake_scores_out))
        adv_loss = autosummary('Loss/scores/adv_loss', adv_loss)

    loss = recon_loss + adv_loss

    return loss, recon_loss, adv_loss

#----------------------------------------------------------------------------
# Discriminator loss function.
def D_logistic_simplegp(E, G, D, reals, r1_gamma=10.0):

    num_layers, latent_dim = G.components.synthesis.input_shape[1:3]
    latent_w = E.get_output_for(reals, phase=True)
    latent_wp = tf.reshape(latent_w, [reals.shape[0], num_layers, latent_dim])
    fake_X = G.components.synthesis.get_output_for(latent_wp, randomize_noise=False)
    real_scores_out = fp32(D.get_output_for(reals, None))
    fake_scores_out = fp32(D.get_output_for(fake_X, None))

    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss_fake = tf.reduce_mean(tf.nn.softplus(fake_scores_out))
    loss_real = tf.reduce_mean(tf.nn.softplus(-real_scores_out))

    with tf.name_scope('R1Penalty'):
        real_grads = fp32(tf.gradients(real_scores_out, [reals])[0])
        r1_penalty = tf.reduce_mean(tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3]))
        r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss_gp = r1_penalty * (r1_gamma * 0.5)
    loss = loss_fake + loss_real + loss_gp
    return loss, loss_fake, loss_real, loss_gp
