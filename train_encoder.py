import os
import dnnlib
from dnnlib import EasyDict
import config
import copy


desc            = 'stylegan-encoder'
train           = EasyDict(run_func_name='training.training_loop_encoder.training_loop')
Encoder         = EasyDict(func_name='training.networks_encoder.Encoder')
E_opt           = EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)
D_opt           = EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)
E_loss          = EasyDict(func_name='training.loss_encoder.E_loss', feature_scale=0.00005, D_scale=0.05, perceptual_img_size=256)
D_loss          = EasyDict(func_name='training.loss_encoder.D_logistic_simplegp', r1_gamma=10.0)
lr              = EasyDict(learning_rate=0.0001, decay_step=30000, decay_rate=0.8, stair=False)
Data_dir        = EasyDict(data_train='', data_test='')
Decoder_pkl     = EasyDict(decoder_pkl='')
tf_config       = {'rnd.np_random_seed': 1000}
submit_config   = dnnlib.SubmitConfig()

# num_gpus = 1; desc += '-1gpu'
# num_gpus = 2; desc += '-2gpu'
# num_gpus = 4; desc += '-4gpu'
num_gpus = 8; desc += '-8gpu'

# image_size = 128;  desc += '-128x128';   total_kimg = 12000
image_size = 256;  desc += '-256x256';   total_kimg = 14000
# image_size = 512;  desc += '-512x512';   total_kimg = 16000
# image_size = 1024; desc += '-1024x1024'; total_kimg = 18000

dataset = 'ffhq';           desc += '-ffhq';             train.mirror_augment = True
# dataset = 'lsun-cat';      desc += '-lsun-cat';         train.mirror_augment = False
# dataset = 'lsun-car';      desc += '-lsun-car';         train.mirror_augment = False
# dataset = 'lsun-bedroom';  desc += '-lsun-bedroom';     train.mirror_augment = False

z_dim = 512

minibatch_per_gpu_train = {128: 16, 256: 16, 512: 8, 1024: 4}
minibatch_per_gpu_test  = {128: 1, 256: 1, 512: 1, 1024: 1}

assert image_size in minibatch_per_gpu_train, 'Invalid image size'
batch_size = minibatch_per_gpu_train.get(image_size) * num_gpus
batch_size_test = minibatch_per_gpu_test.get(image_size) * num_gpus
train.max_iters = int(total_kimg * 1000 / batch_size)

def main():
    kwargs = EasyDict(train)
    kwargs.update(Encoder_args=Encoder, E_opt_args=E_opt, D_opt_args=D_opt, E_loss_args=E_loss, D_loss_args=D_loss, lr_args=lr)
    kwargs.update(dataset_args=Data_dir, decoder_pkl=Decoder_pkl, tf_config=tf_config)
    kwargs.lr_args.decay_step = train.max_iters // 4
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.num_gpus = num_gpus
    kwargs.submit_config.image_size = image_size
    kwargs.submit_config.batch_size = batch_size
    kwargs.submit_config.batch_size_test = batch_size_test
    kwargs.submit_config.z_dim = z_dim
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc

    dnnlib.submit_run(**kwargs)

if __name__ == "__main__":
    main()
