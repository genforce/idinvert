import argparse
import dnnlib
from dnnlib import EasyDict
import config
import copy

def main():
    parser = argparse.ArgumentParser(description='Training the in-domain encoder')
    parser.add_argument('training_data', type=str,
                        help='path to training data (.tfrecords).')
    parser.add_argument('test_data', type=str,
                        help='path to test data (.tfrecords).')
    parser.add_argument('decoder_pkl', default=str,
                        help='path to the stylegan generator, which serves as a decoder here.')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use during training (defaults: 8)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='the image size in training dataset (defaults; 256)')
    parser.add_argument('--dataset_name', type=str, default='ffhq',
                        help='the name of the training dataset (defaults; ffhq)')
    parser.add_argument('--mirror_augment', action='store_false',
                        help='Mirror augment (default: True)')
    args = parser.parse_args()

    train           = EasyDict(run_func_name='training.training_loop_encoder.training_loop')
    Encoder         = EasyDict(func_name='training.networks_encoder.Encoder')
    E_opt           = EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)
    D_opt           = EasyDict(beta1=0.9, beta2=0.99, epsilon=1e-8)
    E_loss          = EasyDict(func_name='training.loss_encoder.E_loss', feature_scale=0.00005, D_scale=0.08, perceptual_img_size=256)
    D_loss          = EasyDict(func_name='training.loss_encoder.D_logistic_simplegp', r1_gamma=10.0)
    lr              = EasyDict(learning_rate=0.0001, decay_step=30000, decay_rate=0.8, stair=False)
    Data_dir        = EasyDict(data_train=args.training_data, data_test=args.test_data)
    Decoder_pkl     = EasyDict(decoder_pkl=args.decoder_pkl)
    tf_config       = {'rnd.np_random_seed': 1000}
    submit_config   = dnnlib.SubmitConfig()

    desc = 'stylegan-encoder'
    desc += '-%dgpu' % (args.num_gpus)
    desc += '-%dx%d' % (args.image_size, args.image_size)
    desc += '-%s' % (args.dataset_name)

    train.mirror_augment = args.mirror_augment
    minibatch_per_gpu_train = {128: 16, 256: 16, 512: 8, 1024: 4}
    minibatch_per_gpu_test  = {128: 1, 256: 1, 512: 1, 1024: 1}
    total_kimgs = {128: 12000, 256: 14000, 512: 16000, 1024: 18000}

    assert args.image_size in minibatch_per_gpu_train, 'Invalid image size'
    batch_size = minibatch_per_gpu_train.get(args.image_size) * args.num_gpus
    batch_size_test = minibatch_per_gpu_test.get(args.image_size) * args.num_gpus
    train.max_iters = int(total_kimgs.get(args.image_size) * 1000 / batch_size)

    kwargs = EasyDict(train)
    kwargs.update(Encoder_args=Encoder, E_opt_args=E_opt, D_opt_args=D_opt, E_loss_args=E_loss, D_loss_args=D_loss, lr_args=lr)
    kwargs.update(dataset_args=Data_dir, decoder_pkl=Decoder_pkl, tf_config=tf_config)
    kwargs.lr_args.decay_step = train.max_iters // 4
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.num_gpus = args.num_gpus
    kwargs.submit_config.image_size = args.image_size
    kwargs.submit_config.batch_size = batch_size
    kwargs.submit_config.batch_size_test = batch_size_test
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc

    dnnlib.submit_run(**kwargs)


if __name__ == "__main__":
    main()
