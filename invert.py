# python 3.6
"""Inverts given images to latent codes with In-Domain GAN Inversion.

Basically, for a particular image (real or synthesized), this script first
employs the domain-guided encoder to produce a initial point in the latent
space and then performs domain-regularized optimization to refine the latent
code.
"""

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from perceptual_model import PerceptualModel
from utils.logger import setup_logger
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import save_image, load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('image_list', type=str,
                      help='List of images to invert.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/inversion/${IMAGE_LIST}` '
                           'will be used by default.')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size. (default: 4)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('-R', '--random_init', action='store_true',
                      help='Whether to use random initialization instead of '
                           'the output from encoder. (default: False)')
  parser.add_argument('-E', '--domain_regularizer', action='store_false',
                      help='Whether to use domain regularizer for '
                           'optimization. (default: True)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--loss_weight_enc', type=float, default=2.0,
                      help='The encoder loss scale for optimization.'
                           '(default: 2.0)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.image_list)
  image_list_name = os.path.splitext(os.path.basename(args.image_list))[0]
  output_dir = args.output_dir or f'results/inversion/{image_list_name}'
  logger = setup_logger(output_dir, 'inversion.log', 'inversion_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs = pickle.load(f)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  input_shape = E.input_shape
  input_shape[0] = args.batch_size
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  x_255 = (tf.transpose(x, [0, 2, 3, 1]) + 1) / 2 * 255
  latent_shape = Gs.components.synthesis.input_shape
  latent_shape[0] = args.batch_size
  wp = tf.get_variable(shape=latent_shape, name='latent_code')
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  x_rec_255 = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) / 2 * 255
  if args.random_init:
    logger.info(f'  Use random initialization for optimization.')
    wp_rnd = tf.random.normal(shape=latent_shape, name='latent_code_init')
    setter = tf.assign(wp, wp_rnd)
  else:
    logger.info(f'  Use encoder output as the initialization for optimization.')
    w_enc = E.get_output_for(x, is_training=False)
    wp_enc = tf.reshape(w_enc, latent_shape)
    setter = tf.assign(wp, wp_enc)

  # Settings for optimization.
  logger.info(f'Setting configuration for optimization.')
  perceptual_model = PerceptualModel([image_size, image_size], False)
  x_feat = perceptual_model(x_255)
  x_rec_feat = perceptual_model(x_rec_255)
  loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
  loss_pix = tf.reduce_mean(tf.square(x - x_rec), axis=[1, 2, 3])
  if args.domain_regularizer:
    logger.info(f'  Involve encoder for optimization.')
    w_enc_new = E.get_output_for(x_rec, is_training=False)
    wp_enc_new = tf.reshape(w_enc_new, latent_shape)
    loss_enc = tf.reduce_mean(tf.square(wp - wp_enc_new), axis=[1, 2])
  else:
    logger.info(f'  Do NOT involve encoder for optimization.')
    loss_enc = 0
  loss = (loss_pix +
          args.loss_weight_feat * loss_feat +
          args.loss_weight_enc * loss_enc)
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss, var_list=[wp])
  tflib.init_uninitialized_vars()

  # Load image list.
  logger.info(f'Loading image list.')
  image_list = []
  with open(args.image_list, 'r') as f:
    for line in f:
      image_list.append(line.strip())

  # Invert images.
  logger.info(f'Start inversion.')
  save_interval = args.num_iterations // args.num_results
  headers = ['Name', 'Original Image', 'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=len(image_list), num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  images = np.zeros(input_shape, np.uint8)
  names = ['' for _ in range(args.batch_size)]
  latent_codes_enc = []
  latent_codes = []
  for img_idx in tqdm(range(0, len(image_list), args.batch_size), leave=False):
    # Load inputs.
    batch = image_list[img_idx:img_idx + args.batch_size]
    for i, image_path in enumerate(batch):
      image = resize_image(load_image(image_path), (image_size, image_size))
      images[i] = np.transpose(image, [2, 0, 1])
      names[i] = os.path.splitext(os.path.basename(image_path))[0]
    inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
    # Run encoder.
    sess.run([setter], {x: inputs})
    outputs = sess.run([wp, x_rec])
    latent_codes_enc.append(outputs[0][0:len(batch)])
    outputs[1] = adjust_pixel_range(outputs[1])
    for i, _ in enumerate(batch):
      image = np.transpose(images[i], [1, 2, 0])
      save_image(f'{output_dir}/{names[i]}_ori.png', image)
      save_image(f'{output_dir}/{names[i]}_enc.png', outputs[1][i])
      visualizer.set_cell(i + img_idx, 0, text=names[i])
      visualizer.set_cell(i + img_idx, 1, image=image)
      visualizer.set_cell(i + img_idx, 2, image=outputs[1][i])
    # Optimize latent codes.
    col_idx = 3
    for step in tqdm(range(1, args.num_iterations + 1), leave=False):
      sess.run(train_op, {x: inputs})
      if step == args.num_iterations or step % save_interval == 0:
        outputs = sess.run([wp, x_rec])
        outputs[1] = adjust_pixel_range(outputs[1])
        for i, _ in enumerate(batch):
          if step == args.num_iterations:
            save_image(f'{output_dir}/{names[i]}_inv.png', outputs[1][i])
          visualizer.set_cell(i + img_idx, col_idx, image=outputs[1][i])
        col_idx += 1
    latent_codes.append(outputs[0][0:len(batch)])

  # Save results.
  os.system(f'cp {args.image_list} {output_dir}/image_list.txt')
  np.save(f'{output_dir}/encoded_codes.npy',
          np.concatenate(latent_codes_enc, axis=0))
  np.save(f'{output_dir}/inverted_codes.npy',
          np.concatenate(latent_codes, axis=0))
  visualizer.save(f'{output_dir}/inversion.html')


if __name__ == '__main__':
  main()
