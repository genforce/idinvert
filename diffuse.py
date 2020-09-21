# python 3.6
"""diffuses target images to context images with In-domain GAN Inversion.

Basically, this script first copies the central region from the target image to
the context image, and then performs in-domain GAN inversion on the stitched
image. Different from `intert.py`, masked reconstruction loss is used in the
optimization stage.

NOTE: This script will diffuse every image from `target_image_list` to every
image from `context_image_list`.
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
from utils.visualizer import load_image, resize_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('target_list', type=str,
                      help='List of target images to diffuse from.')
  parser.add_argument('context_list', type=str,
                      help='List of context images to diffuse to.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/diffusion` will be used by default.')
  parser.add_argument('-s', '--crop_size', type=int, default=110,
                      help='Crop size. (default: 110)')
  parser.add_argument('-x', '--center_x', type=int, default=125,
                      help='X-coordinate (column) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 125)')
  parser.add_argument('-y', '--center_y', type=int, default=145,
                      help='Y-coordinate (row) of the center of the cropped '
                           'patch. This field should be adjusted according to '
                           'dataset and image size. (default: 145)')
  parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size. (default: 4)')
  parser.add_argument('--learning_rate', type=float, default=0.01,
                      help='Learning rate for optimization. (default: 0.01)')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of optimization iterations. (default: 100)')
  parser.add_argument('--num_results', type=int, default=5,
                      help='Number of intermediate optimization results to '
                           'save for each sample. (default: 5)')
  parser.add_argument('--loss_weight_feat', type=float, default=5e-5,
                      help='The perceptual loss scale for optimization. '
                           '(default: 5e-5)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  assert os.path.exists(args.target_list)
  target_list_name = os.path.splitext(os.path.basename(args.target_list))[0]
  assert os.path.exists(args.context_list)
  context_list_name = os.path.splitext(os.path.basename(args.context_list))[0]
  output_dir = args.output_dir or f'results/diffusion'
  job_name = f'{target_list_name}_TO_{context_list_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  logger.info(f'Loading model.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    E, _, _, Gs = pickle.load(f)

  # Get input size.
  image_size = E.input_shape[2]
  assert image_size == E.input_shape[3]
  crop_size = args.crop_size
  crop_x = args.center_x - crop_size // 2
  crop_y = args.center_y - crop_size // 2
  mask = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
  mask[:, crop_y:crop_y + crop_size, crop_x:crop_x + crop_size, :] = 1.0

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  input_shape = E.input_shape
  input_shape[0] = args.batch_size
  x = tf.placeholder(tf.float32, shape=input_shape, name='real_image')
  x_mask = (tf.transpose(x, [0, 2, 3, 1]) + 1) * mask - 1
  x_mask_255 = (x_mask + 1) / 2 * 255
  latent_shape = Gs.components.synthesis.input_shape
  latent_shape[0] = args.batch_size
  wp = tf.get_variable(shape=latent_shape, name='latent_code')
  x_rec = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)
  x_rec_mask = (tf.transpose(x_rec, [0, 2, 3, 1]) + 1) * mask - 1
  x_rec_mask_255 = (x_rec_mask + 1) / 2 * 255

  w_enc = E.get_output_for(x, is_training=False)
  wp_enc = tf.reshape(w_enc, latent_shape)
  setter = tf.assign(wp, wp_enc)

  # Settings for optimization.
  logger.info(f'Setting configuration for optimization.')
  perceptual_model = PerceptualModel([image_size, image_size], False)
  x_feat = perceptual_model(x_mask_255)
  x_rec_feat = perceptual_model(x_rec_mask_255)
  loss_feat = tf.reduce_mean(tf.square(x_feat - x_rec_feat), axis=[1])
  loss_pix = tf.reduce_mean(tf.square(x_mask - x_rec_mask), axis=[1, 2, 3])

  loss = loss_pix + args.loss_weight_feat * loss_feat
  optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
  train_op = optimizer.minimize(loss, var_list=[wp])
  tflib.init_uninitialized_vars()

  # Load image list.
  logger.info(f'Loading target images and context images.')
  target_list = []
  with open(args.target_list, 'r') as f:
    for line in f:
      target_list.append(line.strip())
  num_targets = len(target_list)
  context_list = []
  with open(args.context_list, 'r') as f:
    for line in f:
      context_list.append(line.strip())
  num_contexts = len(context_list)
  num_pairs = num_targets * num_contexts

  # Invert images.
  logger.info(f'Start diffusion.')
  save_interval = args.num_iterations // args.num_results
  headers = ['Target Image', 'Context Image', 'Stitched Image',
             'Encoder Output']
  for step in range(1, args.num_iterations + 1):
    if step == args.num_iterations or step % save_interval == 0:
      headers.append(f'Step {step:06d}')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_pairs, num_cols=len(headers), viz_size=viz_size)
  visualizer.set_headers(headers)

  images = np.zeros(input_shape, np.uint8)
  latent_codes_enc = []
  latent_codes = []
  for target_idx in tqdm(range(num_targets), desc='Target ID', leave=False):
    # Load target.
    target_image = resize_image(load_image(target_list[target_idx]),
                                (image_size, image_size))
    visualizer.set_cell(target_idx * num_contexts, 0, image=target_image)
    for context_idx in tqdm(range(0, num_contexts, args.batch_size),
                            desc='Context ID', leave=False):
      row_idx = target_idx * num_contexts + context_idx
      batch = context_list[context_idx:context_idx + args.batch_size]
      for i, context_image_path in enumerate(batch):
        context_image = resize_image(load_image(context_image_path),
                                     (image_size, image_size))
        visualizer.set_cell(row_idx + i, 1, image=context_image)
        context_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size] = (
            target_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size])
        visualizer.set_cell(row_idx + i, 2, image=context_image)
        images[i] = np.transpose(context_image, [2, 0, 1])
      inputs = images.astype(np.float32) / 255 * 2.0 - 1.0
      # Run encoder.
      sess.run([setter], {x: inputs})
      outputs = sess.run([wp, x_rec])
      latent_codes_enc.append(outputs[0][0:len(batch)])
      outputs[1] = adjust_pixel_range(outputs[1])
      for i, _ in enumerate(batch):
        visualizer.set_cell(row_idx + i, 3, image=outputs[1][i])
      # Optimize latent codes.
      col_idx = 4
      for step in tqdm(range(1, args.num_iterations + 1), leave=False):
        sess.run(train_op, {x: inputs})
        if step == args.num_iterations or step % save_interval == 0:
          outputs = sess.run([wp, x_rec])
          outputs[1] = adjust_pixel_range(outputs[1])
          for i, _ in enumerate(batch):
            visualizer.set_cell(row_idx + i, col_idx, image=outputs[1][i])
          col_idx += 1
      latent_codes.append(outputs[0][0:len(batch)])

  # Save results.
  code_shape = [num_targets, num_contexts] + list(latent_shape[1:])
  np.save(f'{output_dir}/{job_name}_encoded_codes.npy',
          np.concatenate(latent_codes_enc, axis=0).reshape(code_shape))
  np.save(f'{output_dir}/{job_name}_inverted_codes.npy',
          np.concatenate(latent_codes, axis=0).reshape(code_shape))
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
