# python 3.6
"""Interpolates real images with In-domain GAN Inversion.

The real images should be first inverted to latent codes with `invert.py`. After
that, this script can be used for image interpolation.

NOTE: This script will interpolate every image pair from source directory to
target directory.
"""

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from utils.logger import setup_logger
from utils.editor import interpolate
from utils.visualizer import load_image
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('src_dir', type=str,
                      help='Source directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('dst_dir', type=str,
                      help='Target directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/interpolation` will be used by default.')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size. (default: 32)')
  parser.add_argument('--step', type=int, default=5,
                      help='Number of steps for interpolation. (default: 5)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  src_dir = args.src_dir
  src_dir_name = os.path.basename(src_dir.rstrip('/'))
  assert os.path.exists(src_dir)
  assert os.path.exists(f'{src_dir}/image_list.txt')
  assert os.path.exists(f'{src_dir}/inverted_codes.npy')
  dst_dir = args.dst_dir
  dst_dir_name = os.path.basename(dst_dir.rstrip('/'))
  assert os.path.exists(dst_dir)
  assert os.path.exists(f'{dst_dir}/image_list.txt')
  assert os.path.exists(f'{dst_dir}/inverted_codes.npy')
  output_dir = args.output_dir or 'results/interpolation'
  job_name = f'{src_dir_name}_TO_{dst_dir_name}'
  logger = setup_logger(output_dir, f'{job_name}.log', f'{job_name}_logger')

  # Load model.
  logger.info(f'Loading generator.')
  tflib.init_tf({'rnd.np_random_seed': 1000})
  with open(args.model_path, 'rb') as f:
    _, _, _, Gs = pickle.load(f)

  # Build graph.
  logger.info(f'Building graph.')
  sess = tf.get_default_session()
  num_layers, latent_dim = Gs.components.synthesis.input_shape[1:3]
  wp = tf.placeholder(
      tf.float32, [args.batch_size, num_layers, latent_dim], name='latent_code')
  x = Gs.components.synthesis.get_output_for(wp, randomize_noise=False)

  # Load image and codes.
  logger.info(f'Loading images and corresponding inverted latent codes.')
  src_list = []
  with open(f'{src_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{src_dir}/{name}_ori.png')
      src_list.append(name)
  src_codes = np.load(f'{src_dir}/inverted_codes.npy')
  assert src_codes.shape[0] == len(src_list)
  num_src = src_codes.shape[0]
  dst_list = []
  with open(f'{dst_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{dst_dir}/{name}_ori.png')
      dst_list.append(name)
  dst_codes = np.load(f'{dst_dir}/inverted_codes.npy')
  assert dst_codes.shape[0] == len(dst_list)
  num_dst = dst_codes.shape[0]

  # Interpolate images.
  logger.info(f'Start interpolation.')
  step = args.step + 2
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_src * num_dst, num_cols=step + 2, viz_size=viz_size)
  visualizer.set_headers(
      ['Source', 'Source Inversion'] +
      [f'Step {i:02d}' for i in range(1, step - 1)] +
      ['Target Inversion', 'Target']
  )

  inputs = np.zeros((args.batch_size, num_layers, latent_dim), np.float32)
  for src_idx in tqdm(range(num_src), leave=False):
    src_code = src_codes[src_idx:src_idx + 1]
    src_path = f'{src_dir}/{src_list[src_idx]}_ori.png'
    codes = interpolate(src_codes=np.repeat(src_code, num_dst, axis=0),
                        dst_codes=dst_codes,
                        step=step)
    for dst_idx in tqdm(range(num_dst), leave=False):
      dst_path = f'{dst_dir}/{dst_list[dst_idx]}_ori.png'
      output_images = []
      for idx in range(0, step, args.batch_size):
        batch = codes[dst_idx, idx:idx + args.batch_size]
        inputs[0:len(batch)] = batch
        images = sess.run(x, feed_dict={wp: inputs})
        output_images.append(images[0:len(batch)])
      output_images = adjust_pixel_range(np.concatenate(output_images, axis=0))

      row_idx = src_idx * num_dst + dst_idx
      visualizer.set_cell(row_idx, 0, image=load_image(src_path))
      visualizer.set_cell(row_idx, step + 1, image=load_image(dst_path))
      for s, output_image in enumerate(output_images):
        visualizer.set_cell(row_idx, s + 1, image=output_image)

  # Save results.
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
