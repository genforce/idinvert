# python 3.6
"""Mixes styles with In-domain GAN Inversion.

The real images should be first inverted to latent codes with `invert.py`. After
that, this script can be used for style mixing.

NOTE: This script will mix every `style-content` image pair from style
directory to content directory.
"""

import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from dnnlib import tflib

from utils.logger import setup_logger
from utils.editor import mix_style
from utils.visualizer import load_image
from utils.visualizer import adjust_pixel_range
from utils.visualizer import HtmlPageVisualizer


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('model_path', type=str,
                      help='Path to the pre-trained model.')
  parser.add_argument('style_dir', type=str,
                      help='Style directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('content_dir', type=str,
                      help='Content directory, which includes original images, '
                           'inverted codes, and image list.')
  parser.add_argument('-o', '--output_dir', type=str, default='',
                      help='Directory to save the results. If not specified, '
                           '`./results/style_mixing` will be used by default.')
  parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size. (default: 32)')
  parser.add_argument('--mix_layer_start_idx', type=int, default=10,
                      help='0-based layer index. Style mixing is performed '
                           'from this layer to the last layer. (default: 10)')
  parser.add_argument('--viz_size', type=int, default=256,
                      help='Image size for visualization. (default: 256)')
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
  style_dir = args.style_dir
  style_dir_name = os.path.basename(style_dir.rstrip('/'))
  assert os.path.exists(style_dir)
  assert os.path.exists(f'{style_dir}/image_list.txt')
  assert os.path.exists(f'{style_dir}/inverted_codes.npy')
  content_dir = args.content_dir
  content_dir_name = os.path.basename(content_dir.rstrip('/'))
  assert os.path.exists(content_dir)
  assert os.path.exists(f'{content_dir}/image_list.txt')
  assert os.path.exists(f'{content_dir}/inverted_codes.npy')
  output_dir = args.output_dir or 'results/style_mixing'
  job_name = f'{style_dir_name}_STYLIZE_{content_dir_name}'
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
  mix_layers = list(range(args.mix_layer_start_idx, num_layers))

  # Load image and codes.
  logger.info(f'Loading images and corresponding inverted latent codes.')
  style_list = []
  with open(f'{style_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{style_dir}/{name}_ori.png')
      style_list.append(name)
  logger.info(f'Loading inverted latent codes.')
  style_codes = np.load(f'{style_dir}/inverted_codes.npy')
  assert style_codes.shape[0] == len(style_list)
  num_styles = style_codes.shape[0]
  content_list = []
  with open(f'{content_dir}/image_list.txt', 'r') as f:
    for line in f:
      name = os.path.splitext(os.path.basename(line.strip()))[0]
      assert os.path.exists(f'{content_dir}/{name}_ori.png')
      content_list.append(name)
  logger.info(f'Loading inverted latent codes.')
  content_codes = np.load(f'{content_dir}/inverted_codes.npy')
  assert content_codes.shape[0] == len(content_list)
  num_contents = content_codes.shape[0]

  # Mix styles.
  logger.info(f'Start style mixing.')
  viz_size = None if args.viz_size == 0 else args.viz_size
  visualizer = HtmlPageVisualizer(
      num_rows=num_styles + 1, num_cols=num_contents + 1, viz_size=viz_size)
  visualizer.set_headers(
      ['Style'] +
      [f'Content {i:03d}' for i in range(num_contents)]
  )
  for style_idx, style_name in enumerate(style_list):
    style_image = load_image(f'{style_dir}/{style_name}_ori.png')
    visualizer.set_cell(style_idx + 1, 0, image=style_image)
  for content_idx, content_name in enumerate(content_list):
    content_image = load_image(f'{content_dir}/{content_name}_ori.png')
    visualizer.set_cell(0, content_idx + 1, image=content_image)

  codes = mix_style(style_codes=style_codes,
                    content_codes=content_codes,
                    num_layers=num_layers,
                    mix_layers=mix_layers)
  inputs = np.zeros((args.batch_size, num_layers, latent_dim), np.float32)
  for style_idx in tqdm(range(num_styles), leave=False):
    output_images = []
    for idx in range(0, num_contents, args.batch_size):
      batch = codes[style_idx, idx:idx + args.batch_size]
      inputs[0:len(batch)] = batch
      images = sess.run(x, feed_dict={wp: inputs})
      output_images.append(images[0:len(batch)])
    output_images = adjust_pixel_range(np.concatenate(output_images, axis=0))
    for content_idx, output_image in enumerate(output_images):
      visualizer.set_cell(style_idx + 1, content_idx + 1, image=output_image)

  # Save results.
  visualizer.save(f'{output_dir}/{job_name}.html')


if __name__ == '__main__':
  main()
