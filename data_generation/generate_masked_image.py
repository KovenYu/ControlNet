"""
This script is to use generated random masks to generate masked images.
"""
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

load_dir_mask = '/viscam/projects/nerfluid/alis/lhq/lhq_256/perlin_masks'
load_dir_imgs = '/viscam/projects/nerfluid/alis/lhq/lhq_256/lhq_256'
save_dir_masked = '/viscam/projects/nerfluid/alis/lhq/lhq_256/perlin_masked_lhq_256'
os.makedirs(save_dir_masked, exist_ok=True)

for i in tqdm(range(90000)):
    mask_path = os.path.join(load_dir_mask, f'mask_{i:06d}.png')
    # read binary mask image
    mask = Image.open(mask_path)
    mask = np.array(mask)[..., np.newaxis]

    img_path = os.path.join(load_dir_imgs, f'{i:07d}.png')
    # read image
    img = Image.open(img_path)
    img = np.array(img)

    # mask image
    masked_img = np.where(mask == 0, 0, img)

    # save masked image
    masked_img = np.uint8(masked_img)
    img = Image.fromarray(masked_img, 'RGB')
    save_path = os.path.join(save_dir_masked, f'masked_{i:06d}.png')
    img.save(save_path)
    # print('save_path:', save_path)
