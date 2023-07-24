"""
This script is to generate random masks using perlin noise and thresholding to get a binary mask.
These masks are used to generate masked images.
"""
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch
from pyperlin import FractalPerlin2D

save_dir = '/viscam/projects/nerfluid/alis/lhq/lhq_256/perlin_masks'
os.makedirs(save_dir, exist_ok=True)

shape = (1,256,256) #for batch size = 32 and noises' shape = (256,256)
# resolutions = [(2**i,2**i) for i in range(1,4)] #for lacunarity = 2.0
# factors = [0.8**i for i in range(3)] #for persistence = 0.5
resolutions = [(2**i,2**i) for i in range(3,4)]
factors = [0.5**i for i in range(1)]
g_cuda = torch.Generator(device='cuda') #for GPU acceleration
fp = FractalPerlin2D(shape, resolutions, factors, generator=g_cuda)

for i in tqdm(range(90000)):
    noise = fp() #sampling

    # convert noise to an image
    pic_array = (noise[0].cpu().numpy() + 1) / 2
    scaled_pic_array = np.uint8(pic_array * 255)

    # convert noise to a binary mask
    binary_mask = np.where(pic_array > 0.35, 1, 0)
    # save binary_mask as an image
    binary_mask = np.uint8(binary_mask * 255)
    img = Image.fromarray(binary_mask, 'L')
    save_path = os.path.join(save_dir, f'mask_{i:06d}.png')
    # print('save_path:', save_path)
    img.save(save_path)


"""legacy code using CPU"""
# for i in tqdm(range(90000)):
#     noise = PerlinNoise(octaves=7, seed=i)
#     pic = [[noise([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
#     pic_array = (np.array(pic) + 1) / 2
#     scaled_pic_array = np.uint8(pic_array * 255)

#     # img = Image.fromarray(scaled_pic_array, 'L')
#     binary_mask = np.where(pic_array > 0.35, 1, 0)

#     # save binary_mask as an image
#     binary_mask = np.uint8(binary_mask * 255)
#     img = Image.fromarray(binary_mask, 'L')

#     save_path = os.path.join(save_dir, f'mask_{i:06d}.png')
#     print('save_path:', save_path)
#     img.save(save_path)