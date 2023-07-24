from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import argparse

parser = argparse.ArgumentParser(description='Control Stable Diffusion with Depth Maps')

# Mandatory positional arguments
parser.add_argument('--input_image_path', type=str, default='tmp/cat0.png')
parser.add_argument('--prompt', type=str, help='Prompt', default='a cute cat around a window')

# Optional arguments
parser.add_argument('--a_prompt', type=str, default='best quality, extremely detailed', help='Added Prompt')
parser.add_argument('--n_prompt', type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', help='Negative Prompt')
parser.add_argument('--num_samples', type=int, default=1, help='Images')
parser.add_argument('--image_resolution', type=int, default=512, help='Image Resolution')
parser.add_argument('--detect_resolution', type=int, default=384, help='Depth Resolution')
parser.add_argument('--ddim_steps', type=int, default=20, help='Steps')
parser.add_argument('--guess_mode', type=bool, default=False, help='Guess Mode')
parser.add_argument('--strength', type=float, default=1.0, help='Control Strength')
parser.add_argument('--scale', type=float, default=9.0, help='Guidance Scale')
parser.add_argument('--seed', type=int, default=-1, help='Seed')
parser.add_argument('--eta', type=float, default=0.0, help='eta (DDIM)')

args = parser.parse_args()

apply_midas = MidasDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results



input_image = Image.open(args.input_image_path)
input_image = np.array(input_image)

depth_array, generated = process(input_image, args.prompt, args.a_prompt, args.n_prompt, args.num_samples, args.image_resolution, 
        args.detect_resolution, args.ddim_steps, args.guess_mode, args.strength, args.scale, args.seed, args.eta)

depth_array_normalized = ((depth_array - depth_array.min()) * (1/(depth_array.max() - depth_array.min()) * 255)).astype('uint8')

# Convert normalized depth_array to an image
depth_image = Image.fromarray(depth_array_normalized)

# Save the depth image
depth_image.save('tmp/depth_image.png')

# Convert numpy array to PIL Image
img = Image.fromarray(generated)

# Save the image
img.save('tmp/my_image.png')