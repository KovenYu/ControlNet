import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from pypfm import PFMLoader


class MyDataset(Dataset):
    def __init__(self, json_path='./training/fill50k/prompt.json'):
        self.data = []
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.json_path = json_path
        self.pfm_loader = PFMLoader((256, 256), color=False, compress=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        mask_filename = item['mask']
        disp_filename = item['disp']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        mask = cv2.imread(mask_filename)[..., 0:1]
        target = cv2.imread(target_filename)

        pfm = self.pfm_loader.load_pfm(disp_filename)
        pfm = pfm[::-1, :, np.newaxis]

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        pfm[pfm == np.inf] = pfm[pfm != np.inf].max()
        pfm[pfm == -np.inf] = pfm[pfm != -np.inf].min()
        pfm[pfm == np.nan] = pfm[pfm != np.nan].mean()
        pfm = (pfm - pfm.min()) / (pfm.max() - pfm.min() + 1e-6)
        source = np.concatenate([source, mask, pfm], axis=-1)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

