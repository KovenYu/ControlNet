from share import *

import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


def __main__(args):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.model_path).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(gpus=args.gpus, precision=32, callbacks=[logger], default_root_dir=args.default_root_dir,
                         accumulate_grad_batches=args.accumulate_grad_batches, strategy=args.strategy)

    # Train!
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--logger_freq', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--sd_locked', type=bool, default=True)
    parser.add_argument('--only_mid_control', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='ddp')
    args = parser.parse_args()
    __main__(args)