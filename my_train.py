# This is adapted from https://github.com/Stability-AI/generative-models



import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer



from my_common_variable import *
from my_dataset2 import SVDDataset



# This is adapted from generative-models/scripts/sampling/simple_video_sample.py
def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            #model = instantiate_from_config(config.model).to(device).eval()
            model = instantiate_from_config(config.model).to(device) #modified code start end
    else:
        #model = instantiate_from_config(config.model).to(device).eval()
        model = instantiate_from_config(config.model).to(device) #modified code start end

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter



#current directory
print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())



dataset = SVDDataset(dataset_file, frames_folder)

#batch size can be multiples of number of frames in a video
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)



# This is adapted from generative-models/scripts/sampling/simple_video_sample.py
model_config = "/content/generative-models/svd_train.yaml"
device = "cuda"
num_frames = 6
num_steps = 25
verbose = True

model, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
)



#freeze part of model when training, only middle_block is training
for param in model.parameters():
    param.requires_grad = False

for param in model.model.diffusion_model.middle_block.parameters():
    param.requires_grad = True



trainer = Trainer(max_epochs=2)
trainer.fit(model, dataloader)