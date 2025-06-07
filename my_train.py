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
from my_utils import load_model
from my_lora_handler import *



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



#replace ConvXd module to LoraConvXd module
add_lora_into_model(model.model.diffusion_model.middle_block, "model.model.diffusion_model.middle_block", "Conv1d", 0)
add_lora_into_model(model.model.diffusion_model.middle_block, "model.model.diffusion_model.middle_block", "Conv2d", 0)
add_lora_into_model(model.model.diffusion_model.middle_block, "model.model.diffusion_model.middle_block", "Conv3d", 0)



# Disable gradients for frozen layers
for name, module in model.model.diffusion_model.time_embed.named_children():
  for param in module.parameters():
    param.requires_grad = False

for name, module in model.model.diffusion_model.label_emb[0].named_children():
  for param in module.parameters():
    param.requires_grad = False

for param in model.model.diffusion_model.input_blocks.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.output_blocks.parameters():
  param.requires_grad = False

for name, module in model.model.diffusion_model.out.named_children():
  for param in module.parameters():
    param.requires_grad = False



for param in model.model.diffusion_model.middle_block[0].in_layers[2].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[0].emb_layers[1].parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[0].out_layers[3].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[0].time_stack.in_layers[2].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[0].time_stack.emb_layers[1].parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[0].time_stack.out_layers[3].conv.parameters():
  param.requires_grad = False



for name, module in model.model.diffusion_model.middle_block[1].named_children():
  for param in module.parameters():
    param.requires_grad = False

    

for param in model.model.diffusion_model.middle_block[2].in_layers[2].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[2].emb_layers[1].parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[2].out_layers[3].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[2].time_stack.in_layers[2].conv.parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[2].time_stack.emb_layers[1].parameters():
  param.requires_grad = False

for param in model.model.diffusion_model.middle_block[2].time_stack.out_layers[3].conv.parameters():
  param.requires_grad = False



# enable gradients for all lora layers
enable_lora_train(model, "Lora", 0)



#train the model
trainer = Trainer(max_epochs=2)
trainer.fit(model, dataloader)



#save lora weight into .pth
lora_weight = []
save_lora(model, "/content/generative-models/lora_weight.pth", "model", lora_weight)



#load model and update lora weight by .pth
model_test, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
)



load_lora(model_test, "/content/generative-models/lora_weight.pth")





