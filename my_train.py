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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



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
for i in range(3):
  model_name = "model.model.diffusion_model.input_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv1d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv2d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv3d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)

for i in range(4, 6):
  model_name = "model.model.diffusion_model.input_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv1d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv2d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv3d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)

for i in range(7, 9):
  model_name = "model.model.diffusion_model.input_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv1d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv2d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv3d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)

for i in range(10, 12):
  model_name = "model.model.diffusion_model.input_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv1d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv2d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.input_blocks[i], model_name, "Conv3d", 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i)



for i in range(2):
  model_name = "model.model.diffusion_model.middle_block." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[i], model_name, "Conv1d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 + i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[i], model_name, "Conv2d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 + i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[i], model_name, "Conv3d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 + i)

for i in range(1):
  model_name = "model.model.diffusion_model.middle_block." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[2 - i], model_name, "Conv1d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[2 - i], model_name, "Conv2d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.middle_block[2 - i], model_name, "Conv3d", 0, in_model_layer=12 + i, in_model_Unet_up_or_down_layer=12 - i)



for i in range(3):
  model_name = "model.model.diffusion_model.output_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv1d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv2d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv3d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)

for i in range(4, 6):
  model_name = "model.model.diffusion_model.output_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv1d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv2d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv3d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)

for i in range(7, 9):
  model_name = "model.model.diffusion_model.output_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv1d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv2d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv3d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)

for i in range(10, 12):
  model_name = "model.model.diffusion_model.output_blocks." + str(i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv1d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv2d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)
  add_lora_into_model_with_statistic_info(model.model.diffusion_model.output_blocks[i], model_name, "Conv3d", 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i)



#set frozen layer to requires_grad = False
for name, param in model.model.diffusion_model.named_parameters():
  if "lora" in name:
    print(f'parameter name: {name}, param.is_leaf: {param.is_leaf}')
  
  else:
    if param.is_leaf:
      param.requires_grad = False



#early stop callback
class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.callback_metrics.get("early_stop_loss", None) is not None:
            self.monitor = "early_stop_loss"
            self._run_early_stopping_check(trainer)



#train the model
#trainer = Trainer(max_epochs=2)
trainer = Trainer(callbacks=[MyEarlyStopping(monitor="early_stop_loss", mode="min", patience=1)])
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






