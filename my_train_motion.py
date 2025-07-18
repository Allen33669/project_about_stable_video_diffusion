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
from my_dataset2 import SVDDataset, SVDDatasetTwoText
from my_utils import load_model
from my_lora_handler import *



#current directory
print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())



text_file_path = "/content/generative-models/dataset/frames/text_motion.txt"
text_2_file_path = "/content/generative-models/dataset/frames/text_appearence.txt"
dataset = SVDDatasetTwoText(frames_folder=frames_folder, text_file_path=text_file_path, text_2_file_path=text_2_file_path, unordered=False)

#batch size can be multiples of number of frames in a video
dataloader = DataLoader(dataset, batch_size=12, shuffle=False)



# This is adapted from generative-models/scripts/sampling/simple_video_sample.py
model_config = "/content/generative-models/svd_train_motion.yaml"
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



#replace Linear module to LoraLinear module, layer 1, 2, 4, 5, 22, 23, 25, 26, temporal
lora_layer_list_input_blocks = [1, 2, 4, 5]
for i in lora_layer_list_input_blocks:
  in_model_task = "temporal"

  module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2
  module_name = "model.model.diffusion_model.input_blocks[i].time_stack[0].attn2"

  target_module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2.to_q
  target_module_name = "to_q"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=i, in_model_Unet_up_or_down_layer=i, in_model_task=in_model_task)

  target_module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2.to_k
  target_module_name = "to_k"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=i, in_model_Unet_up_or_down_layer=i, in_model_task=in_model_task)

  target_module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2.to_v
  target_module_name = "to_v"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=i, in_model_Unet_up_or_down_layer=i, in_model_task=in_model_task)

  module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2.to_out
  module_name = "model.model.diffusion_model.input_blocks[i].time_stack[0].attn2.to_out"
  target_module = model.model.diffusion_model.input_blocks[i][1].time_stack[0].attn2.to_out[0]
  target_module_name = "to_out"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, target_module_in_list = 0, in_model_layer=i, in_model_Unet_up_or_down_layer=i, in_model_task=in_model_task)



lora_layer_list_output_blocks = [22, 23, 25, 26]
for i in lora_layer_list_output_blocks:
  i = i - 12 - 3
  in_model_task = "temporal"

  module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2
  module_name = "model.model.diffusion_model.output_blocks[i].time_stack[0].attn2"

  target_module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2.to_q
  target_module_name = "to_q"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i, in_model_task=in_model_task)

  target_module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2.to_k
  target_module_name = "to_k"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i, in_model_task=in_model_task)

  target_module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2.to_v
  target_module_name = "to_v"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i, in_model_task=in_model_task)

  module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2.to_out
  module_name = "model.model.diffusion_model.output_blocks[i].time_stack[0].attn2.to_out"
  target_module = model.model.diffusion_model.output_blocks[i][1].time_stack[0].attn2.to_out[0]
  target_module_name = "to_out"
  replace_lora_module_with_statistic_info(module, module_name, target_module, target_module_name, target_module_in_list = 0, in_model_layer=12 + 3 + i, in_model_Unet_up_or_down_layer=11 - i, in_model_task=in_model_task)



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
#trainer = Trainer(max_epochs=2, callbacks=[MyEarlyStopping(monitor="early_stop_loss", mode="min", patience=1)])
trainer.fit(model, dataloader)



#save lora weight into .pth
lora_weight = []
save_lora(model, lora_weight_file, "model", lora_weight)



#load model and update lora weight by .pth
model_test, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
)



load_lora(model_test, lora_weight_file)






