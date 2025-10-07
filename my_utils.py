# This is adapted from https://github.com/Stability-AI/generative-models

# This is adapted from generative-models/scripts/sampling/simple_video_sample.py



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
            model = instantiate_from_config(config.model).to(device) #modified code start end
    else:
        model = instantiate_from_config(config.model).to(device) #modified code start end

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter



#print list, dictionary, tensor, other 
def print_all(target, description):
  print("_" * 60)
  print(f"{description} type(target): {type(target)}")

  if isinstance(target, list):
    print_list(target, description + " list >")
  elif isinstance(target, tuple):
    print_tuple(target, description + " tuple >")
  elif isinstance(target, dict):
    print_dict(target, description + " dict >")
  elif isinstance(target, np.ndarray):
    ndarray_tensor = torch.from_numpy(target)
    print_tensor(ndarray_tensor, description + " np.ndarray >")
  elif isinstance(target, torch.Tensor):
    print_tensor(target, description + " Tensor >")
  else:
    print(f"{description} : {target}")



#print list 
def print_list(target, description):
  print("_" * 60)
  try:
    list_tensor = torch.tensor(target)
    print_tensor(list_tensor, description + " list to tensor >")
  except Exception as e:
    for list_item in target:
      print_all(list_item, description + " list >")



#print tuple 
def print_tuple(target, description):
  print("_" * 60)
  for item in target:
    try:
      item_tensor = torch.tensor(item)
      print_tensor(item_tensor, description + " tuple to tensor >")
    except Exception as e:
      for item in target:
        print_all(item, description + " tuple >")



#print dictionary 
def print_dict(target, description):
  print("_" * 60)
  for key, value in target.items():
    print(f"{description} key: {key}")
    print(f"{description} type(value): {type(value)}")
    print_all(value, description + " " + key + " > value >")



#print tensor
def print_tensor(tensor, description):
  print("_" * 60)
  if isinstance(tensor, torch.Tensor):
    print(f"{description} tensor.shape: {tensor.shape}")
    print(f"{description} tensor.float().mean(): {tensor.float().mean()}")
    print(f"{description} tensor.float().min(): {tensor.float().min()}")
    print(f"{description} tensor.float().max(): {tensor.float().max()}")
  else:
    print(f"{description} is not tensor!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
