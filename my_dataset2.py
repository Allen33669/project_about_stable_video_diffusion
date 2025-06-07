import os
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torchvision import transforms
import torch
import subprocess
from PIL import Image



from my_common_variable import *



#fix PIL.ImageShow Error: no “view“ mailcap rules found for type “image/png“
try:
    subprocess.run(["sudo", "apt-get", "remove", "-y", "xdg-utils"], check=True)
except subprocess.CalledProcessError:
    print("Error removing xdg-utils. Try running the script with sudo.")



"""
params:
-image: image file opened with PIL.Image
-width: width of preprocessed image
-height: height of preprocessed image
return:
-image: preprocessed image with width and height specified
"""
def preprocess_image(image, width=1024, height=576):
  if image.mode != "RGB":
    image = image.convert("RGB")

  # Resize the image
  image = image.resize((1024, 576), Image.LANCZOS)

  #image format to tensor and in chw format
  tensor_image = transforms.ToTensor()(image)

  #scaled to -1 ... 1
  tensor_image = tensor_image * 2.0 - 1.0

  return tensor_image



"""
params:
-dataset_file: svd_dataset.yaml file path
-frames_folder: xxxxxx.jpg images for dataset
-width: width of preprocessed image
-height: height of preprocessed image
-num_samples: set total samples in the dataset, 0: all samples
"""
class SVDDataset(Dataset):
    def __init__(self, dataset_file: str, frames_folder: str, width: int = 1024, height: int = 576, num_samples: int = 0):
        self.dataset_file = dataset_file
        self.frames_folder = frames_folder
        self.jpg = []
        self.cond_frames_without_noise = []
        self.fps_id = []
        self.motion_bucket_id = []
        self.cond_frames = []
        self.cond_aug = []

        dataset = OmegaConf.load(self.dataset_file)
        for frame, params in dataset.items():
          if (num_samples != 0) and (len(self.jpg) >= num_samples):
            break
          frame = Image.open(os.path.join(self.frames_folder, frame))
          frame = preprocess_image(frame, width, height)
          self.jpg.append(frame)
          cond_frame_without_noise = Image.open(os.path.join(self.frames_folder, params["cond_frames_without_noise"]))
          cond_frame_without_noise = preprocess_image(cond_frame_without_noise, width, height)
          self.cond_frames_without_noise.append(cond_frame_without_noise)
          self.fps_id.append(params["fps_id"])
          self.motion_bucket_id.append(params["motion_bucket_id"])
          cond_frame = Image.open(os.path.join(self.frames_folder, params["cond_frames"]))
          cond_frame = preprocess_image(cond_frame, width, height)
          cond_frame = cond_frame + params["cond_aug"] * torch.randn_like(cond_frame)
          self.cond_frames.append(cond_frame)
          self.cond_aug.append(params["cond_aug"])

    def __len__(self):
        return len(self.jpg)

    def __getitem__(self, idx):
        return {
                "jpg": self.jpg[idx],
                "cond_frames_without_noise": self.cond_frames_without_noise[idx],
                "fps_id": self.fps_id[idx],
                "motion_bucket_id": self.motion_bucket_id[idx],
                "cond_frames": self.cond_frames[idx],
                "cond_aug": self.cond_aug[idx],
                }



#dataset = SVDDataset(dataset_file, frames_folder)

