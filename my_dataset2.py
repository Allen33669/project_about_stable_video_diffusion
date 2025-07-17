import os
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torchvision import transforms
import torch
import subprocess
from PIL import Image
import random


from my_common_variable import *



#fix PIL.ImageShow Error: no “view“ mailcap rules found for type “image/png“
try:
    subprocess.run(["sudo", "apt-get", "remove", "-y", "xdg-utils"], check=True)
except subprocess.CalledProcessError:
    print("Error removing xdg-utils. Try running the script with sudo.")



"""
params:
-frames_folder: xxxxxx.jpg images for dataset
-text_file_path: 
-height: resize the frame to the height
-width: resize the frame to the width
-num_frames: number of frames in a sample
-fps_id: frames per second
-motion_bucket_id: control motion strength
-cond_aug: the noise level be added to the frame
-unordered: set to True randomly break up the order of frames
-transform: the function to transform the frame
"""
class SVDDataset(Dataset):
    def __init__(self, frames_folder: str, text_file_path: str, height: int = 512, width: int = 512, num_frames:int = 6, fps_id: int = 6, motion_bucket_id: int = 127, cond_aug: float = 0.02, unordered: bool = False, transform = None):
        #self.dataset_file = dataset_file
        self.jpg = []
        self.cond_frames_without_noise = []
        self.fps_id = []
        self.motion_bucket_id = []
        self.cond_frames = []
        self.cond_aug = []
        self.text = []
        self.transform = transform

        #load frame filenames
        frame_paths = []
        for frame in os.listdir(frames_folder):
          if frame.endswith(('.png', '.jpg', '.jpeg')):
            frame_paths.append(os.path.join(frames_folder, frame))

        #load text lines
        text_lines = []
        with open(os.path.join(frames_folder, text_file_path), "r") as file:
          for line in file:
            text_lines.append(line.strip())

        #zip frame paths and text lines
        if len(frame_paths) != len(text_lines):
          print(f'SVDDataset > __init__ > len(frame_paths) != len(text_lines > len(frame_paths): {len(frame_paths)}, len(frame_paths): {len(frame_paths)}')
        zipped_frame_paths_text_lines = list(zip(frame_paths, text_lines))

        #sort frame paths and text lines by frame paths
        frame_paths_text_lines_sorted = sorted(zipped_frame_paths_text_lines, key=lambda x: x[0])

        #cut the frames number to multiples of number of frames in a sample
        rest = len(frame_paths_text_lines_sorted) % num_frames
        frame_paths_text_lines_sorted = frame_paths_text_lines_sorted[:rest * (-1)]

        #handle unorder request
        if unordered:
          random.shuffle(frame_paths_text_lines_sorted)

        #set frame conditions
        current_num_frames = 0
        current_cond_frames_without_noise = None
        if self.transform is None:
          self.transform = transforms.Compose([
                                               transforms.Resize((height, width)), 
                                               transforms.ToTensor(), 
                                              ])

        for frame_paths_text_lines in frame_paths_text_lines_sorted:
          #preprocess frame
          frame = Image.open(frame_paths_text_lines[0]).convert("RGB")
          frame = self.transform(frame)
          
          #scaled to -1 ... 1
          frame =  frame * 2.0 - 1.0
            
          #set conditions
          self.jpg.append(frame)

          if (current_num_frames % num_frames) == 0:
            current_cond_frames_without_noise = frame

          self.cond_frames_without_noise.append(current_cond_frames_without_noise)
          self.fps_id.append(fps_id)
          self.motion_bucket_id.append(motion_bucket_id)

          current_cond_frame = current_cond_frames_without_noise + cond_aug * torch.randn_like(current_cond_frames_without_noise)
          self.cond_frames.append(current_cond_frame)
          self.cond_aug.append(cond_aug)
          self.text.append(frame_paths_text_lines[1])

          current_num_frames += 1

    def __len__(self):
        return len(self.jpg)

    def __getitem__(self, idx):
        item = {
                "jpg": self.jpg[idx],
                "cond_frames_without_noise": self.cond_frames_without_noise[idx],
                "fps_id": self.fps_id[idx],
                "motion_bucket_id": self.motion_bucket_id[idx],
                "cond_frames": self.cond_frames[idx],
                "cond_aug": self.cond_aug[idx], 
                "text": self.text[idx],
                }
        return item



"""
params:
-frames_folder: xxxxxx.jpg images for dataset
-text_file_path: 
-height: resize the frame to the height
-width: resize the frame to the width
-num_frames: number of frames in a sample
-fps_id: frames per second
-motion_bucket_id: control motion strength
-cond_aug: the noise level be added to the frame
-unordered: set to True randomly break up the order of frames
-transform: the function to transform the frame
"""
class SVDDatasetTwoText(Dataset):
    def __init__(self, frames_folder: str, text_file_path: str, text_2_file_path: str, height: int = 512, width: int = 512, num_frames:int = 6, fps_id: int = 6, motion_bucket_id: int = 127, cond_aug: float = 0.02, unordered: bool = False, transform = None):
        #self.dataset_file = dataset_file
        self.jpg = []
        self.cond_frames_without_noise = []
        self.fps_id = []
        self.motion_bucket_id = []
        self.cond_frames = []
        self.cond_aug = []
        self.text = []
        self.text_2 = []
        self.transform = transform

        #load frame filenames
        frame_paths = []
        for frame in os.listdir(frames_folder):
          if frame.endswith(('.png', '.jpg', '.jpeg')):
            frame_paths.append(os.path.join(frames_folder, frame))

        #load text lines
        text_lines = []
        with open(os.path.join(frames_folder, text_file_path), "r") as file:
          for line in file:
            text_lines.append(line.strip())

        text_2_lines = []
        with open(os.path.join(frames_folder, text_2_file_path), "r") as file:
          for line in file:
            text_2_lines.append(line.strip())

        #zip frame paths and text lines
        if len(frame_paths) != len(text_lines):
          print(f'SVDDataset > __init__ > len(frame_paths) != len(text_lines) > len(frame_paths): {len(frame_paths)}, len(text_lines): {len(text_lines)}')
        if len(frame_paths) != len(text_2_lines):
          print(f'SVDDataset > __init__ > len(frame_paths) != len(text_2_lines) > len(frame_paths): {len(frame_paths)}, len(text_2_lines): {len(text_2_lines)}')
        zipped_frame_paths_text_lines = list(zip(frame_paths, text_lines, text_2_lines))

        #sort frame paths and text lines by frame paths
        frame_paths_text_lines_sorted = sorted(zipped_frame_paths_text_lines, key=lambda x: x[0])

        #cut the frames number to multiples of number of frames in a sample
        rest = len(frame_paths_text_lines_sorted) % num_frames
        frame_paths_text_lines_sorted = frame_paths_text_lines_sorted[:rest * (-1)]

        #handle unorder request
        if unordered:
          random.shuffle(frame_paths_text_lines_sorted)

        #set frame conditions
        current_num_frames = 0
        current_cond_frames_without_noise = None
        if self.transform is None:
          self.transform = transforms.Compose([
                                               transforms.Resize((height, width)), 
                                               transforms.ToTensor(), 
                                              ])

        for frame_paths_text_lines in frame_paths_text_lines_sorted:
          #preprocess frame
          frame = Image.open(frame_paths_text_lines[0]).convert("RGB")
          frame = self.transform(frame)
          
          #scaled to -1 ... 1
          frame =  frame * 2.0 - 1.0
            
          #set conditions
          self.jpg.append(frame)

          if (current_num_frames % num_frames) == 0:
            current_cond_frames_without_noise = frame

          self.cond_frames_without_noise.append(current_cond_frames_without_noise)
          self.fps_id.append(fps_id)
          self.motion_bucket_id.append(motion_bucket_id)

          current_cond_frame = current_cond_frames_without_noise + cond_aug * torch.randn_like(current_cond_frames_without_noise)
          self.cond_frames.append(current_cond_frame)
          self.cond_aug.append(cond_aug)
          self.text.append(frame_paths_text_lines[1])
          self.text_2.append(frame_paths_text_lines[2])


          current_num_frames += 1

    def __len__(self):
        return len(self.jpg)

    def __getitem__(self, idx):
        item = {
                "jpg": self.jpg[idx],
                "cond_frames_without_noise": self.cond_frames_without_noise[idx],
                "fps_id": self.fps_id[idx],
                "motion_bucket_id": self.motion_bucket_id[idx],
                "cond_frames": self.cond_frames[idx],
                "cond_aug": self.cond_aug[idx], 
                "text": self.text[idx], 
                "text_2": self.text_2[idx], 
                }
        return item




text_file_path = "/content/generative-models/dataset/frames/text_motion.txt"
text_2_file_path = "/content/generative-models/dataset/frames/text_appearence.txt"
dataset = SVDDatasetTwoText(frames_folder=frames_folder, text_file_path=text_file_path, text_2_file_path=text_2_file_path)

dataset = SVDDatasetTwoText(frames_folder=frames_folder, text_file_path=text_file_path, text_2_file_path=text_2_file_path, unordered=True)



