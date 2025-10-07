import os
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from torchvision import transforms
import torch
import subprocess
from PIL import Image
import random


from my_common_variable import *
from my_utils import *
from my_context_embedder import *



#fix PIL.ImageShow Error: no “view“ mailcap rules found for type “image/png“
try:
    subprocess.run(["sudo", "apt-get", "remove", "-y", "xdg-utils"], check=True)
except subprocess.CalledProcessError:
    print("Error removing xdg-utils. Try running the script with sudo.")



"""
description: SVD dataset with text prompt 
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
description: SVD dataset with 2 text prompt (like spatial text prompt and temporal text prompt)
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



"""
description: 
  SVD dataset for input concatenation, spatial context of cross attention, temporal context of cross attention
  For spatial and temporal context embedding, generate context with video frames, video frame masks, ref image, ref mask 
params:
-frames_folder: xxxxxx.jpg images, the answer of the training dataset
-video_folder: xxxxxx.jpg images like depth, pose, and etc. 
-video_masks_folder: xxxxxx.jpg images, the masks for the video frames, one mask match one video frame 
-ref_image_file_path: xxxxxx.jpg image, 1 ref image for the video. 
-ref_image_mask_file_path: xxxxxx.jpg image, the mask for the ref image, one mask match one ref image
-text_file_path: the text prompt file, the file includes text of the frame line by line
-height: resize the frame to the height
-width: resize the frame to the width
-num_frames: number of frames in a sample
-fps_id: frames per second
-motion_bucket_id: control motion strength
-cond_aug: the noise level be added to the frame
-unordered: set to True randomly break up the order of frames
-transform: the function to transform the frame
"""
class SVDDatasetContextEmbedder(Dataset):
    def __init__(
      self, 
      frames_folder: str, 
      video_folder: str, 
      video_masks_folder: str, 
      ref_image_file_path: str, 
      ref_image_mask_file_path: str, 
      text_file_path: str, 
      height: int = 512, 
      width: int = 512, 
      num_frames:int = 14, 
      fps_id: int = 6, 
      motion_bucket_id: int = 127, 
      cond_aug: float = 0.02, 
      unordered: bool = False, 
      transform = None, 
      input_image_weight: float = 0, 
      input_text_weight: float = 0, 
      spatial_image_weight: float = 0, 
      spatial_text_weight: float = 0, 
      spatial_conditions_context_weight: float = 1,
      temporal_image_weight: float = 0, 
      temporal_text_weight: float = 0, 
      temporal_conditions_context_weight: float = 1, 
    ):
        self.jpg = []
        self.fps_id = []
        self.motion_bucket_id = []
        self.cond_frames = []
        self.cond_aug = []
        self.transform = transform
        self.cond_frames_without_noise = []
        self.input_image_weight = []
        self.input_text = []
        self.input_text_weight = []
        self.spatial_image = []
        self.spatial_image_weight = []
        self.spatial_text = []
        self.spatial_text_weight = []
        self.spatial_conditions_context = []
        self.spatial_conditions_context_weight = []
        self.temporal_image = []
        self.temporal_image_weight = []
        self.temporal_text = []
        self.temporal_text_weight = []
        self.temporal_conditions_context = []
        self.temporal_conditions_context_weight = []

 

        #load frame filenames
        frame_paths = []
        for frame in os.listdir(frames_folder):
          if frame.endswith(('.png', '.jpg', '.jpeg')):
            frame_paths.append(os.path.join(frames_folder, frame))
        frame_paths = sorted(frame_paths)

        #load text lines
        text_lines = []
        with open(os.path.join(frames_folder, text_file_path), "r") as file:
          for line in file:
            text_lines.append(line.strip())

        #load video frame filenames
        video_frame_paths = []
        for video_frame in os.listdir(video_folder):
          if video_frame.endswith(('.png', '.jpg', '.jpeg')):
            video_frame_paths.append(os.path.join(video_folder, video_frame))
        video_frame_paths = sorted(video_frame_paths)

        #load video mask filenames
        video_mask_paths = []
        for video_mask in os.listdir(video_masks_folder):
          if video_mask.endswith(('.png', '.jpg', '.jpeg')):
            video_mask_paths.append(os.path.join(video_masks_folder, video_mask))
        video_mask_paths = sorted(video_mask_paths)

        #check length between frame_paths, video_frame_paths, video_mask_paths, and text_lines
        if len(frame_paths) != len(video_frame_paths):
          print(f'SVDDatasetContextEmbedder > __init__ > len(frame_paths) != len(video_frame_paths) > len(frame_paths): {len(frame_paths)}, len(video_frame_paths): {len(video_frame_paths)}')
        if len(frame_paths) != len(video_mask_paths):
          print(f'SVDDatasetContextEmbedder > __init__ > len(frame_paths) != len(video_mask_paths) > len(frame_paths): {len(frame_paths)}, len(video_mask_paths): {len(video_mask_paths)}')
        if len(frame_paths) != len(text_lines):
          print(f'SVDDatasetContextEmbedder > __init__ > len(frame_paths) != len(text_lines) > len(frame_paths): {len(frame_paths)}, len(text_lines): {len(text_lines)}')
        


        #prepare condition context
        if len(video_frame_paths) < num_frames:
          print(f'SVDDatasetContextEmbedder > __init__ > len(video_frame_paths) < num_frames > len(video_frame_paths): {len(video_frame_paths)}, num_frames: {num_frames}')
        
        condition_context_list = []
        for i in range((len(video_frame_paths) - num_frames + 1)):
          #prepare video frames
          video_frame_list = []
          for j in range(i, i + num_frames):
            video_frame_tensor = preprocess_condition(video_frame_paths[j], "video") #Shape: [Cv, Hv, Wv], Scale: [-1, 1]

            video_frame_list.append(video_frame_tensor.unsqueeze(0)) #Shape: [1, Cv, Hv, Wv], Scale: [-1, 1]
          video_frame_tensors = torch.cat(video_frame_list, dim=0) #Shape: [Tv, Cv, Hv, Wv], Scale: [-1, 1]

          video_frame_tensors = video_frame_tensors.permute(1, 0, 2, 3) #Shape: [Cv, Tv, Hv, Wv], Scale: [-1, 1]
   
          #prepare video masks
          video_mask_list = []
          for j in range(i, i + num_frames):
            video_mask_tensor = preprocess_condition(video_mask_paths[j], "mask") #Shape: [Cvm, Hvm, Wvm], Scale: [0, 1]

            video_mask_list.append(video_mask_tensor.unsqueeze(0)) #Shape: [1, Cvm, Hvm, Wvm], Scale: [0, 1]
          video_mask_tensors = torch.cat(video_mask_list, dim=0) #Shape: [Tvm, Cvm, Hvm, Wvm], Scale: [0, 1]

          video_mask_tensors = video_mask_tensors.permute(1, 0, 2, 3) #Shape: [Cvm, Tvm, Hvm, Wvm], Scale: [0, 1]

          #prepare ref image
          ref_image_tensor = preprocess_condition(ref_image_file_path, "ref_image") #Shape: [Cr, Hr, Wr], Scale: [-1, 1]

          ref_image_tensor = ref_image_tensor.unsqueeze(0) #Shape: [Tr, Cr, Hr, Wr], Scale: [-1, 1] 

          ref_image_tensor = ref_image_tensor.permute(1, 0, 2, 3) #Shape: [Cr, Tr, Hr, Wr], Scale: [-1, 1] 

          #prepare ref image mask
          ref_image_mask_tensor = preprocess_condition(ref_image_mask_file_path, "mask") #Shape: [Crm, Hrm, Wrm], Scale: [0, 1]

          ref_image_mask_tensor = ref_image_mask_tensor.unsqueeze(0) #Shape: [Trm, Crm, Hrm, Wrm], Scale: [-1, 1] 

          ref_image_mask_tensor = ref_image_mask_tensor.permute(1, 0, 2, 3) #Shape: [Crm, Trm, Hrm, Wrm], Scale: [-1, 1]

          #prepare encode video, ref image
          encoded_video_ref = encode_video_ref(video_frame_tensors, video_mask_tensors, ref_image_tensor, ref_image_mask_tensor) # Shape: [Cv*2, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]

          encoded_mask = encode_mask(video_mask_tensors, ref_image_mask_tensor) # Shape: [Cvm, Tvm+Trm, Hvm, Wvm], Scale: [0, 1]

          encoded_video_ref_mask = concatenate_conditions(encoded_video_ref, encoded_mask) # Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]

          condition_context_list.append(encoded_video_ref_mask)

    

        #zip frame paths, condition_context_list, and text lines
        zipped_frame_paths_text_lines = list(zip(frame_paths, condition_context_list, text_lines))

        #cut the frames number to multiples of number of frames in a sample
        rest = len(zipped_frame_paths_text_lines) % num_frames
        zipped_frame_paths_text_lines = zipped_frame_paths_text_lines[:rest * (-1)]

        #handle unorder request
        if unordered:
          random.shuffle(zipped_frame_paths_text_lines)



        #set frame conditions
        current_num_frames = 0
        current_cond_frames_without_noise = None
        
        #set transform for preprocessing frames
        if self.transform is None:
          self.transform = transforms.Compose([
                                               transforms.Resize((height, width)), 
                                               transforms.ToTensor(), 
                                              ])

        for frame_paths_text_lines in zipped_frame_paths_text_lines:
          frame_path = frame_paths_text_lines[0]
          condition_context = frame_paths_text_lines[1]
          text_line = frame_paths_text_lines[2]

          #preprocess frame
          frame = Image.open(frame_path).convert("RGB")
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
          self.input_image_weight.append(input_image_weight)

          self.input_text.append(text_line)
          self.input_text_weight.append(input_text_weight)
          self.spatial_image.append(current_cond_frames_without_noise)
          self.spatial_image_weight.append(spatial_image_weight)
          self.spatial_text.append(text_line)
          self.spatial_text_weight.append(spatial_text_weight)
          self.spatial_conditions_context.append(condition_context)
          self.spatial_conditions_context_weight.append(spatial_conditions_context_weight)
          self.temporal_image.append(current_cond_frames_without_noise)
          self.temporal_image_weight.append(temporal_image_weight)
          self.temporal_text.append(text_line)
          self.temporal_text_weight.append(temporal_text_weight)
          self.temporal_conditions_context.append(condition_context)
          self.temporal_conditions_context_weight.append(temporal_conditions_context_weight)

          current_num_frames += 1

    def __len__(self):
        return len(self.jpg)

    def __getitem__(self, idx):
        item = {
                "jpg": self.jpg[idx], #the answer of the training dataset
                "fps_id": self.fps_id[idx],
                "motion_bucket_id": self.motion_bucket_id[idx],
                "cond_frames": self.cond_frames[idx], #the first frame of the answer of the training dataset, with random noise
                "cond_aug": self.cond_aug[idx], #for generating random noise which combined with the first frame of the answer of the training dataset
                input_image_key: self.cond_frames_without_noise[idx], #the first frame of the answer of the training dataset, for concatenate input x
                input_image_weight_key: self.input_image_weight[idx], 
                input_text_key: self.input_text[idx], #the text prompt, for concatenate input x
                input_text_weight_key: self.input_text_weight[idx], 
                spatial_image_key: self.spatial_image[idx], 
                spatial_image_weight_key: self.spatial_image_weight[idx], 
                spatial_text_key: self.spatial_text[idx], #the text prompt, for spatial context
                spatial_text_weight_key: self.spatial_text_weight[idx], 
                spatial_conditions_context_key: self.spatial_conditions_context[idx],  #the context generated by conditions like video frames, video masks, ref image, ref image mask, for spatial context or temporal context or both, Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
                spatial_conditions_context_weight_key: self.spatial_conditions_context_weight[idx], 
                temporal_image_key: self.temporal_image[idx], 
                temporal_image_weight_key: self.temporal_image_weight[idx], 
                temporal_text_key: self.temporal_text[idx], #the text prompt, for temporal context
                temporal_text_weight_key: self.temporal_text_weight[idx], 
                temporal_conditions_context_key: self.temporal_conditions_context[idx],  #the context generated by conditions like video frames, video masks, ref image, ref image mask, for spatial context or temporal context or both, Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
                temporal_conditions_context_weight_key: self.temporal_conditions_context_weight[idx], 
                }
        return item


"""
text_file_path = "/content/generative-models/dataset/frames/text_motion.txt"
text_2_file_path = "/content/generative-models/dataset/frames/text_appearence.txt"
dataset = SVDDatasetTwoText(frames_folder=frames_folder, text_file_path=text_file_path, text_2_file_path=text_2_file_path)

dataset = SVDDatasetTwoText(frames_folder=frames_folder, text_file_path=text_file_path, text_2_file_path=text_2_file_path, unordered=True)
"""
"""
dataset = SVDDatasetContextEmbedder(
  frames_folder=frames_folder, 
  video_folder=video_folder, 
  video_masks_folder=video_masks_folder, 
  ref_image_file_path=ref_image_file_path, 
  ref_image_mask_file_path=ref_image_mask_file_path, 
  text_file_path=text_file_path, 
)

print_all(dataset[0], "dataset[0] >")

dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for batch in dataloader:
  print("dataloader----------------------------------------------")
  print_all(batch, "batch >")
"""
