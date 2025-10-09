import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import open_clip



from my_utils import * 



"""
description: text embedder
"""
class TextEmbedder:
  """
  description: set clip model for text embedding
  params:
  return: 
  """
  def __init__(
    self, 
    clip_model: str="ViT-H-14", 
    clip_dataset: str="laion2b_s32b_b79k", 
    clip_device: str="cpu", 
    clip_max_length: int=77, 
    clip_legacy: bool=True,
    text_embedding_device: str="cuda",
  ):

    self.text_embedder, _, self.text_embedder_preprocess = open_clip.create_model_and_transforms(
      model_name=clip_model,
      pretrained=clip_dataset, 
      device=clip_device
    )

    self.text_tokenizer = open_clip.get_tokenizer(clip_model)
    self.clip_max_length = clip_max_length 



  """
  description: embed text prompt 
  params:
  -text: text prompt list, batch number of text prompts
  return: text embedding # Shape: [batch, 1, 1024]
  """
  def embed_text(
    self, 
    text, 
  ): 
    text_tokens = self.text_tokenizer(text, context_length=self.clip_max_length)
    with torch.no_grad():
      text_embedding = self.text_embedder.encode_text(text_tokens) # Shape: [batch, 1024]
    text_embedding = text_embedding.unsqueeze(1) # Shape: [batch, 1, 1024]

    return text_embedding



"""
description: preprocess conditions like video frames, video masks, ref image, and ref image mask
  reshape the image tensor to (C, H, W)
  image tensor scale: [-1, 1] or [0, 1]
params:
-file_path: file path for conditions like video frames, video masks, ref image, and ref image mask
-height: resized image height 
-width: resized image width 
-scale_type: "video": scale: [-1, 1], "ref_image": scale: [-1, 1], "mask": scale: [0, 1]
return: preprocessed tensors, Shape: [C, H, W], Scale: [-1, 1] or [0, 1]
"""
def preprocess_condition(
  file_path: str, 
  scale_type: str, 
  height: int=512, 
  width: int=512,  
):

  transform = transforms.Compose([
    transforms.Resize((height, width)), 
    transforms.PILToTensor()
  ])

  """
  frame_files = sorted([
    f for f in os.listdir(file_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
  ])

  frames = []
  for fname in frame_files:
    img_path = os.path.join(file_folder, fname)
    img = Image.open(img_path).convert("RGB")
    tensor_img = transform(img)  # Shape: [C, H, W], Scale: [0, 255]
    tensor_img = tensor_img / 255  # Shape: [C, H, W], Scale: [0, 1]
    if scale_type == "video" or scale_type == "ref_image":
      tensor_img = tensor_img * 2 - 1 # Shape: [C, H, W], Scale: [-1, 1]
    frames.append(tensor_img) 

  frames_tensor = torch.stack(frames)  # Shape: [T, C, H, W], Scale: [-1, 1]
  frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # Shape: [C, T, H, W], Scale: [-1, 1] or [0, 1]
  """

  img = Image.open(file_path).convert("RGB")
  tensor_img = transform(img)  # Shape: [C, H, W], Scale: [0, 255]
  tensor_img = tensor_img / 255  # Shape: [C, H, W], Scale: [0, 1]
  if scale_type == "video" or scale_type == "ref_image":
    tensor_img = tensor_img * 2 - 1 # Shape: [C, H, W], Scale: [-1, 1]

  return tensor_img



"""
description: encode conditions like video frames, video masks, ref image, and ref image mask into an output 
params:
-video: video frame tensors, Shape: [Cv, Tv, Hv, Wv], Scale: [-1, 1]
-video_masks: video frame mask tensors, Shape: [Cvm, Tvm, Hvm, Wvm], Scale: [0, 1]
-ref_image: ref image tensor, Shape: [Cr, Tr, Hr, Wr], Scale: [-1, 1]
-ref_mask: ref image mask tensor, Shape: [Crm, Trm, Hrm, Wrm], Scale: [0, 1]
return: encoded video and ref image tensor, Shape: [Cv*2, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
"""
def encode_video_ref(
  video: torch.Tensor, 
  video_masks: torch.Tensor, 
  ref_image: torch.Tensor, 
  ref_mask: torch.Tensor, 
):

  video_masks_first_channel = video_masks[0].unsqueeze(0)  # Shape: [1, Tvm, Hvm, Wvm], Scale: [0, 1]

  video_masks_first_channel_binary = (video_masks_first_channel >= 0.5).int()  # Shape: [1, Tvm, Hvm, Wvm], Scale: 0, 1

  inactive_video = video * (1 - video_masks_first_channel_binary)  # Shape: [Cv, Tv, Hv, Wv], Scale: [-1, 1]
  
  active_video = video * video_masks_first_channel_binary  # Shape: [Cv, Tv, Hv, Wv], Scale: [-1, 1]

  cat_video = torch.cat((inactive_video, active_video), dim=0)  # Shape: [Cv*2, Tv, Hv, Wv], Scale: [-1, 1]

  cat_ref_mask = torch.cat((ref_image, ref_mask), dim=0)  # Shape: [Cr*2, Tr, Hr, Wr], Scale: [-1, 1] and [0, 1]  

  cat_all = torch.cat((cat_video, cat_ref_mask), dim=1)  # Shape: [Cv*2, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]

  return cat_all



"""
description: encode conditions like video masks, and ref image mask into an output 
params:
-video_masks: video frame mask tensors, Shape: [Cvm, Tvm, Hvm, Wvm], Scale: [0, 1]
-ref_mask: ref image mask tensor, Shape: [Crm, Trm, Hrm, Wrm], Scale: [0, 1]
return: encoded masks tensor, Shape: [Cvm, Tvm+Trm, Hvm, Wvm], Scale: [0, 1]
"""
def encode_mask(
  video_masks: torch.Tensor,  
  ref_mask: torch.Tensor, 
):

  cat_masks = torch.cat((video_masks, ref_mask), dim=1)  # Shape: [Cvm, Tvm+Trm, Hvm, Wvm], Scale: [0, 1]

  return cat_masks



"""
description: concatenate conditions like video frames, video masks, ref image, and ref image mask into an output 
params:
-cat_video_ref: encoded video and ref image tensor, Shape: [Cv*2, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
-cat_masks: encoded masks tensor, Shape: [Cvm, Tvm+Trm, Hvm, Wvm], Scale: [0, 1]
return: encoded video, ref image, and masks tensor, Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
"""
def concatenate_conditions(
  cat_video_ref: torch.Tensor,  
  cat_masks: torch.Tensor, 
):

  cat_all = torch.cat((cat_video_ref, cat_masks), dim=0)  # Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]

  return cat_all



"""
description: embed spatial context 
"""
class SpatialContextEmbedder(nn.Module):

  """
  description: 
  params:
  return: 
  """
  def __init__(
    self, 
    height: int=512, 
    width: int=512, 
    T: int = 7, # Tv+Tr = num of videos + 1
    device: str = "cuda", 
    conv_in_channels: int=9, 
    conv_out_channels: int=16, 
    conv_kernel_size: int=3, 
    conv_stride: int=1, 
    conv_padding: int=1, 
    conv_downsampling_stride: int=2, 
    conv_downsampling_ch_mul: int=2, 
    conv_downsampling_layers: int=4, 
    self_attn_num_heads: int=8, 
    self_attn_dropout: float=0.1, 
    self_attn_batch_first: bool=True, 
    ff_proj_out_features: int=512, 
    ff_dropout: float=0.1, 
    cross_attn_k_v_dim: int=1024, 
    cross_attn_num_heads: int=8, 
    cross_attn_dropout: float=0.1, 
    cross_attn_batch_first: bool=True, 
    projector_out_features: int=1024,  
  ):
    super(SpatialContextEmbedder, self).__init__()

    self.height = height
    self.width = width
    self.T = T
    self.device = device
    self.conv_out_channels = conv_out_channels

    #Conv2d
    self.conv2d = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)

    #Conv2d downsampling
    conv2d_downsampling_layers = []
    current_conv_downsampling_ch_mul = 1
    for i in range(0, conv_downsampling_layers):
      conv2d_downsampling_layers.append(nn.Conv2d(
                                  conv_out_channels * current_conv_downsampling_ch_mul, 
                                  conv_out_channels * current_conv_downsampling_ch_mul * conv_downsampling_ch_mul, 
                                  kernel_size=conv_kernel_size, 
                                  stride=conv_downsampling_stride, 
                                  padding=conv_padding
                                 )
                       )
      conv2d_downsampling_layers.append(nn.ReLU())
      current_conv_downsampling_ch_mul = current_conv_downsampling_ch_mul * conv_downsampling_ch_mul

    #set params after Conv2d downsampling
    self.current_conv_downsampling_ch_mul = current_conv_downsampling_ch_mul
    height = int(self.height / self.current_conv_downsampling_ch_mul)
    width = int(self.width / self.current_conv_downsampling_ch_mul)
    conv_out_channels = int(self.conv_out_channels * self.current_conv_downsampling_ch_mul)

    self.conv2d_downsampling = nn.Sequential(*conv2d_downsampling_layers)


    #self-attention
    self.self_attn = nn.MultiheadAttention(embed_dim=conv_out_channels, num_heads=self_attn_num_heads, dropout=self_attn_dropout, batch_first=self_attn_batch_first)

    #FeedForward
    self.ff_proj = nn.Linear(conv_out_channels, ff_proj_out_features)
    self.ff_activation = nn.GELU()
    self.ff_dropout = nn.Dropout(ff_dropout)
    self.ff_output = nn.Linear(ff_proj_out_features, conv_out_channels)

    #cross-attention
    self.cross_attn = nn.MultiheadAttention(embed_dim=conv_out_channels, kdim=cross_attn_k_v_dim, vdim=cross_attn_k_v_dim, num_heads=cross_attn_num_heads, dropout=cross_attn_dropout, batch_first=cross_attn_batch_first)

    #downsampling pool
    self.adaptive_avg_pool_3d = nn.AdaptiveAvgPool3d((1, height, width))

    #project output
    projector_in_features = int(conv_out_channels * height * width)
    self.projector = nn.Linear(projector_in_features, projector_out_features)



  """
  description: embed spatial context by vace context and context 
  params:
  -vace_context: encoded video, ref image, and masks tensor, Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
  -context: text embedding, Shape: [1, 1024] ([token, depth])
  return: spatial context embedding
  """
  def forward(
    self, 
    vace_context: torch.Tensor,  
    context: torch.Tensor, 
  ):
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context >")
    print_all(context, "SpatialContextEmbedder > forward > context >")

    vace_context = vace_context.to(self.device)
    context = context.to(self.device)
    context = context.repeat(self.T, 1, 1)  # shape: [7, 1, 1024]
    print_all(context, "SpatialContextEmbedder > forward > context > context.repeat(self.T, 1, 1) >")

    #set params after Conv2d downsampling
    height = int(self.height / self.current_conv_downsampling_ch_mul)
    width = int(self.width / self.current_conv_downsampling_ch_mul)
    conv_out_channels = int(self.conv_out_channels * self.current_conv_downsampling_ch_mul)
    print_all(height, "SpatialContextEmbedder > forward > height > downsampling >")
    print_all(width, "SpatialContextEmbedder > forward > width > downsampling >")
    print_all(conv_out_channels, "SpatialContextEmbedder > forward > conv_out_channels > downsampling >")

    #Conv2d
    vace_context = vace_context.permute(1, 0, 2, 3) # Shape: [Tv+Tr, Cv*3=9, Hv, Wv], Scale: [-1, 1] 
    vace_context = self.conv2d(vace_context) # Shape: [Tv+Tr, C2d=16, Hv=512, Wv=512] 
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.conv2d(vace_context) >")

    #Conv2d downsampling
    vace_context = self.conv2d_downsampling(vace_context) # Shape: [Tv+Tr, C2d=16*2*2*2*2=256, Hv=512/2/2/2/2=32, Wv=512/2/2/2/2=32] 
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.conv2d_downsampling(vace_context) >")

    #self-attention
    vace_context = vace_context.permute(0, 2, 3, 1)  # Shape: [Tv+Tr, Hv=32, Wv=32, C2d=256] 
    vace_context = vace_context.view(self.T, height * width, conv_out_channels)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, C2d=256] 
    vace_context, vace_context_self_attn_weights = self.self_attn(vace_context, vace_context, vace_context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, C2d=256] 
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.self_attn >")

    #FeedForward
    vace_context = self.ff_proj(vace_context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, Cproj=512] 
    vace_context = self.ff_activation(vace_context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, Cproj=512] 
    vace_context = self.ff_dropout(vace_context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, Cproj=512] 
    vace_context = self.ff_output(vace_context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, C2d=256]
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.ff_output(vace_context) >") 

    #cross-attention
    vace_context, vace_context_cross_attn_weights = self.cross_attn(vace_context, context, context)  # Shape: [Tv+Tr, Hv * Wv = 32 * 32, C2d=256] 
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.cross_attn >") 

    #downsampling pool
    vace_context = vace_context.view(self.T, height, width, conv_out_channels)  # Shape: [Tv+Tr, Hv = 32, Wv = 32, C2d=256] 
    vace_context = vace_context.permute(3, 0, 1, 2)  # Shape: [C2d=256, Tv+Tr, Hv = 32, Wv = 32]  
    vace_context = self.adaptive_avg_pool_3d(vace_context) # Shape: [C2d=256, (Tv+Tr)/17=1, Hv = 32, Wv = 32] 
    print_all(vace_context, "SpatialContextEmbedder > forward > vace_context > self.adaptive_avg_pool_3d(vace_context) >")  

    #project output
    vace_context_flat = vace_context.reshape(-1)  # Shape: [256 * 1 * 32 * 32]
    vace_context_embedding = self.projector(vace_context_flat)  # Shape: [1024] 
    vace_context_embedding = vace_context_embedding.unsqueeze(0) # Shape: [1, 1024] 
    print_all(vace_context_embedding, "SpatialContextEmbedder > forward > vace_context_embedding > unsqueeze(0) >")  

    return vace_context_embedding



"""
description: embed temporal context 
"""
class TemporalContextEmbedder(nn.Module):

  """
  description: 
  params:
  return: 
  """
  def __init__(
    self, 
    height: int=512, 
    width: int=512, 
    T: int = 7, # Tv+Tr = num of videos + 1
    device: str = "cuda", 
    conv_in_channels: int=9, 
    conv_out_channels: int=16, 
    conv_kernel_size: int=3, 
    conv_stride: int=1, 
    conv_padding: int=1, 
    conv_downsampling_stride: int=2, 
    conv_downsampling_ch_mul: int=2, 
    conv_downsampling_layers: int=4, 
    self_attn_num_heads: int=8, 
    self_attn_dropout: float=0.1, 
    self_attn_batch_first: bool=True, 
    ff_proj_out_features: int=128, 
    ff_dropout: float=0.1, 
    cross_attn_k_v_dim: int=1024, 
    cross_attn_num_heads: int=8, 
    cross_attn_dropout: float=0.1, 
    cross_attn_batch_first: bool=True, 
    projector_out_features: int=1024,  
  ):
    super(TemporalContextEmbedder, self).__init__()

    self.height = height
    self.width = width
    self.T = T
    self.device = device
    self.conv_out_channels = conv_out_channels

    #Conv3d
    self.conv3d = nn.Conv3d(in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding)

    #Conv2d downsampling
    conv2d_downsampling_layers = []
    current_conv_downsampling_ch_mul = 1
    for i in range(0, conv_downsampling_layers):
      conv2d_downsampling_layers.append(nn.Conv2d(
                                  conv_out_channels * current_conv_downsampling_ch_mul, 
                                  conv_out_channels * current_conv_downsampling_ch_mul * conv_downsampling_ch_mul, 
                                  kernel_size=conv_kernel_size, 
                                  stride=conv_downsampling_stride, 
                                  padding=conv_padding
                                 )
                       )
      conv2d_downsampling_layers.append(nn.ReLU())
      current_conv_downsampling_ch_mul = current_conv_downsampling_ch_mul * conv_downsampling_ch_mul

    #set params after Conv2d downsampling
    self.current_conv_downsampling_ch_mul = current_conv_downsampling_ch_mul
    height = int(self.height / self.current_conv_downsampling_ch_mul)
    width = int(self.width / self.current_conv_downsampling_ch_mul)
    conv_out_channels = int(self.conv_out_channels * self.current_conv_downsampling_ch_mul)

    self.conv2d_downsampling = nn.Sequential(*conv2d_downsampling_layers)

    #self-attention
    self.self_attn = nn.MultiheadAttention(embed_dim=conv_out_channels, num_heads=self_attn_num_heads, dropout=self_attn_dropout, batch_first=self_attn_batch_first)

    #FeedForward
    self.ff_proj = nn.Linear(conv_out_channels, ff_proj_out_features)
    self.ff_activation = nn.GELU()
    self.ff_dropout = nn.Dropout(ff_dropout)
    self.ff_output = nn.Linear(ff_proj_out_features, conv_out_channels)

    #cross-attention
    self.cross_attn = nn.MultiheadAttention(embed_dim=conv_out_channels, kdim=cross_attn_k_v_dim, vdim=cross_attn_k_v_dim, num_heads=cross_attn_num_heads, dropout=cross_attn_dropout, batch_first=cross_attn_batch_first)

    #downsampling pool
    self.adaptive_avg_pool_3d = nn.AdaptiveAvgPool3d((self.T, int(height/2), int(width/2)))

    #project output
    projector_in_features = int(conv_out_channels * self.T * height/2 * width/2)
    self.projector = nn.Linear(projector_in_features, projector_out_features)




  """
  description: embed temporal context by vace context and context
  params:
  -vace_context: encoded video, ref image, and masks tensor, Shape: [Cv*3, Tv+Tr, Hv, Wv], Scale: [-1, 1] and [0, 1]
  -context: text embedding, Shape: [1, 1024] ([token, depth])
  return: temporal context embedding
  """
  def forward(
    self, 
    vace_context: torch.Tensor,  
    context: torch.Tensor, 
  ):
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context >")
    print_all(context, "TemporalContextEmbedder > forward > context >")

    vace_context = vace_context.to(self.device)
    context = context.to(self.device)

    #set params after Conv2d downsampling
    height = int(self.height / self.current_conv_downsampling_ch_mul)
    width = int(self.width / self.current_conv_downsampling_ch_mul)
    conv_out_channels = int(self.conv_out_channels * self.current_conv_downsampling_ch_mul)
    print_all(height, "TemporalContextEmbedder > forward > height > downsampling >")
    print_all(width, "TemporalContextEmbedder > forward > width > downsampling >")
    print_all(conv_out_channels, "TemporalContextEmbedder > forward > conv_out_channels > downsampling >")

    context = context.repeat(height * width, 1, 1)  # shape: [Hv * Wv = 32 * 32, 1, 1024]
    print_all(context, "TemporalContextEmbedder > forward > context > context.repeat(height * width, 1, 1) >")

    #Conv3d
    vace_context = self.conv3d(vace_context)  # Shape: [C3d=16, Tv+Tr, Hv=512, Wv=512] 
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.conv3d(vace_context) >")

    #Conv2d downsampling
    vace_context = vace_context.permute(1, 0, 2, 3) # Shape: [Tv+Tr, C3d=16, Hv=512, Wv=512]
    vace_context = self.conv2d_downsampling(vace_context) # Shape: [Tv+Tr, C2d=16*2*2*2*2=256, Hv=512/2/2/2/2=32, Wv=512/2/2/2/2=32] 
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.conv2d_downsampling(vace_context) >")

    #self-attention
    vace_context = vace_context.permute(2, 3, 0, 1)  # Shape: [Hv=32, Wv=32, Tv+Tr, C2d=256] 
    vace_context = vace_context.view(height * width, self.T, conv_out_channels)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, C2d=256] 
    vace_context, vace_context_self_attn_weights = self.self_attn(vace_context, vace_context, vace_context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, C2d=256]  
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.self_attn >")

    #FeedForward
    vace_context = self.ff_proj(vace_context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, Cproj=512] 
    vace_context = self.ff_activation(vace_context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, Cproj=512] 
    vace_context = self.ff_dropout(vace_context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, Cproj=512] 
    vace_context = self.ff_output(vace_context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, Cproj=256] 
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.ff_output(vace_context) >")

    #cross-attention
    vace_context, vace_context_cross_attn_weights = self.cross_attn(vace_context, context, context)  # Shape: [Hv * Wv = 32 * 32, Tv+Tr, Cproj=256] 
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.cross_attn >")

    #downsampling
    vace_context = vace_context.view(height, width, self.T, conv_out_channels)  # Shape: [Hv=32, Wv=32, Tv+Tr, Cproj=256] 
    vace_context = vace_context.permute(3, 2, 0, 1)  # Shape: [Cproj=256, Tv+Tr, Hv=32, Wv=32]  
    vace_context = self.adaptive_avg_pool_3d(vace_context) # Shape: [Cproj=256, Tv+Tr, Hv=16, Wv=16] 
    print_all(vace_context, "TemporalContextEmbedder > forward > vace_context > self.adaptive_avg_pool_3d(vace_context) >")

    #project
    vace_context_flat = vace_context.reshape(-1)  # Shape: [256 * 7 * 16 * 16]  
    vace_context_embedding = self.projector(vace_context_flat)  # Shape: [1024] 
    vace_context_embedding = vace_context_embedding.unsqueeze(0) # Shape: [1, 1024]
    print_all(vace_context_embedding, "TemporalContextEmbedder > forward > vace_context_embedding > unsqueeze(0) >")

    return vace_context_embedding
