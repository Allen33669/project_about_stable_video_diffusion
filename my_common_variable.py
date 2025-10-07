#folders
project_folder = "/content/generative-models"

#training dataset folder
dataset_folder = "/content/generative-models/dataset"
#dataset_file = "/content/generative-models/dataset/svd_dataset.yaml"
frames_folder = "/content/generative-models/dataset/frames"
video_folder = "/content/generative-models/dataset/frames/depth_map"
video_masks_folder = "/content/generative-models/dataset/frames/depth_map/depth_map_masks"
ref_image_folder = "/content/generative-models/dataset/frames/ref_image"
ref_image_file_path = "/content/generative-models/dataset/frames/ref_image/5020-1_70130_ref_image.jpg"
ref_image_mask_folder = "/content/generative-models/dataset/frames/ref_image/ref_image_mask"
ref_image_mask_file_path = "/content/generative-models/dataset/frames/ref_image/ref_image_mask/mask.jpg"
text_folder = "/content/generative-models/dataset/frames/text"
text_file_path = "/content/generative-models/dataset/frames/text/text.txt"

#inference dataset folder
inference_assets_folder = "/content/generative-models/inference_assets"
#depth_frames_folder = "/content/generative-models/inference_assets/depth_frames"

check_point_folder = "/content/generative-models/checkpoints"
output_folder = "/content/generative-models/outputs"



#for lora statistics in training process
current_timestep_file = "/content/generative-models/current_timestep.tar"
current_batch_step_file = "/content/generative-models/current_batch_step.tar"
current_epoch_step_file = "/content/generative-models/current_epoch_step.tar"
statistic_sample_backward_file = "/content/generative-models/statistic_sample_backward.tar"



#lora weight files
lora_weight_file = "/content/generative-models/lora_weight.pth"
lora_weight_motion_file = "/content/generative-models/lora_weight_motion.pth"
lora_weight_appearence_file = "/content/generative-models/lora_weight_appearence.pth"



#tensor board
tensor_board_folder = "/content/generative-models/tensor_board"



#condition batch key
input_image_key = "cond_frames_without_noise"
input_image_weight_key = "input_image_weight"
input_text_key = "input_text"
input_text_weight_key = "input_text_weight"
#input_depth_frames_key = "input_depth_frames"
#input_depth_frames_weight_key = "input_depth_frames_weight"
spatial_image_key = "spatial_image"
spatial_image_weight_key = "spatial_image_weight"
spatial_text_key = "spatial_text"
spatial_text_weight_key = "spatial_text_weight"
spatial_conditions_context_key = "spatial_conditions_context"
spatial_conditions_context_weight_key = "spatial_conditions_context_weight"
temporal_image_key = "temporal_image"
temporal_image_weight_key = "temporal_image_weight"
temporal_text_key = "temporal_text"
temporal_text_weight_key = "temporal_text_weight"
temporal_conditions_context_key = "temporal_conditions_context"
temporal_conditions_context_weight_key = "temporal_conditions_context_weight"
#conditions_context_key = "conditions_context"



#condition file
input_text_image_file = "/content/generative-models/input_text.png"



#GeneralConditioner output key
input_concat_key = "concat"
input_depth_frames_key = "input_depth_frames"
temporal_crossattn_context_key = "temporal_crossattn_context"
spatial_crossattn_context_key = "crossattn"



#GeneralConditioner output key for SCS
SCS_crossattn_context_motion_key = "crossattn_time_context"
SCS_crossattn_context_appearence_key = "crossattn_time_context_appearence"
