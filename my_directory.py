import os



from my_common_variable import *



print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())

#make new directory
os.makedirs(dataset_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)  
os.makedirs(video_folder, exist_ok=True) 
os.makedirs(video_masks_folder, exist_ok=True) 
os.makedirs(ref_image_folder, exist_ok=True) 
os.makedirs(ref_image_mask_folder, exist_ok=True) 
os.makedirs(text_folder, exist_ok=True) 

os.makedirs(inference_assets_folder, exist_ok=True) 
#os.mkdir(depth_frames_folder, exist_ok=True) 
os.makedirs(check_point_folder, exist_ok=True) 
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tensor_board_folder, exist_ok=True)

if not os.path.exists(dataset_folder):
    print("dataset_folder created successfully!")
else:
    print("dataset_folder already exists.")

if not os.path.exists(frames_folder):
    print("frames_folder created successfully!")
else:
    print("frames_folder already exists.")

if not os.path.exists(video_folder):
    print("video_folder created successfully!")
else:
    print("video_folder already exists.")

if not os.path.exists(video_masks_folder):
    print("video_masks_folder created successfully!")
else:
    print("video_masks_folder already exists.")

if not os.path.exists(ref_image_folder):
    print("ref_image_folder created successfully!")
else:
    print("ref_image_folder already exists.")

if not os.path.exists(ref_image_mask_folder):
    print("ref_image_mask_folder created successfully!")
else:
    print("ref_image_mask_folder already exists.")

if not os.path.exists(text_folder):
    print("text_folder created successfully!")
else:
    print("text_folder already exists.")



if not os.path.exists(inference_assets_folder):
    print("inference_assets_folder created successfully!")
else:
    print("inference_assets_folder already exists.")

"""
if not os.path.exists(depth_frames_folder):
    print("depth_frames_folder created successfully!")
else:
    print("depth_frames_folder already exists.")
"""

if not os.path.exists(check_point_folder):
    print("check_point_folder created successfully!")
else:
    print("check_point_folder already exists.")

if not os.path.exists(output_folder):
    print("output_folder created successfully!")
else:
    print("output_folder already exists.")

if not os.path.exists(tensor_board_folder):
    print("tensor_board_folder created successfully!")
else:
    print("tensor_board_folder already exists.")



#current directory
print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())