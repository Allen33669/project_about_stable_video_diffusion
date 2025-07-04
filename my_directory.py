import os



from my_common_variable import *



print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())

#make new directory
os.mkdir(dataset_folder)
os.mkdir(frames_folder)  
os.mkdir(check_point_folder) 
os.mkdir(output_folder)
os.mkdir(tensor_board_folder)

if not os.path.exists(dataset_folder):
    print("dataset_folder created successfully!")
else:
    print("dataset_folder already exists.")

if not os.path.exists(frames_folder):
    print("frames_folder created successfully!")
else:
    print("frames_folder already exists.")

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