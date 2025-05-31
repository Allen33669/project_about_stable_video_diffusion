import os



from my_common_variable import *



#make new directory
os.mkdir(dataset_folder)
os.mkdir(frames_folder)  
os.mkdir(check_point_folder) 

if not os.path.exists(dataset_folder):
    print("Directory created successfully!")
else:
    print("Directory already exists.")

if not os.path.exists(frames_folder):
    print("Directory created successfully!")
else:
    print("Directory already exists.")

if not os.path.exists(check_point_folder):
    print("Directory created successfully!")
else:
    print("Directory already exists.")



#current directory
print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())