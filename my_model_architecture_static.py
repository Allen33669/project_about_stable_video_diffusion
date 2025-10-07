import os



from my_common_variable import *
#from my_dataset2 import SVDDataset
from my_utils import load_model 



#current directory
print("Current directory:", os.getcwd())
os.chdir(project_folder)
print("Current directory:", os.getcwd())



model_config = "/content/generative-models/svd_train_conditions_context.yaml"
device = "cuda"
num_frames = 14
num_steps = 25
verbose = True

model, _ = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
)



#inspect model statically
print(model)

"""
for name, layer in model.named_modules():
    print(f"layer0:{name}, {layer}")
    
    for name, layer in layer.named_modules():
        print(f"layer1:{name}, {layer}")

        for name, layer in layer.named_modules():
            print(f"layer2:{name}, {layer}")

            for name, layer in layer.named_modules():
                print(f"layer3:{name}, {layer}")

                for name, layer in layer.named_modules():
                    print(f"layer5:{name}, {layer}")
"""