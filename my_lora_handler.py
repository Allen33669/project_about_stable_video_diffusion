import torch
from einops import rearrange



import copy

from my_common_variable import *
from my_lora import *



"""
description: replace module of base model by lora module, copy original weight of module into lora module
params:
-module: target module be searched
-module_name: name of target module be searched
-target_module: target class be replaced in the target module
-target_module_name: name of target class be replaced in the target module
-r: lora rank
-scale: lora scale
return: replaced module
"""
def replace_lora_module(
  module, 
  module_name: str, 
  target_module,
  target_module_name: str,  
  r = 16,
  scale: float = 1.0,
):
  replace_module = None
  
  if target_module.__class__.__name__ == "Linear":
      return
  elif target_module.__class__.__name__ == "Conv1d":
      replace_module = LoraConv1d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                 )
  elif target_module.__class__.__name__ == "Conv2d":
      replace_module = LoraConv2d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                 )
  elif target_module.__class__.__name__ == "Conv3d":
      replace_module = LoraConv3d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                 )

  module._modules[target_module_name] = replace_module

  print(f'replace_lora_module > parent module: {module_name}, target_module_name: {target_module_name}, target_module_class_name: {target_module.__class__.__name__}')

  return module._modules[target_module_name]



"""
description: add lora into base model, replace module of base model by lora module
params:
-model: target model be searched
-model_name: name of target model be searched
-target: target class name be searched in the target module like Conv2d
-depth: current depth layer from start point
-depth_max: max depth layer from start point
"""
def add_lora_into_model(
  model,
  model_name: str,  
  target: str,
  depth: int, 
  depth_max: int = 20, 
):
  if depth >= depth_max:
    return

  for name, module in model.named_children():
    if module.__class__.__name__ == target:
      replace_lora_module(model, model_name, module, name)
    else:
      depth += 1
      add_lora_into_model(module, name, target, depth)



"""
description: module backward hook for generate lora statistics samples and store the statistics samples
params:
return: 
"""
def save_statistic_sample_backward_hook(module, grad_input, grad_output):
  current_timestep = torch.load(current_timestep_file)['current_timestep']

  current_batch_step = torch.load(current_batch_step_file)['current_batch_step']

  current_epoch_step = torch.load(current_epoch_step_file)['current_epoch_step']

  statistic_sample_backward_list = torch.load(statistic_sample_backward_file)
  
  grad_input_mean_abs = []
  for i, tensor in enumerate(grad_input):
    if tensor is not None:
      grad_input_mean_abs.append(torch.mean(torch.abs(tensor)))
    else:
      grad_input_mean_abs.append(None)

  grad_output_mean_abs = torch.mean(torch.abs(grad_output[0]))

  grad_input_mean_square = []
  for i, tensor in enumerate(grad_input):
    if tensor is not None:
      grad_input_mean_square.append(torch.mean(torch.square(tensor)))
    else:
      grad_input_mean_square.append(None)

  grad_output_mean_square = torch.mean(torch.square(grad_output[0]))

  lora_down_weight_mean_abs = module.lora_down.weight.abs().mean().item()
  lora_down_weight_mean_square = (module.lora_down.weight ** 2).mean().item()

  lora_up_weight_mean_abs = module.lora_up.weight.abs().mean().item()
  lora_up_weight_mean_square = (module.lora_up.weight ** 2).mean().item()

  statistic_sample_backward_new = {
                                   "in_model_layer": module.in_model_layer, 
                                   "in_model_toal_layer": module.in_model_toal_layer, 
                                   "in_model_Unet_up_or_down_layer": module.in_model_Unet_up_or_down_layer, 
                                   "in_model_position": module.in_model_position, 
                                   "in_model_replaced_module": module.in_model_replaced_module, 
                                   "in_model_task": module.in_model_task, 
                                   "current_timestep": current_timestep, 
                                   "current_batch_step" : current_batch_step, 
                                   "current_epoch_step": current_epoch_step, 
                                   "grad_input_mean_abs": grad_input_mean_abs, 
                                   "grad_output_mean_abs": grad_output_mean_abs, 
                                   "grad_input_mean_square": grad_input_mean_square, 
                                   "grad_output_mean_square": grad_output_mean_square,
                                   "lora_down_weight_mean_abs": lora_down_weight_mean_abs,
                                   "lora_down_weight_mean_square": lora_down_weight_mean_square,
                                   "lora_up_weight_mean_abs": lora_up_weight_mean_abs,
                                   "lora_up_weight_mean_square": lora_up_weight_mean_square, 
                                  }
  statistic_sample_backward_list.append(statistic_sample_backward_new)

  torch.save(statistic_sample_backward_list, statistic_sample_backward_file)



"""
description: replace module of base model by lora module, copy original weight of module into lora module with statistics info
params:
-module: target module be searched
-module_name: name of target module be searched
-target_module: target class be replaced in the target module
-target_module_name: name of target class be replaced in the target module
-r: lora rank
-scale: lora scale
-in_model_layer: statistics info in lora module
-in_model_Unet_up_or_down_layer: statistics info in lora module
-in_model_task: handle spacial, temporal, or etc. like "spatial", "temporal", "original", and etc.
return: replaced module
"""
def replace_lora_module_with_statistic_info(
  module, 
  module_name: str, 
  target_module,
  target_module_name: str, 
  target_module_in_list: int = -1, 
  r = 16,
  scale: float = 1.0, 
  in_model_layer: int = None, 
  in_model_Unet_up_or_down_layer: int = None, 
  in_model_task: str = None,
  register_statistic_sample_backward_hook: bool = True, 
):
  replace_module = None
  
  if target_module.__class__.__name__ == "Linear":
      replace_module = LoraLinear(
                                  target_module.in_features, 
                                  target_module.out_features, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                  in_model_layer = in_model_layer, 
                                  in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer, 
                                  in_model_position = target_module_name, 
                                  in_model_replaced_module = target_module.__class__.__name__, 
                                  in_model_task = in_model_task, 
                                 )
  elif target_module.__class__.__name__ == "Conv1d":
      replace_module = LoraConv1d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                  in_model_layer = in_model_layer, 
                                  in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer, 
                                  in_model_position = target_module.__class__.__name__, 
                                  in_model_replaced_module = target_module.__class__.__name__, 
                                  in_model_task = in_model_task, 
                                 )
  elif target_module.__class__.__name__ == "Conv2d":
      replace_module = LoraConv2d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                  in_model_layer = in_model_layer, 
                                  in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer, 
                                  in_model_position = target_module.__class__.__name__, 
                                  in_model_replaced_module = target_module.__class__.__name__, 
                                  in_model_task = in_model_task, 
                                 )
  elif target_module.__class__.__name__ == "Conv3d":
      replace_module = LoraConv3d(
                                  target_module.in_channels, 
                                  target_module.out_channels, 
                                  target_module.kernel_size, 
                                  target_module.stride, 
                                  target_module.padding, 
                                  target_module.dilation, 
                                  target_module.groups, 
                                  target_module.bias is not None, 
                                  r, 
                                  scale, 
                                  in_model_layer = in_model_layer, 
                                  in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer, 
                                  in_model_position = target_module.__class__.__name__, 
                                  in_model_replaced_module = target_module.__class__.__name__, 
                                  in_model_task = in_model_task, 
                                 )
  
  if target_module_in_list == -1:
    module._modules[target_module_name] = replace_module
    module._modules[target_module_name].register_full_backward_hook(save_statistic_sample_backward_hook)
  else:
    module[target_module_in_list] = replace_module
    module[target_module_in_list].register_full_backward_hook(save_statistic_sample_backward_hook)

  """
  if register_statistic_sample_backward_hook:
    module._modules[target_module_name].register_full_backward_hook(save_statistic_sample_backward_hook)
  """

  print(f'replace_lora_module_with_statistic_info > parent module: {module_name}, target_module_name: {target_module_name}, target_module_class_name: {target_module.__class__.__name__}')

  #return module._modules[target_module_name]



"""
description: add lora into base model, replace module of base model by lora module with statistics info
params:
-model: target model be searched
-model_name: name of target model be searched
-target: target class name be searched in the target module like Conv2d
-depth: current depth layer from start point
-depth_max: max depth layer from start point
-in_model_layer: statistics info in lora module
-in_model_Unet_up_or_down_layer: statistics info in lora module
"""
def add_lora_into_model_with_statistic_info(
  model,
  model_name: str,  
  target: str,
  depth: int, 
  depth_max: int = 20, 
  in_model_layer: int = None, 
  in_model_Unet_up_or_down_layer: int = None, 
  in_model_task: str = None, 
):
  if depth >= depth_max:
    return

  for name, module in model.named_children():   
    #replace the module to lora module 
    if module.__class__.__name__ == target:
      replace_lora_module_with_statistic_info(model, model_name, module, name, in_model_layer=in_model_layer, in_model_Unet_up_or_down_layer=in_model_Unet_up_or_down_layer, in_model_task=in_model_task)
    else:
      depth += 1
      add_lora_into_model_with_statistic_info(module, name, target, depth, in_model_layer=in_model_layer, in_model_Unet_up_or_down_layer=in_model_Unet_up_or_down_layer, in_model_task=in_model_task)



"""
description: find loras in the module and append lora weights into the list
params:
-model: model be searched
-lora_weight: the list be appended by lora weight
-module_name: name of module be searched
-depth: current depth layer from start point
-depth_max: max depth layer from start point
"""
def find_lora_weight(
  model, 
  lora_weight: list, 
  model_name: str, 
  depth: int, 
  depth_max: int = 50,  
):
  if depth >= depth_max:
    return

  for name, module in model.named_children():
    current_module_name = model_name + "." + name

    if "Lora" in module.__class__.__name__:
      print(f'find_lora_weight > module_name: {current_module_name}')
      print(f'find_lora_weight > class_name: {module.__class__.__name__}')
      lora_dict = {
                   "module_name": current_module_name, 
                   "class_name": module.__class__.__name__, 
                   "lora_down_weight": module.lora_down.weight.data, 
                   "lora_up_weight": module.lora_up.weight.data,
                   }
      lora_weight.append(lora_dict)
    else:
      depth += 1
      find_lora_weight(module, lora_weight, current_module_name, depth)

 

"""
description: save lora weight of the model into .pth
params:
-model: model be searched
-file_path: lora module weight file .pth
-model_name: name of model be searched
-lora_weight: the list be appended by lora weight
"""
def save_lora(
  model,
  file_path: str,
  model_name: str,
  lora_weight: list, 
):
  find_lora_weight(model, lora_weight, model_name, 0)

  torch.save(lora_weight, file_path)

  

"""
description: replace module of base model by lora module, load lora weight from .pth into lora module
params:
-module_x: module be updated by lora weight
-lora_x: lora dictionary with lora information
"""
def update_lora_weight_direct(
  module_x,
  lora_x,
):
  lora_down_weight_x = lora_x["lora_down_weight"]
  lora_up_weight_x = lora_x["lora_up_weight"]
  weight_x = module_x.weight.data

  #calculate update lora weight by lora down weight and lora up weight
  if lora_x["class_name"] == "LoraConv1d":
    lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c l -> l o_c i_c')
    lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c l -> l o_c i_c')
    lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
    lora_weight_x = rearrange(lora_weight_x_rearrange, 'l o_c i_c -> o_c i_c l').to("cuda")
    update_weight_x = weight_x + lora_weight_x
  elif lora_x["class_name"] == "LoraConv2d":
    lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c h w -> h w o_c i_c')
    lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c h w -> h w o_c i_c')
    lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
    lora_weight_x = rearrange(lora_weight_x_rearrange, 'h w o_c i_c -> o_c i_c h w').to("cuda")
    update_weight_x = weight_x + lora_weight_x
  elif lora_x["class_name"] == "LoraConv3d":
    lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c d h w -> d h w o_c i_c')
    lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c d h w -> d h w o_c i_c')
    lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
    lora_weight_x = rearrange(lora_weight_x_rearrange, 'd h w o_c i_c -> o_c i_c d h w').to("cuda")
    update_weight_x = weight_x + lora_weight_x
  elif lora_x["class_name"] == "LoraLinear":
    lora_weight_x = torch.matmul(lora_up_weight_x, lora_down_weight_x).to("cuda")
    update_weight_x = weight_x + lora_weight_x

  #update module weight by lora weight
  module_x.weight.data = update_weight_x

  euclidean_distance_before_after_weight_x = torch.dist(weight_x, update_weight_x, p=2)
  print(f'update_lora_weight_direct > euclidean_distance_before_after_weight_x: {euclidean_distance_before_after_weight_x}')



"""
description: replace module of base model by lora module, load lora weight from .pth into lora module
params:
-model: base model
-file_path: lora module weight file .pth 
"""
def load_lora(
  model,
  file_path: str, 
):

  lora_weight = torch.load(file_path)
  
  for lora in lora_weight:
    print(f'load_lora > module_name: {lora["module_name"]}')
    print(f'load_lora > class_name: {lora["class_name"]}')
    
    module_name_list = lora["module_name"].split(".")

    module_name_list.pop(0)
    length = len(module_name_list)
    module_x = model
    for i in range(length):
      module_name_x = module_name_list.pop(0)
      module_x = module_x._modules[module_name_x]

    update_lora_weight_direct(module_x, lora)



"""
description: save model weight with relevant layers in the lora weight file
params:
-model: base model
-lora_file_path: lora module weight file .pth 
-save_file_path: save file path
-actions:
  "save model weight": save original model weight
  "save model lora weight": save model weight updated by lora
"""
def save_weight(
  model,
  lora_file_path: str, 
  save_file_path: str, 
  actions: str, 
):
  #print(f'save_weight > save_file_path: {save_file_path}')
  #print(f'save_weight > actions: {actions}')

  lora_weight = torch.load(lora_file_path)
  model_weight = []
  
  for lora in lora_weight:
    #print(f'save_weight > module_name: {lora["module_name"]}')
    #print(f'save_weight > class_name: {lora["class_name"]}')
    
    #find target module
    module_name_list = lora["module_name"].split(".")

    module_name_list.pop(0)
    length = len(module_name_list)
    module_x = model
    for i in range(length):
      module_name_x = module_name_list.pop(0)
      module_x = module_x._modules[module_name_x]

    #print(f'save_weight > module_name: {lora["module_name"]}')
    #print(f'save_weight > class_name: {lora["class_name"]}')
    #print(f'save_weight > module_x.__class__.__module__: {module_x.__class__.__module__}')
    #print(f'save_weight > module_x.__class__.__name__: {module_x.__class__.__name__}')

    #calculate weight and add new weight into model weight list
    if actions == "save model weight":
      #print(f'save_weight > save model weight > module_x.weight.data.shape: {module_x.weight.data.shape}')
      #print(f'save_weight > save model weight > module_x.weight.data.mean(): {module_x.weight.data.mean()}')

      new_model_weight  = {
                           "module_name": lora["module_name"], 
                           "class_name": lora["class_name"], 
                           "module_weight": module_x.weight.data,  
                          }
      model_weight.append(new_model_weight)

    elif actions == "save model lora weight":
      lora_down_weight_x = lora["lora_down_weight"]
      lora_up_weight_x = lora["lora_up_weight"]
      weight_x = module_x.weight.data

      #print(f'save_weight > save model lora weight > module_x.weight.data.shape: {module_x.weight.data.shape}')
      #print(f'save_weight > save model lora weight > module_x.weight.data.mean(): {module_x.weight.data.mean()}')
      #print(f'save_weight > save model lora weight > lora_down_weight_x.shape: {lora_down_weight_x.shape}')
      #print(f'save_weight > save model lora weight > lora_down_weight_x.mean(): {lora_down_weight_x.mean()}')
      #print(f'save_weight > save model lora weight > lora_up_weight_x.shape: {lora_up_weight_x.shape}')
      #print(f'save_weight > save model lora weight > lora_up_weight_x.mean(): {lora_up_weight_x.mean()}')

      #calculate update lora weight by lora down weight and lora up weight
      if lora["class_name"] == "LoraConv1d":
        lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c l -> l o_c i_c')
        lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c l -> l o_c i_c')
        lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
        lora_weight_x = rearrange(lora_weight_x_rearrange, 'l o_c i_c -> o_c i_c l').to("cuda")
        update_weight_x = weight_x + lora_weight_x
      elif lora["class_name"] == "LoraConv2d":
        lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c h w -> h w o_c i_c')
        lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c h w -> h w o_c i_c')
        lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
        lora_weight_x = rearrange(lora_weight_x_rearrange, 'h w o_c i_c -> o_c i_c h w').to("cuda")
        update_weight_x = weight_x + lora_weight_x
      elif lora["class_name"] == "LoraConv3d":
        lora_down_weight_x_rearrange = rearrange(lora_down_weight_x, 'o_c i_c d h w -> d h w o_c i_c')
        lora_up_weight_x_rearrange = rearrange(lora_up_weight_x, 'o_c i_c d h w -> d h w o_c i_c')
        lora_weight_x_rearrange = torch.matmul(lora_up_weight_x_rearrange, lora_down_weight_x_rearrange)
        lora_weight_x = rearrange(lora_weight_x_rearrange, 'd h w o_c i_c -> o_c i_c d h w').to("cuda")
        update_weight_x = weight_x + lora_weight_x
      elif lora["class_name"] == "LoraLinear":
        lora_weight_x = torch.matmul(lora_up_weight_x, lora_down_weight_x).to("cuda")
        update_weight_x = weight_x + lora_weight_x

      #print(f'save_weight > save model lora weight > update_weight_x.shape: {update_weight_x.shape}')
      #print(f'save_weight > save model lora weight > update_weight_x.mean(): {update_weight_x.mean()}')

      new_model_weight  = {
                           "module_name": lora["module_name"], 
                           "class_name": lora["class_name"], 
                           "module_weight": update_weight_x, 
                          }
      model_weight.append(new_model_weight)


  #save model weight list
  torch.save(model_weight, save_file_path)




"""
description: load weight to the model
params:
-model: base model
-file_path: weight file .pth 
"""
def load_weight(
  model,
  file_path: str, 
):

  weights = torch.load(file_path)
  
  for weight in weights:
    #print(f'load_weight > module_name: {weight["module_name"]}')
    #print(f'load_weight > class_name: {weight["class_name"]}')
    
    #find target module
    module_name_list = weight["module_name"].split(".")

    module_name_list.pop(0)
    length = len(module_name_list)
    module_x = model
    for i in range(length):
      module_name_x = module_name_list.pop(0)
      module_x = module_x._modules[module_name_x]

    #update module weight
    #print(f'load_weight > weight["module_name"]: {weight["module_name"]}')
    #print(f'load_weight > weight["class_name"]: {weight["class_name"]}')
    #print(f'load_weight > module_x.__class__.__module__: {module_x.__class__.__module__}')
    #print(f'load_weight > module_x.__class__.__name__: {module_x.__class__.__name__}')
    #print(f'load_weight > weight["module_weight"].shape: {weight["module_weight"].shape}')
    #print(f'load_weight > weight["module_weight"].mean(): {weight["module_weight"].mean()}')
    #print(f'load_weight > module_x.weight.data.shape: {module_x.weight.data.shape}')
    #print(f'load_weight > module_x.weight.data.mean(): {module_x.weight.data.mean()}')

    #module_x_weight_min = torch.min(module_x.weight.data)
    #module_x_weight_max = torch.max(module_x.weight.data)
    module_x.weight.data = copy.deepcopy(weight["module_weight"])
    #module_x.weight.data = torch.clamp(module_x.weight.data, min=module_x_weight_min, max=module_x_weight_max)

    #print(f'load_weight > updated module_x.weight.data.shape: {module_x.weight.data.shape}')
    #print(f'load_weight > updated module_x.weight.data.mean(): {module_x.weight.data.mean()}')

    
  
"""
description: Enable gradients for lora module in model
params:
-model: model with lora module
-key_word: search key word in module class name
-depth: current depth layer from start point
-depth_max: max depth layer from start point
"""
def enable_lora_train(
  model, 
  key_word: str, 
  depth: int, 
  depth_max: int = 50,  
):
  if depth >= depth_max:
    return

  for name, module in model.named_children():
    if key_word in module.__class__.__name__:
      print(f'enable_lora_train > module_name: {name}')
      print(f'enable_lora_train > module.__class__.__name__: {module.__class__.__name__}')

      for name, param in module.named_parameters():
        if "lora" in name:
          param.requires_grad = True
        print(f'enable_lora_train > name: {name}, param.requires_grad: {param.requires_grad}')        
    else:
      depth += 1
      enable_lora_train(module, key_word, depth)



"""
description: disable gradients for origianl conv in lora module in model
params:
-model: model with lora module
-key_word: search key word in module class name
-depth: current depth layer from start point
-depth_max: max depth layer from start point
"""
def disable_lora_conv_train(
  model, 
  key_word: str, 
  depth: int, 
  depth_max: int = 50,  
):
  if depth >= depth_max:
    return

  for name, module in model.named_children():
    if key_word in module.__class__.__name__:
      print(f'disable_lora_conv_train > module_name: {name}')
      print(f'disable_lora_conv_train > module.__class__.__name__: {module.__class__.__name__}')

      for name, param in module.named_parameters():
        if "conv" in name:
          param.requires_grad = False
        print(f'disable_lora_conv_train > name: {name}, param.requires_grad: {param.requires_grad}')        
    else:
      depth += 1
      disable_lora_conv_train(module, key_word, depth)
