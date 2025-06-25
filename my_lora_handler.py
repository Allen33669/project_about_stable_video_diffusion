import torch
from einops import rearrange



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
  #print(f'add_lora_into_model > depth: {depth}')
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
  current_timestep = torch.load("/content/generative-models/current_timestep.tar")['current_timestep']
  #print(f'save_statistic_sample_backward_hook > current_timestep: {current_timestep}')
  #print(f'save_statistic_sample_backward_hook > module.in_model_layer: {module.in_model_layer}')

  statistic_sample_backward_list = torch.load("/content/generative-models/statistic_sample_backward.tar")
  #print(f'save_statistic_sample_backward_hook > len(grad_input): {len(grad_input)}')
  
  grad_input_mean_abs = []
  for i, tensor in enumerate(grad_input):
    if tensor is not None:
      grad_input_mean_abs.append(torch.mean(torch.abs(tensor)))
    else:
      grad_input_mean_abs.append(None)

  grad_output_mean_abs = torch.mean(torch.abs(grad_output[0]))
  statistic_sample_backward_new = {
                                   "in_model_layer": module.in_model_layer, 
                                   "in_model_toal_layer": module.in_model_toal_layer, 
                                   "in_model_Unet_up_or_down_layer": module.in_model_Unet_up_or_down_layer, 
                                   "in_model_position": module.in_model_position, 
                                   "in_model_replaced_module": module.in_model_replaced_module, 
                                   "in_model_task": module.in_model_task, 
                                   "current_timestep": current_timestep, 
                                   "grad_input_mean_abs": grad_input_mean_abs, 
                                   "grad_output_mean_abs": grad_output_mean_abs, 
                                  }
  statistic_sample_backward_list.append(statistic_sample_backward_new)

  #print(f'save_statistic_sample_backward_hook > len(statistic_sample_backward_list): {len(statistic_sample_backward_list)}')
  #print(f'save_statistic_sample_backward_hook > statistic_sample_backward_list[len(statistic_sample_backward_list) - 1]: {statistic_sample_backward_list[len(statistic_sample_backward_list) - 1]}')

  torch.save(statistic_sample_backward_list, "/content/generative-models/statistic_sample_backward.tar")



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
return: replaced module
"""
def replace_lora_module_with_statistic_info(
  module, 
  module_name: str, 
  target_module,
  target_module_name: str,  
  r = 16,
  scale: float = 1.0, 
  in_model_layer: int = None, 
  in_model_Unet_up_or_down_layer: int = None, 
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
                                  in_model_layer = in_model_layer, 
                                  in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer, 
                                  in_model_position = target_module.__class__.__name__, 
                                  in_model_replaced_module = target_module.__class__.__name__, 
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
                                 )

  module._modules[target_module_name] = replace_module
  module._modules[target_module_name].register_full_backward_hook(save_statistic_sample_backward_hook)

  print(f'replace_lora_module_with_statistic_info > parent module: {module_name}, target_module_name: {target_module_name}, target_module_class_name: {target_module.__class__.__name__}')

  return module._modules[target_module_name]



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
  in_model_Unet_up_or_down_layer:int = None, 
):
  #print(f'add_lora_into_model_with_statistic_info > depth: {depth}')
  if depth >= depth_max:
    return

  for name, module in model.named_children():   
    #replace the module to lora module 
    if module.__class__.__name__ == target:
      replace_lora_module_with_statistic_info(model, model_name, module, name, in_model_layer=in_model_layer, in_model_Unet_up_or_down_layer=in_model_Unet_up_or_down_layer)
    else:
      depth += 1
      add_lora_into_model_with_statistic_info(module, name, target, depth, in_model_layer=in_model_layer, in_model_Unet_up_or_down_layer=in_model_Unet_up_or_down_layer)



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
      #print(f'find_lora_weight > depth: {depth}')
      print(f'find_lora_weight > module_name: {current_module_name}')
      print(f'find_lora_weight > class_name: {module.__class__.__name__}')
      #print(f'find_lora_weight > lora_down_weight.shape: {module.lora_down.weight.data.shape}')
      #print(f'find_lora_weight > lora_up_weight.shape: {module.lora_up.weight.data.shape}')
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
  #print(f'update_lora_weight_direct > lora_down_weight_x.shape: {lora_down_weight_x.shape}')
  #print(f'update_lora_weight_direct > lora_up_weight_x.shape: {lora_up_weight_x.shape}')
  #print(f'update_lora_weight_direct > weight_x.shape: {weight_x.shape}')

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

  #print(f'update_lora_weight_direct > lora_down_weight_x_rearrange.shape: {lora_down_weight_x_rearrange.shape}')
  #print(f'update_lora_weight_direct > lora_up_weight_x_rearrange.shape: {lora_up_weight_x_rearrange.shape}')
  #print(f'update_lora_weight_direct > lora_weight_x_rearrange.shape: {lora_weight_x_rearrange.shape}')
  #print(f'update_lora_weight_direct > lora_weight_x.shape: {lora_weight_x.shape}')
  #print(f'update_lora_weight_direct > update_weight_x.shape: {update_weight_x.shape}')

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
    #print(f'load_lora > lora_down_weight.shape: {lora["lora_down_weight"].shape}')
    #print(f'load_lora > lora_up_weight.shape: {lora["lora_up_weight"].shape}')
    
    module_name_list = lora["module_name"].split(".")

    module_name_list.pop(0)
    length = len(module_name_list)
    module_x = model
    for i in range(length):
      module_name_x = module_name_list.pop(0)
      module_x = module_x._modules[module_name_x]
    
    update_lora_weight_direct(module_x, lora)
    

  
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
      print(f'enable_lora_train > depth: {depth}')
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
      print(f'disable_lora_conv_train > depth: {depth}')
      print(f'disable_lora_conv_train > module_name: {name}')
      print(f'disable_lora_conv_train > module.__class__.__name__: {module.__class__.__name__}')

      for name, param in module.named_parameters():
        if "conv" in name:
          param.requires_grad = False
        print(f'disable_lora_conv_train > name: {name}, param.requires_grad: {param.requires_grad}')        
    else:
      depth += 1
      disable_lora_conv_train(module, key_word, depth)
