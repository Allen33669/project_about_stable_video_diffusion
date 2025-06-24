import torch
import torch.nn as nn
import torch.nn.functional as F



"""
lora for Conv channels 

"""
class LoraConv1d(nn.Module):
    """
    params:
    -in_model_layer: the module in the layer of the model. like 0, 1, 2, or etc.
    -in_model_toal_layer: the total layer of the model. like 16, 32, or etc.
    -in_model_Unet_up_or_down_layer: the module in the up or down layer of the Unet architecture model. Up layer is from first layer of model. Down layer is from last layer of model. like 0, 1, 2, or etc.
    -in_model_position: the module position in the layer of the model. like "Q", "K", "V", "ConvFirst", "ConvSecond", "FFNFirst", "FFNSecond", and etc.
    -in_model_replaced_module: the class name of the original module be replaced by lora module. like "Conv1d", "Conv2d", "Linear", and etc.
    -in_model_task: handle spacial, temporal, or etc. like "spatial", "temporal", "original", and etc.
    -current_time_step: current time step in sampling process.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 16,
        scale: float = 1.0,
        in_model_layer: int = -1,
        in_model_toal_layer: int = 27, #12 (input_blocks) + 3 (middle_block) + 12 (output_blocks) = 27
        in_model_Unet_up_or_down_layer: int = None,
        in_model_position: str = None,
        in_model_replaced_module: str = None,
        in_model_task: str = None,
        current_time_step: float = None, 
        dropout_p: float = 0.2, 
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            print(f"LoRA rank {r} is too large. setting to: {min(in_channels, out_channels)}")
            r = min(in_channels, out_channels)

        self.r = r
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
 
        self.lora_up = nn.Conv1d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.scale = scale
        self.dropout = nn.Dropout(p=dropout_p)

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

        self.in_model_layer = in_model_layer
        self.in_model_toal_layer = in_model_toal_layer
        self.in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer
        self.in_model_position = in_model_position
        self.in_model_replaced_module = in_model_replaced_module
        self.in_model_task = in_model_task
        self.current_time_step = current_time_step

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.dropout(self.lora_down(input)))
            * self.scale
        )

    """
    return: base module weight params, lora down weight params, lora up weight params
    """
    def get_params(self):
        return self.conv.weight.data, self.lora_down.weight.data, self.lora_up.weight.data

    """
    print: module, base module, lora down, lora up
    """
    def get_module_architecture(self):
        print(self)
        print(self.conv)
        print(self.lora_down)
        print(self.lora_up)



"""
lora for Conv channels 
"""
class LoraConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 16,
        scale: float = 1.0,
        in_model_layer: int = -1,
        in_model_toal_layer: int = 27, #12 (input_blocks) + 3 (middle_block) + 12 (output_blocks) = 27
        in_model_Unet_up_or_down_layer: int = None,
        in_model_position: str = None,
        in_model_replaced_module: str = None,
        in_model_task: str = None,
        current_time_step: float = None, 
        dropout_p: float = 0.2, 
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            print(f"LoRA rank {r} is too large. setting to: {min(in_channels, out_channels)}")
            r = min(in_channels, out_channels)

        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
 
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.scale = scale
        self.dropout = nn.Dropout(p=dropout_p)

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

        self.in_model_layer = in_model_layer
        self.in_model_toal_layer = in_model_toal_layer
        self.in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer
        self.in_model_position = in_model_position
        self.in_model_replaced_module = in_model_replaced_module
        self.in_model_task = in_model_task
        self.current_time_step = current_time_step

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.dropout(self.lora_down(input)))
            * self.scale
        )

    """
    return: base module weight params, lora down weight params, lora up weight params
    """
    def get_params(self):
        return self.conv.weight.data, self.lora_down.weight.data, self.lora_up.weight.data

    """
    print: module, base module, lora down, lora up
    """
    def get_module_architecture(self):
        print(self)
        print(self.conv)
        print(self.lora_down)
        print(self.lora_up)



"""
lora for Conv channels 
"""
class LoraConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 16,
        scale: float = 1.0,
        in_model_layer: int = -1,
        in_model_toal_layer: int = 27, #12 (input_blocks) + 3 (middle_block) + 12 (output_blocks) = 27
        in_model_Unet_up_or_down_layer: int = None,
        in_model_position: str = None,
        in_model_replaced_module: str = None,
        in_model_task: str = None,
        current_time_step: float = None, 
        dropout_p: float = 0.2, 
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            print(f"LoRA rank {r} is too large. setting to: {min(in_channels, out_channels)}")
            r = min(in_channels, out_channels)

        self.r = r
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.lora_down = nn.Conv3d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
 
        self.lora_up = nn.Conv3d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.scale = scale
        self.dropout = nn.Dropout(p=dropout_p)

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

        self.in_model_layer = in_model_layer
        self.in_model_toal_layer = in_model_toal_layer
        self.in_model_Unet_up_or_down_layer = in_model_Unet_up_or_down_layer
        self.in_model_position = in_model_position
        self.in_model_replaced_module = in_model_replaced_module
        self.in_model_task = in_model_task
        self.current_time_step = current_time_step

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.dropout(self.lora_down(input)))
            * self.scale
        )



    """
    return: base module weight params, lora down weight params, lora up weight params
    """
    def get_params(self):
        return self.conv.weight.data, self.lora_down.weight.data, self.lora_up.weight.data



    """
    print: module, base module, lora down, lora up
    """
    def get_module_architecture(self):
        print(self)
        print(self.conv)
        print(self.lora_down)
        print(self.lora_up)
