import torch
import torch.nn as nn
import torch.nn.functional as F



"""
lora for Conv channels 
"""
class LoraConv1d(nn.Module):
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

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.lora_down(input))
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

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.lora_down(input))
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

        #initialize lora weight
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)
        #nn.init.normal_(self.lora_up.weight, std=1 / r)

    def forward(self, input):
        return (
            self.conv(input)
            + self.lora_up(self.lora_down(input))
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
