import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"



from my_common_variable import * #modified code start end
from my_utils import * #modified code start end



class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:

        print_all(x, "OpenAIWrapper > forward > x >")

        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                #context=c.get("crossattn", None),
                context=c.get(spatial_crossattn_context_key, None), #modified code start end
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None), 
                time_context=c.get(temporal_crossattn_context_key, None), #modified code start end
                **kwargs,
            )
        else:
            return self.diffusion_model(
                x,
                timesteps=t,
                #context=c.get("crossattn", None),
                context=c.get(spatial_crossattn_context_key, None), #modified code start end
                y=c.get("vector", None), 
                time_context=c.get(temporal_crossattn_context_key, None), #modified code start end
                **kwargs,
            )
