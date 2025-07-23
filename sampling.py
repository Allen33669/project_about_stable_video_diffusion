"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (get_ancestral_step,
                                                        linear_multistep_coeff,
                                                        to_d, to_neg_log_sigma,
                                                        to_sigma)
from ...util import append_dims, default, instantiate_from_config



from my_lora_handler import * #modified code start end



DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        #modified code start
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2

        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 5, Formula (5)
        #produce sigmas
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2
        #σ(t) = t
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: σ(t) = t")
        #print(f"sigmas: {sigmas}")

        #unconditions
        uc = default(uc, cond)

        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2
        #sample x0
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: sample x0")
        #print(f"original x: {x[:1, :1, :1]}")
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        #print(f"sampled x: {x[:1]}")

        num_sigmas = len(sigmas)

        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2
        #s(t) = 1
        s_in = x.new_ones([x.shape[0]])
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: s(t) = 1")
        #print(f"s(t): {s_in}")

        #modified code end

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        
        #modified code start
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2
        #Take Euler step from tˆi to t
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: Take Euler step from tˆi to t")
        #print(f"x: {x[:1, :1, :1]}")
        #print(f"d: {d[:1, :1, :1]}")
        #print(f"dt: {dt[:1, :1, :1]}")
        #print(f"x + dt * d: {(x + dt * d)[:1, :1, :1]}")
        return x + dt * d
        #modified code end


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        #modified code start
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2
        #used for calculate γ
        self.s_churn = s_churn
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: used for calculate γ")
        #print(f"s_churn: {self.s_churn}")


        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        #modified code end

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        #modified code start
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2

        #Select temporarily increased noise level t
        sigma_hat = sigma * (gamma + 1.0)
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: Select temporarily increased noise level t")
        #print(f"t: {sigma}")
        #print(f"gamma: {gamma}")
        #print(f"t_hat: {sigma_hat}")

        if gamma > 0:
            #sample new noise
            eps = torch.randn_like(x) * self.s_noise
            #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: sample new noise")
            #print(f"eps: {eps[:1, :1, :1]}")

            #Add new noise to move from ti to t
            #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: Add new noise to move from ti to t")
            #print(f"original x: {x[:1, :1, :1]}")
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
            #print(f"new x: {x[:1, :1, :1]}")
        
        #calculate denoised result
        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)

        #Evaluate dx/dt at t
        d = to_d(x, sigma_hat, denoised)
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: Evaluate dx/dt at t")
        #print(f"d: {d[:1, :1, :1]}")
        
        #calculate dt
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: calculate dt")
        #print(f"dt: {dt[:1, :1, :1]}")

        # Take Euler step from ti to ti+ 
        euler_step = self.euler_step(x, d, dt)

        # Apply 2nd order correction unless σ goes to zero
        #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: Apply 2nd order correction unless σ goes to zero")
        #print(f"original x: {x[:1, :1, :1]}")
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        #print(f"new x: {x[:1, :1, :1]}")

        #modified code end

        return x








    """
    description: EDM sampling, optional: Spatial-Temporal Collaborative Sampling (SCS) (VideoMage)
    params:
    -denoiser: base model
    -x: input
    -cond: conditions
    -uc: un-conditions
    -num_steps: number of sampling steps
    -scs_t: Spatial-Temporal Collaborative Sampling (SCS) (VideoMage)
      = 0: use original sampling
      > 0: use SCS
      i: generate new motion input and appearence input in i steps
    -model: diffusion model, used for update weight and sample
    -x_motion: motion input
    -cond_motion: motion conditions
    -uc_motion: motion un-conditions
    -x_appearence: appearence input
    -cond_appearence: appearence conditions
    -uc_appearence: appearence un-conditions
    -beta_motion: used for SCS combining motion input and appearence input
    -beta_appearence: used for SCS combining motion input and appearence input
    -alpha: used for SCS update motion input and appearence input in scs_t steps
    """
    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, 
                 scs_t: float = -1,  #modified code start end
                 model = None, #modified code start end
                 x_motion = None, cond_motion = None, uc_motion =  None, #modified code start end
                 x_appearence = None, cond_appearence = None, uc_appearence=None, #modified code start end
                 beta_motion: float = 0.5, beta_appearence: float = 0.5, #modified code start end
                 alpha: float = 0.1, #modified code start end
                ):
        #modified code start
        #Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2

        #prepare parameters for starting sampling loop, parameters includes x0, s_in, sigmas, num sigmas, condition, uncondition 
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        """ 
        print(f'EDMSampler > __call__ > self.prepare_sampling_loop > type(x): {type(x)}')
        if isinstance(x, torch.Tensor):
          print(f'EDMSampler > __call__ > self.prepare_sampling_loop > x.shape: {x.shape}')
        print(f'EDMSampler > __call__ > self.prepare_sampling_loop > s_in: {s_in}')
        print(f'EDMSampler > __call__ > self.prepare_sampling_loop > sigmas: {sigmas}')
        print(f'EDMSampler > __call__ > self.prepare_sampling_loop > num_sigmas: {num_sigmas}')

        for key, value in cond.items():
          print(f"EDMSampler > __call__ > cond > key: {key}")
          print(f"EDMSampler > __call__ > cond > type(value): {type(value)}")

          if isinstance(value, torch.Tensor):
            print(f"EDMSampler > __call__ > cond > value.shape: {value.shape}")
            print(f"EDMSampler > __call__ > cond > value.mean(): {value.mean()}")
          else:
            print(f"EDMSampler > __call__ > cond > value: {value}")

        for key, value in uc.items():
          print(f"EDMSampler > __call__ > uc > key: {key}")
          print(f"EDMSampler > __call__ > uc > type(value): {type(value)}")

          if isinstance(value, torch.Tensor):
            print(f"EDMSampler > __call__ > uc > value.shape: {value.shape}")
            print(f"EDMSampler > __call__ > uc > value.mean(): {value.mean()}")
          else:
            print(f"EDMSampler > __call__ > uc > value: {value}")
        """



        #EDMSampler sampling loop
        for i in self.get_sigma_gen(num_sigmas):
            print(f'EDMSampler > __call__ > i: {i}')

            #calculate γ
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            #print(f"Elucidating the Design Space of Diffusion-Based Generative Models, Page 7, algorithm 2: calculate γ")
            #print(f"s_churn: {self.s_churn}")
            #print(f"num_sigmas: {num_sigmas}")
            #print(f"gamma: {gamma}")

            #modified code start
            if scs_t > i:
              print(f'EDMSampler > __call__ > scs_t: {scs_t}')

              #restore model original weight
              load_weight(model, file_path="/content/generative-models/model_original_weight_motion.pth")
              load_weight(model, file_path="/content/generative-models/model_original_weight_appearence.pth")

              #calculate model output with motion condition
              #print(f'EDMSampler > __call__ > torch.isnan(x).sum(): {torch.isnan(x).sum()}')
              #print(f'EDMSampler > __call__ > torch.isnan(cond_motion).sum(): {torch.isnan(cond_motion).sum()}')
              #print(f'EDMSampler > __call__ > torch.isnan(uc_motion).sum(): {torch.isnan(uc_motion).sum()}')
              
              x_cond_motion = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x,
                  cond_motion,
                  uc_motion,
                  gamma,
              )
              
              #print(f'EDMSampler > __call__ > x_cond_motion.shape: {x_cond_motion.shape}')
              #print(f'EDMSampler > __call__ > x_cond_motion.mean(): {x_cond_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_cond_motion).sum(): {torch.isnan(x_cond_motion).sum()}')

              #x_cond_motion = torch.where(torch.isnan(x_cond_motion), torch.tensor(0.0), x_cond_motion)
              #print(f'EDMSampler > __call__ > x_cond_motion.shape: {x_cond_motion.shape}')
              #print(f'EDMSampler > __call__ > x_cond_motion.mean(): {x_cond_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_cond_motion).sum(): {torch.isnan(x_cond_motion).sum()}')

              #calculate model output with appearence condition
              x_cond_appearence = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x,
                  cond_appearence,
                  uc_appearence,
                  gamma,
              )
              #print(f'EDMSampler > __call__ > x_cond_appearence.shape: {x_cond_appearence.shape}')
              #print(f'EDMSampler > __call__ > x_cond_appearence.mean(): {x_cond_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_cond_appearence).sum(): {torch.isnan(x_cond_appearence).sum()}')

              #x_cond_appearence = torch.where(torch.isnan(x_cond_appearence), torch.tensor(0.0), x_cond_appearence)
              #print(f'EDMSampler > __call__ > x_cond_appearence.shape: {x_cond_appearence.shape}')
              #print(f'EDMSampler > __call__ > x_cond_appearence.mean(): {x_cond_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_cond_appearence).sum(): {torch.isnan(x_cond_appearence).sum()}')

              #update model weight to model motion
              load_weight(model, file_path="/content/generative-models/model_lora_weight_motion.pth")

              #calculate motion model output with motion condition
              x_motion = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x,
                  cond_motion,
                  uc_motion,
                  gamma,
              )
              #print(f'EDMSampler > __call__ > x_motion.shape: {x_motion.shape}')
              #print(f'EDMSampler > __call__ > x_motion.mean(): {x_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_motion).sum(): {torch.isnan(x_motion).sum()}')

              #x_motion = torch.where(torch.isnan(x_motion), torch.tensor(0.0), x_motion)
              #print(f'EDMSampler > __call__ > x_motion.shape: {x_motion.shape}')
              #print(f'EDMSampler > __call__ > x_motion.mean(): {x_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_motion).sum(): {torch.isnan(x_motion).sum()}')

              #update model weight to model appearence
              load_weight(model, file_path="/content/generative-models/model_original_weight_motion.pth")
              load_weight(model, file_path="/content/generative-models/model_lora_weight_appearence.pth")

              #calculate appearence model output with appearence condition
              x_appearence = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x,
                  cond_appearence,
                  uc_appearence,
                  gamma,
              )
              #print(f'EDMSampler > __call__ > x_appearence.shape: {x_appearence.shape}')
              #print(f'EDMSampler > __call__ > x_appearence.mean(): {x_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_appearence).sum(): {torch.isnan(x_appearence).sum()}')

              #x_appearence = torch.where(torch.isnan(x_appearence), torch.tensor(0.0), x_appearence)
              #print(f'EDMSampler > __call__ > x_appearence.shape: {x_appearence.shape}')
              #print(f'EDMSampler > __call__ > x_appearence.mean(): {x_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_appearence).sum(): {torch.isnan(x_appearence).sum()}')



              #update x_motion
              x_motion = x_motion - alpha * (x_appearence - x_cond_motion)
              #x_motion = torch.clamp(x_motion, min=1e-10)

              #update x_appearence
              x_appearence = x_appearence - alpha * (x_cond_appearence - x_motion)
              #x_appearence = torch.clamp(x_appearence, min=1e-10)



            """
            #EDMSampler sampling one step
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )
            """

            #modified code start
            if scs_t > 0: 
              #restore model original weight
              load_weight(model, file_path="/content/generative-models/model_original_weight_motion.pth")
              load_weight(model, file_path="/content/generative-models/model_original_weight_appearence.pth")

              #update model weight to model motion
              load_weight(model, file_path="/content/generative-models/model_lora_weight_motion.pth")

              #calculate new x_motion and x_appearence, combine x_motion and x_appearence
              x_motion = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x_motion,
                  cond_motion,
                  uc_motion,
                  gamma,
              )
              #print(f'EDMSampler > __call__ > final x_motion.shape: {x_motion.shape}')
              #print(f'EDMSampler > __call__ > final x_motion.mean(): {x_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_motion).sum(): {torch.isnan(x_motion).sum()}')

              #x_motion = torch.where(torch.isnan(x_motion), torch.tensor(0.0), x_motion)
              #print(f'EDMSampler > __call__ > x_motion.shape: {x_motion.shape}')
              #print(f'EDMSampler > __call__ > x_motion.mean(): {x_motion.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_motion).sum(): {torch.isnan(x_motion).sum()}')



              #update model weight to model appearence
              load_weight(model, file_path="/content/generative-models/model_original_weight_motion.pth")
              load_weight(model, file_path="/content/generative-models/model_lora_weight_appearence.pth")

              x_appearence = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x_appearence,
                  cond_appearence,
                  uc_appearence,
                  gamma,
              )
              #print(f'EDMSampler > __call__ > final x_appearence.shape: {x_appearence.shape}')
              #print(f'EDMSampler > __call__ > final x_appearence.mean(): {x_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_appearence).sum(): {torch.isnan(x_appearence).sum()}')

              #x_appearence = torch.where(torch.isnan(x_appearence), torch.tensor(0.0), x_appearence)
              #print(f'EDMSampler > __call__ > x_appearence.shape: {x_appearence.shape}')
              #print(f'EDMSampler > __call__ > x_appearence.mean(): {x_appearence.mean()}')
              #print(f'EDMSampler > __call__ > torch.isnan(x_appearence).sum(): {torch.isnan(x_appearence).sum()}')



              x = beta_motion * x_motion + beta_appearence * x_appearence
              #x = torch.clamp(x, min=1e-10)
              #print(f'EDMSampler > __call__ > final x.shape: {x.shape}')
              #print(f'EDMSampler > __call__ > final x.mean(): {x.mean()}')

            else:
              #EDMSampler sampling one 
              x = self.sampler_step(
                  s_in * sigmas[i],
                  s_in * sigmas[i + 1],
                  denoiser,
                  x,
                  cond,
                  uc,
                  gamma,
              )

            #modified code end



        #modified code end
        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x