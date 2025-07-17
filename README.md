<head>

</head>

<body>
<h3>Stable Video Diffusion is licensed under the Stable Video Diffusion Research License, Copyright (c) Stability AI Ltd. All Rights Reserved.</h3>
<ul>
  <li><a href=https://github.com/Allen33669/stable_video_diffusion_project>source github</a></li>
  <li><a href=https://github.com/Allen33669/stable_video_diffusion_project/blob/main/LICENSE.txt>license</a></li>
  <li><a href=https://github.com/Allen33669/stable_video_diffusion_project/blob/main/Notice.txt>notice</a></li>
</ul>
<br>
<br>
<br>
<h1>References:</h1>
<ul>
<blockquote><h3>paper: </h3><a href=https://arxiv.org/abs/2311.15127>Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets</a> </blockquote>
  <li><a href=https://github.com/Stability-AI/generative-models>github</a></li>
  <li><a href=https://github.com/Stability-AI/generative-models/blob/main/LICENSE-CODE>license</a></li>
  <li><a href=https://github.com/Stability-AI/generative-models/blob/main/model_licenses/LICENSE-SVD>model license</a></li>
  <li><a href=https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/tree/main>hugging face</a></li>
  <li><a href=https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE.md>hugging face license Powered by Stability AI</a></li>
</ul>
<br>
<ul>
<blockquote><h3>paper: </h3><a href=https://arxiv.org/abs/2206.00364>Elucidating the Design Space of Diffusion-Based Generative Models</a></blockquote>
</ul>
<br>
<br>
<br>
<h1>Personal portfolio space:</h1>
<h2>project about stable video diffusion:</h2>
<br>
<br>
<br>
<blockquote>
<h3>Features: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD LoRA ConvXd fine-tuning, add text condition, simple lora analysis<br>
  <blockquote>
  project_about_stable_video_diffusion_2_9_0.ipynb<br>
    A brief introduction:<br> 
      framework<br>
        <blockquote>
        decouple the components of the framework<br>
        sampling method<br>
        continuous-time common Diffusion Model framework<br>
        find differentiation of x<br>
        neural network's target is score function<br>
        </blockquote>
      model architecture<br>
        <blockquote>
        transformer for video<br>
        residual block for video<br>
        decoder<br>
        </blockquote>
      conditions<br>
        <blockquote>
        embedding style<br>
        fuse style<br>
        </blockquote>
   </blockquote>
   <blockquote>
    code:<br>
      dataset: for unordered appearence dataset and ordered motion dataset<br>
      model:<br>
        <blockquote>
        text condition: add text condition as time context<br>
        LoRA:<br>
          LoRA ConvXd class<br>
          LoRA Linear class in attention (Q, K, V, FFN)<br>
          train with LoRA ConvXd, LoRA Linear<br>
          save lora weight<br>
          load lora weight by update weight of the original layer of the base model<br>
        </blockquote>
      optimizer:<br>
        <blockquote>
        different learning rate: lora down learning rate > lora up learning rate<br>
        </blockquote>
      regularization:<br>
        <blockquote>
        dropout<br>
        weight decay<br>
        early stopping<br>
        </blockquote>
      tensor board:<br>
        <blockquote>
        train batch loss<br>
        train epoch loss<br>
        latent image output with channels<br>
        </blockquote>
      generate lora analysis samples<br>
        <blockquote>
        model layer<br>
        model Unet layer<br>
        model position<br>
        replaced module<br>
        module task<br>
        current time step<br>
        current_batch_step<br>
        current_epoch_step<br>
        gradient input<br>
        gradient output<br>
        lora_down_weight<br>
        lora_up_weight<br>
        </blockquote>
      lora analysis (only specific layers):<br>
        <blockquote>
        heatmap<br>
        ols<br>
        </blockquote>
  project_about_stable_video_diffusion_2_8_2.ipynb
    <blockquote>
    LoRA ConvXd class<br>
    lora analysis (all layers):<br>
    </blockquote>
  </blockquote>
PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation): A brief introduction to LoRA<br>
  <blockquote>
  project_about_stable_video_diffusion_1_3_0.ipynb<br>
    PEFT brief introduction and Why LoRA?<br>
    LoRA position / layer<br>
    LoRA rank<br>
    LoRA learning rate<br>
    LoRA architecture<br>
    Multiple LoRA / Multi-task<br>
    LoRA composition<br>
    LoRA generalized framework<br>
    LoRA mixture of experts brief introduction<br>
    LoRA MoE<br>
    LoRA application: Video generation<br>
    REFERENCES<br>
  </blockquote>
</blockquote>
<br>
<br>
<br>
<h3>2.9.0: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD LoRA ConvXd fine-tuning, add text condition, simple lora analysis<br>
  project_about_stable_video_diffusion_2_9_0.ipynb:<br>
    LoRA Linear class in attention (Q, K, V, FFN)<br>
    generate lora analysis samples:<br>
      in_model_replaced_module<br>
      in_model_task<br>
</blockquote>
<br>
<h3>2.8.2: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD LoRA ConvXd fine-tuning, add text condition, simple lora analysis<br>
  project_about_stable_video_diffusion_2_8_2.ipynb:<br>
    correct conditions: fuse style<br>
    dataset for unordered appearence dataset and ordered motion dataset<br>
    add text condition, train with appearence and motion<br>
    tensor board:<br>
      train batch loss<br>
      train epoch loss<br>
      latent image output with channels<br>
    generate lora analysis samples:<br>
      current_batch_step<br>
      current_epoch_step<br>
      lora_down_weight<br>
      lora_up_weight<br>
</blockquote>
<br>
<h3>2.6.1: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD LoRA ConvXd fine-tuning and simple lora analysis<br>
project_about_stable_video_diffusion_2_6_1.ipynb:<br>
    LoRA ConvXd class<br>
    train with LoRA ConvXd using regularization skill: dropout, weight decay, and early Stopping<br>
    different learning rate: lora down learning rate > lora up learning rate<br>
    generate lora analysis samples:<br>
      model layer<br>
      model Unet layer<br>
      model position<br>
      current time step<br>
      gradient input<br>
      gradient output<br>
    lora analysis:<br>
      heatmap<br>
      ols<br>
</blockquote>
<br>
<h3>2.2.1: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD LoRA fine-tuning<br>
project_about_stable_video_diffusion_2_2_1.ipynb:<br>
    LoRA ConvXd class<br>
    train with LoRA ConvXd<br>
    save lora weight<br>
    load lora weight by update weight of the original layer of the base model<br>
</blockquote>
<br>
<h3>2.0.1: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: SVD partial fine-tuning<br>
project_about_stable_video_diffusion_2_0_1.ipynb:<br>
  prepare training dataset<br>
  training configuration<br>
</blockquote>
<br>
<h3>1.3.0: </h3>
<blockquote>
PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation): A brief introduction to LoRA<br>
project_about_stable_video_diffusion_1_3_0.ipynb:<br>
  LoRA application: Video generation<br>
</blockquote>
<br>
<h3>1.2.0: </h3>
<blockquote>
PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation): A brief introduction to LoRA<br>
project_about_stable_video_diffusion_1_2_0.ipynb:<br>
  LoRA mixture of experts brief introduction<br>
  LoRA MoE<br>
</blockquote>
<br>
<h3>1.1.0: </h3>
<blockquote>
PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation): A brief introduction to LoRA<br>
project_about_stable_video_diffusion_1_1_0.ipynb:<br>
  Multiple LoRA / Multi-task<br>
  LoRA composition<br>
  LoRA generalized framework<br>
</blockquote>
<br>
<h3>1.0.0: </h3>
<blockquote>
PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation): A brief introduction to LoRA<br>
project_about_stable_video_diffusion_1_0_0.ipynb:<br>
    PEFT brief introduction and Why LoRA?<br>
    LoRA position / layer<br>
    LoRA rank<br>
    LoRA learning rate<br>
    LoRA architecture<br>
    REFERENCES<br>
</blockquote>
<br>
<h3>0.1.0: </h3>
<blockquote>
Stable Video Diffusion: A brief introduction to the paper and SVD source code: inference<br>
project_about_stable_video_diffusion_0_1_0.ipynb<br>
</blockquote>
</blockquote>
<br>








