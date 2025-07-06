# simple_diffusion_model

A first-step PyTorch implementation of a denoising diffusion probabilistic model (DDPM).  
This repo trains a DDPM for a fixed number of timesteps, visualizes loss curves, and generates sample grids to inspect how the model learns structure over time.

------ Run this in terminal ------

python main_script.py --experiment_name T10 --num_timesteps 10
python main_script.py --experiment_name T100 --num_timesteps 100
python main_script.py --experiment_name T300 --num_timesteps 300
python main_script.py --experiment_name T100_cosine --num_timesteps 100 --beta_schedule cosine
