# multiscale-neural-op: Learning Fast and Grid-independent PDE Solvers
This repository implements multiscale neural operator. A fast and grid-independent surrogate model of multiscale PDEs by combining neural operators with coarse-grained simulations. 

The repository contains the code of the following papers:
- Multiscale Neural Operator: Learning Fast and Grid-independent PDE Solvers
- Spectral PINNs: Fast Uncertainty Propagation with Physics-Informed Neural Networks

# TODO:
- Anonymize via removing LICENSE, Readme references, .gitmodules, .gitignore, doc, data

# Structure
- See [here](https://github.com/drivendata/cookiecutter-data-science) for folder structure.
- Data, config folders are structured: .../<experiment>/<diffeq>/<model>

# Install
```
git clone git@github.com:<author>/pce-pinns.git
cd pce-pinns
conda env create -f environment.yml # This may take ~20min
conda activate pce-pinns
pip install -e .
wandb login # Login to ML logging tool

pip install ray # tested with 1.4.1
pip install -U pydantic
git clone https://github.com/<author>/deepxde.git pce_pinns/deepxde # TODO: get rid of this installation requirement by installing deepxde as submodule
python pce_pinns/deepxde/deepxde/backend/set_default_backend.py pytorch                             
```

# Run Multiscale Neural Operator
```
conda activate pce-pinns
# MNO: Create data, train, and test MNO
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type "mno" 
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type "mno" --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type "mno" --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"
python main_lorenz96.py --n_samples 1300 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --model_type "mno" --eval_model_digest e323bd65c0 --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_1k_t1200" --parallel
# Train on existing data 
--load_data_path "data/raw/neurips22/lorenz96/fcnn/lorenz96_5k" 
--load_data_path "data/raw/temp/lorenz96/mno/lorenz96_5k_t200"
--load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"
# Evaluate
--load_data_path "data/raw/temp/lorenz96/mno/lorenz96_5k_t1000" # on supercloud
--eval_model_digest 67b659f901
# Sweep
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --est_param_nn "u" --parallel --model_type "mno" --no_test --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"

# FCNN: 
python main_lorenz96.py --n_samples 5000 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'fcnn' --overwrite --no_plot
--load_data_path "data/raw/neurips22/lorenz96/fcnn/lorenz96_5k" 
--eval_model_digest "fcnn_l02u0032e020lr01000n005000k4"

python main_lorenz96.py --n_samples 1300 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'fcnn' --overwrite --no_plot --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_1k_t1200"  --eval_model_digest "fcnn_l02u0032e002lr01000n005000k64"

# Climatology: 
python main_lorenz96.py --n_samples 5000 --mode_target "X" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'climatology' --overwrite --no_plot
--load_data_path "data/raw/neurips22/lorenz96/fcnn/lorenz96_5k_t1000" 

# Polynomial param:
python main_lorenz96.py --n_samples 5000 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'poly' --overwrite --no_plot 
--load_data_path "data/raw/temp/lorenz96/mno/lorenz96_5k_t1000"

# Pure ML param: 
python main_lorenz96.py --n_samples 5000 --mode_target "X" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type "mno" --no_plot --load_data_path "data/raw/neurips22/lorenz96/fcnn/lorenz96_5k" 

python create_icml_results.py --eval_results --eval_mno_digest 0007875792 --eval_fcnn_digest fcnn_l02u0032e020lr01000n005000k4 --eval_clim_digest climatology_n5000 --eval_poly_digest poly_n5000
--eval_runtime 
python create_icml_results.py --eval_results --eval_mno_digest 3d3510bff3 --eval_clim_digest climatology_n10000 --eval_fcnn_digest fcnn_l02u0032e002lr01000n005000k64
python create_icml_results.py --eval_results --eval_mno_digest e323bd65c0 --eval_clim_digest climatology_n1300 --eval_fcnn_digest fcnn_l02u0032e002lr01000n001300k64 --eval_poly_digest poly_n1300

```
# Run MNO on Quasi-Geostrophic Turbulence
```
# Install
cd ../torchqg
pip install -e .
# Generate data (on supercomputer)
python main.py # For spinup data
python main.py # For warmed data
# Train MNO
cd ../pce-pinns
python main_qgturb.py --no_wandb --model_type="mno" --load_data_path="../torchqg/output/warmed_100_dump" # Test run 
python main_qgturb.py --no_wandb --model_type="mno" --load_processed_data_path="data/processed/temp/qgturb/mno/n50_t20" # Full training run
# Evaluate
python main_qgturb.py --no_wandb --model_type="mno" --eval_model_digest "b4e8334aa9" --load_processed_data_path="data/processed/temp/qgturb/mno/n5_t20"  # 903804a78e
## eofe7 trained d5c949d18de with train loss 0.03, depth 3, n_ch 5, n_modes 64,64, lr 0.001, step = 20, gamma = 0.9, n_ep 500, warmed_100
```

# Run on server
```
export WANDB_MODE=offline
```

# Debug
```
# Debug local advection diffusion equation
python main_localAdvDiffEq.py --nn_pce --pce_dim 3 --est_param_nn "u" --n_samples 5 --silence_wandb --debug
# Debug lorenz96
python pce_pinns/deepxde/deepxde/backend/set_default_backend.py pytorch                             
python main_lorenz96.py --n_samples 2 --debug
```

## On MIT Supercloud supercomputer
```
ssh -L8848:localhost:8848 -L6007:localhost:6007 [username]@txe1-login.mit.edu # ssh [username]@eofe7.mit.edu
conda deactivate
cd pce-pinns
git fetch
git reset --hard origin/main
git pull
# conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
module load cuda/11.0 # module load anaconda3/2020.11
tmux new -s pce-sess # tmux at -t pce-sess
setw -g mode-mouse on
LLsub -i -s 20 -g volta:1 # LLsub -i full #  On eofe7 run $ srun -p sched_mit_darwin2 -n 28 --mem-per-cpu=8000 --pty /bin/bash # -N 1
# module load cuda/11.0
conda activate pce-pinns
export WANDB_MODE='offline'
# export CUDA_VISIBLE_DEVICES=0
python main_lorenz96.py --n_samples 5000 --seed 1 --no_wandb --mode_target "no-scale-sep-param-no-mem" --parallel --no_plot --overwrite 
python main_helmholtz.py  --n_samples 128 --load_data_path 'data/raw/temp/helmholtz' --seed 1 --no_plot
python main_localAdvDiffEq.py --nn_pce --pce_dim 3 --est_param_nn "u" --parallel --no_rand_insts --n_samples 100 # --silence_wandb
python main_localAdvDiffEq.py --nn_pce --pce_dim 3 --est_param_nn "u" --parallel --n_samples 10000 # --silence_wandb
ctrl+b -> d # Detach tmux session
tmux -> Ctrl+B and % to open new pane
tmux attach-session -t pce-sess # Reattach session
wandb sync wandb/offline-run-20210907_194341-18bscuvt 
scp kerberos@txe1-login.mit.edu:/home/gridsan/kerberos/pce-pinns/doc/figures/localAdvDiff/nn_pred_2D.png /mnt/c/Users/bjoern/ 
scp lorenz96.pickle kerberos@eofe7.mit.edu:/home/kerberos/pce-pinns/data/interim/temp/lorenz96
scp local_path/to/yourfiles kerberos@txe1-login.mit.edu:/home/gridsan/groups/EarthIntelligence/datasets/floods/raw/
# Access plots via https://txe1-portal.mit.edu/jupyter/jupyter_notebook.php
# Or on eofe7 via launching jupyter on https://engaging-ood.mit.edu/ --> Interactive Apps
# Check status via LLstat # sinfo
# Run SLURM scheduler
sbatch train.sh
tail -f runs/train.sh.log 
# Monitor memory usage
scontrol show node node761 # Displays allocated memory
du -sh folder # Displays size of folder
```
## FNO
```
conda create --name pce-pinns-fno --clone pce-pinns
conda activate pce-pinns-fno
conda install pytorch torchvision torchaudio cpuonly -c pytorch # Update to torch>=1.9
python pce_pinns/models/fno_train.py+

python main_helmholtz.py --seed 1 --n_samples 128 --load_data_path "data/raw/temp/helmholtz" --no_wandb --eval_model_digest a43eee0178
```

## MNO
```
# Create Runtime plot
python main_icml_results.py --create_runtime_plot
```

# References 
```
@article{lutjens2022mno,
    title = {Multiscale Neural Operators: Learning Fast and Grid-independent PDE Solvers},
    authors = {Bj{\"o}rn L{\"u}tjens and Catherine H. Crawford and Campbell Watson and Chris Hill and Dava Newman},
    journal = {arxiv},
    year = 2022,
    url = {},
}
@article{lutjens2021spectralpinns,
    title = {Spectral PINNs: Fast Uncertainty Propagation with Physics-Informed Neural Networks},
    authors = {Bj{\"o}rn L{\"u}tjens and Catherine H. Crawford and Mark Veillette and Dava Newman},
    journal = {Conference on Neural Information Processing Systems Workshop on the Symbiosis of Deep Learning and Differential Equations (NeurIPS DLDE)},
    year = 2021,
    url = {https://openreview.net/forum?id=218sl_mPChc}
}
```

# Archive

# Run PCE-PINNs
```
# Train NN to learn polynomial coefficients of deg. 3, fitting a gaussian process prior of diffusion, k
python main_stochDiffEq.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 1000 --load_data_path pce_1k.pickle --est_param_nn k
# Fit NN to observed diffusion, k  
python main_stochDiffEq.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 8 --est_param_nn k_true # -batch_size 151 n_epochs 100
python main_stochDiffEq.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 1000 --est_param_nn k_true #--batch_size 151 n_epochs 5
# Fit NN to eigvecs of GP k
python main_stochDiffEq.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 8 --est_param_nn k_eigvecs --load_data_path pce_1k.pickle
# Test NN to eigvecs of GP k
python main_stochDiffEq.py --rand_flux_bc --pce --nn_pce --pce_dim 3 --n_samples 8 --est_param_nn k_eigvecs --load_data_path pce_1k.pickle
# Local advection-diffusion equation
python main_localAdvDiffEq.py --nn_pce --pce_dim 3 --est_param_nn "u" --n_samples 10000 
```
