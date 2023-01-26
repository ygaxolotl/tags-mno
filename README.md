# multiscale-neural-op: Learning Fast and Grid-independent PDE Solvers
This repository implements multiscale neural operator. A fast and grid-independent surrogate model of multiscale PDEs by combining neural operators with coarse-grained simulations. 

The repository contains the code of the following papers:
- Multiscale Neural Operator: Learning Fast and Grid-independent PDE Solvers
- Spectral PINNs: Fast Uncertainty Propagation with Physics-Informed Neural Networks

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

# Run Multiscale Neural Operator for Lorenz96
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
```

# Run on server
```
export WANDB_MODE=offline
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