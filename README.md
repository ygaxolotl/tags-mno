# multiscale-neural-op: Learning Fast and Grid-independent PDE Solvers
This repository implements multiscale neural operator. A fast and grid-independent surrogate model of multiscale PDEs by combining neural operators with coarse-grained simulations. 

The repository contains the code of the following papers:
- Multiscale Neural Operator: Learning Fast and Grid-independent PDE Solvers

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

# In case the above code is exhibiting problems please try executing the following:
pip install ray # tested with 1.4.1
pip install -U pydantic
git clone https://github.com/<author>/deepxde.git pce_pinns/deepxde # TODO: get rid of this installation requirement by installing deepxde as submodule
python pce_pinns/deepxde/deepxde/backend/set_default_backend.py pytorch                             
```

# Run Multiscale Neural Operator for Lorenz96
```
conda activate pce-pinns
## MNO: Create data, train, and test MNO
# Train MNO. This will also create the raw dataset and save it as interim data. 
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type "mno" 
# (opt:) Start train on existing raw data if process is interrupted. Add flag:
--load_data_path "data/raw/temp/lorenz96/mno/lorenz96_5k_t1000"
# (opt:) Start train on existing interim data if process is interrupted. Add flag:
--load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"
# (opt:) Run evaluation script on MNO model:
python main_lorenz96.py --n_samples 1300 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --model_type "mno" --eval_model_digest e323bd65c0 --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_1k_t1200" --parallel
# (opt:) Run hyper parameter sweep
python main_lorenz96.py --n_samples 5000 --mode_target "no-scale-sep-param-no-mem" --seed 1 --no_plot --est_param_nn "u" --parallel --model_type "mno" --no_test --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_5k"

# Train FCNN baseline: 
python main_lorenz96.py --n_samples 5000 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'fcnn' --overwrite --no_plot
--load_data_path "data/raw/temp/lorenz96/fcnn/lorenz96_5k" 
--eval_model_digest "fcnn_l02u0032e020lr01000n005000k4"
# (opt:) Run evaluation script on FCNN model:
python main_lorenz96.py --n_samples 1300 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'fcnn' --overwrite --no_plot --load_interim_data_path "data/interim/temp/lorenz96/lorenz96_n64_1k_t1200"  --eval_model_digest "fcnn_l02u0032e002lr01000n005000k64"

# Train Climatology baseline: 
python main_lorenz96.py --n_samples 5000 --mode_target "X" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'climatology' --overwrite --no_plot
--load_data_path "data/raw/temp/lorenz96/fcnn/lorenz96_5k_t1000" 

# Train polynomial parametrization baseline:
python main_lorenz96.py --n_samples 5000 --mode_target "param-no-mem" --seed 1 --no_plot --no_wandb --est_param_nn "u" --parallel --model_type 'poly' --overwrite --no_plot 
--load_data_path "data/raw/temp/lorenz96/mno/lorenz96_5k_t1000"

# Create result plots in paper:
python create_paper_results.py --eval_results --eval_mno_digest a6a5acd859 --eval_fcnn_digest fcnn_l02u0032e020lr01000n005000k4 --eval_clim_digest climatology_n5000 --eval_poly_digest poly_n5000
--eval_runtime 
```
# Run Multiscale Neural Operator (MNO) for Quasi-Geostrophic Turbulence
```
# Install numerical solver to generate data:
cd ..
git clone git@github.com:hrkz/torchqg.git # Note: The linked code to generate the training data is not anonymized. But anonymization of the NeurIPS23 paper still holds, because the linked repository is a public repository that is not part of the contribution. The authors of the linked code might or might not be the same authors of the NeurIPS23 paper.
cd torchqg
pip install -e .
# Generate data (on supercomputer)
python main.py # To generate spinup data. 
# Change the main.py to lines:
init_data_path = 'output/spinup_10000_dump'
raw_data_path = '<path/to/spinupdata>'
python main.py # To generated warmed up data.
# Train MNO on quasi-geostrophic turbulence data:
cd ../pce-pinns
# (opt:) Test if training script works on small dataset
python main_qgturb.py --no_wandb --model_type="mno" --load_data_path="../torchqg/output/warmed_100_dump" # Test run 
# Train MNO on raw dataset. This also creates the processed dataset.
python main_qgturb.py --no_wandb --model_type="mno" --load_data_path="../torchqg/output/warmed_20000_dump"
# (opt:) Start training on processeed data. Add flag:
--load_processed_data_path="data/processed/temp/qgturb/fno-pure-ml-sol/n1000_t20"
# (opt:) Evaluate MNO vs. null parametrization baseline. 
python main_qgturb.py --no_wandb --model_type="mno" --eval_model_digest "03a25874ca" --load_processed_data_path="data/processed/temp/qgturb/mno/n5_t20"  # 903804a78e
# Train pure ML FNO baseline and null parametrizatio
python main_qgturb.py --no_wandb --model_type="fno-pure-ml-sol" --mode_target="pure-ml-sol"  --load_processed_data_path="data/processed/temp/qgturb/mno/n5_t20"
# Evaluate MNO vs. pure ML vs. null parametrization
python main_qgturb.py --no_wandb --model_type="fno-pure-ml-sol" --eval_model_digest "9dfe08e0ca" --load_processed_data_path="data/processed/temp/qgturb/mno/n5_t20"
```

# To run on server without internet connection
```
export WANDB_MODE=offline
```

## MNO
```
# Create Runtime plot
python main_paper_results.py --create_runtime_plot
```