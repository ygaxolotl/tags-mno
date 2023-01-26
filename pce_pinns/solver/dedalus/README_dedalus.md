# Dedalus 

This project contains an ongoing integration of the Dedalus solver to generate target datasets. The integration in pce_pinns.solver.dedalus is not well tested and in progress. 

# Install Dedalus
```
# Intall conda env via pce_pinns.solver.install_conda.sh and https://dedalus-project.readthedocs.io/en/latest/pages/installation.html
conda activate dedalus_instal
conda install -c anaconda cython
wget https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_conda.sh
cp pce_pinns/solver/dedalus/install_conda.sh install_conda.sh
bash install_conda.sh
rm install_conda.sh
# If error is thrown
conda activate dedalus
python3 -m pip install --no-cache --no-build-isolation dedalus
```
Dedalus3
```
conda create --name dedalus3_install
conda activate dedalus3_install
curl https://raw.githubusercontent.com/DedalusProject/dedalus_conda/master/install_dedalus3_conda.sh --output install_dedalus3_conda.sh
# Edit line53 in install_dedalus3_conda.sh "base"-> "dedalus3_install"
bash install_dedalus3_conda.sh
conda deactivate
conda activate dedalus3
python3 -m pip install --no-cache http://github.com/dedalusproject/dedalus/zipball/master/
```
