############## 1) create environment with torch
ENV_NAME='detection'
conda env remove -n $ENV_NAME -y
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c conda-forge tensorboard -y
conda install mkl=2024.0.0 -y # needs to downgraded because of some bug in other version

############## 2) get further required packages
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

############## 3) initialize submodules to the commits specified in this repo
############## mmdetection commit hash: cfd5d3a985b0249de009b67d04f37263e11cdf3d
############## mmyolo commit hash: 8c4d9dc503dc8e327bec8147e8dc97124052f693
git submodule update --init --recursive

####### optionally update submodules if newer versions of mmdetection or mmyolo are available with new detection models (has not been tested and might require debugging)
# cd mmdetection
# git fetch
# git checkout main
# git pull
# cd ..

# cd mmyolo
# git fetch
# git checkout main
# git pull
# cd ..

############## 4) build mmdetection, mmyolo and detection_utils
cd mmdetection
pip install -v -e .
cd ..

cd mmyolo
pip install -r requirements/albu.txt
pip install albumentations==1.3.1 # this downgrade is necessary to avoid conflicts with mmyolo
mim install -v -e .
cd ..

pip install -v -e .

# other conda packages
conda install jupyter -y
conda deactivate