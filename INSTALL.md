### Set up the python environment



```
conda create -n medical python=3.7
conda activate medical

conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch 
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
export CUDA_HOME="/usr/local/cuda-11.0"
python setup.py install --cuda_ext --cpp_ext

pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
export CUDA_HOME="/usr/local/cuda-11.0"
python setup.py install --cuda_ext --cpp_ext
```

### Compile cuda extensions under `lib/csrc`

```
ROOT=/path/to/snake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-11.0"
cd DCNv2
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```

### Set up datasets annotation
python convert_sbd.py 
#make sure to set accurate address in lib/datasets/dataset_catalog.py ('ann_file')
