
# Environment Setup

This project uses **CUDA 11.8** and recommends **venv** for environment management. 


**a. Create a virtual environment**
```
cd SparseWorld
python3.9 -m venv env
source env/bin/activate
```

**Conda** is also completely feasible, with only the difference being in the creation of the environment.

**b. Download and compile some core packages**
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -v -e .
pip install -r requirements.txt
cd mmdet3d/models/sparsedetectors/csrc
python setup.py build_ext --inplace
```