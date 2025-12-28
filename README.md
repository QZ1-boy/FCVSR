# FCVSR

The code of the paper "A Frequency-aware Method for Compressed Video Super-Resolution".

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## Environment
```python
conda create -n FCVSR python=3.7 -y && conda activate FCVSR

git clone --depth=1 https://github.com/QZ1-boy/FCVSR && cd QZ1-boy/FCVSR/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## Dataset Download

Our work was evlauated in three training datasets, i.e., [CVCP](https://auguste7.github.io/CVCP-database/), [REDS](https://seungjunnah.github.io/Datasets/reds.html), and [Vimeo-90K](https://github.com/anchen1011/toflow), and three testing datasets, i.e., [CVCP10](https://auguste7.github.io/CVCP-database/), [REDS4](https://seungjunnah.github.io/Datasets/reds.html), and [Vid4](https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA).
The uncompressed datasets (Ground-Truth) can be downloaded from thier official links.
Moreover, we provide two datasets (including training and testing data) from our Baiduyun [REDS](https://pan.baidu.com/s/18VO1G63zn1081mwZbV29Tw) [code:t9k8] and [Vimeo90k](https://pan.baidu.com/s/1n7ThUrJtNcLOGkPFcW4ohQ) [code:mdbe]. CVCP dataset (including training and testing data)  can be downloaded from [CVCP](https://auguste7.github.io/CVCP-database/).


# Train
For CVCP dataset:
```python
cd CVSR_train
python train_LD_37.py
```
For REDS or Vimeo90K dataset:
```python
cd mmedit_train
python train_LD_37.py
```
# Test
For CVCP10 dataset:
```python
cd CVSR_train
python test_LD_37.py 
```
For REDS4 or Vid4 dataset:
```python
cd mmedit_train
python test_LD_37.py 
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2025fcvsr,
  title={FCVSR: A Frequency-aware Method for Compressed Video Super-Resolution},
  author={Zhu, Qiang and Zhang, Fan and Chen, Feiyu and Zhu, Shuyuan and Bull, David and Zeng, Bing},
  journal={arXiv preprint arXiv:2502.06431},
  year={2025}
}
```
