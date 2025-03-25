# FCVSR

The code of the paper "A frequency-aware Method for Compressed Video Super-Resolution".

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
Training Datasets:

[CVCP dataset] [CVCP] (https://auguste7.github.io/CVCP-database/)

[REDS dataset] [REDS] (https://seungjunnah.github.io/Datasets/reds.html)

[Vimeo-90K dataset] [Vimeo-90K] (https://github.com/anchen1011/toflow)

Testing Datasets:

[CVCP10 dataset] [CVCP10] (https://auguste7.github.io/CVCP-database/)

[REDS4 dataset] [REDS4] (https://seungjunnah.github.io/Datasets/reds.html)

[Vid4 dataset] [Vid4] (https://drive.google.com/drive/folders/1An6hF1oYkeWxfOBxxKm073mvgIFrBNDA)


# Train
```python
python train_LD_37.py
```
# Test
```python
python test_LD_37.py 
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2024deep,
  title={Deep compressed video super-resolution with guidance of coding priors},
  author={Zhu, Qiang and Chen, Feiyu and Liu, Yu and Zhu, Shuyuan and Zeng, Bing},
  journal={IEEE Transactions on Broadcasting},
  year={2024},
  publisher={IEEE}
}
```
