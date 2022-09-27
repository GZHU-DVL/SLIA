# SLIA
Code for [Decision-Based Black-box Attack Specific to Large-Size Images] by Dan Wang and Yuan-Gen Wang.

## Dependencies
The code for LHS-BA runs with Python and requires Tensorflow of version 1.2.1 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `keras`
- `pywt`

## Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run SLIA on a ResNet-50 pre-trained on ImageNet. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/GZHU-DVL/SLIA
cd SLIA
############################################### 
# Carry out L2 based untargeted attack on 76 iterations.
python SLIA.py --constraint l2 --attack_type untargeted --num_iterations 76
# Carry out L2 based targeted attack on 76 iterations.
python SLIA.py --constraint l2 --attack_type targeted --num_iterations 76

# Results are stored in imagenet+resnet50/DWT/. 
```

See `SLIA.py` for details. 
## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/xxxxxxxxxxxxxxxxxxx):
```
@article{wang2022slia,
  title={Decision-Based Black-box Attack Specific to Large-Size Images},
  author={Dan Wang and Yuan-Gen Wang.},
  journal={Asian Conference on Computer Vision},
  year={2022}
}
```
