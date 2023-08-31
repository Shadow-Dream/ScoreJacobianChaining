# Score Jacobian Chaining Toy Demo
## Introduction
这个工程是对Score Jacobian Chaining（与DreamFusion相似）的简单理论实现，用作三维物体生成<br>
This is a simple implementation of the algorithm in the Score Jacobian Chaining (The algorithm is also similar as the one of Dreamfusion), which is designed for 3D generation.

## Environment
1. 至少需要安装 Pytorch, OpenCV-Python, Numpy and Tqdm 库<br>
Pytorch, OpenCV-Python, Numpy and Tqdm are required.

2. 需要CUDA环境<br>
CUDA is needed. 

3. 详细的Conda环境在environment.yml中，仅供参考<br>
The detailed conda environment is in the environment.yml file, and for reference only.

4. 数据集可自行设计（请修改Dataset类中的加载脚本），也可以下载简单训练数据<br>
Simple training data can be downloaded in the Netdisk <br>
datasets文件夹放在本目录下即可 <br>
Place the datasets folder in this directory <br>
链接(Link)：https://pan.baidu.com/s/1MjsTYkg5p2sQ1fepaBUfNQ <br>
提取码(Password)：b7p1

5. 运行脚本前请在本目录下新建weights文件夹用以存放网络权重<br>
Before Running the scripts, please create a "weights" folder in this directory

## Scripts
### test_chain
SJC 训练结果测试<br>
Test Script for the result of SJC.
### test_diffusion
DDPM 训练结果测试<br>
Test Script for the result of DDPM.
### test_nerf
NeRF 训练结果测试（与test_chain一样）<br>
Test Script for the result of NeRF. The same as test_chain
### train_chain
SJC方法训练NeRF（请先训练DDPM再训练SJC）<br>
Train Script for SJC, please train DDPM first.
### train_diffusion
训练DDPM<br>
Train Script for DDPM.
### train_nerf
训练NeRF（这是不使用SJC，直接用数据集图像训练NeRF）<br>
Train Script for NeRF. This directly uses dataset images for training, rather than using SJC.
