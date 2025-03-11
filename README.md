# Directional Label Diffusion Model for Learning from Noisy Labels

## 1. Preparing python environment
Install requirements.<br />
```
conda env create -f environment.yml
conda activate DLD-env
conda env update --file environment.yml --prune
```
The name of the environment is set to **DLD-env** by default. You can modify the first line of the `environment.yml` file to set the new environment's name.

## 2. Pre-trained model & Checkpoints
* ViT pre-trained models are available in the python package at [here](https://github.com/openai/CLIP). Install without dependency: <br />
```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git  --no-dependencies
```

Trained checkpoints for the directional diffusion models will be available soon.

## 3. Generate the Instance-Dependent Noisy (IDN) Labels
The IDN used in our experiments are provided in folder `noise_label_IDN`. The noisy labels are generated following the original [paper](https://arxiv.org/abs/2012.05458).

## 4. Run demo scripts to train the DLD models
### 4.1 CIFAR-10 and CIFAR-100<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_on_CIFAR.py --device cuda:0 --noise_type cifar10-idn-0.1 --nepoch 200 --warmup_epochs 5 --log_name cifar10-idn-0.1.log
```
### 4.2 Animal10N<br />
The dataset should be downloaded according to the instruction here: [Aniaml10N](https://dm.kaist.ac.kr/datasets/animal-10n/)<br />
Default values for input arguments are given in the code. An example command is given:
```
python train_on_Animal10N.py --device cuda:0 --nepoch 200 --warmup_epochs 5 --log_name Animal10N.log
```

### 4.3 WebVision and ILSVRC2012<br />
Download [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/download.html) and the validation set of [ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/) datasets. 
```
python train_on_WebVision.py --gpu_devices 0 1 2 3 4 5 6 7 --nepoch 200 --warmup_epochs 5  --log_name Webvision.log

python test_on_ILSVRC2012.py --gpu_devices 0 1 2 3 4 5 6 7 --log_name ILSVRC2012.log
```

### 4.4 Clothing1M<br />
The dataset should be downloaded according to the instruction here: [Clothing1M](https://github.com/Cysu/noisy_label). Default values for input arguments are given in the code. <br />

```
python train_Clothing1M.py --gpu_devices 0 1 2 3 4 5 6 7 --nepoch 200 --warmup_epochs 5  --log_name Clothing1M.log
```



