# Rethinking Individual Fairness in Deepfake Detection
_________________

This repository is the official implementation of our paper "Rethinking Individual Fairness in Deepfake Detection"

## 1. Installation
You can run the following script to configure the necessary environment:

```
cd Individual-Fairness
conda create -n IndividualFairness python=3.9.0
conda activate IndividualFairness
pip install -r requirements.txt
```

## 2. Dataset Preparation

The FF++, Celeb-DF, DFD, DFDC with demographic annotations are from [paper](https://arxiv.org/pdf/2208.05845.pdf). The datasets can be downloaded from their official links: The FF++ dataset can be downloaded from [link](https://github.com/ondyari/FaceForensics/); DFDC from [link](https://ai.meta.com/datasets/dfdc/); DFD from [link](https://research.google/blog/contributing-data-to-deepfake-detection-research/); Celeb-DF from [link](https://github.com/yuezunli/celeb-deepfakeforensics); and AI-Face dataset from [link](https://github.com/Purdue-M2/AI-Face-FairnessBench). The train, val, and test CSVs for each dataset are available through this [link](https://drive.google.com/drive/folders/1YoSsQGO5bMxAtv0H9x-1uBeredCK8VQx?usp=drive_link).

## 3. Load Pretrained Weights
Before running the training code, make sure you load the pre-trained weights. We provide pre-trained weights under [`./training/pretrained`](./training/pretrained). 

To load the checkpoints we used in our paper to the different models, you can find them from this [link](https://drive.google.com/drive/folders/14IozQOpEbecTWCX12R9bGnYYTzMkHUj7?usp=drive_link).

## 4. Train
To run the training code, you should first go to the [`./training/`](./training/) folder, then you can train our model by running [`train.py`](training/train.py):

```
cd training

python train.py 
```

You can adjust the parameters in [`train.py`](training/train.py) to specify the parameters, *e.g.,* batchsize, model, *etc*.

`--test_batchsize`: batch size, default is 32.

`--train_batchsize`: batch size, default is 32.

`--checkpoints`: checkpoint path, default is ''.

`--mode`: method used ['naive', 'ours'], default is 'naive'.

`--model`: detector name ['xception', 'efficientnet', 'resnet''], default is 'xception'.

## 5. Test
* For model testing, we provide a python file to test our model by running `python test.py`. 

	`--test_batchsize`: batch size, default is 32.

	`--train_batchsize`: batch size, default is 32.

	`--checkpoints`: checkpoint path, default is ''.

	`--mode`: method used ['naive', 'ours'], default is 'naive'.

	`--model`: detector name ['xception', 'efficientnet', 'resnet''], default is 'xception'.

* After testing, for metric calculation, we provide the AUC, $L_{\text{ind}}^{\text{naive}}$, and $L_{\text{ind}}^*$.

## Provided Backbones
|                  | File name                               | Paper                                                                                                                                                                                                                                                                                                                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Xception          | [xception.py](./training/networks/xception.py)         | [Xception: Deep learning with depthwise separable convolutions](https://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html) |
| ResNet50          | [resnet50.py](training/networks/resnet50.py)       | [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)                                                                                                                                                                                                                                                                                              |
| EfficientNet-B3      | [efficientnetb3.py](./training/networks/efficientnetb3.py) | [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html)                                                                                                                                           
