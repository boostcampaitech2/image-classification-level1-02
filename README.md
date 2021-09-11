# Level1 P-stage Image classification

### 💡 **Team: boos2**

## Project Overview

- Predict **mask / gender / age**
- Input : 384 x 512 Image
- Output : 0~17 classes 
    - Mask : Wear, Incorrect, Not Wear
    - Gender : Male, Female
    - Age : <30, >=30 and <60, >=60

## Archive contents

```
image-classification-level1-02/
├── input/
│   └── train/
|		├── images/
|		└── train.csv
│   └── eval/
|		├── images/
|		└── info.csv
├── Dataset.py
├── Model.py
├── Test.py
├── functions.py
└── main.py
```

- ```input/data/train``` : train dataset images
- ```input/data/test``` : used in evaluation
- ```input/``` : download from [https://stages.ai/](https://stages.ai/)

### Requirements

- Ubuntu 18.04.5
- Python 3.8.5
- pytorch 1.7.1
- torchvision 0.8.2

Install packages :  `pip install -r requirements.txt` 

#### Hardware

- CPU: 8 x Intel(R) Xeon(R) Gold 5220 CPU
- GPU: V100
- RAM: 88GB

## Train Model & Test & Save Result

```bash
sh run.sh
```
### run.sh
```bash
python main.py \
--PATH '../input/data' \
--BATCH_SIZE 128 \
--SAVE true \
--SAVE_PATH 'saved/' \
--EPOCH 20
```
- ```--PATH```: ```train/```, ```eval/``` parent directory
- ```--BATCH_SIZE```: Batch size (default=128)
- ```--SAVE```: If want to save weights while training else remove this line
- ```--SAVE_PATH```: Saving weights directory
- ```--EPOCH```: Train epoch (default=20)

### Trained Model

- Pretrained ResNet18 model
- No data augmentation
- Hyperparameter
    - Learning rate : 1e-5
    - Optimizer : Adam
    - Batch size : 128

## Contributors

| **Name** @github                                             | **Project link**                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **안명철** [@abbymark](https://github.com/abbymark)          | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/amc_T2126) |
| **김민진** [@kkmjkim](https://github.com/kkmjkim)            | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/minjin) |
| **박상현** [@hyun06000](https://github.com/hyun06000)        | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/Sang-hyun) |
| **심세령** [@seryoungshim17](https://github.com/seryoungshim17) | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/seryoung) |
| **이유진** [@YoojLee](https://github.com/YoojLee)            | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/yujin) |
| **정두해** [@Doohae](https://github.com/Doohae)              | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/doohae) |
| **홍지연** [@hongjourney](https://github.com/hongjourney)    | [image classification](https://github.com/boostcampaitech2/image-classification-level1-02/tree/jiyeon) |
