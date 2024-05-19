# RiCo: Semi-supervised Segmentation through Rival Networks Collaboration with Saliency Map in Diabetic Retinopathy

This repository is the implementation to review RiCo Network for IDRiD Dataset.

## 1. git clone

## 2. Set the environment.
```
pip install -r requirements.txt
```

## 3. Dataset

## 3-1. Download the IDRiD Dataset in public.

## 3-2. Preprocess the IDRiD Dataset.
```
python RiCo/code/preprocess.py
```

## 3-3. Add prepared dataset path in configure file.
```
configs/data_module/train_ssl.yaml
configs/data_module/train_sup.yaml
```

## 4. Train models.

## 4-1. Train supervised models
c : gpu device id
m : method
t : seed tag
s : sampling proportion
Below commends are for 10% dataset and seed 43. 
```
PYTHONPATH='.' bash train_script.sh -c 0 -m sup_unet -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 1 -m sup_unetplusplus -t 43 -s 0.1
``` 

## 4-2. Train semi-supervised models
```
PYTHONPATH='.' bash train_script.sh -c 0 -m ours -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 1 -m dhc -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 2 -m mcf -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 2 -m cps -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 3 -m cld -t 43 -s 0.1
PYTHONPATH='.' bash train_script.sh -c 3 -m uamt -t 43 -s 0.1
``` 


