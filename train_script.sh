#!/bin/bash

while getopts 'm:c:i:o:s:t:k:' OPT; do
    case $OPT in
        m) method=$OPTARG;;
        c) cuda=$OPTARG;;
        i) modelresultpath=$OPTARG;;
        o) outputpath=$OPTARG;;
        s) sampleprop=$OPTARG;;
        t) seedtype=$OPTARG;;
        k) ckptpath=$OPTARG;;
    esac
done

echo $method
echo $cuda
echo $modelresultpath
echo $outputpath
echo $sampleprop
echo $seedtype

# PYTHONPATH='.' bash train_script.sh -c 0 -m sup_unet -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 1 -m sup_unetplusplus -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 0 -m ours -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 1 -m dhc -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 2 -m mcf -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 2 -m cps -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 3 -m cld -t 43 -s 0.1
# PYTHONPATH='.' bash train_script.sh -c 3 -m uamt -t 43 -s 0.1

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=${cuda} python trainer/train.py experiment=train_${method}.yaml seed=${seedtype} data_module.sample_prop=${sampleprop}
