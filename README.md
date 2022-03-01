# SwinOCSR



# Install

Please see INSTALL.md

## Datasets

### Image

The our **datasets** are available in SMILES format to generate images

Download form "dataset"

User GenerateBatchimage to generate the images  
User Binary_images to generate binary images  

Saving binary images to  "Data/500wan/500wanBinarizationPNG/"

### Label

download form https://www.kaggle.com/gogogogo11/datasetlabel

The directory Data should look like:

```
Data
├── 500wan
│   ├── 500wanBinarizationPNG
│   ├── 500wan_shuffle_DeepSMILES_test_category_0.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_category_1.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_category_2.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_category_3.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_length_0.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_length_1.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_length_2.pkl
│   ├── 500wan_shuffle_DeepSMILES_test_length_3.pkl
│   ├── 500wan_shuffle_DeepSMILES_test.pkl
│   ├── 500wan_shuffle_DeepSMILES_train.pkl
│   ├── 500wan_shuffle_DeepSMILES_val.pkl
│   ├── 500wan_shuffle_DeepSMILES_word_map
```

## Trained models

| Name                         | Accuracy | Tanimoto | Model                                    |
| ---------------------------- | -------- | -------- | ---------------------------------------- |
| Swin Transformer(ce loss)    | 0.9736   | 0.9965   | https://www.kaggle.com/gogogogo11/moedel |
| ResNet-50                    | 0.8917   | 0.9879   | https://www.kaggle.com/gogogogo11/moedel |
| EfficientNet-B3              | 0.8670   | 0.9846   | https://www.kaggle.com/gogogogo11/moedel |
| Swin Transformer(focal loss) | 0.9858   | 0.9977   | https://www.kaggle.com/gogogogo11/moedel |





## Train model

```
this is a example for Swin-transformer-celoss
cd model/Swin-transformer-celoss
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc_per_node 1 --master_port 29500 main.py   --resume <checkpoint-file> 


```



## Eval model

```
this is a example for Swin-transformer-celoss
cd model/Swin-transformer-celoss
CUDA_VISIBLE_DEVICES="1" python -m torch.distributed.launch --nproc_per_node 1 --master_port 29500 main.py --eval  --resume <checkpoint-file> --test_dir <label-file>


```
