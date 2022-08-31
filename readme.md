# AMOS2022
Code for the our paper: "[AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical Image Segmentation](https://arxiv.org/abs/2206.08023)"

## Installation

```bash
git clone ***/amos22
cd amos22/
pip install -e .
pip install -r requirements.txt
```

## Dataset
### Prepare your AMOS22 dataset -> ./DATASET/AMOS22
```bash
source setup_dataset.sh
```

    ./DATASET/
    ├── AMOS22/
        ├── imagesTr/
        ├── imagesTs/
        ├── labelsTs/
        ├── labelsTs/
        ├── task1_dataset.json
        ├── task2_dataset.json
    ├── AMOS22_raw/
        ├── AMOS22_raw_data/
            ├── Task01_AMOS_CT/
                ├── imagesTr/
                ├── labelsTr/
                ├── dataset.json
            ├── Task02_AMOS_MR/
                ├── imagesTr/
                ├── labelsTr/
                ├── dataset.json
            ├── Task03_AMOS_CT+MR/
                ├── imagesTr/
                ├── labelsTr/
                ├── dataset.json
        ├── AMOS22_cropped_data/
    ├── AMOS22_preprocessed/

### Setup environment variables
```bash
export nnUNet_raw_data_base="./DATASET/AMOS22_raw"
export nnUNet_preprocessed="./DATASET/AMOS22_preprocessed"
export RESULTS_FOLDER="./DATASET/AMOS_trained_models"
```

### Data preprocessing

```bash
nnUNet_convert_decathlon_task -i ./DATASET/AMOS22_raw/AMOS22_raw_data/Task01_AMOS_CT
nnUNet_plan_and_preprocess -t 1
```

```bash
nnUNet_convert_decathlon_task -i ./DATASET/AMOS22_raw/AMOS22_raw_data/Task02_AMOS_MR
nnUNet_plan_and_preprocess -t 2
```

```bash
nnUNet_convert_decathlon_task -i ./DATASET/AMOS22_raw/AMOS22_raw_data/Task03_AMOS_CT+MR
nnUNet_plan_and_preprocess -t 3
```

    ./DATASET/
    ├── AMOS22/
    ├── AMOS22_raw/
        ├── AMOS22_raw_data/
            ├── Task01_AMOS_CT/
            ├── Task02_AMOS_MR/
            ├── Task03_AMOS_CT+MR/
            ├── Task001_AMOS_CT/
            ├── Task002_AMOS_MR/
            ├── Task003_AMOS_CT+MR/
        ├── AMOS22_cropped_data/
            ├── Task001_AMOS_CT/
            ├── Task002_AMOS_MR/
            ├── Task003_AMOS_CT+MR/
    ├── AMOS22_preprocessed/
        ├── Task001_AMOS_CT/
        ├── Task002_AMOS_MR/
        ├── Task003_AMOS_CT+MR/

## Training
Training with different **MODELS:** nnunet, nnformer, swin-unetr, unetr, vnet, cotr 
and different **MODALITIES:** ct, mr, ct+mr, **TASK:** 1 for ct, 2 for mr, and 3 for ct+mr
```bash
nnUNet_train 3d_fullres nnUNetTrainer $TASK preset $MODELS $MODALITIES
```
## Inference 

```bash
nnUNet_predict -i ./test_data -o ./output_dir -t $TASK -tr nnUNetTrainer -m 3d_fullres -f preset -a $MODEL --modality $MODALITIES -chk model_best
```










