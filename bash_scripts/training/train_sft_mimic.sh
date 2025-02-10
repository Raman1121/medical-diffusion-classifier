source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TRAIN_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TRAIN.xlsx"
TEST_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TEST.xlsx"
DATASET='mimic'
MODEL="resnet50" # resnet50, vit_base_patch16_224
NUM_CLASSES=2

LR=1e-4
BATCH_SIZE=256
EPOCHS=30
OUTPUT_DIR="OUTPUT_SFT"

CUDA_VISIBLE_DEVICES=0,1,2
NUM_PROC=3

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$NUM_PROC --master_port=12345 train_sft.py \
                                            --train_csv=$TRAIN_CSV \
                                            --test_csv=$TEST_CSV \
                                            --dataset=$DATASET \
                                            --model=$MODEL \
                                            --num-classes=$NUM_CLASSES \
                                            --pretrained \
                                            --hflip=0.5 \
                                            --batch-size=$BATCH_SIZE \
                                            --grad-accum-steps=1 \
                                            --epochs=$EPOCHS \
                                            --output=$OUTPUT_DIR \
                                            --lr=$LR \                                    