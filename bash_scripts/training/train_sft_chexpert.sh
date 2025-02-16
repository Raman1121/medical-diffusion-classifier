source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier


DATASET='chexpert'
MODEL="resnet50" # resnet50, vit_base_patch16_224
NUM_CLASSES=2
TRAIN_CSV="/raid/s2198939/Chexpert/train_frontal_with_prompts.csv"
TEST_CSV="/raid/s2198939/Chexpert/test_frontal_with_prompts.csv"
LR=5e-4
BATCH_SIZE=256
EPOCHS=30
OUTPUT_DIR="OUTPUT_SFT"

CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROC=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=$NUM_PROC --master_port=1234 train_sft.py \
                                            --train_csv=$TRAIN_CSV \
                                            --test_csv=$TEST_CSV \
                                            --dataset=$DATASET \
                                            --model=$MODEL \
                                            --num-classes=$NUM_CLASSES \
                                            --pretrained \
                                            --batch-size=$BATCH_SIZE \
                                            --grad-accum-steps=1 \
                                            --epochs=$EPOCHS \
                                            --output=$OUTPUT_DIR \
                                            --lr=$LR \                                    