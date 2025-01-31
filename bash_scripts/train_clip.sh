source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TRAIN_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/test.csv"
OUTPUT_DIR="OUTPUT_CLIP"
LR=5e-6
WD=0.1
EPOCHS=30
BATCH_SIZE=256
MODEL="RN50"

CUDA_VISIBLE_DEVICES=6,7

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_clip.py \
    --batch-size $BATCH_SIZE \
    --precision amp \
    --workers 8 \
    --report-to tensorboard \
    --save-frequency 4 \
    --logs $OUTPUT_DIR \
    --dataset-type csv \
    --csv-separator="," \
    --train-data $TRAIN_CSV \
    --val-data $TEST_CSV \
    --csv-img-key path \
    --csv-caption-key Text \
    --warmup 1000 \
    --lr=$LR \
    --wd=$WD \
    --epochs=$EPOCHS \
    --model=$MODEL \
    --pretrained='openai' \
