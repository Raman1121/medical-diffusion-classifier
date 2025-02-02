source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier/

TRAIN_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/test.csv"
DATASET='fundus'
NUM_CLASSES=2

BATCH_SIZE=32
EPOCHS=60
RESOLUTION=512

OUTPUT_DIR="OUTPUT_CCND"

CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROC=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nnodes=1 --nproc_per_node=$NUM_PROC -m DiT.train \
                                            --model DiT-XL/2 \
                                            --train_csv=$TRAIN_CSV \
                                            --test_csv=$TEST_CSV \
                                            --results-dir $OUTPUT_DIR \
                                            --dataset=$DATASET \
                                            --image-size $RESOLUTION \
                                            --num-classes=$NUM_CLASSES \
                                            --epochs=$EPOCHS \
                                            --ckpt-every 500 \
                                            --global-batch-size=$BATCH_SIZE \



