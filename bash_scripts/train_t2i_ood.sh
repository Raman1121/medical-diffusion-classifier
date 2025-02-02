source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

RESOLUTION=512
BATCH_SIZE=20
GRAD_ACC_STEPS=1
LR=1e-5
WARMUP_STEPS=0
# MAX_TRAIN_STEPS=1000
NUM_TRAIN_EPOCHS=30
VALIDATION_EPOCHS=5
TRAINING_SETTING="OOD"
DATASET="fundus"
TRAIN_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/test.csv"

MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="TEST"

CUDA_VISIBLE_DEVICES=4,5,6,7

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --multi_gpu --main_process_port 12345 train_text_to_image.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --dataset_name=$DATASET \
  --resolution=$RESOLUTION \
  --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --learning_rate=$LR \
  --training_setting=$TRAINING_SETTING \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=$WARMUP_STEPS \
  --output_dir=$OUTPUT_DIR \
  --enable_xformers_memory_efficient_attention \
  --checkpoints_total_limit=4
