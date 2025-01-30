source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

RESOLUTION=512
BATCH_SIZE=16
GRAD_ACC_STEPS=4
LR=1e-5
WARMUP_STEPS=0
MAX_TRAIN_STEPS=1000
VALIDATION_EPOCHS=10
TRAINING_SETTING="OOD"
TRAIN_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/test.csv"

MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="OUTPUT"

CUDA_VISIBLE_DEVICES=2

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --use_ema \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --resolution=$RESOLUTION \
  --center_crop --random_flip \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACC_STEPS \
  --gradient_checkpointing \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --learning_rate=$LR \
  --training_setting=$TRAINING_SETTING \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=$WARMUP_STEPS \
  --output_dir=$OUTPUT_DIR \
  --enable_xformers_memory_efficient_attention
