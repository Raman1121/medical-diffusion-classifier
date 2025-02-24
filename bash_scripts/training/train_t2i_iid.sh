source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

RESOLUTION=512
BATCH_SIZE=20
GRAD_ACC_STEPS=1
LR=1e-5
WARMUP_STEPS=0
NUM_TRAIN_EPOCHS=30
VALIDATION_EPOCHS=5
# MAX_TRAIN_STEPS=1000
TRAINING_SETTING="IID"
DATASET="fundus"
TRAIN_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/test.csv"
# RESUME_FROM_CKPT="/raid/s2198939/medical-diffusion-classifier/OUTPUT/IID/512/checkpoint-19000"

MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="OUTPUT"

CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --multi_gpu train_text_to_image.py \
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
  --checkpoints_total_limit=4 --checkpointing_steps=1000

  # --resume_from_checkpoint=$RESUME_FROM_CKPT \