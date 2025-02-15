source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

RESOLUTION=512
BATCH_SIZE=32
GRAD_ACC_STEPS=1
LR=5e-6
WARMUP_STEPS=0
NUM_TRAIN_EPOCHS=50
VALIDATION_EPOCHS=10
# MAX_TRAIN_STEPS=64
TRAINING_SETTING="IID"
DATASET="chexpert"
TRAIN_CSV="/raid/s2198939/Chexpert/train_frontal_with_prompts.csv"
TEST_CSV="/raid/s2198939/Chexpert/test_frontal_with_prompts.csv"
IMG_COL="Path"
CAPTION_COL="Simple_prompt"

MODEL_NAME="stabilityai/stable-diffusion-2"
OUTPUT_DIR="OUTPUT_Chexpert_512"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --multi_gpu --main_process_port 12345 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --dataset_name=$DATASET \
  --image_column=$IMG_COL \
  --caption_column=$CAPTION_COL \
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
  --checkpoints_total_limit=4 \
  --checkpointing_steps=1000 \

  # --resume_from_checkpoint=$RESUME_FROM_CKPT \
  # --use_ema \
  # 