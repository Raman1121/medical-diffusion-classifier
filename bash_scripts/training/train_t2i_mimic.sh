source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

RESOLUTION=512
BATCH_SIZE=64
GRAD_ACC_STEPS=1
LR=5e-6
WARMUP_STEPS=500
NUM_TRAIN_EPOCHS=15
VALIDATION_EPOCHS=5
# MAX_TRAIN_STEPS=1000
TRAINING_SETTING="IID"
DATASET="mimic"
TRAIN_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TRAIN.xlsx"
TEST_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TEST.xlsx"
IMG_DIR="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
CAPTION_COL="simple_prompt"

MODEL_NAME="radedit"
OUTPUT_DIR="OUTPUT_MIMIC_simple_prompt"

CUDA_VISIBLE_DEVICES=0,1,2,6

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --multi_gpu --main_process_port 12345 train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision="fp16" \
  --train_csv=$TRAIN_CSV \
  --test_csv=$TEST_CSV \
  --dataset_name=$DATASET \
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
  --checkpoints_total_limit=4 --checkpointing_steps=1000 \

  # --resume_from_checkpoint=$RESUME_FROM_CKPT \
  # --use_ema \
  # 