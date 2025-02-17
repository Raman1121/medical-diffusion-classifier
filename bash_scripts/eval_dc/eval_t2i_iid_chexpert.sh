source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

pretrained_model_name_or_path="/raid/s2198939/medical-diffusion-classifier/OUTPUT_Chexpert_256/IID/256/SD-V2_IID_256"

TRAIN_CSV="/raid/s2198939/Chexpert/train_frontal_with_prompts.csv"
TEST_CSV="/raid/s2198939/Chexpert/test_frontal_with_prompts.csv"
DATASET="chexpert"
TRAINING_SETTING="IID"

NUM_TRIALS_PER_TIMESTEP=1
BATCH_SIZE=32
LOSS="l1"
PROMPT_KEY="Simple_prompt"
LABEL_KEY="Unhealthy"
IMG_SIZE=256
SENSITIVE_ATTRIBUTE="Sex"
# MAX_SAMPLES=300

# NUM_TSTEPS_TO_SAMPLE_PER_IMAGE=6          # This can be increased for a more accurate estimate
# NUM_PROMPTS_TO_KEEP_AFTER_EACH_ROUND=1  # We want to keep only 1 class in the final round since binary classification

CUDA_VISIBLE_DEVICES=4
NUM_WORKERS=12
WORKER_IDX=11

# EXTRA="original_prompt"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORKER_IDX: $WORKER_IDX"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval_prob_adaptive.py --dataset $DATASET --split test \
  --pretrained_model_name_or_path $pretrained_model_name_or_path \
  --version $pretrained_model_name_or_path \
  --dataset $DATASET --extra $TRAINING_SETTING \
  --loss $LOSS \
  --img_size $IMG_SIZE --batch_size $BATCH_SIZE \
  --n_trials $NUM_TRIALS_PER_TIMESTEP \
  --prompt_path $TEST_CSV \
  --prompt_key $PROMPT_KEY \
  --label_key $LABEL_KEY \
  --to_keep 1 \
  --n_samples 4 \
  --n_workers $NUM_WORKERS \
  --worker_idx $WORKER_IDX \
  --sensitive_attribute $SENSITIVE_ATTRIBUTE
  # --max_samples $MAX_SAMPLES \
  # --extra $EXTRA