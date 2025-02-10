source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

pretrained_model_name_or_path="radedit"

TRAIN_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TRAIN.xlsx"
TEST_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TEST.xlsx"
DATASET="mimic"
TRAINING_SETTING="IID"

NUM_TRIALS_PER_TIMESTEP=1
BATCH_SIZE=32
LOSS="l1"
PROMPT_KEY="text"
LABEL_KEY="Unhealthy"
IMG_SIZE=512
MAX_SAMPLES=300

NUM_TSTEPS_TO_SAMPLE_PER_IMAGE=16          # This can be increased for a more accurate estimate
NUM_PROMPTS_TO_KEEP_AFTER_EACH_ROUND=1  # We want to keep only 1 class in the final round since binary classification

CUDA_VISIBLE_DEVICES=7
NUM_WORKERS=10
WORKER_IDX=9

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "WORKER_IDX: $WORKER_IDX"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval_prob_adaptive.py --dataset fundus --split test \
  --pretrained_model_name_or_path $pretrained_model_name_or_path \
  --version $pretrained_model_name_or_path \
  --dataset $DATASET --extra $TRAINING_SETTING \
  --loss $LOSS \
  --img_size $IMG_SIZE --batch_size $BATCH_SIZE \
  --n_trials $NUM_TRIALS_PER_TIMESTEP \
  --prompt_path $TEST_CSV \
  --prompt_key $PROMPT_KEY \
  --label_key $LABEL_KEY \
  --to_keep $NUM_PROMPTS_TO_KEEP_AFTER_EACH_ROUND \
  --n_samples $NUM_TSTEPS_TO_SAMPLE_PER_IMAGE \
  --n_workers $NUM_WORKERS \
  --worker_idx $WORKER_IDX \
  --max_samples $MAX_SAMPLES \