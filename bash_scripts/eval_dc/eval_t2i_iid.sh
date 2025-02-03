source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

pretrained_model_name_or_path="/raid/s2198939/medical-diffusion-classifier/OUTPUT/IID/512/SD-V1-4_IID_512"

TRAIN_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/test.csv"
DATASET="fundus"

NUM_TRIALS_PER_TIMESTEP=1
BATCH_SIZE=32
LOSS="l1"
PROMPT_KEY="Text"
LABEL_KEY="Unhealthy"
IMG_SIZE=512
MAX_SAMPLES=10

NUM_TSTEPS_TO_SAMPLE_PER_IMAGE=8          # This can be increased for a more accurate estimate
NUM_PROMPTS_TO_KEEP_AFTER_EACH_ROUND=1  # We want to keep only 1 class in the final round since binary classification

CUDA_VISIBLE_DEVICES=6

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python eval_prob_adaptive.py --dataset fundus --split test \
  --pretrained_model_name_or_path $pretrained_model_name_or_path \
  --dataset $DATASET \
  --loss $LOSS \
  --img_size $IMG_SIZE --batch_size $BATCH_SIZE \
  --n_trials $NUM_TRIALS_PER_TIMESTEP \
  --prompt_path $TEST_CSV \
  --prompt_key $PROMPT_KEY \
  --label_key $LABEL_KEY \
  --max_samples $MAX_SAMPLES \
  --to_keep $NUM_PROMPTS_TO_KEEP_AFTER_EACH_ROUND \
  --n_samples $NUM_TSTEPS_TO_SAMPLE_PER_IMAGE