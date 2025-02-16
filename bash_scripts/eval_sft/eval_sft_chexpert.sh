source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TEST_CSV="/raid/s2198939/Chexpert/test_frontal_with_prompts.csv"
DATASET='chexpert'
MODEL="resnet50" # vit_base_patch16_224, resnet50
NUM_CLASSES=2
RESULTS_DIR="/raid/s2198939/medical-diffusion-classifier/RESULTS/SFT_RESULTS"
RESULTS_FILE="sft_results.csv"
CKPT="/raid/s2198939/medical-diffusion-classifier/OUTPUT_SFT/chexpert/20250216-145146-resnet50-224/model_best.pth.tar"
SENSITIVE_ATTRIBUTE='Sex'

CUDA_VISIBLE_DEVICES=0 python test_sft.py \
                                --test_csv=$TEST_CSV \
                                --dataset=$DATASET \
                                --model=$MODEL \
                                --num-classes=$NUM_CLASSES \
                                --batch-size=64 \
                                --results-file=$RESULTS_FILE \
                                --results-dir=$RESULTS_DIR \
                                --checkpoint=$CKPT \
                                --sensitive_attribute=$SENSITIVE_ATTRIBUTE \
                                --no-prefetcher True
                                # --max_samples=$MAX_SAMPLES 