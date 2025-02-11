source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TEST_CSV="/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSVs/FINAL_TEST.xlsx"
DATASET='mimic'
MODEL="vit_base_patch16_224" # vit_base_patch16_224, resnet50
NUM_CLASSES=2
RESULTS_DIR="/raid/s2198939/medical-diffusion-classifier/RESULTS/SFT_RESULTS"
RESULTS_FILE="sft_results.csv"
CKPT="/raid/s2198939/medical-diffusion-classifier/OUTPUT_SFT/mimic/20250210-183015-vit_base_patch16_224-224/model_best.pth.tar"
MAX_SAMPLES=300

CUDA_VISIBLE_DEVICES=0 python test_sft.py \
                                --test_csv=$TEST_CSV \
                                --dataset=$DATASET \
                                --model=$MODEL \
                                --num-classes=$NUM_CLASSES \
                                --batch-size=64 \
                                --results-file=$RESULTS_FILE \
                                --results-dir=$RESULTS_DIR \
                                --checkpoint=$CKPT \
                                --max_samples=$MAX_SAMPLES 