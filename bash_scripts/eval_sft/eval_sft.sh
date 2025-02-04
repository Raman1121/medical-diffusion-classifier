source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TEST_CSV="/raid/s2198939/Fundus_Images/In-Distribution-Splits/test_balanced.csv"
DATASET='fundus'
MODEL="resnet50"
NUM_CLASSES=2
RESULTS_DIR="/raid/s2198939/medical-diffusion-classifier/RESULTS/SFT_RESULTS"
RESULTS_FILE="sft_results.csv"
CKPT="/raid/s2198939/medical-diffusion-classifier/OUTPUT_SFT/20250131-124821-resnet50-224/model_best.pth.tar"

CUDA_VISIBLE_DEVICES=0 python test_sft.py \
                                --test_csv=$TEST_CSV \
                                --dataset=$DATASET \
                                --model=$MODEL \
                                --num-classes=$NUM_CLASSES \
                                --batch-size=64 \
                                --results-file=$RESULTS_FILE \
                                --results-dir=$RESULTS_DIR \
                                --checkpoint=$CKPT