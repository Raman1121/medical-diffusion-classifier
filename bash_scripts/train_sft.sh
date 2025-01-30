source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TRAIN_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/test.csv"

CUDA_VISIBLE_DEVICES=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_sft.py