source /raid/s2198939/miniconda3/bin/activate demm

cd /raid/s2198939/medical-diffusion-classifier

TRAIN_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/train.csv"
TEST_CSV="/raid/s2198939/Fundus_Images/OOD-Splits/test.csv"

CUDA_VISIBLE_DEVICES=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_clip.py \
    --output_dir ./clip-roberta-finetuned \
    --model_name_or_path openai/clip-vit-base-patch32 \
    --image_column path \
    --caption_column Text \
    --train_csv=$TRAIN_CSV \
    --test_csv=$TEST_CSV \
    --do_train  \
    --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" \
    --warmup_steps="0" \
    --weight_decay 0.1 \
    --overwrite_output_dir \
