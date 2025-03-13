model=$1
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu \
    python evaluation/eval.py \
    --benchmark jec-qa-1-multi-choice \
    --model_path $model \
    --max_tokens 4096 \
    --tensor_parallel_size 1 \
    --system_message r1-lawyer \
    --force_generate