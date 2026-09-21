CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero_low_mem_finetune \
    --exp-name pi05_libero_lora \
    --batch-size 64 \
    --num-workers 8 \
    --num-train-steps 30000 \
    --log-interval 20 \
    --save-interval 3000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_toolbox_low_mem_finetune \
    --exp-name pi05_toolbox_lora \
    --batch-size 64 \
    --num-workers 8 \
    --num-train-steps 30000 \
    --log-interval 20 \
    --save-interval 5000

CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python scripts/benchmark_pi05_latency.py \
    --gpu 0 \
    --batch-size 1 \
    --denoise-steps 10 \
    --warmup 10 \
    --repeats 100
