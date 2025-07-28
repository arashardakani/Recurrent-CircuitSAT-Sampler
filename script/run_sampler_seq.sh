OPTIMIZER=$1
DEVICES=6
CUDA_VISIBLE_DEVICES=$DEVICES python ./src/pytorch/run.py \
    -d "../data/ISCAS/s27.bench" -l\
    --optimizer "mse"\
    --lr 15e0 \
    --problem_type "BLIF" \
    --circuit_type "seq" \
    --num_steps 1 \
    --batch_size 1000000 \
    --num_clock_cycles 50 \
    --start_point 0 \
    --wandb_entity "ucb-hcrl" \
    --wandb_project "gdsampler" \
    --wandb_group "debug" \
    --wandb_tags "seed=0"
