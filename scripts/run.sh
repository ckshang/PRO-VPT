model_root=<MODEL_ROOT>
data_path=<DATA_PATH>
output_dir=<OUTPUT_DIR>
for seed in "18" "25" "42"; do
    python train.py \
        --config-file configs/base-prompt.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.ADAPTIVE "True" \
        MODEL.PROMPT.PPO "True" \
        MODEL.PROMPT.GATE_GRADS_IDENT "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        MODEL.PROMPT.NUM_TOKENS "20" \
        MODEL.PROMPT.REWARD_WARMUP_EPOCH "1" \
        MODEL.PROMPT.PPO_K_EPOCH "60" \
        SOLVER.SCHEDULER "None" \
        DATA.FEATURE "sup_vitb16" \
        DATA.NAME 'vtab-cifar(num_classes=100)' \
        DATA.NUMBER_CLASSES "100" \
        SOLVER.OPTIMIZER "sgd" \
        SOLVER.BASE_LR "2.5" \
        SOLVER.WEIGHT_DECAY "0.0001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}"
done