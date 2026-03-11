datapath="/path/to/imagenet"
ckpt="/path/to/model"

evit_versions=("evit_original")
rates=(0.7)
locs=("(0, 3, 6)")

for rate in "${rates[@]}"; do
  for loc in "${locs[@]}"; do
    for evit_ver in "${evit_versions[@]}"; do
      sed -i "s/^from evit_.* import EViT, _cfg/from ${evit_ver} import EViT, _cfg/" models.py
      echo "Switched models.py to use: ${evit_ver}"
      loc_clean=$(echo "$loc" | tr -d '(),' | tr -s ' ' '_')

      logdir="./deit_base/eval_log/${evit_ver}_with_fused_token_${rate}_${loc_clean}"

      echo "========================================================"
      echo "Current Experiment: EViT=$evit_ver, Rate=$rate, Drop Loc=$loc"
      echo "Output Dir: $logdir"
      echo "========================================================"

      python3 -m torch.distributed.launch --nproc_per_node=1 --use_env \
          main.py \
          --model deit_small_patch16_shrink_base \
          --fuse_token \
          --eval \
          --base_keep_rate "$rate" \
          --input-size 224 \
          --batch-size 1024 \
          --shrink_start_epoch 0 \
          --finetune $ckpt \
          --data-path $datapath \
          --output_dir $logdir \
          --drop_loc "$loc"
    done
  done
done