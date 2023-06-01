TARGETS=("tracking_shuffled_objects" "date_understanding" "coin_flip" "last_letter_concatenation" "commonsense_qa" "strategy_qa"
         "single_eq" "addsub" "multiarith" "svamp" "gsm8k" "aqua")
MODELS=("t5_small" "t5_base" "t5_large" "t5_3b")
DEVICES="0"

for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft_cot" --devices $DEVICES --batch_size 8 --inference_batch_size 32
  done
done
