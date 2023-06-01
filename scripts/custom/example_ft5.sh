TARGETS=("tracking_shuffled_objects" "date_understanding" "coin_flip" "last_letter_concatenation" "commonsense_qa" "strategy_qa"
         "single_eq" "addsub" "multiarith" "svamp" "gsm8k" "aqua")
MODELS=("flan_t5_small" "flan_t5_base" "flan_t5_large" "flan_t5_xl")
DEVICES="0"


for MODEL in ${MODELS[@]}; do
  for TARGET in ${TARGETS[@]}; do
    python custom_train.py --dataset_key $TARGET --model_key $MODEL --train_key "ft_cot" --devices $DEVICES --batch_size 8 --inference_batch_size 32
  done
done
