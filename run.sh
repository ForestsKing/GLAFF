for hist_len in 96 48 24; do
  for dataset in Traffic Electricity; do
    for model in Informer DLinear TimesNet iTransformer; do
      for flag in Standard Plugin; do
        python -u main.py --model $model --flag $flag --dataset $dataset --hist_len $hist_len --pred_len 192
      done
    done
  done
done
