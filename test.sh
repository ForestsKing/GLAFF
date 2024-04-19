python -u main.py --model Informer --flag Standard --dataset Traffic --hist_len 96 --pred_len 192 --only_test
python -u main.py --model Informer --flag Plugin --dataset Traffic --hist_len 96 --pred_len 192 --only_test

python -u main.py --model DLinear --flag Standard --dataset Traffic --hist_len 96 --pred_len 192 --only_test
python -u main.py --model DLinear --flag Plugin --dataset Traffic --hist_len 96 --pred_len 192 --only_test

python -u main.py --model TimesNet --flag Standard --dataset Traffic --hist_len 96 --pred_len 192 --only_test
python -u main.py --model TimesNet --flag Plugin --dataset Traffic --hist_len 96 --pred_len 192 --only_test

python -u main.py --model iTransformer --flag Standard --dataset Traffic --hist_len 96 --pred_len 192 --only_test
python -u main.py --model iTransformer --flag Plugin --dataset Traffic --hist_len 96 --pred_len 192 --only_test
