
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
seq_len=720
model_name=FlowCast

python -u run_expirement.py \
  --is_training 1 \
  --train_epochs 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'96 \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 1 \
  --num_trials 2 \
  --des 'Exp' \
  --features 'S' \
  --itr 1 --batch_size 256  >logs/$model_name'_'$seq_len'_'96.log

# python -u run_expirement.py \
#   --is_training 1 \
#   --train_epochs 1 \
#   --root_path ./dataset/ \
#   --data_path weather.csv \
#   --model_id weather_$seq_len'_'192 \
#   --model $model_name \
#   --seq_len $seq_len \
#   --pred_len 192 \
#   --enc_in 1 \
#   --num_trials 2 \
#   --des 'Exp' \
#   --features 'S' \
#   --itr 1 --batch_size 16  >logs/$model_name'_'$seq_len'_'192.log

# python -u run_expirement.py \
#   --is_training 1 \
#   --train_epochs 1 \
#   --root_path ./dataset/ \
#   --data_path weather.csv \
#   --model_id weather_$seq_len'_'336 \
#   --model $model_name \
#   --seq_len $seq_len \
#   --pred_len 336 \
#   --enc_in 1 \
#   --num_trials 2 \
#   --des 'Exp' \
#   --features 'S' \
#   --itr 1 --batch_size 16  >logs/$model_name'_'$seq_len'_'336.log

# python -u run_expirement.py \
#   --is_training 1 \
#   --train_epochs 1 \
#   --root_path ./dataset/ \
#   --data_path weather.csv \
#   --model_id weather_$seq_len'_'720 \
#   --model $model_name \
#   --seq_len $seq_len \
#   --pred_len 720 \
#   --enc_in 1 \
#   --num_trials 2 \
#   --des 'Exp' \
#   --features 'S' \
#   --itr 1 --batch_size 16  >logs/$model_name'_'$seq_len'_'720.log