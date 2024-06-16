model_name=StockLLM
train_epochs=10
learning_rate=0.01
llama_layers=32

master_port=01097
num_process=4
batch_size=2
d_model=16
d_ff=32
llm_dim=1536
comment='StockLLM'
use_amp='True'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_stock_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /home/userroot/projs/Time-LLM/etl/index_300 \
  --data_path *.csv \
  --model_id stock_14_3 \
  --model $model_name \
  --data stock \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 3 \
  --patch_len 16 \
  --stride 1 \
  --prompt_domain 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --llm_dim $llm_dim

