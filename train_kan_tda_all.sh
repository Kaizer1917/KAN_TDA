#!/bin/bash

echo "🚀 Starting comprehensive KAN_TDA training on multiple datasets..."

# Configuration
model_name=KAN_TDA
epochs=10
batch_size=32
learning_rate=0.0001

# Function to train on a dataset
train_dataset() {
    local dataset=$1
    local data_path=$2
    local enc_in=$3
    local c_out=$4
    
    echo "📊 Training KAN_TDA on $dataset dataset..."
    
    for pred_len in 96 192 336 720; do
        echo "  🎯 Training for prediction length: $pred_len"
        
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./data/ \
            --data_path $data_path \
            --model_id ${dataset}_96_${pred_len} \
            --model $model_name \
            --data $dataset \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in $enc_in \
            --dec_in $enc_in \
            --c_out $c_out \
            --d_model 64 \
            --d_ff 128 \
            --begin_order 2 \
            --down_sampling_layers 2 \
            --down_sampling_window 2 \
            --down_sampling_method avg \
            --moving_avg 25 \
            --des 'KAN_TDA_Production' \
            --itr 1 \
            --train_epochs $epochs \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --use_gpu False \
            --gpu_type cpu
            
        echo "  ✅ Completed $dataset prediction length $pred_len"
    done
    
    echo "🎉 Completed all training for $dataset dataset!"
}

# Train on ETTh1 dataset
train_dataset "ETTh1" "ETT-small/ETTh1.csv" 7 7

# Train on ETTh2 dataset  
train_dataset "ETTh2" "ETT-small/ETTh2.csv" 7 7

# Train on ETTm1 dataset
train_dataset "ETTm1" "ETT-small/ETTm1.csv" 7 7

# Train on ETTm2 dataset
train_dataset "ETTm2" "ETT-small/ETTm2.csv" 7 7

echo "🏆 All KAN_TDA training completed successfully!"
echo "📈 Results saved to: ./results/"
echo "💾 Model checkpoints saved to: ./checkpoints/"
echo "📊 Performance logs saved to: result_long_term_forecast.txt" 