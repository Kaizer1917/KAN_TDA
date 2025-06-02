#!/bin/bash

echo "🎯 Training KAN_TDA with optimized hyperparameters for MSE < 0.35"

model_name=KAN_TDA

# Optimized hyperparameters based on KAN_TDA paper
echo "🔧 Using optimized configuration:"
echo "  - Larger model dimension (d_model=128)"
echo "  - Higher KAN order (begin_order=3)"
echo "  - More layers (e_layers=3)"
echo "  - Optimized learning rate schedule"
echo "  - Longer training (20 epochs)"
echo "  - Smaller batch size for better convergence"

# Configuration 1: High Performance Setup
train_optimized() {
    local pred_len=$1
    local config_name=$2
    
    echo "🚀 Training optimized KAN_TDA for prediction length: $pred_len ($config_name)"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_${pred_len}_optimized_${config_name} \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 3 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 128 \
        --d_ff 256 \
        --begin_order 3 \
        --down_sampling_layers 3 \
        --down_sampling_window 2 \
        --down_sampling_method avg \
        --moving_avg 25 \
        --des 'KAN_TDA_Optimized' \
        --itr 1 \
        --train_epochs 20 \
        --batch_size 16 \
        --learning_rate 0.0005 \
        --lradj type3 \
        --patience 7 \
        --use_gpu False \
        --gpu_type cpu
        
    echo "✅ Completed optimized training for prediction length $pred_len"
}

# Configuration 2: Ultra High Performance Setup
train_ultra_optimized() {
    local pred_len=$1
    
    echo "🔥 Training ULTRA optimized KAN_TDA for prediction length: $pred_len"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_${pred_len}_ultra \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 4 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 256 \
        --d_ff 512 \
        --begin_order 4 \
        --down_sampling_layers 4 \
        --down_sampling_window 2 \
        --down_sampling_method avg \
        --moving_avg 25 \
        --des 'KAN_TDA_Ultra' \
        --itr 1 \
        --train_epochs 30 \
        --batch_size 8 \
        --learning_rate 0.001 \
        --lradj type3 \
        --patience 10 \
        --use_gpu False \
        --gpu_type cpu
        
    echo "🏆 Completed ULTRA optimized training for prediction length $pred_len"
}

# Configuration 3: Paper-based Best Setup
train_paper_config() {
    local pred_len=$1
    
    echo "📄 Training with paper's best configuration for prediction length: $pred_len"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/ETT-small/ \
        --data_path ETTh1.csv \
        --model_id ETTh1_96_${pred_len}_paper \
        --model $model_name \
        --data ETTh1 \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model 16 \
        --d_ff 32 \
        --begin_order 1 \
        --down_sampling_layers 0 \
        --down_sampling_window 1 \
        --down_sampling_method avg \
        --moving_avg 25 \
        --des 'KAN_TDA_Paper' \
        --itr 1 \
        --train_epochs 10 \
        --batch_size 16 \
        --learning_rate 0.001 \
        --lradj type1 \
        --patience 3 \
        --use_gpu False \
        --gpu_type cpu
        
    echo "📊 Completed paper configuration training for prediction length $pred_len"
}

echo "🎯 Starting optimized training runs..."

# Test different configurations on prediction length 96
echo "=== Testing Optimized Configuration ==="
train_optimized 96 "config1"

echo "=== Testing Ultra Optimized Configuration ==="
train_ultra_optimized 96

echo "=== Testing Paper Configuration ==="
train_paper_config 96

echo "🏁 All optimized training completed!"
echo "📊 Check results in result_long_term_forecast.txt"
echo "🎯 Target: MSE < 0.35" 