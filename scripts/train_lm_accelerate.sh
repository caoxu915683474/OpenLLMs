basic_dir=`pwd`
cd ${basic_dir}/../workflows/

accelerate=/root/autodl-tmp/miniconda3/envs/python310/bin/accelerate
accelerate_config=../conf/accelerate/zero3.yaml

CUDA_VISIBLE_DEVICES=0,1 ${accelerate} launch --num_cpu_threads_per_process 2 \
                     --config_file ${accelerate_config} \
                     lmflow.py \
                        --do_train True \
                        --probs 0.5,0.5 \
                        --report_to none \
                        --disable_tqdm True \
                        --include_tokens_per_second True \
                        --include_num_input_tokens_seen True \
                        --dataset baiduyidian1,baiduyidian2 \
                        --path bigscience/bloom-560m \
                        --template qwen_v1  \
                        --output_dir "../experinments/test_exp1/" \
                        --max_steps -1 \
                        --num_train_epochs 1 \
                        --dispatch_batches True \
                        --logging_steps 1 \
                        --dataloader_pin_memory True \
                        --save_steps 50 \
                        --save_only_model False \
                        --per_device_train_batch_size 2 \
                        --gradient_accumulation_steps 2 \
                        --finetuning_type full \
                        --dataloader_num_workers 2 \
                        --num_shards 2 \
                        --dataloader_pin_memory False \
                        --bf16 True \
                        --bf16_full_eval True \
                        --streaming False \
                        --mix_strategy interleave_over \
                        --gradient_checkpointing False

# CUDA_VISIBLE_DEVICES=0,1 ${accelerate} launch --num_cpu_threads_per_process 2 \
#                      --config_file ${accelerate_config} \
#                      lmflow.py \
#                      --path bigscience/bloom-560m \
#                      --template qwen_v1 \
#                      --dataset alpaca_data_zh_51k,alpaca_gpt4_data_zh,baiduyidian \
#                      --output_dir "../experinments/exp_20240401/" \
#                      --probs 0.4,0.4,0.2