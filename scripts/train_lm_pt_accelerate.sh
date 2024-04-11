basic_dir=`pwd`
cd ${basic_dir}/../workflows/

accelerate=/home/data/anaconda3/envs/python310/bin/accelerate

accelerate=/home/share/caoxu/miniconda3/envs/python310/bin/accelerate
accelerate_config=../conf/accelerate/zero3.yaml
CUDA_VISIBLE_DEVICES=0,1 ${accelerate} launch --num_cpu_threads_per_process 2 \
                     --config_file ${accelerate_config} \
                     lmflow.py \
                        --do_train True \
                        --probs 0.4,0.4,0.2 \
                        --report_to none \
                        --disable_tqdm True \
                        --include_tokens_per_second True \
                        --dataset alpaca_data_zh_51k,alpaca_gpt4_data_zh,baiduyidian \
                        --path /home/share/LLMs/llms/Qwen/Qwen-7B \
                        --template qwen_v1  \
                        --output_dir "../experinments/test_exp1/" \
                        --max_steps 10 \
                        --dispatch_batches False \
                        --logging_steps 1 \
                        --save_steps 100 \
                        --per_device_train_batch_size 2 \
                        --gradient_accumulation_steps 2 \
                        --finetuning_type full

