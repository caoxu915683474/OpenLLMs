basic_dir=`pwd`
cd ${basic_dir}/../workflows/

python=/home/data/anaconda3/envs/python310/bin/python
CUDA_VISIBLE_DEVICES=0 ${python}  lmflow.py \
                                        --do_train \
                                        --probs 0.3,0.7 \
                                        --dataset alpaca_data_zh_51k,alpaca_gpt4_data_zh\
                                        --path /home/share/LLMs/llms/Qwen/Qwen-7B \
                                        --template qwen_v1  \
                                        --output_dir "../experinments/test_exp1/" \
                                        --max_steps 10000 \
                                        --logging_steps 20 \
                                        --save_steps 100 \
                                        --per_device_train_batch_size 1 \
                                        --gradient_accumulation_steps 2 \
                                        --finetuning_type full
            
