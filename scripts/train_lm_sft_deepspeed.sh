basic_dir=`pwd`
cd ${basic_dir}/../workflows/

deepspeed=/home/data/anaconda3/envs/python310/bin/deepspeed
CUDA_VISIBLE_DEVICES=0,1,2,3 ${deepspeed}  --num_gpus 4 \
                                    lmflow.py \
                                        --deepspeed ../conf/deepspeed/ds_z3_config.json \
                                        --do_train True \
                                        --probs 0.3,0.7 \
                                        --dataset alpaca_data_zh_51k_100p2,alpaca_data_zh_51k_100p \
                                        --path /home/share/LLMs/llms/Qwen/Qwen-7B \
                                        --template qwen_v1  \
                                        --output_dir "../experinments/test_exp1/" \
                                        --max_steps 10000 \
                                        --dispatch_batches False\
                                        --logging_steps 20 \
                                        --save_steps 100 \
                                        --per_device_train_batch_size 8 \
                                        --gradient_accumulation_steps 4 \
                                        --finetuning_type full
            
