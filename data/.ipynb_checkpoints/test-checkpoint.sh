# python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
# ${python}  test.py \
#                --path Qwen/Qwen-7B-Chat \
#                --template qwen_v1 \
#                --dataset yidian1 \
#                --output_dir "../experinments/exp_20240401/" \
#                --probs 0.3,0.7

# python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
# ${python}  test.py \
#                  --path bigscience/bloom-560m \
#                  --template qwen_v1 \
#                  --dataset alpaca_data_zh_51k,alpaca_gpt4_data_zh,baiduyidian \
#                  --output_dir "../experinments/exp_20240401/" \
#                  --probs 0.4,0.4,0.2 \


# accelerate=/root/autodl-tmp/miniconda3/envs/python310/bin/accelerate

# accelerate_config=../conf/accelerate/zero3.yaml

# CUDA_VISIBLE_DEVICES=0,1 ${accelerate} launch --num_cpu_threads_per_process 2 \
#                      --config_file ${accelerate_config} \  
#                         test.py \
#                              --path bigscience/bloom-560m \
#                              --template qwen_v1 \
#                              --dataset baiduyidian1,baiduyidian2 \
#                              --output_dir "../experinments/exp_20240401/" \
#                              --probs 0.5,0.5 \

python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
${python}  test.py \
                 --path bigscience/bloom-560m \
                 --template qwen_v1 \
                 --dataset baiduyidian1,baiduyidian2 \
                 --output_dir "../experinments/exp_20240401/" \
                 --probs 0.5,0.5 \
                 --num_shards 2 \
                 --streaming True \
                 --mix_strategy interleave_over