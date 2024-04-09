python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
${python}  test.py \
                --path Qwen/Qwen-7B-Chat \
                --template qwen_v1 \
                --dataset yidian1 \
                --output_dir "../experinments/exp_20240401/" \
                --probs 0.3,0.7

# python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
# ${python}  test.py \
#                 --path Qwen/Qwen-7B-Chat \
#                 --template qwen_v1 \
#                 --dataset alpaca_data_zh_51k_100p2 \
#                 --output_dir "../experinments/exp_20240401/" \
#                 --probs 0.3,0.7