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