basic_dir=`pwd`
cd ${basic_dir}/../workflows/

python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
${python}  lmflow.py \
            --do_train \
            --path Qwen/Qwen-7B-Chat \
            --template qwen_v1  \
            --dataset yidian1,yidian2 \
            --probs 0.3,0.7 \
            --output_dir "../experinments/exp_20240401/" \
            --max_steps 10000 \
            --logging_steps 20 \
            --save_steps 100 \
            --per_device_train_batch_size 8 \
            --gradient_accumulation_steps 4