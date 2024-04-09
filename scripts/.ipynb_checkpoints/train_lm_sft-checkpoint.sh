basic_dir=`pwd`
cd ${basic_dir}/../workflows/

python=/root/autodl-tmp/miniconda3/envs/python310/bin/python
${python}  lmflow.py \
            --do_train \
            --path Qwen/Qwen-7B \
            --template qwen_v1  \
            