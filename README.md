### Setup instructions for running evaluation
1. Git clone lm-evaluation-harness in this directory
   
   ```git clone https://github.com/EleutherAI/lm-evaluation-harness.git```
   

2. Create know_dist.yaml in the lm-evaluation/lm_eval/tasks/benchmarks with the following code:
   
    ```YAML
   group: know_dist
    task:
    - headqa_en
    - headqa_es
    - paws_en
    - paws_es
    - paws_zh
    - xnli_en
    - xnli_es
    - xnli_zh
    - xstorycloze_en
    - xstorycloze_es
    - xstorycloze_zh
    - lambada_openai_mt_en
    - lambada_openai_mt_es

3. run the install.sh script in your environment
4. change directory to /jobs and sbatch the model

Note: if you need to change the job script or create a new one then there are three things that you are crucial when you change it:

1. model_name: name that the results will have, mostly for properly keeping track of which model is which
2. model_path: where the model is located. This is the huggingface `pretrained_model_name_or_path` so you can also provide it with for example `bigscience/bloom-3B` to load a model from huggingface.   
3. output_path: location where the results will be saved. 


Most other things should not be touched except for SBATCH settings in case you don't have red queue access or something similar. Each evaluation job script should run for roughly 30 min.
