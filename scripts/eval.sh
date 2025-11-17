#!/bin/bash
models=(
    "llama3"
    "typhoon15"
    "llama31"
    "gemma2"
    "sailor"
    "qwen15"
    "qwen25"
    "otg"
)
temp=1.0
poss_lang=(
    "en"
    "th"
)

for model in "${models[@]}"; do
    for inst_lang in "${poss_lang[@]}"; do
        for in_lang in "${poss_lang[@]}"; do
            for out_lang in "${poss_lang[@]}"; do
                echo "${model}_${temp}_${inst_lang}-it_${in_lang}-in_${out_lang}-out"
                python scripts/eval/rouge.py \
                    -f "data/output/WTI_CQA/${model}/${model}_WTI_CQA_temp_${temp}_${inst_lang}-it_${in_lang}-in_${out_lang}-out_answers_with_lang.jsonl" \
                    -task cqa \
                    -out "${out_lang}"
                python scripts/eval/uncertainty.py \
                    -f "data/output/WTI_CQA/${model}/${model}_WTI_CQA_temp_${temp}_${inst_lang}-it_${in_lang}-in_${out_lang}-out_answers_with_lang.jsonl" \
                    -out "${out_lang}"
            done
        done
    done
done
