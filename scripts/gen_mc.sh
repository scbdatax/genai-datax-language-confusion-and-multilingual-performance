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
tasks=(
    "thai_exam"
    "wti_mc"
)
poss_lang=(
    "en"
    "th"
)

for model in "${models[@]}"; do
    task="mmlu"
    python src/generation/unified_data_gen.py \
        -t "${temp}" \
        -task "${task}" \
        -m "${model}" \
        -itl en \
        -inl en \
        -oul en

    for task in "${tasks[@]}"; do
        for inst_lang in "${poss_lang[@]}"; do
            for in_lang in "${poss_lang[@]}"; do
                echo "${task}_${model}_${temp}_${inst_lang}-it_${in_lang}-in"
                python src/generation/unified_data_gen.py \
                    -t "${temp}" \
                    -task "${task}" \
                    -m "${model}" \
                    -itl "${inst_lang}" \
                    -inl "${in_lang}"
            done
        done
    done
done
