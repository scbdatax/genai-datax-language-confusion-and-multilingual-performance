# Language Confusion and Multilingual Performance: A Case Study of Thai-Adapted Large Language Models

This repository contains the implementation and experimental setup for our research on multilingual adaptability of large language models (LLMs), with a focus on Thai as a low-resource language case study.

## Abstract

This paper presents a comprehensive study on the multilingual adaptability of large language models (LLMs), with a focus on the interplay between training strategies and prompt design. Using Thai as a case study of a low-resource language, we examine:

- **RQ1**: The extent to which pre-trained models (Base) can adapt to a low-resource language like Thai through additional fine-tuning
- **RQ2**: How continual pre-training (CPT) compares to multilingual pre-training (MLLM) in terms of performance on downstream tasks
- **RQ3**: How language variation within different components of a structured prompt—*task instruction*, *context input*, and *output instruction*—influences task performance in cross-lingual settings

Our findings reveal that CPT proves to be a promising strategy for enhancing model performance in low-resource languages like Thai in monolingual settings, particularly for models that initially lack strong linguistic capabilities. In cross-lingual scenarios, MLLMs exhibit superior robustness compared to Base and CPT models.

## Repository Structure

```
.
├── data/
│   ├── input/              # Input datasets and questions
│   ├── output/             # Generated model responses
│   │   ├── MMLU/
│   │   ├── THAI_EXAM/
│   │   ├── WTI_MC/
│   │   ├── WTI_CQA/
│   │   └── WTI_SUM/
│   ├── analysis/           # Processed analysis data
│   └── prep_main_questions/ # Question preparation notebooks
├── src/
│   ├── generation/         # Response generation scripts
│   ├── evaluation/         # Evaluation metrics (IFHR, uncertainty, ROUGE, etc.)
│   ├── translation/        # Translation utilities
│   ├── llm/               # LLM API wrappers
│   └── utils/             # Utility functions
├── scripts/
│   ├── gen_mc.sh          # Generate multiple-choice responses
│   ├── gen_long_form.sh   # Generate long-form responses
│   └── eval/              # Evaluation scripts
├── analysis/              # Jupyter notebooks for analysis
│   ├── 00_translation.ipynb
│   ├── 01_agg_results.ipynb
│   ├── 02_base_vs_cpt.ipynb
│   ├── 03_prompt_variation.ipynb
│   ├── 04_question_level.ipynb
│   └── 05_model_type_level.ipynb
└── figure/               # Generated figures
```

## Tasks & Datasets

The repository supports five main tasks:

1. **MMLU**: Massive Multitask Language Understanding (English benchmark)
2. **THAI_EXAM**: Thai standardized exam questions (A-level, O-NET, TPAT)
3. **WTI_MC**: WangchanThaiInstruct Multiple Choice
4. **WTI_CQA**: WangchanThaiInstruct Closed Question Answering
5. **WTI_SUM**: WangchanThaiInstruct Summarization

## Models

The study evaluates 8 LLM variants across three categories:

- **Base Models**: Llama-3-8B, Qwen1.5-7B, Qwen2.5-7B
- **CPT Models**: Typhoon-v1.5-7B, Sailor-7B, OpenThaiGPT-1.5-7B
- **MLLM Models**: Llama-3.1-8B, Gemma-2-9B

## Installation

```bash
# Install dependencies using Poetry
poetry install
poetry shell
```

## Environment Setup

Create a `.env` file based on `.env.template` with your API keys:

```bash
OPENAI_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
TYPHOON_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Reproducing Results

### 1. Generate Model Responses

#### Multiple Choice Tasks (MMLU, THAI_EXAM, WTI_MC)

```bash
bash scripts/gen_mc.sh
```

#### Long-form Tasks (WTI_CQA, WTI_SUM)

```bash
bash scripts/gen_long_form.sh
```

Or generate for a specific configuration:

```bash
python src/generation/unified_data_gen.py \
    -t 1.0 \
    -task wti_cqa \
    -m llama3 \
    -itl en \
    -inl en \
    -oul en \
    -n 10
```

**Parameters:**
- `-t`: Temperature (default: 1.0)
- `-task`: Task name (mmlu, thai_exam, wti_mc, wti_cqa, wti_sum)
- `-m`: Model name
- `-itl`: Instruction language (en/th)
- `-inl`: Input language (en/th)
- `-oul`: Output language (en/th)
- `-n`: Number of generations per question (default: 10)

### 2. Evaluate Responses

#### Language Identification

```bash
python scripts/eval/lang_id.py \
    -f data/output/WTI_CQA/llama3/llama3_WTI_CQA_temp_1.0_en-it_en-in_en-out_answers.jsonl \
    -out en
```

#### Uncertainty Estimation

```bash
python scripts/eval/uncertainty.py \
    -f data/output/WTI_CQA/llama3/llama3_WTI_CQA_temp_1.0_en-it_en-in_en-out_answers.jsonl \
    -out en
```

#### ROUGE Score

```bash
python scripts/eval/rouge.py \
    -f data/output/WTI_SUM/llama3/llama3_WTI_SUM_temp_1.0_en-it_en-in_en-out_answers.jsonl \
    -task WTI_SUM \
    -out en
```

### 3. Analyze Results

Run the analysis notebooks in order:

1. `analysis/01_agg_results.ipynb` - Aggregate results across all experiments
2. `analysis/02_base_vs_cpt.ipynb` - Compare Base vs CPT models
3. `analysis/03_prompt_variation.ipynb` - Analyze prompt language effects
4. `analysis/04_question_level.ipynb` - Question-level analysis
5. `analysis/05_model_type_level.ipynb` - Model type comparisons

## Evaluation Metrics

- **IFHR (Instruction Following & Hallucination Rate)**: Measures adherence to instructions
- **Uncertainty**: Semantic entropy calculated from response variations
- **Accuracy**: Task-specific accuracy metrics
- **WLE (Word-Level Entropy)**: Language mixing detection
- **ROUGE-1**: Content overlap for summarization tasks

## Key Components

### Generation (`src/generation/`)
- `unified_data_gen.py` - Main generation script
- `data_gen_api.py` - API-based generation
- `data_gen_host.py` - Hosted model generation

### Evaluation (`src/evaluation/`)
- `language_id.py` - Language identification using fastText
- `ifhr.py` - Instruction following metrics
- `similarity.py` - Similarity calculations for uncertainty

### LLM Wrappers (`src/llm/`)
- `api.py` - API clients (OpenAI, Together, Typhoon)
- `host.py` - vLLM hosting for open-source models
