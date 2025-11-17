import argparse
from src.llm.api import LLM
from src.generation.data_gen_api import MMLU
from src.generation.data_gen_api import THAI_EXAM
from src.generation.data_gen_api import WTI_MC
from src.generation.data_gen_api import WTI_CQA
from src.generation.data_gen_api import WTI_SUM


# Parser
parser = argparse.ArgumentParser(
    description="Generate response for the specified task and model at defined temperature -t and numgen -n"
)
parser.add_argument(
    '-t',
    '--temp', 
    help='temperature', 
    required=True
)
parser.add_argument(
    '-task',
    '--task', 
    help='task', 
    required=True
)
parser.add_argument(
    '-m',
    '--model', 
    help='model', 
    required=True
)
parser.add_argument(
    '-n',
    '--numgen', 
    help='the number of responses per question', 
    default=10, 
    required=False
)
parser.add_argument(
    '-r',
    '--resume', 
    help='resume id', 
    default=0, 
    required=False
)
parser.add_argument(
    '-itl',
    '--instlang', 
    help='instruction language', 
    default="th", 
    required=True
)
parser.add_argument(
    '-inl',
    '--inputlang', 
    help='input language', 
    default="th", 
    required=True
)
parser.add_argument(
    '-oul',
    '--outputlang', 
    help='output language', 
    default="th", 
    required=False
)
args = vars(parser.parse_args())

temperature = float(args['temp'])
task = str(args['task'])
model = str(args['model'])
num_gen = int(args['numgen'])
resume = int(args['resume'])
inst_lang = str(args['instlang'])
input_lang = str(args['inputlang'])
output_lang = str(args['outputlang'])

# Model
if model.lower() == "llama3":
    llm = LLM(
        client_name="together", 
        model_name="meta-llama/Llama-3-8b-chat-hf",
    )
elif model.lower() == "typhoon15":
    llm = LLM(
        client_name="typhoon", 
        model_name="typhoon-v1.5-instruct",
    )
elif model.lower() == "llama31":
    llm = LLM(
        client_name="together", 
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", # quantization to fp8.
    )
elif model.lower() == "gemma2":
    llm = LLM(
        client_name="together", 
        model_name="google/gemma-2-9b-it",
    )

else: # asuuming that the rest is meant to be an open-source model
    from src.llm.host import vLLM
    from src.generation.data_gen_host import MMLU
    from src.generation.data_gen_host import THAI_EXAM
    from src.generation.data_gen_host import WTI_MC
    from src.generation.data_gen_host import WTI_CQA
    from src.generation.data_gen_host import WTI_SUM
    llm = vLLM(model_name=model)

# Task
if task.lower() == "mmlu":
    generator = MMLU(
        split="test",
        lang="en",
    )
elif task.lower() == "thai_exam":
    generator = THAI_EXAM(
        inst_lang=inst_lang,
        input_lang=input_lang,
    )
elif task.lower() == "wti_mc":
    generator = WTI_MC(
        inst_lang=inst_lang,
        input_lang=input_lang
    )
elif task.lower() == "wti_cqa":
    generator = WTI_CQA(
        inst_lang=inst_lang,
        input_lang=input_lang,
        output_lang=output_lang,
    )
elif task.lower() == "wti_sum":
    generator = WTI_SUM(
        inst_lang=inst_lang,
        input_lang=input_lang,
        output_lang=output_lang,
    )
else: 
    raise Exception("Please choose the task to be either 'thai_exam', 'wti_mc', 'wti_cqa', 'wti_oqa', or 'wti_sum'")

generator.generate_responses(
    llm=llm, 
    llm_name=model,
    temperature=temperature,
    num_gen=num_gen,
    resume_id=resume,
)
