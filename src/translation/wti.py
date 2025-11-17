import os
import logging
import time
import pandas as pd
from tqdm import tqdm
from src.llm.api import LLM


# log
logging.basicConfig(
    filename="translation.log",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger()
failures = []

# llm
llm = LLM(
    client_name="openai",
    model_name="gpt-4o-2024-05-13",
)
prompts = {
    "question": "Translate the following Thai quesstion into English one\n\nThai: {content}\n\nEnglish: ",
    "content": "Translate the following Thai content into English one\n\nThai: {content}\n\nEnglish: ",
}

# dataset
ds = pd.read_parquet(
    "hf://datasets/airesearch/WangchanThaiInstruct_7.24/data/train-00000-of-00001.parquet"
)
tasks = ["Multiple choice", "Closed QA", "Summarization"]
ds = ds.query(f" Task_type in {tasks} ").sample(200)
ds_en = {
    "ID": [],
    "en_instruction": [],
    "en_input": [],
    "en_output": [],
}

# start
for idx, item in tqdm(ds.iterrows(), total=ds.shape[0]):
    task = item["Task_type"]
    question_id = item["ID"]

    res = {}
    success = {}
    for c in ["Instruction", "Input", "Output"]:
        if item[c] is None: 
            res[c] = None
            success[c] = True
        else:
            if task in ["Multiple choice", "Closed QA"] and c == "Instruction":
                prompt = prompts["question"]
            else:
                prompt = prompts["content"]

            content = item[c]
            try:
                res[c] = llm.get_response(
                    prompt.format(content=content),
                    max_tokens=2000,
                    temperature=0,
                )
                success[c] = True
            except:
                try:
                    time.sleep(30)
                    res[c] = llm.get_response(
                        prompt.format(content=content),
                        max_tokens=2000,
                        temperature=0,
                    )
                    success[c] = True
                except:
                    failures.append(f"{question_id}_{c}")
                    success[c] = False

    if success["Instruction"] and success["Input"] and success["Output"]:
        ds_en["ID"].append(question_id)
        ds_en["en_instruction"].append(res["Instruction"])
        ds_en["en_input"].append(res["Input"])
        ds_en["en_output"].append(res["Output"])

        (
            pd.DataFrame(ds_en)
            .to_excel("data/input/translation.xlsx", index=False)
        )
        logger.info(f"Finished question_id : {question_id}")
logger.info("failure: {}".format(", ".join(failures)))
