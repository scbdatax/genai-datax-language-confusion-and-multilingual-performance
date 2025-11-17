import os
import argparse
import logging
import time
import pandas as pd
from tqdm import tqdm
from src.llm import LLM


# Parser
parser = argparse.ArgumentParser(
    description="Translate TH responses for file (-f for input the file path)"
)
parser.add_argument(
    '-f',
    '--file_name', 
    help='input file', 
    required=True
)
args = vars(parser.parse_args())

file_name = args['file_name']
df = pd.read_excel(file_name)

df_en = {
    "ID": [],
    "en_question": [],
    "en_a": [],
    "en_b": [],
    "en_c": [],
    "en_d": [],
    "en_e": [],
}

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
    "question": "Translate the following Thai quesstion into English\n\nThai: {content}\n\nEnglish: ",
    "content": "Translate the following Thai content into English\n\nThai: {content}\n\nEnglish: ",
}

# start
for idx, item in tqdm(df.iterrows(), total=df.shape[0]):
    question_id = item["ID"]
    subject = item["subject"]
    res = {}

    if subject not in ["Quant", "math"]:
        for c in ["question", "a", "b", "c", "d", "e"]:
            if c == "question":
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
            except:
                try:
                    time.sleep(30)
                    res[c] = llm.get_response(
                        prompt.format(content=content),
                        max_tokens=2000,
                        temperature=0,
                    )
                except:
                    failures.append(f"{question_id}")
                    res[c] = "X"
    else:
        res = {
            choice: item[choice]
            for choice in ["a", "b", "c", "d", "e"]
        }
        prompt = prompts["question"]
        content = item["question"]
        try:
            res["question"] = llm.get_response(
                prompt.format(content=content),
                max_tokens=2000,
                temperature=0,
            )
        except:
            try:
                time.sleep(30)
                res["question"] = llm.get_response(
                    prompt.format(content=content),
                    max_tokens=2000,
                    temperature=0,
                )
            except:
                failures.append(f"{question_id}")
                res["question"] = "X"

    df_en["ID"].append(question_id)
    df_en["en_question"].append(res["question"])
    df_en["en_a"].append(res["a"])
    df_en["en_b"].append(res["b"])
    df_en["en_c"].append(res["c"])
    df_en["en_d"].append(res["d"])
    df_en["en_e"].append(res["e"])

    (
        pd.DataFrame(df_en)
        .to_excel(f'{file_name[:-5]}_translated.xlsx', index=False)
    )
    logger.info(f"Finished question_id : {question_id}")
logger.info("failure: {}".format(", ".join(failures)))
