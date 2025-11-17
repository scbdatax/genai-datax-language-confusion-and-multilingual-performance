import argparse
import pandas as pd
from tqdm import tqdm
from src.evaluation.language_id import LanguageID

# Parser
parser = argparse.ArgumentParser(
    description="Language identification file"
)
parser.add_argument(
    '-f',
    '--file_name', 
    help='input file', 
    required=True
)
args = vars(parser.parse_args())

file_name = args['file_name']

df = pd.read_json(file_name, lines=True)

df_with_lang = {
    "ID": [],
    "answer_pred": [],
    "lang_pred": [],
    "lang_pred_word_level": []
}

lid = LanguageID()
file_name = file_name.rsplit(".", 1)[0]

for i, rows in tqdm(df.iterrows(), total=df.shape[0]):
    id = rows['ID']
    answer_pred = rows['answer_pred']
    answer_pred = str(answer_pred).replace("\n"," ")

    lang = lid.predict(answer_pred)[0]
    word_level_langs = lid.pred_word_level(answer_pred)

    df_with_lang['ID'].append(id)
    df_with_lang['answer_pred'].append(answer_pred)
    df_with_lang['lang_pred'].append(lang)
    df_with_lang['lang_pred_word_level'].append(word_level_langs)

    # save csv
    if i % 1000 == 0:
        pd.DataFrame(df_with_lang).to_json(f"{file_name}_with_lang.jsonl")

pd.DataFrame(df_with_lang).to_json(f"{file_name}_with_lang.jsonl")