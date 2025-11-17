import argparse
from tqdm import tqdm
from pandas import read_json
from src.evaluation import cal_rouge_score


# parser
parser = argparse.ArgumentParser(
    description="Rouge-1 calculation file"
)
parser.add_argument(
    "-f",
    "--file_name", 
    help="input file", 
    required=True,
)
parser.add_argument(
    "-task",
    "--task", 
    help="task", 
    required=True,
)
parser.add_argument(
    "-out",
    "--out_lang", 
    help="output language", 
    required=False,
)
args = vars(parser.parse_args())

# load variables
file_name = args["file_name"]
task = args["task"].upper()
out_lang = args["out_lang"] if args["out_lang"] is not None else file_name.rsplit("_", 4)[1].split("-")[0]

# get data and its ground truth
df = (
    read_json(file_name, lines=True)
    .dropna(subset=["answer_pred"])
)
df_context = read_json(f"data/output/{task}/{task}_questions_gt_{out_lang}.jsonl")
context_col = "answer"
rouge_col = "rouge"

# get scores at response level
score = []
for idx, item in tqdm(df.iterrows(), total=df.shape[0]):
    question_id = item["ID"]
    response = item["answer_pred"]
    reference = df_context.loc[
        df_context["ID"]==question_id, context_col
    ].values[0]

    rouge = cal_rouge_score(
        reference=reference,
        candidate=response,
        lang=out_lang,
    )
    score.append(rouge)

df["rouge"] = score
output_path = file_name.rsplit("_", 2)[0]
df.to_json(f"{output_path}.jsonl")

# # =============================================================================
# # get scores at question level (because we generate 10 responses for each question)
# # use mean as the main aggregation function
# rouge = (
#     df[["ID", rouge_col]]
#     .groupby(
#         by="ID",
#         as_index=False,
#     )
#     .mean()
# )
# rouge.to_csv(
#     f"{file_name[:-4]}_with_rouge.csv", 
#     index=False,
# )
