import argparse
from pandas import Series
from pandas import read_csv
from src.evaluation import EigenUncertainty
from src.evaluation.similarity import Jaccard


# parser
parser = argparse.ArgumentParser(
    description="Uncertainty estimation file (-f for input the file path)"
)
parser.add_argument(
    "-f",
    "--file_name", 
    help="input file", 
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
out_lang = args["out_lang"] if args["out_lang"] is not None else "th"

# get data
df = (
    read_csv(file_name, lineterminator="\n")
    .dropna(subset=["answer_pred"])
)

# get pairwise similarity calculator
uncertainty_estimator = EigenUncertainty()
pairwise_sim = {
    "en": Jaccard(method="simple"),
    "th": Jaccard(method="pythai"),
}

# define aggregation function
def cal_selfcheck(
    group, 
    res_col, 
):    
    # selfcheck with entropy
    # out_lang = group[lang_col].values[0]
    W = pairwise_sim[out_lang].get_W_mat(group[res_col])
    U = uncertainty_estimator.estimate(W)
    return Series({"uncertainty": U})

# combine all record-level results
uncertainty = (
    df
    .groupby(
        by=["ID"], 
        as_index=False,
    )
    .apply(
        lambda x: cal_selfcheck(
            group=x, 
            res_col="answer_pred",
        ), 
        include_groups=False,
    )
)
file_name = file_name.rsplit(".", 1)[0]
uncertainty.to_json(
    f"{file_name}_with_uncertainty.jsonl",
)
