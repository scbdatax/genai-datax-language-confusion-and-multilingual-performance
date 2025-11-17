from numpy import unique
from ast import literal_eval
from scipy.stats import entropy


def cal_entropy_long(x) -> float:
    if type(x) != list:
        x = literal_eval(x)
    count = unique(x, return_counts=True)[1]
    p = count / sum(count)
    return entropy(p)

def cal_entropy(x) -> float:
    count = unique(x, return_counts=True)[1]
    p = count / sum(count)
    return entropy(p)

def check_ifhr(x: str) -> int:
    x = str(x).strip()
    choices = ["A", "B", "C", "D", "E"]
    if x in choices:
        return 0
    else:
        return 1

def revise_answer(x: str) -> str:
    x = x.strip()
    choices = ["A", "B", "C", "D", "E"]
    for idx, choice in enumerate(choices):
        x = x.replace(str(idx), choice)

        # just contains the answer at anywhere in the response
        if choice in x:
            return choice
        
    return x

def aggregate_results(result, by=["model", "experiment"]):
    return (
        result
        .drop(columns=["ID"])
        .groupby(
            by=by,
            as_index=False,
        )
        .mean()
    )
