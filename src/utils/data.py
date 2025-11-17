from tqdm import tqdm
from pandas import concat
from pandas import read_json
from pandas import DataFrame
from src.utils.misc_functions import check_ifhr
from src.utils.misc_functions import cal_entropy
from src.utils.misc_functions import revise_answer
from src.utils.misc_functions import cal_entropy_long


def get_long_form_data(model, task, temp="1.0"):
    task = task.upper()
    poss_lang = ["en", "th"]
    result = DataFrame()
    for inst_lang in poss_lang:
        for in_lang in poss_lang:
            for out_lang in poss_lang:
                # load data
                u_input_file = f"../data/output/{task}/{model}/{model}_{task}_temp_{temp}_{inst_lang}-it_{in_lang}-in_{out_lang}-out_answers_with_lang_with_uncertainty.jsonl"
                r_input_file = f"../data/output/{task}/{model}/{model}_{task}_temp_{temp}_{inst_lang}-it_{in_lang}-in_{out_lang}-out_answers.jsonl"

                u = read_json(u_input_file, lines=True)
                r = (
                    read_json(r_input_file, lines=True)
                    .dropna(subset=["lang_pred_word_level"])
                )

                # ifhr
                r["ifhr"] = r["lang_pred"].apply(
                    lambda x: 0 if x == out_lang else 1
                )

                # wle
                r["wle"] = r["lang_pred_word_level"].apply(cal_entropy_long)

                # aggregate
                r = (
                    r[["ID", "rouge", "ifhr", "wle"]]
                    .groupby(
                        by="ID",
                        as_index=False,
                    )
                    .mean()
                )

                # merge all metrics
                df = u.merge(r, on="ID", how="inner")
                df["model"] = model
                df["temperature"] = float(temp)
                df["experiment"] = f"{inst_lang}_{in_lang}_{out_lang}"

                # concatenate all experiments from all models
                result = concat(
                    objs=[result, df],
                    ignore_index=True,
                )

    return result


def get_short_form_data(model, task, temp="1.0"):
    task = task.upper()
    poss_lang = ["en"] if task == "MMLU" else ["en", "th"]

    df_gt = read_json(
        f"../data/output/{task}/{task}_questions_gt_{poss_lang[0]}.jsonl", 
        lines=True,
    )
    df_gt = df_gt[["ID", "answer"]]

    result = DataFrame()
    for inst_lang in poss_lang:
        for in_lang in poss_lang:
            filename = f"../data/output/{task}/{model}/{model}_{task}_temp_{temp}_{inst_lang}-it_{in_lang}-input_answers.jsonl"
            df = (
                read_json(filename, lines=True)
                .merge(df_gt, on="ID", how="left")
                .dropna(subset=["answer_pred"])
            )

            # check accuracy
            df["accuracy"] = df[["answer_pred", "answer"]].apply(
                lambda x: 1 if x["answer_pred"] == x["answer"] else 0,
                axis=1,
            )

            # ifhr
            df["ifhr"] = df["answer_pred"].apply(check_ifhr)

            # accuracy2
            df["answer_pred_revised"] = df["answer_pred"].apply(revise_answer)
            df["accuracy_revised"] = df[["answer_pred_revised", "answer"]].apply(
                lambda x: 1 if x["answer_pred_revised"] == x["answer"] else 0,
                axis=1,
            )

            # calculate mean acc
            df = (
                df
                .groupby(
                    by="ID",
                    as_index=False
                )
                .agg({
                    "accuracy": "mean",
                    "answer_pred": cal_entropy,
                    "accuracy_revised": "mean",
                    "answer_pred_revised": cal_entropy,
                    "ifhr": "mean",
                })
            )
            
            # provide metadata
            df["model"] = model
            df["temperature"] = float(temp)
            df["experiment"] = f"{inst_lang}_{in_lang}"
            df = df.rename({"answer_pred": "uncertainty"}, axis=1)

            # concat each experiment
            result = concat(
                objs=[result, df],
                ignore_index=True,
            )
    return result


def get_common_questions(result: DataFrame, models: list):
    for i, model in enumerate(models):
        tmp_set = set(
            result
            .loc[
                result["model"]==model, "ID"
            ]
            .unique()
        )
        if i == 0:
            common_questions = tmp_set
        else: 
            common_questions = common_questions.intersection(tmp_set)
            
    return list(common_questions)


def get_all_data(task, temp="1.0"):
    models = ["llama3", "typhoon15", "qwen15", "sailor", "qwen25", "otg", "gemma2", "llama31"]
    get_data_fn = get_short_form_data if task in ["mmlu", "thai_exam", "wti_mc"] else get_long_form_data

    df_list = []
    for model in tqdm(models, total=len(models), desc=f"Loading data for [{task.upper()}]"):
        df_list.append(
            get_data_fn(model=model, task=task, temp=temp)
        )

    result = concat(df_list)
    common_questions = get_common_questions(result, models)
    result = result.query(f"ID in {common_questions}")
    print(f"Common questions for [{task.upper()}]:", len(common_questions))

    return result


def normalize_df(df, tasks):
    model_types = {
        "Base": ["llama3", "qwen15", "qwen25"],
        "CPT": ["typhoon15", "sailor", "otg"],
        "MLLM": ["llama31", "gemma2"]
    }

    for idx, (model_type, model_list) in enumerate(model_types.items()):
        df.loc[df["model"].isin(model_list), "model_type"] = model_type
        df.loc[df["model"].isin(model_list), "sort_order1"] = idx

    for idx, task in enumerate(tasks):
        df.loc[df["task"] == task, "sort_order2"] = idx

    df = (
        df
        .sort_values(by=["sort_order2", "sort_order1"])
        .reset_index(drop=True)
        .drop(columns=["sort_order2", "sort_order1"])
    )

    df["task"] = df["task"].replace({
        "mmlu": "MMLU",
        "thai_exam": "ThaiExam",
        "wti_mc": "WTI - MC",
        "wti_cqa": "WTI - CQA",
        "wti_sum": "WTI - SUM",
    })
    return df


def compare_diff_task_inst(
    df: DataFrame,
    task: str,
    model: str,
    req_cols=["experiment"],
):
    if task in ["WTI - SUM", "WTI - CQA", "long"]:
        exprs = ["en_en_en", "th_en_en",  "en_en_th", "th_en_th", "en_th_en", "th_th_en", "en_th_th", "th_th_th"]
        sel_cols = ["ifhr", "uncertainty", "wle", "rouge"]
    else:
        exprs = ["en_en", "th_en", "en_th", "th_th"]
        sel_cols = ["ifhr", "uncertainty", "accuracy_revised"]

    if task not in ["short", "long"]:
        df = df.query(f"task == '{task}'")

    df = (
        df
        .query(f"model == '{model}'")
        .sort_values(by="experiment", key=lambda x: x.map({expr: i for i, expr in enumerate(exprs)}))
        .reset_index(drop=True)
        .loc[:, req_cols+sel_cols]
    )

    df_diff = df[sel_cols].diff().iloc[1::2]
    df_diff.insert(0, "experiment", "diff") # delta = \u0394
    df_combined = (
        concat([df, df_diff])
        .sort_index(kind="merge")
        .reset_index(drop=True)
    )
    
    return df_combined


def compare_diff_task_inst_by_model_type(df: DataFrame, task: str):
    if task in ["WTI - SUM", "WTI - CQA", "long"]:
        metrics = ["ifhr", "uncertainty", "wle", "rouge"]
    else:
        metrics = ["ifhr", "uncertainty", "accuracy_revised"]

    data = {
        "base": compare_diff_task_inst(df=df, task=task, model="llama3"),
        "cpt": compare_diff_task_inst(df=df, task=task, model="typhoon15"),
        "mllm": compare_diff_task_inst(df=df, task=task, model="llama31"),
    }

    comparison = data["base"][["experiment"]].copy()
    for metric in metrics:
        for model_type in ["base", "cpt", "mllm"]:
            comparison[f"{model_type}_{metric}"] = data[model_type][metric]
            
    return comparison
