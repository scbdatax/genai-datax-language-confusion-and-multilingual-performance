import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")


palette = [
    "#992990", 
    "#2E82A8", 
    "#191919", 
    "#F7AD45", 
    "#328E6E",
    "#27548A",
    "#8E1616",
    "#7A7A73",
]

def _get_marker(r, g, b, marker="o"):
    return Line2D(
        [0], 
        [0], 
        marker=marker, 
        color="w", 
        markerfacecolor=[r/255, g/255, b/255], 
        markersize=10,
    )

def _plot_legend(labels, marker="o"):
    custom_legend = [
        _get_marker(153, 41, 144, marker),
        _get_marker(46, 130, 168, marker),
        _get_marker(25, 25, 25, marker),
        _get_marker(247, 173, 69, marker),
        _get_marker(50, 142, 110, marker),
        _get_marker(39, 84, 138, marker),
        _get_marker(142, 22, 22, marker),
        _get_marker(122, 122, 115, marker),
    ]
    plt.figlegend(custom_legend, labels, loc="lower center", ncol=4)

def plot_kde(
    df,
    labels,
    temp = 1,
    xlim = [0.5, 7.5],
    ylim = [0, 0.7],
):
    poss_lang = ["en", "th"]
    exprs = [
        f"{inst_lang}_{in_lang}_{out_lang}"
        for inst_lang in poss_lang
        for in_lang in poss_lang
        for out_lang in poss_lang
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, expr in enumerate(exprs):
        ax = axes[i//4, i%4]
        tmp_data = (
            df
            .query(f" temperature == {temp} ")
            .query(f" experiment == '{expr}' ")
        )
        sns.kdeplot(
            data=tmp_data,
            x="uncertainty",
            y="rouge", 
            ax=ax,
            hue="model",
            hue_order=labels,
            palette=palette,
            alpha=0.5,
        )
        ax.set_title(expr)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Rouge-1")
        ax.get_legend().remove()

    _plot_legend(labels)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_barchart(result_agg, models, metric, rotation=0):
    metric_mapping = {
        "uncertainty": r"Uncertainty $(\downarrow)$",
        "rouge": r"Rouge-1 $(\uparrow)$",
        "accuracy": r"Accuracy $(\uparrow)$",
        "ifhr": r"IFHR $(\downarrow)$",
        "wle": r"WLE $(\downarrow)$",
    }

    r'$\rho/\rho_{ref}\;\rightarrow$'

    sns.barplot(
        data=result_agg,
        x="experiment",
        y=metric,
        hue="model",
        hue_order=models,
        palette=palette,
    )

    plt.xlabel("Experiment")
    plt.ylabel(metric_mapping[metric])
    plt.xticks(rotation=rotation)
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc="upper center", ncol=4)
    plt.show()

def subplot_histogram_curve(result, metric, models, experiments, temperature=1):
    # plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, model in enumerate(models):
        ax = axes[i//4, i%4]
        for j, experiment in enumerate(experiments):
            filtered_data = result.query(f"model == '{model}' & temperature == {temperature} & experiment == '{experiment}'")
            sns.kdeplot(filtered_data[metric], bw_adjust=0.5, label=experiment, ax=ax, color=palette[j])
        ax.set_title(f"model={model}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Density")

    _plot_legend(experiments, marker="s")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_histogram_curve(result, model, metric, experiments, temperature=1):
    # plt.figure(figsize=(10, 6))
    for experiment in experiments:
        filtered_data = result.query(f"model == '{model}' & temperature == {temperature} & experiment == '{experiment}'")
        sns.kdeplot(filtered_data[metric], bw_adjust=0.5, label=experiment)
    plt.title(f"model={model}")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.legend(title="Experiment")
    plt.show()


def handle_metric_name(metric, use_symbol: bool = True):
    metric_mapping = {
        "Accuracy": "accuracy_revised",
        "Uncertainty": "uncertainty",
        "IFHR": "ifhr",
        "ROUGE-1": "rouge",
        "WLE": "wle",
        "ROUGE-1 | Accuracy": "performance",
    }

    metric_to_use = metric_mapping[metric]
    metric_to_show = metric

    if use_symbol:
        if metric in ["Accuracy", "ROUGE-1", "ROUGE-1 | Accuracy"]:
            metric_to_show += " (\u2191)" # r" $(\uparrow)$"
        elif metric in ["Uncertainty", "IFHR", "WLE"]:
            metric_to_show += " (\u2193)" # r" $(\downarrow)$"
        
    return metric_to_show, metric_to_use


def plot_barchat_by_metric(df, metric, hue="model_type", legend_title="Model Type", *args, **kwargs) -> None:
    metric_to_show, metric_to_use = handle_metric_name(metric)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="task", 
        y=metric_to_use,
        data=df, 
        hue=hue, 
        errorbar=None,
        palette=["#992990", "#2E82A8", "#404040"],
        *args,
        **kwargs,
    )
    plt.xlabel("Task")
    plt.ylabel(metric_to_show)
    plt.legend(title=legend_title)
    plt.show()


def plot_by_pair(df, focus_models, expr="short"):
    temp_df = df.query(f"model in {focus_models}")
    # temp_df = df.query("model in @focus_models")
    expr_mapping = {
        "short": ["IFHR", "Uncertainty", "Accuracy"],
        "long": ["IFHR", "Uncertainty",  "WLE", "ROUGE-1"],
    }
    for metric in expr_mapping[expr]:
        plot_barchat_by_metric(temp_df, metric, hue="model", hue_order=focus_models, legend_title="Model")


def plot_by_experiment(df, metric, focus_models, task, *args, **kwargs):
    temp_df = df.query(f"task == '{task}' and model in {focus_models}")
    metric_to_show, metric_to_use = handle_metric_name(metric)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="experiment", 
        y=metric_to_use, 
        data=temp_df, 
        hue="model", 
        errorbar=None,
        hue_order=focus_models,
        *args,
        **kwargs,
    )
    plt.title(f"{task}")
    plt.xlabel("Experiment")
    plt.ylabel(metric_to_show)
    plt.legend(title="Model", loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    plt.show()


def holistic_plot_by_experiment(df, focus_models, task, expr="short"):
    expr_mapping = {
        "short": ["IFHR", "Uncertainty", "Accuracy"],
        "long": ["IFHR", "Uncertainty",  "WLE", "ROUGE-1"],
    }
    for metric in expr_mapping[expr]:
        plot_by_experiment(df, metric, focus_models, task=task)


def plot_base_vs_cpt(
    df,
    expr="en_en",
    task: str = None,
    models=["llama3", "typhoon15", "qwen15", "sailor", "qwen25", "otg"],
    mllm_model="llama31",
    use_legend: bool = False,
    fix_yaxis: bool = True,
    save_result: bool = True,
):
    # prep viz config
    sns.set_theme(style="darkgrid")
    positions = [0, 0.7, 2, 2.7, 4, 4.7]

    # prep data
    base_data = (
        df
        .query(f"experiment == '{expr}'")
        .sort_values(by="model", key=lambda x: x.map({model: i for i, model in enumerate(models)}))
    )

    if task not in ["short", "long"]:
        base_data = base_data.query(f"task == '{task}'")

    data = base_data[base_data["model"].isin(models)].copy()

    if task in ["MMLU", "ThaiExam", "WTI - MC", "short"]:
        metrics=["IFHR", "Uncertainty", "Accuracy"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    else:
        metrics=["IFHR", "Uncertainty", "WLE", "ROUGE-1"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        metric_to_show, metric_to_use = handle_metric_name(metric)
        
        bars = ax.bar(
            positions,
            data[metric_to_use], 
            width=0.6, 
            tick_label=models,
            color=palette[:2],
        )
        
        # Add a horizontal dashed line for mllm results
        mllm_value = base_data.query(f"model == '{mllm_model}'")[metric_to_use].values[0]
        ax.axhline(y=mllm_value, color=palette[2], linestyle="--")

        # Adjust x-ticks to be centered under each bar
        ax.set_xticks(positions)
        ax.set_title(metric_to_show)

        # Set y-axis limits based on the metric
        if fix_yaxis:
            if metric == "Uncertainty":
                ax.set_ylim(0, 2.5)
            elif metric in ["IFHR", "Accuracy"]:
                ax.set_ylim(0, 1)

    # Add a legend at the bottom of the graph
    if use_legend:
        legend_elements = [
            Patch(facecolor=palette[0], label="Base"),
            Patch(facecolor=palette[1], label="CPT"),
            plt.Line2D([0], [0], color=palette[2], linestyle="--", label="MLLM"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=True,
        )

    plt.tight_layout()

    if save_result:
        base_dir = "figure"
        task = task.replace(" ", "").replace("-", "_").lower()
        save_path = f"{base_dir}/base_vs_cpt_{task}_{expr}.png"
        plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()


def plot_prompt_variation_by_metric(
    df,
    metric,
    task,
    exprs=["en_en_en", "en_en_th", "en_th_en", "en_th_th"],
    models=["llama3", "typhoon15", "qwen15", "sailor", "qwen25", "otg", "llama31", "gemma2"],
    bar_width = 0.8,
    use_legend: bool = False,
):
    # prep viz config
    sns.set_theme(style="darkgrid")

    # prep data
    data = (
        df
        .query(f"experiment in {exprs}")
        .query(f"task == '{task}'")
        .sort_values(by="model", key=lambda x: x.map({model: i for i, model in enumerate(models)}))
    )

    positions = [
        i * (len(models) + 2) + j * bar_width 
        for i in range(len(exprs)) 
        for j in range(len(models))
    ]

    metric_to_show, metric_to_use = handle_metric_name(metric)


    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot bars for each experiment
    for i, expr in enumerate(exprs):
        expr_data = data.query(f"experiment == '{expr}'")
        expr_positions = positions[i * len(models):(i + 1) * len(models)]
        ax.bar(
            expr_positions,
            expr_data[metric_to_use],
            width=bar_width,
            label=expr,
            color=palette,
        )

    # Adjust x-ticks to be centered under each group of bars
    group_centers = [
        sum(positions[i * len(models):(i + 1) * len(models)]) / len(models) 
        for i in range(len(exprs))
    ]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(exprs)

    # Add labels and title
    ax.set_ylabel(metric_to_show)
    ax.set_xlabel("Experiments")

    if use_legend:
        legend_elements = [
            Patch(facecolor=palette[i], label=models[i])
            for i in range(len(models))
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=4,
            frameon=True,
        )

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_prompt_variation(
    df,
    task,
    exprs=["en_en_en", "en_en_th", "en_th_en", "en_th_th"],
    models=["llama3", "typhoon15", "qwen15", "sailor", "qwen25", "otg", "llama31", "gemma2"],
    bar_width = 0.8,
    use_legend: bool = False,
    save_result: bool = True,
):
    # prep viz config
    sns.set_theme(style="darkgrid")

    # prep data
    data = (
        df
        .query(f"experiment in {exprs}")
        .sort_values(by="model", key=lambda x: x.map({model: i for i, model in enumerate(models)}))
    )

    if task not in ["short", "long"]:
        data = data.query(f"task == '{task}'")

    positions = [
        i * (len(models) + 2) + j * bar_width 
        for i in range(len(exprs)) 
        for j in range(len(models))
    ]

    # Create subplots for each metric
    if task in ["MMLU", "ThaiExam", "WTI - MC", "short"]:
        metrics=["IFHR", "Uncertainty", "Accuracy"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    else:
        metrics=["IFHR", "Uncertainty", "WLE", "ROUGE-1"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        metric_to_show, metric_to_use = handle_metric_name(metric)
        ax = axes[idx]

        # Plot bars for each experiment
        for i, expr in enumerate(exprs):
            expr_data = data.query(f"experiment == '{expr}'")
            expr_positions = positions[i * len(models):(i + 1) * len(models)]
            ax.bar(
                expr_positions,
                expr_data[metric_to_use],
                width=bar_width,
                label=expr,
                color=palette,
            )

        # Adjust x-ticks to be centered under each group of bars
        group_centers = [
            sum(positions[i * len(models):(i + 1) * len(models)]) / len(models) 
            for i in range(len(exprs))
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(exprs)
        ax.set_title(metric_to_show)

    # Add a legend if required
    if use_legend:
        legend_elements = [
            Patch(facecolor=palette[i], label=models[i])
            for i in range(len(models))
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=8,
            frameon=True,
        )

    # Adjust layout and show the plot
    plt.tight_layout()

    if save_result:
        base_dir = "figure"
        task = task.replace(" ", "").replace("-", "_").lower()
        inst_lang = exprs[0].split("_")[0]
        save_path = f"{base_dir}/prompt_varied_{task}_{inst_lang}.png"
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()


def plot_base_vs_cpt_kde(
    data,
    task: str,
    models: List[str],
    x: str = "Uncertainty",
    y: str = "Rouge-1",
    save_result: bool = True,
):
    sns.set_theme(style="darkgrid")
    poss_lang = ["en", "th"]

    if task in ["wti_sum", "wti_cqa"]:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        exprs = [
            f"{inst_lang}_{in_lang}_{out_lang}"
            for inst_lang in poss_lang
            for in_lang in poss_lang
            for out_lang in poss_lang
        ]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        axes = axes.reshape(1, 4)
        exprs = [
            f"{inst_lang}_{in_lang}"
            for inst_lang in poss_lang
            for in_lang in poss_lang
        ]

    metric_to_show_x, metric_to_use_x = handle_metric_name(x)
    metric_to_show_y, metric_to_use_y = handle_metric_name(y)

    for i, expr in enumerate(exprs):
        ax = axes[i//4, i%4]
        tmp_data = (
            data[task]
            .query(f"experiment == '{expr}'")
            .query(f"model in {models}")
        )
        sns.kdeplot(
            data=tmp_data,
            x=metric_to_use_x,
            y=metric_to_use_y, 
            ax=ax,
            hue="model",
            hue_order=models,
            palette=palette,
        )

        if task in ["wti_sum", "wti_cqa"]:
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0, 0.8)
        else:
            ax.set_xlim(0, 2.5)
            ax.set_ylim(0, 1)

        ax.set_title(expr)
        ax.set_xlabel(metric_to_show_x)
        ax.set_ylabel(metric_to_show_y)
        ax.legend_.remove()

    legend_elements = [
        Patch(facecolor=palette[i], label=model)
        for i, model in enumerate(models)
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(models),
        frameon=True,
    )
    plt.tight_layout()

    if save_result:
        base_dir = "figure"
        model_pair = "_".join(models)
        save_path = f"{base_dir}/kde_{task}_{model_pair}.png"
        plt.savefig(save_path, dpi=500, bbox_inches="tight")

    plt.show()


def plot_overall(df, save_result: bool = True):
    metrics = ["IFHR", "Uncertainty", "ROUGE-1 | Accuracy", "WLE"]
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for ax, metric in zip(axes.flatten(), metrics):
        if metric == "WLE":
            df = df[df["task"].isin(["WTI - CQA", "WTI - SUM"])]

        metric_to_show, metric_to_use = handle_metric_name(metric)
        sns.barplot(
            x="task", 
            y=metric_to_use,
            data=df, 
            hue="model_type", 
            errorbar=None,
            palette=palette,
            ax=ax,
        )
        ax.set_title(metric_to_show)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.get_legend().remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.tight_layout()

    if save_result:
        plt.savefig(f"figure/expr_type_results_overall.png", dpi=500, bbox_inches="tight")

    plt.show()


def plot_agg_expr(df, task_type, use_legend: bool = False, save_result: bool = True):

    sns.set_theme(style="darkgrid")
    if task_type == "short":
        metrics=["IFHR", "Uncertainty", "ROUGE-1 | Accuracy"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    else:
        metrics=["IFHR", "Uncertainty", "WLE", "ROUGE-1 | Accuracy"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes = axes.flatten()
    
    tmp_df = df.query(f"task_type == '{task_type}'")
    for ax, metric in zip(axes, metrics):
        metric_to_show, metric_to_use = handle_metric_name(metric)

        if metric == "ROUGE-1 | Accuracy":
            if task_type == "short":
                metric_to_show = "Accuracy (\u2191)"
            else:
                metric_to_show = "ROUGE-1 (\u2191)"

        sns.barplot(
            x="expr_type", 
            y=metric_to_use,
            data=tmp_df, 
            hue="model_type", 
            errorbar=None,
            palette=palette,
            order=["Pure English", "Pure Thai", "Mixed"],
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(metric_to_show)
        ax.get_legend().remove()

    if use_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.05), 
            ncol=3,
        )

    plt.tight_layout()

    if save_result:
        plt.savefig(f"figure/expr_type_results_{task_type}.png", dpi=500, bbox_inches="tight")

    plt.show()
