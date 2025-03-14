import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def configure_plots() -> None:
    plt.rc("font", family="serif")
    # Setting for matplotlib
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 14,
        }
    )


def load_allowed_duplicates():
    # Get all allowed duplicate_results in the original dataset
    raw_data = load_results("data-splitted/*.tsv")
    raw_data.columns = ["code", "sample"]
    allowed_duplicates = raw_data[
        raw_data["sample"].str.strip().duplicated(keep="first")
    ]["sample"].to_list()
    return allowed_duplicates


def load_results(pattern: str):
    # Combine all tsv files to a single dataframe
    files = glob.glob(pattern)
    df = pd.concat([pd.read_csv(file, sep="\t") for file in files], ignore_index=True)
    return df


def get_statistics(patterns: list[str]):
    allowed_duplicates = load_allowed_duplicates()

    df_stat = pd.DataFrame(
        columns=[
            "Model",
            "Max Change",
            "Success Rate",
            "Hate Score",
            "Num Steps",
            "Distance",
            "Distance Ratio",
        ]
    )

    for pattern in patterns:
        results = load_results(pattern)

        # Calcaulte distance ratio
        results["distance_ratio"] = 1 - (
            results["Distance"]
            / (
                len(results["Initial_Sample"].str.strip().values)
                + len(results["Final_Sample"].str.strip())
            )
        )

        # Handle duplicates
        duplicate_results = results[
            results["Initial_Sample"].str.strip().duplicated(keep="first")
        ]
        not_allowed_list = set(duplicate_results["Initial_Sample"].str.strip()) - set(
            allowed_duplicates
        )
        # Drop duplicates not allowed but keep the first one for each instance of duplicates
        mask = results["Initial_Sample"].str.strip().isin(not_allowed_list)
        results = pd.concat(
            [
                results[~mask],
                results[mask].drop_duplicates(subset=["Initial_Sample"], keep="first"),
            ]
        )
        results = results.reset_index(drop=True)

        name = pattern.split("/")[1]
        model = name.split("_")[0]
        max_change = name.replace("*", "").split(".tsv")[0].split("_")[-1]

        # Plots
        # Distance is normalized between [50, 250] to keep dots visible but not too large
        distance_size = (results["Distance"] - results["Distance"].min()) / (
            results["Distance"].max() - results["Distance"].min()
        ) * 200 + 50
        distance_size = distance_size.astype(np.float64)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            data=results,
            x="Num_Steps",
            y="Final_Score",
            c=1-results["Success"],
            cmap="bwr",  # Blue for success, red for fail
            s=distance_size,  # size of the dot
            edgecolor="k",
            alpha=0.5,
        )
        plt.xlabel("Number of Steps")
        plt.ylabel("Hate Score")
        plt.title(f"{model} - Max distance: {max_change}")

        # Manually create legend handles
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Success",
                markerfacecolor="blue",
                markersize=8,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Fail",
                markerfacecolor="red",
                markersize=8,
            ),
        ]

        # Set the legend with proper handles
        plt.legend(
            handles=legend_elements,
            title="Adversarial classification",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
        )
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f"exp_stat/{model}_{max_change}_scatter.png")

        # Statistics
        sucess_rate = results["Success"].mean() * 100

        result_success = results[results["Success"] == 1]
        hate_score = result_success["Final_Score"].mean()
        hate_score_std = result_success["Final_Score"].std()

        num_steps = result_success["Num_Steps"].mean()
        num_steps_std = result_success["Num_Steps"].std()

        distance = result_success["Distance"].mean()
        distance_std = result_success["Distance"].std()

        distance_ratio = result_success["distance_ratio"].mean() * 100
        distance_ratio_std = result_success["distance_ratio"].std() * 100

        # Create a dictionary with the statistics to save to dataframe
        stats = {
            "Model": model,
            "Max Change": max_change,
            "Success Rate": f"{round(sucess_rate, 2)}",
            "Hate Score": f"{round(hate_score, 2)} ± {round(hate_score_std, 2)}",
            "Num Steps": f"{round(num_steps, 2)} ± {round(num_steps_std, 2)}",
            "Distance": f"{round(distance, 2)} ± {round(distance_std, 2)}",
            "Distance Ratio": f"{round(distance_ratio, 2)} ± {round(distance_ratio_std, 2)}",
        }

        df_stat = pd.concat([df_stat, pd.DataFrame([stats])])

    return df_stat


def main():
    configure_plots()
    patterns = [
        "results/Mistral-7B-Instruct-v0.2_INF*.tsv",
        "results/Mistral-7B-Instruct-v0.2_10*.tsv",
        "results/gemma-2b-it_INF*.tsv",
        # "results/gemma-2b-it_10*.tsv",
        # "results/Llama-3.1-8b_INF*.tsv",
        # "results/Llama-3.1-8b_10*.tsv",
    ]
    summary_stat = get_statistics(patterns)
    summary_stat.to_csv("exp_stat/summary_stat.csv", index=False)


if __name__ == "__main__":
    main()
