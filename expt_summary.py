import glob
import pandas as pd


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

        # Statistics
        sucess_rate = results["Success"].mean() * 100

        hate_score = results["Final_Score"].mean()
        hate_score_std = results["Final_Score"].std()

        num_steps = results["Num_Steps"].mean()
        num_steps_std = results["Num_Steps"].std()

        distance = results["Distance"].mean()
        distance_std = results["Distance"].std()

        distance_ratio = results["distance_ratio"].mean() * 100
        distance_ratio_std = results["distance_ratio"].std() * 100

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
    patterns = [
        "results/Mistral-7B-Instruct-v0.2_INF*.tsv",
        "results/Mistral-7B-Instruct-v0.2_10*.tsv",
        # "results/gemma-7B-Instruct-v0.2_INF*.tsv",
        # "results/gemma-7B-Instruct-v0.2_10*.tsv",
        # "results/Llama-3.1-8b_INF*.tsv",
        # "results/Llama-3.1-8b_10*.tsv",
    ]
    summary_stat = get_statistics(patterns)
    summary_stat.to_csv("exp_stat/summary_stat.csv", index=False)


if __name__ == "__main__":
    main()
