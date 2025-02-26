import pandas as pd
import os


def split_tsv(input_file: str, output_dir: str, num_splits: int):
    """Splits the input TSV file into multiple TSV files."""
    df = pd.read_csv(input_file, sep="\t")
    split_size = len(df) // num_splits
    for i in range(num_splits):
        split = df.iloc[i * split_size : (i + 1) * split_size]
        file_name = f"split_{i + 1}.tsv"
        split.to_csv(os.path.join(output_dir, file_name), sep="\t", index=False)
        print("Split saved to: ", os.path.join(output_dir, file_name))


if __name__ == "__main__":
    split_tsv(
        input_file="./data/hate_speech_samples.tsv",
        output_dir="./data-splitted",
        num_splits=8,
    )
