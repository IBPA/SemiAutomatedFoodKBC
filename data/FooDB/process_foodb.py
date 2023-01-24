from pathlib import Path
import pandas as pd


def main():
    food_filepath = "./Food.csv"
    output_filepath = "./foodb_foods.txt"

    df = pd.read_csv(food_filepath)
    df.dropna(subset="ncbi_taxonomy_id", inplace=True)
    df = df[["name", "name_scientific", "ncbi_taxonomy_id"]]
    df["ncbi_taxonomy_id"] = df["ncbi_taxonomy_id"].astype(int)
    df.drop_duplicates(inplace=True)
    df.to_csv(output_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
