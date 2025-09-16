import pandas as pd

def test_weights():
    df = pd.read_parquet("project_data/YambdaData50m/interactions_with_weights.parquet")
    print(df.head())


if __name__ == "__main__":
    test_weights()