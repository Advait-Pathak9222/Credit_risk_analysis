import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df1, df2):
    try:
        # Remove nulls
        df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

        # Drop columns with too many nulls
        columns_to_be_removed = []
        for col in df2.columns:
            if df2[df2[col] == -99999].shape[0] > 10000:
                columns_to_be_removed.append(col)
        df2 = df2.drop(columns_to_be_removed, axis=1)

        # Remove rows with -99999
        for col in df2.columns:
            df2 = df2.loc[df2[col] != -99999]
        
        logging.info("Data preprocessing completed")
        return df1, df2
    except Exception as e:
        logging.error(f"Error in preprocessing data: {e}")
        raise

if __name__ == "__main__":
    df1, df2 = pd.read_csv("data/processed/df1.csv"), pd.read_csv("data/processed/df2.csv")
    df1, df2 = preprocess_data(df1, df2)
    df1.to_csv("data/processed/df1_cleaned.csv", index=False)
    df2.to_csv("data/processed/df2_cleaned.csv", index=False)
