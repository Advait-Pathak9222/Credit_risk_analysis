import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

def load_data(file_path1, file_path2):
    try:
        a1 = pd.read_excel(file_path1)
        a2 = pd.read_excel(file_path2)
        logging.info("Data loaded successfully")
        return a1, a2
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    df1, df2 = load_data("data/raw/case_study1.xlsx", "data/raw/case_study2.xlsx")
