import pandas as pd
import logging
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)

def merge_data(df1, df2):
    try:
        df = pd.merge(df1, df2, how='inner', on='PROSPECTID')
        logging.info("Data merged successfully")
        return df
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def feature_selection(df):
    try:
        # Chi-square test
        categorical_columns = ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
        for col in categorical_columns:
            chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[col], df['Approved_Flag']))
            if pval > 0.05:
                df.drop(col, axis=1, inplace=True)
        
        # VIF for numerical columns
        numeric_columns = [col for col in df.columns if df[col].dtype != 'object' and col not in ['PROSPECTID','Approved_Flag']]
        vif_data = df[numeric_columns]
        columns_to_be_kept = []
        for col in numeric_columns:
            vif_value = variance_inflation_factor(vif_data.values, vif_data.columns.get_loc(col))
            if vif_value <= 6:
                columns_to_be_kept.append(col)

        df = df[columns_to_be_kept + categorical_columns + ['Approved_Flag']]
        logging.info("Feature selection completed")
        return df
    except Exception as e:
        logging.error(f"Error in feature selection: {e}")
        raise

def label_encode(df):
    try:
        df['EDUCATION'] = df['EDUCATION'].replace({'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3, 'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3})
        df = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])
        logging.info("Label encoding completed")
        return df
    except Exception as e:
        logging.error(f"Error in label encoding: {e}")
        raise

def scale_features(df):
    try:
        scaler = StandardScaler()
        columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment','max_recent_level_of_deliq','recent_level_of_deliq','time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']
        df[columns_to_be_scaled] = scaler.fit_transform(df[columns_to_be_scaled])
        logging.info("Feature scaling completed")
        return df
    except Exception as e:
        logging.error(f"Error in feature scaling: {e}")
        raise

if __name__ == "__main__":
    df1 = pd.read_csv("data/processed/df1_cleaned.csv")
    df2 = pd.read_csv("data/processed/df2_cleaned.csv")
    df = merge_data(df1, df2)
    df = feature_selection(df)
    df = label_encode(df)
    df = scale_features(df)
    df.to_csv("data/processed/df_final.csv", index=False)
