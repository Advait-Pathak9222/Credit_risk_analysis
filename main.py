import logging

from sklearn.model_selection import train_test_split
from src.data_ingestion import load_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import merge_data, feature_selection, label_encode, scale_features
from src.model_building import train_random_forest, train_xgboost, train_decision_tree
from src.model_evaluation import evaluate_model
from src.utils import save_model

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Data Ingestion
    df1, df2 = load_data("data/raw/case_study1.xlsx", "data/raw/case_study2.xlsx")
    df1.to_csv("data/processed/df1.csv", index=False)
    df2.to_csv("data/processed/df2.csv", index=False)

    # Data Preprocessing
    df1, df2 = preprocess_data(df1, df2)
    df1.to_csv("data/processed/df1_cleaned.csv", index=False)
    df2.to_csv("data/processed/df2_cleaned.csv", index=False)

    # Feature Engineering
    df = merge_data(df1, df2)
    df = feature_selection(df)
    df = label_encode(df)
    df = scale_features(df)
    df.to_csv("data/processed/df_final.csv", index=False)

    # Model Training
    y = df['Approved_Flag']
    x = df.drop(['Approved_Flag'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(x_train, y_train)
    save_model(rf_model, "models/rf_model.joblib")

    xgb_model = train_xgboost(x_train, y_train)
    save_model(xgb_model, "models/xgb_model.joblib")

    dt_model = train_decision_tree(x_train, y_train)
    save_model(dt_model, "models/dt_model.joblib")

    # Model Evaluation
    logging.info("Evaluating Random Forest Model")
    evaluate_model(rf_model, x_test, y_test)
    logging.info("Evaluating XGBoost Model")
    evaluate_model(xgb_model, x_test, y_test)
    logging.info("Evaluating Decision Tree Model")
    evaluate_model(dt_model, x_test, y_test)
