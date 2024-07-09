import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)

def train_random_forest(x_train, y_train):
    try:
        rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_classifier.fit(x_train, y_train)
        logging.info("Random Forest model trained")
        return rf_classifier
    except Exception as e:
        logging.error(f"Error training Random Forest model: {e}")
        raise

def train_xgboost(x_train, y_train):
    try:
        xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)
        xgb_classifier.fit(x_train, y_train)
        logging.info("XGBoost model trained")
        return xgb_classifier
    except Exception as e:
        logging.error(f"Error training XGBoost model: {e}")
        raise

def train_decision_tree(x_train, y_train):
    try:
        dt_classifier = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
        dt_classifier.fit(x_train, y_train)
        logging.info("Decision Tree model trained")
        return dt_classifier
    except Exception as e:
        logging.error(f"Error training Decision Tree model: {e}")
        raise

if __name__ == "__main__":
    df = pd.read_csv("data/processed/df_final.csv")
    y = df['Approved_Flag']
    x = df.drop(['Approved_Flag'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(x_train, y_train)
    xgb_model = train_xgboost(x_train, y_train)
    dt_model = train_decision_tree(x_train, y_train)
