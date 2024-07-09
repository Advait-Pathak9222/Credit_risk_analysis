import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, x_test, y_test):
    try:
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

        logging.info(f"Accuracy: {accuracy:.2f}")
        for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
            logging.info(f"Class {v}: Precision: {precision[i]}, Recall: {recall[i]}, F1 Score: {f1_score[i]}")
        return accuracy, precision, recall, f1_score
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise

if __name__ == "__main__":
    from model_building import train_random_forest, train_xgboost, train_decision_tree
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/processed/df_final.csv")
    y = df['Approved_Flag']
    x = df.drop(['Approved_Flag'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(x_train, y_train)
    xgb_model = train_xgboost(x_train, y_train)
    dt_model = train_decision_tree(x_train, y_train)

    logging.info("Evaluating Random Forest Model")
    evaluate_model(rf_model, x_test, y_test)
    logging.info("Evaluating XGBoost Model")
    evaluate_model(xgb_model, x_test, y_test)
    logging.info("Evaluating Decision Tree Model")
    evaluate_model(dt_model, x_test, y_test)
