import logging

logging.basicConfig(level=logging.INFO)

def save_model(model, model_path):
    import joblib
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def load_model(model_path):
    import joblib
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
