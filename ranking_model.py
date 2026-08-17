
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_ranker(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, path='model.joblib'):
    joblib.dump(model, path)

def load_model(path='model.joblib'):
    return joblib.load(path)
