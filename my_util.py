import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import( accuracy_score ,roc_auc_score , precision_score , recall_score, f1_score, matthews_corrcoef)


def load_and_preprocess_data(filepath='diadiabetes_data_upload.csv'):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"{filepath} doesn't exist")
    target_col = 'class'
    label_encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = label_encoder.fit_transform(df[col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2 ,random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train ,X_test, y_train, y_test
