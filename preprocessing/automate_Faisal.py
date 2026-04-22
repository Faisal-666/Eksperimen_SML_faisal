from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import os
from utils import zero_to_nan

DATA_PATH = 'diabetes_raw.csv'

os.makedirs('preprocessing/diabetes_preprocessing', exist_ok=True)

TRAIN_PATH = 'preprocessing/diabetes_preprocessing/train.csv'
TEST_PATH = 'preprocessing/diabetes_preprocessing/test.csv'
PREPROCESSOR_PATH = 'preprocessing/diabetes_preprocessing/preprocessor.joblib'

cols = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

def run():
    data = pd.read_csv(DATA_PATH)

    x = data.drop('Outcome', axis=1)
    y = data['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    cols_pipeline = Pipeline(steps=[
        ('zero_to_nan', FunctionTransformer(zero_to_nan, feature_names_out='one-to-one')),
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cols_proc', cols_pipeline, cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    x_train_processed = full_pipeline.fit_transform(x_train)
    x_test_processed = full_pipeline.transform(x_test)

    feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()

    x_train_processed = pd.DataFrame(x_train_processed, columns=feature_names)
    x_test_processed = pd.DataFrame(x_test_processed, columns=feature_names)

    train_df = x_train_processed.copy()
    train_df['Outcome'] = y_train.values

    test_df = x_test_processed.copy()
    test_df['Outcome'] = y_test.values

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    joblib.dump(full_pipeline, PREPROCESSOR_PATH)

if __name__ == '__main__':
    run()