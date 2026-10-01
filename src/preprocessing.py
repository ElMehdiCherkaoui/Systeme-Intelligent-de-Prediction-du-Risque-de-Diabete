import os
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def load_clean_and_scale(input_path="data/raw/dataset-diabete.csv", scaler_output_path="models/scaler.joblib"):

    df = pd.read_csv(input_path)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    df = df.drop_duplicates()

    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
    
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    Q1 = df_imputed.quantile(0.25)
    Q3 = df_imputed.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df_cleaned = df_imputed.clip(lower=lower_limit, upper=upper_limit, axis=1)

    df_cleaned['Insulin_Glucose_Ratio'] = df_cleaned['Insulin'] / df_cleaned['Glucose']
    df_cleaned['BMI-Category'] = df_cleaned['BMI'].apply(lambda x: 1 if 18.5 <= x < 25 else (2 if 25 <= x < 30 else 3))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cleaned)
    df_scaled = pd.DataFrame(scaled_data, columns=df_cleaned.columns)

    os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)
    joblib.dump(scaler, scaler_output_path)
    
    return df_scaled

final_df = load_clean_and_scale("../data/raw/dataset-diabete.csv", "../models/scaler.joblib")
print("Pipeline complete. Scaled dataframe shape:", final_df.shape)