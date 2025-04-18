import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess():
    # Load data
    df = pd.read_csv('/opt/ml/processing/input/train.csv')
    
    # Example preprocessing: scaling features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('target', axis=1))
    
    # Save the scaler
    joblib.dump(scaler, '/opt/ml/processing/model/scaler.joblib')
    
    # Save the processed data
    processed_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    processed_df['target'] = df['target']
    processed_df.to_csv('/opt/ml/processing/output/train.csv', index=False)

if __name__ == "__main__":
    preprocess()
