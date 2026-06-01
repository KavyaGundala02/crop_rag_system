# ingestion/load_data.py

import pandas as pd
import os

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the crop recommendation CSV dataset."""
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    return df

def explore_dataset(df: pd.DataFrame):
    """Print basic info about the dataset."""
    
    print("="*50)
    print("DATASET    OVERVIEW")
    print("="*50)
    
    print(f"\n Total rows    : {len(df)}")
    print(f" Total columns : {len(df.columns)}")
    
    print(f"\n Columns: {list(df.columns)}")
    
    print(f"\n Unique crops ({df['label'].nunique()}):")
    print(df['label'].unique())
    
    print(f"\n Sample rows:")
    print(df.head(3))
    
    print(f"\n Basic statistics:")
    print(df.describe())
    
    print(f"\n Any missing values?")
    print(df.isnull().sum())

if __name__ == "__main__":
    df = load_dataset("data/Crop_recommendation.csv")
    explore_dataset(df)