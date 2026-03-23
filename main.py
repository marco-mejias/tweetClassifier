import pandas as pd
import numpy as np
import math
from collections import defaultdict
from naive_bayes import NaiveBayes

dataset_path = "FinalStemmedSentimentAnalysisDataset.csv"

def load_data(path, train_frac=0.8):
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    print(f"Columnes: {list(df.columns)}")


    df = df[['tweetText', 'sentimentLabel']]
    df['sentimentLabel'] = pd.to_numeric(df['sentimentLabel'], errors='coerce')

    print(f"Columnes: {list(df.columns)}")

    # Eliminem files amb valors nuls
    df = df.dropna(subset=['tweetText', 'sentimentLabel'])

    #Separem df per clases i les balancejem perque tinguin el mateix nombre d'exemples
    pos_df = df[df['sentimentLabel'] == 1]
    neg_df = df[df['sentimentLabel'] == 0]

    min_size = min(len(pos_df), len(neg_df))

    pos_df = pos_df.sample(n=min_size, random_state=42)
    neg_df = neg_df.sample(n=min_size, random_state=42)

    # Combinem les dades balancejades
    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Dividir les dades en train i test
    split_index = int(len(balanced_df) * train_frac)
    train_df = balanced_df[:split_index]  
    test_df = balanced_df[split_index:]  

    print("Distribució en Train:")
    print(train_df['sentimentLabel'].value_counts())

    print(f"Nombre d'exemples d'entrenament: {len(train_df)}")
    print(f"Nombre d'exemples de test: {len(test_df)}")

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_data(dataset_path)
    print("\n--- ENTRENEM MODEL PART C ---")

    nb = NaiveBayes()
    nb.train(train_df)
    
    # C. Validar (Accuracy global)
    print("\n--- EVALUACIÓ GLOBAL ---")
    acc = nb.evaluate(test_df)
    print(f"Accuracy en Test: {acc*100:.2f}%")