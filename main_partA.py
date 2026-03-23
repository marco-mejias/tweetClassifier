import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from naive_bayes_partA import NaiveBayes

dataset_path = "FinalStemmedSentimentAnalysisDataset.csv"

def load_data(path, train_frac=0.8):
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    except:
        df = pd.read_csv(path, sep=';', error_bad_lines=False)
        
    df = df[['tweetText', 'sentimentLabel']]
    df['sentimentLabel'] = pd.to_numeric(df['sentimentLabel'], errors='coerce')
    df = df.dropna(subset=['tweetText', 'sentimentLabel'])

    pos_df = df[df['sentimentLabel'] == 1]
    neg_df = df[df['sentimentLabel'] == 0]

    min_size = min(len(pos_df), len(neg_df))

    pos_df = pos_df.sample(n=min_size, random_state=42)
    neg_df = neg_df.sample(n=min_size, random_state=42)

    balanced_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(balanced_df) * train_frac)
    train_df = balanced_df[:split_index]  
    test_df = balanced_df[split_index:]  

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df, test_df

# --- EXPERIMENT PART A: Laplace Smoothing ---
def run_part_a_experiments(train_df, test_df):

    print("PART A: LAPLACE SMOOTHING")
    nb_smooth = NaiveBayes(smoothing=True) # <--- ACTIVEM LAPLACE

    # 1. Comparativa bàsica amb tot el vocabulari
    print("Entrenament amb Laplace Smoothing...")
    nb_smooth.train(train_df)
    acc = nb_smooth.evaluate(test_df)
    print(f"  Accuracy AMB Laplace: {acc:.4f}")
    
    # 2. Re-execució de l'experiment de Vocabulari Part B (per veure la diferència)
    print("\n-> Repetint Experiment Vocabulari Amb Laplace...")
    original_vocab = nb_smooth.vocab.copy()
    max_vocab = len(original_vocab)
    sizes = [1000, 5000, 10000, 50000, max_vocab]
    accuracies = []

    for size in sizes:
        nb_smooth.vocab = original_vocab.copy()
        if size < max_vocab:
            nb_smooth.limit_vocab(size)
        
        acc = nb_smooth.evaluate(test_df)
        accuracies.append(acc)
        print(f"   Vocabulari: {size} | Accuracy: {acc:.4f}")

    # Gràfica
    plt.figure()
    plt.plot(sizes, accuracies, marker='o', color='purple')
    plt.title('Part A: Accuracy vs Vocab Size (Amb Laplace)')
    plt.xlabel('Num Paraules')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('resultat_part_a_laplace.png')
    print("Gràfica guardada: resultat_part_a_laplace.png")

if __name__ == "__main__":
    train_df, test_df = load_data(dataset_path)
    run_part_a_experiments(train_df, test_df)