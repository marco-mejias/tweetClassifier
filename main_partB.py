import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from naive_bayes_partB import NaiveBayes

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

# --- EXPERIMENT 1: Mida del Train ---
def experiment_train_size(full_train_df, test_df):
    print("\n--- EXPERIMENT 1: Variant mida del Train ---")
    fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
    accuracies = []

    for frac in fractions:
        # Agafem una mostra del train segons la fracció
        subset_train = full_train_df.sample(frac=frac, random_state=42)
        
        nb = NaiveBayes()
        nb.train(subset_train)
        acc = nb.evaluate(test_df)
        accuracies.append(acc)
        print(f"Fracció: {frac*100}% | Accuracy: {acc:.4f}")

    # La idea es veure que amb menys dades d'entrenament, pitjor accuracy. Pero començara a estabilitzar-se.
    # Plot
    plt.figure()
    plt.plot([f*100 for f in fractions], accuracies, marker='o')
    plt.title('Accuracy vs Mida del Train (%)')
    plt.xlabel('% Dades Train')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('resultat_exp1_train.png')
    print("Gràfica guardada: resultat_exp1_train.png")

# --- EXPERIMENT 2: Mida del Vocabulari ---
def experiment_vocab_size(train_df, test_df):
    print("\n--- EXPERIMENT 2: Variant mida del Vocabulari ---")
    
    # Entrenem UNA vegada amb tot
    nb_full = NaiveBayes()
    nb_full.train(train_df)
    
    # Guardem el vocabulari original complet per restaurar-lo
    original_vocab = nb_full.vocab.copy()
    max_vocab = len(original_vocab)
    
    sizes = [1000, 5000, 10000, 50000, 100000, 200000, 300000, max_vocab]
    accuracies = []

    for size in sizes:
        # Restaurem el vocab original abans de retallar
        nb_full.vocab = original_vocab.copy()
        
        if size < max_vocab:
            nb_full.limit_vocab(size)
            
        acc = nb_full.evaluate(test_df)
        accuracies.append(acc)
        print(f"Vocabulari: {size} paraules | Accuracy: {acc:.4f}")

    # Plot
    plt.figure()
    plt.plot(sizes, accuracies, marker='o', color='r')
    plt.title('Accuracy vs Mida del Vocabulari')
    plt.xlabel('Num Paraules')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('resultat_exp2_vocab.png')
    print("Gràfica guardada: resultat_exp2_vocab.png")

# --- EXPERIMENT 3: Vocabulari FIX vs Train VARIABLE ---
def experiment_fixed_vocab(full_train_df, test_df):
    print("\n--- EXPERIMENT 3: Vocabulari fix i train variable ---")
    
    # 1. Creem diccionari fixe
    print(" Vocabulari fixe: Top 50.000...")
    nb_temp = NaiveBayes()
    nb_temp.train(full_train_df)
    nb_temp.limit_vocab(50000) # Fixem tamany. Agafem 50.000 perque es la mesura que dona millor resultats en el test 2
    
    # Fem copia vocab per anar pasant a models
    FIXED_VOCAB = nb_temp.vocab.copy()
    
    fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
    accuracies = []

    for frac in fractions:
        # Agafem part de les dades i l'entrenem. EN RESUM: En aquest apartar fem un train NOMÉS per agafar el diccionari TOP 50K y després fem els splits de train normals
        subset_train = full_train_df.sample(frac=frac, random_state=42)
        
        nb = NaiveBayes()
        nb.train(subset_train, fixed_vocab=FIXED_VOCAB)
        
        acc = nb.evaluate(test_df)
        accuracies.append(acc)
        print(f"Fracció: {frac*100}% | Vocab: {len(nb.vocab)} (Fix) | Accuracy: {acc:.4f}")

    # Gráfica
    plt.figure()
    plt.plot([f*100 for f in fractions], accuracies, marker='o', color='g')
    plt.title('Vocabulari fix i train variable')
    plt.xlabel('% Dades Train')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('resultat_exp3_fixed.png')
    print("Gràfica guardada: resultat_exp3_fixed.png")

if __name__ == "__main__":
    # 1. Cargar Dades
    train_df, test_df = load_data(dataset_path)
    
    # 2. Executar Experiments Part B
    #experiment_train_size(train_df, test_df)
    #experiment_vocab_size(train_df, test_df)
    experiment_fixed_vocab(train_df, test_df)