import pandas as pd
import os
from naive_bayes_def import NaiveBayesDef

dataset_path = "FinalStemmedSentimentAnalysisDataset.csv"

def clear_screen():
    # Neteja la consola
    os.system('cls' if os.name == 'nt' else 'clear')

# Carregar les dades
def load_data(path, train_frac=0.8):
    print(f"\nLlegint dades des de {path}...")
    df = pd.read_csv(path, sep=';', on_bad_lines='skip')
    
    df = df[['tweetText', 'sentimentLabel']]
    df['sentimentLabel'] = pd.to_numeric(df['sentimentLabel'], errors='coerce')

    # Eliminem files amb valors nuls
    df = df.dropna(subset=['tweetText', 'sentimentLabel'])

    # Separem df per clases i les balancejem perque tinguin el mateix nombre d'exemples
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

    print("\n[📊] Distribució en Train:")
    print(train_df['sentimentLabel'].value_counts())
    print(f"Nombre d'exemples d'entrenament: {len(train_df)}")
    print(f"Nombre d'exemples de test: {len(test_df)}")

    return train_df, test_df

def print_menu():
    print("\n" + "="*50)
    print(" SENTIMENT TWITTER CLASSIFIER ")
    print("="*50)
    print("1. 🧠 Entrenar un nou model (llegeix el CSV)")
    print("2. 📂 Carregar un model guardat prèviament")
    print("3. 💬 Analitzar el meu propi Tweet")
    print("4. 🚪 Sortir")
    print("="*50)

def main():
    # Inicialitzem model amb LaPlace
    model = NaiveBayesDef(smoothing=True)
    model_loaded = False

    while True:
        print_menu()
        opcion = input("Tria una opció (1-4): ")

        if opcion == '1':
            clear_screen()
            print("--- MODE ENTRENAMENT ---")
            if not os.path.exists(dataset_path):
                print(f"Error: No es troba el fitxer {dataset_path}")
                continue
            
            # Cridem a la teva funció
            train_df, test_df = load_data(dataset_path)
            
            # Entrenem
            model.train(train_df)
            
            # Avaluem per mostrar a l'usuari que tan bo és el model que acaba de crear
            print("\ nAvaluant el model..")
            acc = model.evaluate(test_df)
            print(f"-> Accuracy global aconseguit: {acc*100:.2f}%")
            
            # Guardem la "memòria" del model a l'ordinador
            print("\n[💾] Guardant el model...")
            model.save_model("sentiment_model_professional.pkl")
            model_loaded = True

        elif opcion == '2':
            clear_screen()
            print("--- MODE CÀRREGA ---")
            model_loaded = model.load_model("sentiment_model_professional.pkl")

        elif opcion == '3':
            clear_screen()
            if not model_loaded:
                print("[⚠️] ¡Atenció! No hi ha cap model preparat.")
                print("Si us plau, entrena un model (Opció 1) o carrega'n un (Opció 2) primer.")
                continue
            
            print("\n--- MODE ANÀLISI INTERACTIU ---")
            print("Escriu 'sortir' per tornar al menú principal.")
            while True:
                user_tweet = input("\n👉 Introdueix un tweet (en anglès): ")
                
                if user_tweet.lower() == 'sortir':
                    clear_screen()
                    break
                
                if not user_tweet.strip():
                    continue

                # Passem is_raw_input=True perquè la classe netegi el tweet
                prediction = model.predict(user_tweet, is_raw_input=True)
                
                if prediction == 1:
                    print("🤖 Resultat: TWEET POSITIU 🟩")
                else:
                    print("🤖 Resultat: TWEET NEGATIU 🟥")

        elif opcion == '4':
            print("\n Sortint...\n")
            break

        else:
            print("\n[❌] Opció no vàlida. Torna-ho a intentar.")

if __name__ == "__main__":
    clear_screen()
    main()