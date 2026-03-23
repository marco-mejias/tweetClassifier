import math
import pickle
import re
from collections import defaultdict, Counter
import nltk
from nltk.stem import LancasterStemmer

# Descarreguem NTLK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
}

class NaiveBayesDef:
    def __init__(self, smoothing=True): # Lo ponemos a True por defecto para el modelo final
        # Diccionaris: paraula -> {classe0: count, classe1: count}
        self.word_counts = defaultdict(lambda: {'0': 0, '1': 0})
        
        # Total de paraules per clase
        self.class_total_words = {'0': 0, '1': 0}
        
        # Priors de cada clase
        self.priors = {}
        
        # Guardem el vocabulari
        self.vocab = set()

        #Laplace smoothing
        self.smoothing = smoothing
        
        # Eina per reduir paraules a la seva arrel (per als teus propis tweets)
        self.stemmer = LancasterStemmer()

    # Funció per tuits inputs
    def preprocess_text(self, text):
        text = str(text).lower()
        # Eliminar puntuació i caràcters estranys
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        
        # Filtrar stopwords i aplicar el stemmer
        clean_words = [self.stemmer.stem(w) for w in words if w not in STOPWORDS]
        return clean_words

    def train(self, df_train, fixed_vocab=None):
        # Reiniciem estructures per si re-entrenem
        self.word_counts = defaultdict(lambda: {'0': 0, '1': 0})
        self.class_total_words = {'0': 0, '1': 0}

        # EXPERIMENT 3: Si tenim vocabulari fixe l'utilitzem, sino creem un buit
        if fixed_vocab is not None:
            self.vocab = fixed_vocab
            use_fixed = True
        else:
            self.vocab = set()
            use_fixed = False

        print("Inici train...")
        total_docs = len(df_train)
        
        # 1. Calculem priors -> Probabilitat inical de positiu/negatiu
        neg_count = len(df_train[df_train['sentimentLabel'] == 0])
        pos_count = len(df_train[df_train['sentimentLabel'] == 1])
        
        self.priors['0'] = math.log(neg_count / total_docs) if neg_count > 0 else -1e9
        self.priors['1'] = math.log(pos_count / total_docs) if pos_count > 0 else -1e9
        
        for _, row in df_train.iterrows():
            label = str(int(row['sentimentLabel']))
            text = str(row['tweetText'])
            words = text.split()
            
            #Per cada paraula, l'afegim al diccionari i sumem al camp(positiu/negatiu) corresponent 
            for word in words:
                if word.lower() in STOPWORDS:
                    continue
                
                # LOGICA EXPERIMENT 3: Si tenim vocabulari fixe, ignorem paraules que no estiguin en ell
                if use_fixed and word not in self.vocab:
                    continue
                
                # Si no es fixe, aprenem paraula
                if not use_fixed:
                    self.vocab.add(word)

                self.word_counts[word][label] += 1
                self.class_total_words[label] += 1
                
        print(f"Train complet.")
        print(f"Total: {len(self.vocab)} paraules.")

    def limit_vocab(self, top_n):
        print(f"Reduint vocabulari a {top_n} paraules...")
        total_freq = Counter()
        for word in self.vocab:
            freq = self.word_counts[word]['0'] + self.word_counts[word]['1']
            total_freq[word] = freq
            
        most_common = dict(total_freq.most_common(top_n))
        self.vocab = set(most_common.keys())
        print(f"   -> Vocabulari actualitzat: {len(self.vocab)}")

    def get_word_prob(self, word, label):
        # Utilitzem .get per evitar errors si la paraula es nova i la busquem al default dict
        count_w_c = self.word_counts.get(word, {'0': 0, '1': 0})[label]
        count_total_c = self.class_total_words[label]

        if self.smoothing:
            vocab_size = len(self.vocab)
            prob = (count_w_c + 1) / (count_total_c + vocab_size)
            return math.log(prob)
        else:
            if count_w_c == 0: 
                return -1e9 
            return math.log(count_w_c / count_total_c)
        
    def predict(self, text, is_raw_input=False):
        # Comprovem si es un tweet input del usuari o si prové del dataset
        if is_raw_input:
            words = self.preprocess_text(text)
        else:
            words = str(text).split()

        score_0 = self.priors['0']
        score_1 = self.priors['1']
        
        for word in words:
            if word.lower() in STOPWORDS:
                continue
            
            # Si estem usant Smoothing, avaluem inclús si no està al diccionari
            if word in self.vocab or self.smoothing:
                score_0 += self.get_word_prob(word, '0')
                score_1 += self.get_word_prob(word, '1')
        
        return 1 if score_1 > score_0 else 0

    def evaluate(self, df_test):
        correct = 0
        total = len(df_test)
        for _, row in df_test.iterrows():
            prediction = self.predict(row['tweetText'])
            if prediction == row['sentimentLabel']:
                correct += 1
        return correct / total

    # Funcions per carregar y descarregar el model (accelerar procés cada vegada que obrim programa)
    def save_model(self, filename="sentiment_model.pkl"):
        """Guarda la 'memòria' del model en un fitxer"""
        print(f"Guardant el model a '{filename}'...")
        with open(filename, 'wb') as f:
            pickle.dump({
                'word_counts': dict(self.word_counts), # Convertim el defaultdict a dict normal
                'class_total_words': self.class_total_words,
                'priors': self.priors,
                'vocab': self.vocab,
                'smoothing': self.smoothing
            }, f)
        print("Model guardat correctament!")

    def load_model(self, filename="sentiment_model.pkl"):
        """Carrega un model prèviament entrenat"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                # Reconstruïm el defaultdict
                self.word_counts = defaultdict(lambda: {'0': 0, '1': 0}, data['word_counts'])
                self.class_total_words = data['class_total_words']
                self.priors = data['priors']
                self.vocab = data['vocab']
                self.smoothing = data['smoothing']
            print(f"Model carregat correctament! Vocabulari: {len(self.vocab)} paraules.")
            return True
        except FileNotFoundError:
            print(f"Error: No s'ha trobat el fitxer '{filename}'. Has d'entrenar el model primer.")
            return False