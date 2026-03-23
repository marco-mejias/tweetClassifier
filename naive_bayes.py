import math
from collections import defaultdict

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

class NaiveBayes:
    def __init__(self):
        # Diccionaris: paraula -> {classe0: count, classe1: count}
        self.word_counts = defaultdict(lambda: {'0': 0, '1': 0})
        
        # Total de paraules per clase
        self.class_total_words = {'0': 0, '1': 0}
        
        # Priors de cada clase
        self.priors = {}
        
        # Guardem el vocabulari
        self.vocab = set()

    def train(self, df_train):

        print("Inici train...")
        
        # 1. Calculem priors -> Probabilitat inical de positiu/negatiu
        neg_count = len(df_train[df_train['sentimentLabel'] == 0])
        pos_count = len(df_train[df_train['sentimentLabel'] == 1])
        
        self.priors['0'] = math.log(neg_count / len(df_train))
        self.priors['1'] = math.log(pos_count / len(df_train))
        
        # 2. Omplim diccionari (Bag of Words)
        for _, row in df_train.iterrows():
            label = str(int(row['sentimentLabel']))
            text = str(row['tweetText'])
            words = text.split()
            
            #Per cada paraula, l'afegim al diccionari i sumem al camp(positiu/negatiu) corresponent 
            for word in words:
                if word.lower() in STOPWORDS:
                    continue
                self.vocab.add(word)
                self.word_counts[word][label] += 1
                self.class_total_words[label] += 1
                
        print(f"Train complet.")
        print(f"Total: {len(self.vocab)} paraules en el diccionari.")

    def get_word_prob(self, word, label):
        #Calculem probabilitat davant una clase -> Ex: P('hola'|positiu)

        # Nombre de vegades que la paraula ha aparegut en la clase
        count_w_c = self.word_counts[word][label]
        # Nombre total de paraules en la clase
        count_total_c = self.class_total_words[label]
        
        # Si la paraula no ha aparegut mai en la clase (positiva o negativa), retornem un valor molt baix per evitar problemes de log(0)
        if count_w_c == 0:
            return -1e9 
        
        return math.log(count_w_c / count_total_c)
    
    def predict(self, text):
       # Classifiquem un tweet donat
        words = str(text).split()

        #Calculem priors
        score_0 = self.priors['0']
        score_1 = self.priors['1']
        
        #Agafem cada paraula, mirem cuantes vegades a aparegut en cada clase i sumem al score
        for word in words:
            #NOTA: Això es posiblament redundant, ja que les paraules en el STOPWORD no s'afegeixen al vocab, pero potser es mes eficient
            if word.lower() in STOPWORDS:
                continue
            # Evaluem només les paraules conegudes
            if word in self.vocab:
                score_0 += self.get_word_prob(word, '0')
                score_1 += self.get_word_prob(word, '1')

        #Si el conjunt de paraules son totes desconegudes, retornariem 0
        # Gana la classe amb més score
        return 1 if score_1 > score_0 else 0

    def evaluate(self, df_test):
        # Calculam l'accuracy sobre el conjunt de test
        correct = 0
        total = len(df_test)
        
        for _, row in df_test.iterrows():
            prediction = self.predict(row['tweetText'])
            if prediction == row['sentimentLabel']:
                correct += 1
        
        return correct / total