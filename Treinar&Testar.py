from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle 
import pandas as pd

### Lendo o .CSV
df = pd.read_csv('Dataset.csv')

### Separando as colunas
x = df.drop('class', axis=1) # features
y = df['class'] # target value

### Arrumando a escala dos valores
# StdSc = StandardScaler()
# StdSc = StdSc.fit(x)
# x = StdSc.transform(x)

### Separando o conjunto de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

### Treinando o classificador
rfClassifier = RandomForestClassifier()
rfClassifier.fit(x_train, y_train)

rfPredictions = rfClassifier.predict(x_test)
rfScore = accuracy_score(y_test, rfPredictions)
print("RF Score was: ", rfScore)

### Exportando o classificador
with open('signs.pkl', 'wb') as f:
    pickle.dump(rfClassifier, f)