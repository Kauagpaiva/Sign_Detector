from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle 
import pandas as pd

### Reading the .CSV
df = pd.read_csv('Dataset.csv')

### Separating the columns
x = df.drop('class', axis=1) # features
y = df['class'] # target value

### Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

### Training the classifier
rfClassifier = RandomForestClassifier()
rfClassifier.fit(x_train, y_train)

### Testing the classifier
rfPredictions = rfClassifier.predict(x_test)
rfScore = accuracy_score(y_test, rfPredictions)
print("RF Score was: ", rfScore)

### Exporting the classifier
with open('signs.pkl', 'wb') as f:
    pickle.dump(rfClassifier, f)
