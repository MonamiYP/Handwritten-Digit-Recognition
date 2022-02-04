import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("train.csv")
# Turn into numpy array
data = data.to_numpy()

# Create an object of the decision tree model
classifier = DecisionTreeClassifier()

X_train = data[0:21000, 1:] # I want the first 21000 rows of data, without the first column (label)
X_train_label = data[0:21000, 0] # Get the labels column for the first 21000 data

classifier.fit(X_train, X_train_label) # Create the decision tree

# Test data
X_test = data[21000:, 1:] # Get the test data info for the remaining rows
true_label = data[21000:, 0] # Get the actual number thast represented by the test points

# Just to see what it's like:
d = X_test[8] # Get a random data row
d.shape = (28, 28) # Turn it into a matrix (28, 28)
print(classifier.predict( [X_test[8]] )) # What does the model predict? (SHould be 3)
plt.imshow(255 - d, cmap='gray') # Show data, the number is 3
plt.show()

# Measureing accuracy of model:
prediction = classifier.predict(X_test)

count = 0
for i in range(0, 21000):
    count += 1 if prediction[i] == true_label[i] else 0

print("Total correct answers:", count)
print("Accuracy:", (count/21000) * 100)