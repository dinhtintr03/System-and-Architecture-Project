import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load data
data_dict = pickle.load(open('data_processed2.pickle', 'rb'))
data = np.asarray(data_dict['data'], dtype=object)

labels = np.asarray(data_dict['labels'], dtype=object)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize SVM model
model = SVC()

# Train the model in segments with progress bar
batch_size = 100  # Choose an appropriate batch size
for i in tqdm(range(0, len(x_train), batch_size), desc="Training Progress"):
    batch_x = x_train[i:i + batch_size]
    batch_y = y_train[i:i + batch_size]
    model.fit(batch_x, batch_y)

# Predict on test data
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f"{score*100:.2f}% accuracy")

# Save the model
with open('src/saved_weights/model2.p', 'wb') as f:
    pickle.dump({'model': model}, f)
