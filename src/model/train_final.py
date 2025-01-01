# import pickle
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load data
# data_dict = pickle.load(open('final_data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Use the best hyperparameters directly
# best_params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}

# # Create an SVM classifier with the best parameters and verbose set to True
# svm_model = SVC(**best_params, verbose=True)

# # Train the SVM model
# svm_model.fit(x_train, y_train)

# # Evaluate the model on the test set
# y_predict = svm_model.predict(x_test)
# score = accuracy_score(y_predict, y_test)
# print('{}% of samples were classified correctly!'.format(score * 100))

# # Save the best SVM model 
# with open('model_final.p', 'wb') as f:
#     pickle.dump({'model': svm_model}, f)
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load data
data_dict = pickle.load(open('final_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Use the best hyperparameters directly
best_params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}

# Create an SVM classifier with the best parameters
svm_model = SVC(**best_params, verbose=True)

# Train the SVM model
svm_model.fit(x_train, y_train)

# Predict using the model on the test set
y_predict = svm_model.predict(x_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')  # 'weighted' handles multi-class
recall = recall_score(y_test, y_predict, average='weighted')  # 'weighted' handles multi-class
f1 = f1_score(y_test, y_predict, average='weighted')  # 'weighted' handles multi-class

# Print out the evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')

# Save the best SVM model
with open('model_final.p', 'wb') as f:
    pickle.dump({'model': svm_model}, f)
