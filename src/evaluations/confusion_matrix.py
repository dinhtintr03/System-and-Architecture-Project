# import pickle
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load data
# data_dict = pickle.load(open('final_data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Use the best hyperparameters directly
# best_params = {'C': 100, 'gamma': 1, 'kernel': 'rbf'}

# # Create an SVM classifier with the best parameters
# svm_model = SVC(**best_params)

# # Train the SVM model
# svm_model.fit(x_train, y_train)

# # Evaluate the model on the test set
# y_predict = svm_model.predict(x_test)
# score = accuracy_score(y_predict, y_test)
# print('{}% of samples were classified correctly!'.format(score * 100))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_predict)

# # Visualize the Confusion Matrix using Seaborn heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# # Save the model
# with open('src\\saved_weights\\model_test.p', 'wb') as f:
#     pickle.dump({'model': svm_model}, f)


import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Tạo từ điển ánh xạ số sang chữ
label_map = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'I', 7: 'K', 
    8: 'L', 9: 'O', 10: 'U', 11: 'V', 12: 'W', 13: 'Y'
}

# Load model and test data
with open('src\\saved_weights\\model_final.p', 'rb') as f:
    model_data = pickle.load(f)
svm_model = model_data['model']

# Load your test data (assuming you have already split into X_test and y_test)
data_dict = pickle.load(open('final_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Example of splitting the data (replace this with your actual test set)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Predict the labels using the trained model
y_predict = svm_model.predict(x_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_predict)

# Convert the confusion matrix to label names
cm_labels = np.array([label_map[i] for i in range(len(label_map))])

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print classification report
print(classification_report(y_test, y_predict, target_names=[label_map[i] for i in range(len(label_map))]))
