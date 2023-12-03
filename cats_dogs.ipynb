import os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage import io, transform
from sklearn.preprocessing import StandardScaler

train_path = r"F:\prodigy\cats and dogs\train"
test_path = r"F:\prodigy\cats and dogs\test"

def load_images(folder_path, target_size=(50, 50)):
    images = []
    labels = []
    for label, category in enumerate(['cats', 'dogs']):
        category_path = os.path.join(folder_path, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = io.imread(img_path)
            img = transform.resize(img, target_size)
            images.append(img.flatten())
            labels.append(label)
    return np.vstack(images), np.array(labels)

X_train, y_train = load_images(train_path)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

clf = svm.SVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

X_test, y_test = load_images(test_path)

X_test = scaler.transform(X_test)

y_test_pred = clf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
