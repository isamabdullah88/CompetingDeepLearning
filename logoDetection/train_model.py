import pickle

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

dataset = np.load('../processedData/dataset5.npy').item()
print(np.array(dataset['images'][0]).shape)

images = np.array(dataset['images']).reshape(-1, 2916)
labels = np.array(dataset['labels']).reshape(-1, 1)

data_frame = np.hstack((images, labels))
np.random.shuffle(data_frame)

print('data set size: ', len(images))
percentage = 80
partition = int(len(images) * percentage / 100)
x_train, x_test = data_frame[:partition, :-1], data_frame[partition:, :-1]
y_train, y_test = data_frame[:partition, -1:].ravel(), data_frame[partition:, -1:].ravel()


def train_model():
    # clf = svm.SVC()
    # clf = RandomForestClassifier()
    clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    # clf = GaussianNB()
    # clf = MLPClassifier(hidden_layer_sizes=(1000,))

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
    print('\n')
    print(classification_report(y_test, y_pred))

    # Dumping model
    model_name = '../learnedModels/learned5_svm.sav'
    pickle.dump(clf, open(model_name, 'wb'))


train_model()
