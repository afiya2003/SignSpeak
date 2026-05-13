import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("model/cnn_model.keras")
labels = np.load("model/labels.npy")

X_test, y_test = [], []

for i, label in enumerate(labels):
    data = np.load(f"signs/sign_{label}.npy")
    for sample in data:
        if len(sample) == 63:
            sample = np.concatenate([sample, np.zeros(63)])
        X_test.append(sample)
        y_test.append(i)

X_test = np.array(X_test).reshape(len(X_test), 126, 1)
y_test = np.array(y_test)

preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%")
