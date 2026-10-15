# CV2026

[Homework1-1](https://youtu.be/jQBcBZ859xE?si=d2F2WXn2HzMVACrh)  MediaPipe Selfie Segmentation

[Homework1-2](https://youtu.be/KaF4AkbqQrc?si=Xq0XGuHCcy_kVV27) Yolo

Homework2




import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

model1 = make_pipeline(
    StandardScaler(),
    MLPClassifier(hidden_layer_sizes=(), activation='logistic',
                  solver='lbfgs', max_iter=3000, random_state=42)
)
model1.fit(X, y)
lpy = model1.predict(X)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))

ax[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
ax[0].set_title("Original Data")

ax[1].set_title("Model 1: 1 Neuron")
ax[1].text(0.5, 0.7, "Input: 2 features", ha='center', transform=ax[1].transAxes)
ax[1].text(0.5, 0.5, "Output: 1 neuron", ha='center', transform=ax[1].transAxes)
ax[1].text(0.5, 0.3, "No hidden layer", ha='center', transform=ax[1].transAxes)
ax[1].axis('off')

DecisionBoundaryDisplay.from_estimator(
    model1, X, response_method='predict',
    cmap='coolwarm', alpha=0.35, ax=ax[2]
)
ax[2].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
ax[2].set_title("Decision Boundary")

plt.suptitle(f"Model 1: One Neuron | Accuracy: {np.mean(lpy == y) * 100:.2f}%")
plt.tight_layout()
plt.show()




