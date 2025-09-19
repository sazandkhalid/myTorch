# main.py
import numpy as np
from models import LinearModel, LogisticRegression, DenseFeedForward
from optimizers import mean_squared_error, BinaryCrossEntropy, SGD, RMSprop, train

# # Tiny toy dataset (4 samples, 2 features)
X_toy = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_toy = np.array([0,1,1,0], dtype=np.float32)

toy_model = LogisticRegression(input_dim=2)
toy_optimizer = SGD(lr=0.1, momentum=0.0, early_stopping_patience=3)

train_losses, val_losses = train(
    toy_model,
    toy_optimizer,
    BinaryCrossEntropy,
    X_toy, y_toy,
    X_toy, y_toy,       # use same data for val
    max_iters=5,
    two_sided=True
)
print("Grad test finished ✅")
