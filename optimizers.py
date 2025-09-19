import numpy as np
from typing import Callable
Array = np.ndarray
from models import Model

#switched to forward finite 
def compute_gradient(model, loss_fn, X, Y, eps=1e-5, two_sided=False):
    theta = model.theta.copy()
    grad = np.zeros_like(theta)
    
    # Compute base loss once (saves a ton of time)
    model.set_theta(theta)
    base_loss = loss_fn(model.forward(X), Y)
    
    for i in range(theta.size):
        t_plus = theta.copy()
        t_plus[i] += eps
        model.set_theta(t_plus)
        L_plus = loss_fn(model.forward(X), Y)
        grad[i] = (L_plus - base_loss) / eps  # forward difference
    
    model.set_theta(theta)
    return grad

# def compute_gradient(model:Model, loss_fn, X, Y, eps=1e-5, two_sided = True):
#     #a model instance with parameters 
#     #chosen loss function 
#     #input data X 
#     #target data Y 
#     #output: the numerical gradient of the loss with respect to each model parameter
#     #parameters stored in model.theta 
#     theta = model.theta.copy()
#     grad = np.zeros_like(theta)
#     #eps => espilon 
#     if not two_sided:
#         model.set_theta(theta)
#         base_loss = loss_fn(model.forward(X),Y)
#     for i in range(theta.size):
#         t_plus = theta.copy()
#         t_plus[i] += eps
#         model.set_theta(t_plus)
#         L_plus = loss_fn(model.forward(X), Y)
#         if two_sided:
#             t_minus = theta.copy()
#             t_minus[i] -= eps 
#             model.set_theta(t_minus)
#             L_minus = loss_fn(model.forward(X), Y)
#             grad[i] = (L_plus - L_minus ) / (2 * eps) 
#         else:
#             grad[i] = (L_plus - base_loss) / eps
#     model.set_theta(theta) #restoring the values at the end
#     return grad  


def mean_squared_error(y_pred : Array, y_true: Array) -> float:
    #for regression
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_pred - y_true) ** 2))


def BinaryCrossEntropy(y_pred : Array, y_true :Array):
    #geeksforgeeks used to see equation for BinaryCrossEntropy syntax
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)
    #clipping to make sure we can take log 
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce 

def categorical_cross_entropy(y_pred : Array, y_true : Array, eps=1e-7):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_true = np.asarray(y_true, dtype=float)
    return float(-np.sum(y_true * np.log(y_pred)) / y_true.shape[0])



class Optimizer:
    def __init__(self, lr: float = 0.01, l1: float = 0.0, l2: float = 0.0, early_stopping_patience=None):
        self.lr = lr
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_patience = early_stopping_patience
        self.val_loss_history = []

        
    def step(self, model: Model, grad: Array) -> None:
        raise NotImplementedError
    #regularization function from class
    def _apply_regularization(self, grad: Array, theta: Array) -> Array:
        if self.l1 > 0:
            grad += self.l1 * np.sign(theta)
        if self.l2 > 0:
            grad += self.l2 * theta
        return grad
    #asked chatgpt to help code early stopping functionality manually 
    def check_early_stopping(self, val_loss: float) -> bool:
        if self.early_stopping_patience is None:
            return False
        self.val_loss_history.append(val_loss)
        if len(self.val_loss_history) > self.early_stopping_patience:
            recent = self.val_loss_history[-self.early_stopping_patience:]
            if recent[-1] > np.mean(recent[:-1]):
                return True
        return False

class SGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.0, **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        self.momentum = momentum
        self.v = None 

    def step(self, model: Model, grad: Array) -> None:
        theta = model.theta
        grad = self._apply_regularization(grad, theta)
        if self.momentum > 0:
            if self.v is None:
                self.v = np.zeros_like(theta)
            self.v = self.momentum * self.v - self.lr * grad
            theta_new = theta + self.v
        else:
            theta_new = theta - self.lr * grad
        model.set_theta(theta_new)


class RMSprop(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8, **kwargs) -> None:
        super().__init__(lr=lr, **kwargs)
        self.beta = beta
        self.eps = eps
        self.sq_grad_avg = None


    def step(self, model: Model, grad: Array) -> None:
        theta = model.theta
        grad = self._apply_regularization(grad, theta)
        if self.sq_grad_avg is None:
            self.sq_grad_avg = np.zeros_like(theta)
        self.sq_grad_avg = self.beta * self.sq_grad_avg + (1 - self.beta) * grad ** 2
        theta_new = theta - self.lr * grad / (np.sqrt(self.sq_grad_avg) + self.eps)
        model.set_theta(theta_new)

#function template from class notes
def train(model: Model, optimizer: Optimizer, loss_fn: Callable, X_train: Array, y_train: Array, X_val: Array = None, y_val: Array = None, max_iters: int = 100, eps: float = 1e-5, two_sided: bool = True, verbose: bool = True,):
    train_losses, val_losses = [], []
    for it in range(max_iters):
        grad = compute_gradient(model, loss_fn, X_train, y_train, eps, two_sided)
        optimizer.step(model,grad) #from class code 
        y_pred = model.forward(X_train)
        train_loss = loss_fn(y_pred, y_train)
        train_losses.append(train_loss)
        if X_train is not None and y_val is not None:
            y_val_pred = model.forward(X_val)
            val_loss = loss_fn(y_val_pred, y_val)
            val_losses.append(val_loss)
            # Check for Early stopping
            if optimizer.check_early_stopping(val_loss):
                if verbose:
                    print(f"Early stopping at iteration {it}, val_loss={val_loss:.4f}")
                break
        if verbose and (it % 10 == 0 or it == max_iters -1):
            msg = f"Iter {it:03d} | Train Loss: {train_loss:.6f}"
            if val_losses:
                msg += f" | Val Loss: {val_losses[-1]:.6f}"
            print(msg)
    return train_losses, val_losses

