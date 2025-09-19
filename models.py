from typing import Sequence
import numpy as np
Array = np.ndarray

def sigmoid(z: Array) -> Array:
    z_clipped = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def relu(z: Array) -> Array:
    return np.maximum(0.0, z)

def softmax(z: Array) -> Array:
    z = np.clip(z, -30, 30)  # avoid overflow
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

ACTIVATIONS = {
    "identity": lambda x: x,
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
}

class Model:
    def __init__(self, input_dim : int, output_dim : int):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
    def forward(self,x):
        #what model should we return? depends on output_dim
        raise NotImplementedError
    @property
    def theta(self) -> Array:
        return self._params_to_flat()
    def set_theta(self, theta: Array) -> None:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        self._flat_to_params(theta)
    def num_params(self) -> int:
        return int(self.theta.size)
    #these are only placeholders because they are implemented by the subclasses
    def _params_to_flat(self) -> Array:
        raise NotImplementedError
    def _flat_to_params(self, theta: Array) -> None: 
        raise NotImplementedError

class LinearModel(Model):
    def __init__(self, input_dim :int, output_dim : int=1):
        super().__init__(input_dim, output_dim)
        rng = np.random.default_rng()
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.W = rng.uniform(-limit, limit, size=(input_dim, output_dim))
        self.b = np.zeros((output_dim,), dtype=float)

    def forward(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        assert x.shape[1] == self.input_dim, f"Expected input dim {self.input_dim}, got {x.shape[1]}"
        return x @ self.W + self.b #linear equation 
      
    def _params_to_flat(self) -> Array:
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def _flat_to_params(self, theta: Array) -> None:
        nW = self.input_dim * self.output_dim
        self.W = theta[:nW].reshape(self.input_dim, self.output_dim)
        self.b = theta[nW:nW + self.output_dim].reshape(self.output_dim)

class LogisticRegression(Model):
    def __init__(self, input_dim:int):
        super().__init__(input_dim,output_dim=1)
        rng = np.random.default_rng()
        limit = np.sqrt(6.0 / (input_dim + 1))
        self.W = rng.uniform(-limit, limit, size=(input_dim, 1))
        self.b = np.zeros((1,), dtype=float)

    def forward(self, x:Array) -> Array:
        x = np.asanyarray(x,dtype=float)
        if x.ndim ==1:
            x = x.reshape(1,-1)
        z = x @ self.W + self.b
        return sigmoid(z)
    
    def _params_to_flat(self) -> Array:
        return np.concatenate([self.W.ravel(), self.b.ravel()])


    def _flat_to_params(self, theta: Array) -> None:
        nW = self.input_dim * 1
        self.W = theta[:nW].reshape(self.input_dim, 1)
        self.b = theta[nW:nW + 1].reshape(1) 


class DenseFeedForward(Model):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        output_dim: int,
        *,
        hidden_activation: str = "relu",
        output_activation: str = "identity",
        rng: np.random.Generator = None,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden_layers = list(hidden_layers)
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation
        self.f_hidden = ACTIVATIONS[hidden_activation]
        self.f_out = ACTIVATIONS[output_activation]
        self.rng = rng or np.random.default_rng()

        # Layer sizes [input -> hidden(s) -> output]
        sizes = [input_dim] + self.hidden_layers + [output_dim]
        self.W, self.B = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He init for ReLU, Xavier otherwise, from class notes
            if hidden_activation == "relu":
                limit = np.sqrt(6.0 / fan_in)
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W.append(self.rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.B.append(np.zeros((fan_out,), dtype=float))

    def forward(self, x: Array) -> Array:
        h = np.asarray(x, dtype=float)
        if h.ndim == 1:
            h = h.reshape(1, -1)

        for i in range(len(self.hidden_layers)):
            h = self.f_hidden(h @ self.W[i] + self.B[i])
        # output layer
        h = self.f_out(h @ self.W[-1] + self.B[-1])
        return h

    def _params_to_flat(self) -> Array:
        vecs = [w.ravel() for w in self.W] + [b.ravel() for b in self.B]
        return np.concatenate(vecs)

    def _flat_to_params(self, theta: Array) -> None:
        theta = np.asarray(theta, dtype=float).ravel()
        sizes = [self.input_dim] + self.hidden_layers + [self.output_dim]
        offset = 0
        new_W, new_B = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            nW = fan_in * fan_out
            new_W.append(theta[offset:offset + nW].reshape(fan_in, fan_out))
            offset += nW
        for fan_out in sizes[1:]:
            new_B.append(theta[offset:offset + fan_out].reshape(fan_out))
            offset += fan_out
        self.W, self.B = new_W, new_B

    def layer_shapes(self):
        return [(w.shape, b.shape) for w, b in zip(self.W, self.B)]
