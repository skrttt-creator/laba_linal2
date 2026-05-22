import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
        self.train_loss = []
        self.val_loss = []
      

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, init_mode='small'):
        n_samples, n_features = X_train.shape

        if init_mode == 'small':
            self.w = np.random.randn(n_features) * 0.01
        elif init_mode == 'zero':
            self.w = np.zeros(n_features)
        elif init_mode == 'large':
            self.w = np.random.randn(n_features) * 10
        
        self.b = 0.0
        self.train_loss = []
        self.val_loss = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
          

            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                y_pred_batch = self.forward(X_batch)
                
                dw = np.dot(X_batch.T, (y_pred_batch - y_batch)) / len(y_batch)
                db = np.mean(y_pred_batch - y_batch)

                self.w -= lr * dw
                self.b -= lr * db

            train_loss = self.compute_loss(y_train, self.forward(X_train))
            val_loss = self.compute_loss(y_val, self.forward(X_val))

          
            
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(int)
      


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_decision_boundary(X, y, model, title="Decision Boundary"):
  
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    w1, w2 = model.w
    b = model.b

    if w2 != 0:
        x_values = np.array([x_min, x_max])
        y_values = -(w1 * x_values + b) / w2
        plt.plot(x_values, y_values, 'k--', label='Boundary')

    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                               n_informative=2, random_state=42, n_clusters_per_class=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("базовое обучение")
    model = Perceptron()
    model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32)

    train_acc = accuracy(y_train, model.predict(X_train_scaled))
    test_acc = accuracy(y_test, model.predict(X_test_scaled))
    print(f"Accuracy (Train): {train_acc:.4f}")
    print(f"Accuracy (Test): {test_acc:.4f}")

  

    plt.figure(figsize=(8, 5))
    plt.plot(model.train_loss, label='Train Loss')
    plt.plot(model.val_loss, label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
  
    plt.legend()
    plt.grid()
    plt.show()

    plot_decision_boundary(X_test_scaled, y_test, model, "edge")

  
    print("\n влияние скорости обучения :")
    lrs = [0.001, 0.01, 0.5, 1.0]
    plt.figure(figsize=(10, 6))
    print(f"{'LR':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 35)
    for lr in lrs:
        exp_model = Perceptron()
        exp_model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=lr, batch_size=32)
        plt.plot(exp_model.val_loss, label=f'lr = {lr}')
        print(f"{lr:<10} | {accuracy(y_train, exp_model.predict(X_train_scaled)):<10.4f} | {accuracy(y_test, exp_model.predict(X_test_scaled)):<10.4f}")
    plt.title('Validation Loss for different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    print("\n влияние размера батча (batch size):")
    batch_sizes = [1, 16, 64, 256]
    plt.figure(figsize=(10, 6))
    print(f"{'Batch':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 35)
    for bs in batch_sizes:
        exp_model = Perceptron()
        exp_model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=bs)
        plt.plot(exp_model.val_loss, label=f'batch_size = {bs}')
        print(f"{bs:<10} | {accuracy(y_train, exp_model.predict(X_train_scaled)):<10.4f} | {accuracy(y_test, exp_model.predict(X_test_scaled)):<10.4f}")
    plt.title('Validation Loss for different Batch Sizes')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    print("\n влияние инициализации весов:")
    inits = ['zero', 'small', 'large']
    plt.figure(figsize=(10, 6))
    print(f"{'Init':<10} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 35)
    for init in inits:
        exp_model = Perceptron()
        exp_model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32, init_mode=init)
        plt.plot(exp_model.val_loss, label=f'init = {init}')
        print(f"{init:<10} | {accuracy(y_train, exp_model.predict(X_train_scaled)):<10.4f} | {accuracy(y_test, exp_model.predict(X_test_scaled)):<10.4f}")
    plt.title('Validation Loss for different Weight Initializations')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
