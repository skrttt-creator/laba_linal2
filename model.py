import numpy as np
from metrics import accuracy

class Perceptron:
    def __init__(self, loss_type='bce', l2_lambda=0.0, momentum=0.0):
        self.w = None
        self.b = None
        self.train_losses = []
        self.val_losses = []
        
        self.loss_type = loss_type   # 'bce' или 'hinge'
        self.l2_lambda = l2_lambda   # Коэфф L2-регуляризации
        self.momentum = momentum     # Коэфф импульса
        
        self.v_w = None 
        self.v_b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250))) 

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def compute_loss(self, X, y):
        z = np.dot(X, self.w) + self.b
        l2_penalty = self.l2_lambda * np.sum(self.w ** 2) 
        
        if self.loss_type == 'bce':
            y_pred = self.sigmoid(z)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            base_loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        elif self.loss_type == 'hinge':
            y_shift = np.where(y == 0, -1, 1)
            margin = y_shift * z
            base_loss = np.mean(np.maximum(0, 1 - margin))
            
        return base_loss + l2_penalty

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.1, batch_size=32):
        n_samples, n_features = X_train.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        
        self.v_w = np.zeros_like(self.w)
        self.v_b = 0.0
        
        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                z = np.dot(X_batch, self.w) + self.b

                if self.loss_type == 'bce':
                    y_pred_batch = self.sigmoid(z)
                    dw = np.dot(X_batch.T, (y_pred_batch - y_batch)) / len(y_batch)
                    db = np.mean(y_pred_batch - y_batch)
                
                elif self.loss_type == 'hinge':
                    y_shift = np.where(y_batch == 0, -1, 1)
                    margin = y_shift * z
                    condition = margin < 1 
                    
                    dw = np.dot(X_batch.T, -y_shift * condition) / len(y_batch)
                    db = np.mean(-y_shift * condition)

                dw += 2 * self.l2_lambda * self.w
                
                self.v_w = self.momentum * self.v_w + lr * dw
                self.v_b = self.momentum * self.v_b + lr * db
                
                self.w -= self.v_w
                self.b -= self.v_b

            self.train_losses.append(self.compute_loss(X_train, y_train))
            if X_val is not None and y_val is not None:
                self.val_losses.append(self.compute_loss(X_val, y_val))

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

def k_fold_cv(X, y, k=5, lr=0.1, batch_size=32, momentum=0.0):
    print(f"Запуск {k}-Fold CV...")
    n_samples = len(X)
    fold_size = n_samples // k
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    val_accuracies = []
    
    for i in range(k):
        start, end = i * fold_size, (i + 1) * fold_size
        X_val, y_val = X[start:end], y[start:end]
        
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        
        model = Perceptron(loss_type='bce', momentum=momentum)
        model.fit(X_train, y_train, epochs=50, lr=lr, batch_size=batch_size)
        
        acc = accuracy(y_val, model.predict(X_val))
        val_accuracies.append(acc)
        
    return np.mean(val_accuracies)
