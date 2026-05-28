import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#  Задача №1
def generate_gaussian_data(n_samples=500, distance=2.5, seed=42):
    np.random.seed(seed)
    n_class = n_samples // 2
    
    X0 = np.random.randn(n_class, 2)
    y0 = np.zeros(n_class)
    
    X1 = np.random.randn(n_class, 2) + np.array([distance, distance])
    y1 = np.ones(n_class)
    
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    return X[indices], y[indices]

# Задача №3)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_recall_f1(y_true, y_pred):
    print("расчет Precision, Recall и F1-score")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def plot_roc_curve(y_true, y_probs):
    print("Построение roc-кривой")
    thresholds = np.linspace(0, 1, 100)
    tpr_list, fpr_list = [], []
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    plt.figure(figsize=(6, 5))

    plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid()

    plt.show()




class Perceptron:
    def __init__(self, loss_type='bce', l2_lambda=0.0, momentum=0.0):
        self.w = None
        self.b = None
        self.train_losses = []
        self.val_losses = []
        
        # Задача №2 и №4
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


# cross validation (Задача №5)

def k_fold_cv(X, y, k=5, lr=0.1, batch_size=32, momentum=0.0):
    print("разбивает данные на K частей, обучает K моделей и возвращает среднюю точность")
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





if __name__ == "main":
    X, y = generate_gaussian_data(n_samples=600, distance=2.0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std


    print("5-Fold Cross-Validation")
    best_lr = 0.1
    best_acc = 0
    for lr_test in [0.01, 0.1, 0.5]:
        avg_acc = k_fold_cv(X_train, y_train, k=5, lr=lr_test, momentum=0.9)
        print(f"LR: {lr_test:<5} | CV Accuracy: {avg_acc:.4f}")
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_lr = lr_test
    print(f"Лучший learning rate: {best_lr}\n")


    print("обучение финальной модели (Hinge + L2 + Momentum)")
    model = Perceptron(loss_type='hinge', l2_lambda=0.01, momentum=0.9)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=best_lr, batch_size=32)


    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)
    
    acc = accuracy(y_test, y_pred)
    prec, rec, f1 = precision_recall_f1(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    # График ошибки
    plt.figure(figsize=(8, 5))
    plt.plot(model.train_losses, label='Train Loss')
    plt.plot(model.val_losses, label='Test Loss')
    plt.title('Loss History (Hinge + L2)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Разделяющая граница
    plot_decision_boundary(X_test, y_test, model, "разделяющая граница")

    # ROC Кривая
    plot_roc_curve(y_test, y_probs)
