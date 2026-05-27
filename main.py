import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class BinaryClassifierNode:
    def __init__(self, seed=None):
        self._rng = np.random.default_rng(seed)
        self.weights = None
        self.bias = None
        self.loss_history_train = []
        self.loss_history_val = []

    def _logistic_function(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _forward_pass(self, features):
        return self._logistic_function(np.dot(features, self.weights) + self.bias)

    def _calculate_bce(self, target, prediction):
        epsilon = 1e-15
        pred_clipped = np.clip(prediction, epsilon, 1.0 - epsilon)
        return -np.mean(target * np.log(pred_clipped) + (1.0 - target) * np.log(1.0 - pred_clipped))

    def optimize(self, x_tr, y_tr, x_v, y_v, max_passes, alpha_step, chunk_size, init_mode='small'):
        n_records, n_dims = x_tr.shape

        if init_mode == 'small':
            self.weights = self._rng.normal(0.0, 0.01, size=n_dims)
        elif init_mode == 'zero':
            self.weights = np.zeros(n_dims)
        elif init_mode == 'large':
            self.weights = self._rng.normal(0.0, 10.0, size=n_dims)
        else:
            raise ValueError("Unknown init_mode")

        self.bias = 0.0
        self.loss_history_train = []
        self.loss_history_val = []

        for _ in range(max_passes):
            shuffle_idx = self._rng.permutation(n_records)
            
            for start_pos in range(0, n_records, chunk_size):
                batch_indices = shuffle_idx[start_pos:start_pos + chunk_size]
                x_chunk = x_tr[batch_indices]
                y_chunk = y_tr[batch_indices]

                y_hat = self._forward_pass(x_chunk)
                errors = y_hat - y_chunk

                grad_weights = np.dot(x_chunk.T, errors) / len(y_chunk)
                grad_bias = np.mean(errors)

                self.weights -= alpha_step * grad_weights
                self.bias -= alpha_step * grad_bias

            train_loss = self._calculate_bce(y_tr, self._forward_pass(x_tr))
            val_loss = self._calculate_bce(y_v, self._forward_pass(x_v))
            self.loss_history_train.append(train_loss)
            self.loss_history_val.append(val_loss)

    def infer(self, features, threshold=0.5):
        probs = self._forward_pass(features)
        return (probs >= threshold).astype(int)

def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def visualize_boundary(model, X, y, title_text):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    w1, w2 = model.weights
    b = model.bias

    if w2 != 0:
        x_vals = np.array([x_min, x_max])
        y_vals = -(w1 * x_vals + b) / w2
        plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='Decision Boundary')

    plt.xlim(x_min, x_max)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
    plt.title(title_text)
    plt.legend()
    plt.grid(True)
    plt.show()

def run_lab_experiments():
    # 1. Data Prep
    X_raw, y_target = make_classification(
        n_samples=500, n_features=2, n_redundant=0, 
        n_informative=2, random_state=42, n_clusters_per_class=1
    )

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_raw, y_target, test_size=0.3, stratify=y_target, random_state=101
    )

    mu_train = np.mean(X_train_orig, axis=0)
    sigma_train = np.std(X_train_orig, axis=0)
    X_tr_norm = (X_train_orig - mu_train) / sigma_train
    X_te_norm = (X_test_orig - mu_train) / sigma_train

    # 3. Base Training
    print("--- 3. Base Training ---")
    base_model = BinaryClassifierNode(seed=42)
    base_model.optimize(
        X_tr_norm, y_train, X_te_norm, y_test, 
        max_passes=100, alpha_step=0.1, chunk_size=32, init_mode='small'
    )

    acc_train = evaluate_accuracy(y_train, base_model.infer(X_tr_norm))
    acc_test = evaluate_accuracy(y_test, base_model.infer(X_te_norm))
    print(f"Base Model - Train Acc: {acc_train:.4f}, Test Acc: {acc_test:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(base_model.loss_history_train, label='Train Loss', color='blue')
    plt.plot(base_model.loss_history_val, label='Validation Loss', color='orange')
    plt.title("Base Training: Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    visualize_boundary(base_model, X_te_norm, y_test, "Decision Boundary (Test Set)")

    # 4. Experiments
    print("\n--- 4. Experiments ---")
    
    # 4.1 Learning Rate
    print("\n[ Learning Rate Experiment ]")
    lr_options = [0.001, 0.01, 0.5, 1.0]
    plt.figure(figsize=(10, 6))
    print(f"{'LR':<10} | {'Train Acc':<12} | {'Test Acc':<12}")
    print("-" * 38)
    for lr_val in lr_options:
        temp_net = BinaryClassifierNode(seed=42)
        temp_net.optimize(X_tr_norm, y_train, X_te_norm, y_test, 100, lr_val, 32)
        plt.plot(temp_net.loss_history_val, label=f'LR={lr_val}')
        t_acc = evaluate_accuracy(y_train, temp_net.infer(X_tr_norm))
        v_acc = evaluate_accuracy(y_test, temp_net.infer(X_te_norm))
        print(f"{lr_val:<10} | {t_acc:<12.4f} | {v_acc:<12.4f}")
    plt.title("Validation Loss across Learning Rates")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # 4.2 Batch Size
    print("\n[ Batch Size Experiment ]")
    batch_options = [1, 16, 64, 256]
    plt.figure(figsize=(10, 6))
    print(f"{'Batch':<10} | {'Train Acc':<12} | {'Test Acc':<12}")
    print("-" * 38)
    for bs in batch_options:
        temp_net = BinaryClassifierNode(seed=42)
        temp_net.optimize(X_tr_norm, y_train, X_te_norm, y_test, 100, 0.1, bs)
        plt.plot(temp_net.loss_history_val, label=f'Batch={bs}')
        t_acc = evaluate_accuracy(y_train, temp_net.infer(X_tr_norm))
        v_acc = evaluate_accuracy(y_test, temp_net.infer(X_te_norm))
        print(f"{bs:<10} | {t_acc:<12.4f} | {v_acc:<12.4f}")
    plt.title("Validation Loss across Batch Sizes")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print("\n[ Initialization Experiment ]")
    init_options = ['zero', 'small', 'large']
    plt.figure(figsize=(10, 6))
    print(f"{'Init Type':<10} | {'Train Acc':<12} | {'Test Acc':<12}")
    print("-" * 38)
    for i_mode in init_options:
        temp_net = BinaryClassifierNode(seed=42)
        temp_net.optimize(X_tr_norm, y_train, X_te_norm, y_test, 100, 0.1, 32, init_mode=i_mode)
        plt.plot(temp_net.loss_history_val, label=f'Init={i_mode}')
        t_acc = evaluate_accuracy(y_train, temp_net.infer(X_tr_norm))
        v_acc = evaluate_accuracy(y_test, temp_net.infer(X_te_norm))
        print(f"{i_mode:<10} | {t_acc:<12.4f} | {v_acc:<12.4f}")
    plt.title("Validation Loss across Weight Initializations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_lab_experiments()
