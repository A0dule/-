import numpy as np

class Trainer:
    def __init__(self, model, lr=0.01, lr_decay=0.5, decay_every=10):
        self.model = model
        self.base_lr = lr
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.train_loss = []
        self.val_acc = []

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-7)
        return np.sum(log_likelihood) / m

    def train(self, x_train, y_train, x_val, y_val, epochs=50, batch_size=64):
        for epoch in range(epochs):
            if epoch > 0 and epoch % self.decay_every == 0:
                self.lr *= self.lr_decay
                print(f"学习率衰减! 新学习率: {self.lr:.5f}")

            permutation = np.random.permutation(x_train.shape[0])
            epoch_loss = 0

            for i in range(0, x_train.shape[0], batch_size):
                batch_idx = permutation[i:i + batch_size]
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]

                logits = self.model.forward(x_batch, is_training=True)
                probs = self.softmax(logits)
                loss = self.cross_entropy_loss(probs, y_batch)
                epoch_loss += loss

                grad = probs.copy()
                grad[range(len(grad)), y_batch.argmax(axis=1)] -= 1
                grad /= len(grad)

                grad = self.model.fc3.backward(grad, self.lr)
                grad = relu_backward(grad, self.model.fc2_output)
                grad = self.model.fc2.backward(grad, self.lr)
                grad = relu_backward(grad, self.model.fc1_output)
                grad = self.model.fc1.backward(grad, self.lr)

                grad = grad.reshape(x_batch.shape[0], 5, 5, 16)
                grad = self.model.pool2.backward(grad)
                grad = self.model.bn2.backward(grad, self.lr)
                grad = relu_backward(grad, self.model.conv2_output)
                grad = self.model.conv2.backward(grad, self.lr)

                grad = self.model.pool1.backward(grad)
                grad = self.model.bn1.backward(grad, self.lr)
                grad = relu_backward(grad, self.model.conv1_output)
                _ = self.model.conv1.backward(grad, self.lr)

            val_pred = self.predict(x_val, is_training=False)
            val_acc = np.mean(val_pred == y_val.argmax(1))
            self.val_acc.append(val_acc)

            avg_loss = epoch_loss / (x_train.shape[0] // batch_size)
            self.train_loss.append(avg_loss)

            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        return self.train_loss, self.val_acc

    def predict(self, x, is_training=False):
        logits = self.model.forward(x, is_training=is_training)
        return logits.argmax(axis=1)

    def evaluate(self, x_test, y_test):
        preds = self.predict(x_test, is_training=False)
        return np.mean(preds == y_test.argmax(1))

# Helper function
def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx
