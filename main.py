import numpy as np
import matplotlib.pyplot as plt
from data_processor import load_mnist_images, load_mnist_labels, one_hot_encode
from lenet import LeNet5
from trainer import Trainer

def plot_results(train_loss, val_acc):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(val_acc)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

def visualize_predictions(x, y_true, y_pred, num=8):
    indices = np.random.choice(len(x), num)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num, i + 1)
        img = x[idx].squeeze()
        plt.imshow(img, cmap='gray')
        color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        plt.title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

if __name__ == "__main__":
    # 加载数据
    x_train = load_mnist_images('train-images.idx3-ubyte')
    y_train = one_hot_encode(load_mnist_labels('train-labels.idx1-ubyte'))
    x_test = load_mnist_images('t10k-images.idx3-ubyte')
    y_test = one_hot_encode(load_mnist_labels('t10k-labels.idx1-ubyte'))

    # 划分验证集
    val_size = 10000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # 初始化模型
    model = LeNet5()
    trainer = Trainer(model, lr=0.01, lr_decay=0.5, decay_every=10)

    # 训练
    train_loss, val_acc = trainer.train(x_train, y_train, x_val, y_val, epochs=50, batch_size=64)

    # 测试集评估
    test_acc = trainer.evaluate(x_test, y_test)
    print(f"\n测试集准确率: {test_acc * 100:.2f}%")

    # 可视化
    plot_results(train_loss, val_acc)
    test_preds = trainer.predict(x_test[:8], is_training=False)
    visualize_predictions(x_test[:8], y_test[:8].argmax(1), test_preds)
