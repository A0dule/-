import numpy as np
import struct

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        buf = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", buf[:16])
    images = np.frombuffer(buf[16:], dtype=np.uint8).reshape(num, rows, cols)
    return images[..., np.newaxis].astype(np.float32) / 255.0

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        buf = f.read()
    return np.frombuffer(buf[8:], dtype=np.uint8)

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]