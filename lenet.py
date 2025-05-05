import numpy as np

# --- Activation Functions ---
def relu(x):
    return np.maximum(0, x)

def relu_backward(dout, x):
    dx = dout.copy()
    dx[x <= 0] = 0
    return dx

# --- Layers ---
class BatchNorm:

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.zeros(dim)

    def forward(self, x, is_training=True):
        if is_training:
            # 沿批次、高度、宽度计算均值和方差
            self.mu = np.mean(x, axis=(0, 1, 2))
            self.var = np.var(x, axis=(0, 1, 2))
            self.x_hat = (x - self.mu.reshape(1, 1, 1, -1)) / np.sqrt(self.var.reshape(1, 1, 1, -1) + self.eps)
            out = self.gamma.reshape(1, 1, 1, -1) * self.x_hat + self.beta.reshape(1, 1, 1, -1)

            # 更新运行均值和方差
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            x_hat = (x - self.running_mean.reshape(1, 1, 1, -1)) / np.sqrt(self.running_var.reshape(1, 1, 1, -1) + self.eps)
            out = self.gamma.reshape(1, 1, 1, -1) * x_hat + self.beta.reshape(1, 1, 1, -1)
        self.cache = (x, self.x_hat)
        return out

    def backward(self, dout, lr=0.01):
        x, x_hat = self.cache
        m = np.prod(x.shape[:3])  # 批次大小 × 高度 × 宽度

        # 计算梯度和参数更新
        dbeta = np.sum(dout, axis=(0, 1, 2))
        dgamma = np.sum(dout * x_hat, axis=(0, 1, 2))

        dxhat = dout * self.gamma.reshape(1, 1, 1, -1)
        dvar = np.sum(dxhat * (x - self.mu.reshape(1, 1, 1, -1)) * (-0.5) * (self.var + self.eps) ** (-1.5),
                      axis=(0, 1, 2))
        dmu = np.sum(dxhat * (-1 / np.sqrt(self.var + self.eps)), axis=(0, 1, 2)) + dvar * np.mean(
            -2 * (x - self.mu.reshape(1, 1, 1, -1)), axis=(0, 1, 2))

        dx = dxhat / np.sqrt(self.var.reshape(1, 1, 1, -1) + self.eps)
        dx += (dvar.reshape(1, 1, 1, -1) * 2 * (x - self.mu.reshape(1, 1, 1, -1)) / m)
        dx += dmu.reshape(1, 1, 1, -1) / m

        # 更新参数
        self.gamma -= lr * dgamma
        self.beta -= lr * dbeta
        return dx

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=0):
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size ** 2))
        self.weights = np.random.randn(kernel_size, kernel_size, in_channels, out_channels) * scale
        self.bias = np.zeros(out_channels)
        self.cache = None

    def forward(self, x):
        x = np.pad(x, [(0,0), (self.padding,self.padding), (self.padding,self.padding), (0,0)], 'constant')
        batch, in_h, in_w, in_c = x.shape
        k = self.weights.shape[0]
        out_h = in_h - k + 1
        out_w = in_w - k + 1

        cols = im2col(x, k, k, 1, 0)
        cols_w = self.weights.reshape(-1, self.weights.shape[3])
        output = cols @ cols_w + self.bias
        self.cache = (x, cols, cols_w)

        return output.reshape(batch, out_h, out_w, -1)

    def backward(self, dout, lr=0.01):
        x, cols, cols_w = self.cache
        batch, in_h, in_w, in_c = x.shape
        k = self.weights.shape[0]

        dout_flat = dout.transpose(0, 3, 1, 2).reshape(-1, dout.shape[3])
        dW = cols.T @ dout_flat
        dW = dW.reshape(k, k, in_c, -1)
        db = np.sum(dout, axis=(0, 1, 2))

        cols_d = dout_flat @ cols_w.T
        dx = col2im(cols_d, x.shape, k, k, 1, 0)

        self.weights -= lr * dW
        self.bias -= lr * db

        return dx

class MaxPool2D:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        batch, h, w, c = x.shape
        out_h = h // 2
        out_w = w // 2

        self.mask = np.zeros_like(x)
        output = np.zeros((batch, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                h_start = 2 * i
                h_end = h_start + 2
                w_start = 2 * j
                w_end = w_start + 2

                region = x[:, h_start:h_end, w_start:w_end, :]
                max_val = np.max(region, axis=(1, 2), keepdims=True)
                mask = (region == max_val)
                self.mask[:, h_start:h_end, w_start:w_end, :] = mask
                output[:, i, j, :] = max_val.squeeze()
        return output

    def backward(self, dout):
        dout_new = np.zeros_like(self.mask)
        batch, out_h, out_w, c = dout.shape

        for i in range(out_h):
            for j in range(out_w):
                h_start = 2 * i
                h_end = h_start + 2
                w_start = 2 * j
                w_end = w_start + 2

                dout_region = dout[:, i:i+1, j:j+1, :]
                mask_region = self.mask[:, h_start:h_end, w_start:w_end, :]
                dout_new[:, h_start:h_end, w_start:w_end, :] += dout_region * mask_region

        return dout_new

class Affine:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout, lr=0.01):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        self.W -= lr * dW
        self.b -= lr * db
        return dx

# --- Model ---
class LeNet5:
    def __init__(self):
        self.conv1 = Conv2D(1, 6, kernel_size=5, padding=2)
        self.bn1 = BatchNorm(6)
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(6, 16, kernel_size=5)
        self.bn2 = BatchNorm(16)
        self.pool2 = MaxPool2D()

        self.fc1 = Affine(16*5*5, 120)
        self.fc2 = Affine(120, 84)
        self.fc3 = Affine(84, 10)

    def forward(self, x, is_training=True):
        x = self.conv1.forward(x)
        self.conv1_output = x
        x = relu(x)
        x = self.bn1.forward(x, is_training)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        self.conv2_output = x
        x = relu(x)
        x = self.bn2.forward(x, is_training)
        x = self.pool2.forward(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        self.fc1_output = x
        x = relu(x)
        x = self.fc2.forward(x)
        self.fc2_output = x
        x = relu(x)
        x = self.fc3.forward(x)
        return x

# --- Utility Functions (im2col/col2im) ---
def im2col(x, kh, kw, stride=1, pad=0):
    batch, h, w, c = x.shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1
    img_padded = np.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)], 'constant')
    col = np.zeros((batch, oh, ow, kh, kw, c))

    for y in range(kh):
        y_max = y + stride * oh
        for x in range(kw):
            x_max = x + stride * ow
            col[:, :, :, y, x, :] = img_padded[:, y:y_max:stride, x:x_max:stride, :]

    return col.reshape(batch * oh * ow, -1)

def col2im(col, x_shape, kh, kw, stride=1, pad=0):
    batch, h, w, c = x_shape
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (w + 2 * pad - kw) // stride + 1

    col_reshaped = col.reshape(batch, oh, ow, kh, kw, c)
    img = np.zeros((batch, h + 2 * pad, w + 2 * pad, c))

    for y in range(kh):
        y_max = y + stride * oh
        for x in range(kw):
            x_max = x + stride * ow
            img[:, y:y_max:stride, x:x_max:stride, :] += col_reshaped[:, :, :, y, x, :]

    return img[:, pad:h + pad, pad:w + pad, :]
