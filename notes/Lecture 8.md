# Lecture 8 | Deep Learning Software

## CPU vs GPU

|  | **CPU** | **GPU** |
| --- | --- | --- |
| 코어 수 | 적음 | 많음 |
| Clock Speed | 빠름(높은 주파수) | 느림(낮은 주파수) |
| 코어 | 빠르고 독립적 작동 | 느리고 완전한 독립 작동 불가능
일반적으로 한 과업을 병렬화 |
| 메모리 | 시스템과 공유 | 자기 메모리 내장 |
| 역할 | → 모든 프로세스 실행 가능 | → (특히) 행렬곱 특화 |
- CUDA (NVIDIA only)
    - GPU에서 바로 작동하는 C-like 코드
    - 하이레벨 API: cuBLAS, cuFFT, cuDNN 등
    
    → CUDA 코드를 직접 작성하기보다는 다른 사람이 최적화 해놓은 것을 쓰게 될 일이 많음
    
    → 최적화된 cuDNN와 CUDA 간의 성능 차이도 약 3배 정도 존재. cuDNN을 쓸일이 많을 것임
    
- OpenCL
    - 어떤 것에서든 작동하나 보통 느림

### CPU / GPU Communication

- 병목 현상 해결
    - 적은 데이터량 or 큰 용량의 메모리
    - 하드디스크 대신 SSD 사용
    - CPU의 멀티스레드 기능 이용하여 저장장치로부터 GPU로 계속 데이터 공급

# 딥러닝 프레임워크(Deep Learning Framework)

### 장점

- 큰 계산 그래프(computational graph)를 구축하기 편함
- 계산 그래프에서 그래디언트를 계산하기 편함
(순전파(forward pass)만 작성하면 역전파(back propagation)는 자동으로 해줌)
- GPU에서 효율적으로 작동함

## 계산 그래프

![image.png](assets/Lecture%208/image.png)

### Numpy

```python
import numpy as np
np.random.seed(0)

N, D = 3, 4

x = np.random.randn(N, D)
y = np.random.randn(N, D)
z = np.random.randn(N, D)

a = x * y
b = a + z
c = np.sum(b)

grad_c = 1.0
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
```

- 그래디언트를 직접 계산해야 함
- GPU를 사용할 수 없음

### TensorFlow

```python
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3, 4

with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)

    a = x * y
    b = a + z
    c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
        z: np.random.randn(N, D),
    }
    out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
```

- 그래디언트를 자동으로 계산해줌
- `with` 를 통해 CPU에서 작동시킬 건지 GPU에서 작동시킬 건지 선택 가능

### PyTorch

```python
import torch
from torch.autograd import Variable

N, D = 3, 4

x = Variable(torch.randn(N, D).cuda(), requires_grad=True)
y = Variable(torch.randn(N, D).cuda(), requires_grad=True)
z = Variable(torch.randn(N, D).cuda(), requires_grad=True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```

- 그래디언트를 자동으로 계산해줌
- `.cuda()` 를 통해(typecast) GPU에서 작동시킬 수 있음

# TensorFlow

> two-layer ReLU with L2 loss를 랜덤 데이터로 학습시키는 코드
> 

```python
import numpy as np
import tensorflow as tf

N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

############### 𐌣 계산 그래프 정의 부분 ###############
###############   ↆ 그래프 실행 부분   ###############

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
        w1: np.random.randn(D, H),
        w2: np.random.randn(H, D),
        y: np.random.randn(N, D),
    }
    learning_rate = 1e-5
    for t in range(50):
        out = sess.run([loss, grad_w1, grad_w2], feed_dict=values)
        loss_val, grad_w1_val, grad_w2_val = out
        ########## 그래프 학습 ##########
        values[w1] -= learning_rate * grad_w1_val
        values[w2] -= learning_rate * grad_w2_val
```

### 그래프 구축 부분

- `.placeholder()`
    - 그래프의 입력 슬롯(input slot) → 실행시킬 때 데이터를 입력
    - 실제로 메모리가 할당되는 것이 아니라 심볼을 생성하는 것
- `.matmul()` , `.maximum()` , `.reduce_sum()` , `.reduce_mean()` 등
    - L2 distance를 계산하는 과정에서의 연산들
    - 실제로 계산이 이루어지는 것이 아니라 그래프가 이렇게 동작한다고 선언하는 것 (= 그래프 구축)
- `.gradients()`
    - 자동으로 그래디언트 계산
    - 실제로 계산이 이루어지는 것이 아니라 이후 그래프 동작 시 그래디언트 계산을 추가하는 것

### 그래프 실행 부분

- `.Session()`
    - 그래프를 실행시키기 위해 Session으로 진입
- `values`
    - 구체적인 입력값 지정. TensorFlow는 일반적으로 Numpy array를 입력으로 받음
- `Session.run()`
    - 그래프 실행. 매개변수의 값들을 Numpy array로 반환함
    - `feed_dict` 를 통해 입력값 전달

> 문제) CPU의 Numpy array를 GPU로 동작하는 그래프에 입력하고 그래프의 출력을 다시 Numpy array로 받음
→ 매 스텝마다 CPU, GPU간 가중치 행렬의 복사가 발생
→ 메모리 이동 병목이 될 가능성
> 

```python
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
w2 = tf.Variable(tf.random_normal((H, D)))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
    }
    for t in range(50):
        loss_val, = sess.run([loss], feed_dict=values)
```

- `.Variable()`
    - 가중치 행렬을 그래프 안에서 유지되도록 선언. 초기화 방법을 지정해줘야 함.
    - 실제로 메모리가 할당되거나 초기화되는 것이 아니라 심볼을 생성하는 것
- `.assign()`
    - 가중치 행렬이 그래프 안에서 업데이트되도록 선언
- `Session.run(tf.global_variables_initializer())`
    - 처음에 가중치 행렬을 실제 값으로 초기화하기 위해 한 번만 실행

> 문제) TensorFlow는 최적화로 인해 필요없는 연산은 수행하지 않기 때문에 실제로 가중치의 업데이트가 발생하지 않음
> 

```python
...
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
updates = tf.group(new_w1, new_w2)

with tf.Session() as sess:
    ...
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
```

- 더미 노드(updates)를 생성하고, `Session.run()` 의 반환값에 포함시킴으로써 TensorFlow가 이를 계산해야 하는 값으로 인식하도록 한다.

## 여러 기법을 사용한 코드

```python
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))

init = tf.contrib.layers.xavier_initializer()
h = tf.layers.dense(inputs=x, units=H,
                    activation=tf.nn.relu, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h, units=D, kernel_initializer=init)

loss = tf.losses.mean_squared_error(y_pred, y)

optimizer = tf.train.GradientDescentOptimizer(1e0)
updates = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    values = {
        x: np.random.randn(N, D),
        y: np.random.randn(N, D),
    }
    for t in range(50):
        loss_val, _ = sess.run([loss, updates], feed_dict=values)
```

- Optimizer
    - 자동으로 변화할 수 있는 값(.Variable)을 인식하고 그래디언트를 계산하여 최적화 수행
    - 더미 노드를 생성해 TensorFlow가 이를 실행시키도록 해야 함
- Loss
    - 자동으로 loss를 계산
- Initializer
    - 자동으로 초기화 (e.g. Xavier 등)
- Layer
    - 자동으로 레이어 생성
    - 자동으로 가중치, 편향(bias)을 초기화, 활성 함수(activation) 등도 설정 가능
- 상위 레벨 Wrapper
    - Keras, tf.layers, TF-Slim, Pretty Tensor, Sonnet 등
- 여러 도움되는 것들
    - Pretrained Models
    - Tensorboard
    - Distributed Version

# PyTorch

### 3가지 추상화

- `Tensor` : Numpy의 array와 유사하지만 GPU에서 실행 가능
- `Variable` : 계산 그래프의 노드 → 값 저장, 그래디언트 계산 등
- `Module` : 신경망 레이어 → 상태나 가중치 등을 저장할 수 있음, Module을 조합하여 네트워크 구축

| **PyTorch** | **TensorFlow** |
| --- | --- |
| Tensor | Numpy array |
| Variable | Tensor, Variable, Placeholder |
| Module | tf.layers, TFSlim, TFLearn, Sonnet, etc. |

> two-layer ReLU with L2 loss를 랜덤 데이터로 학습시키는 코드
> 

```python
import torch

dtype = torch.cuda.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum()

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

※ Numpy 없이 PyTorch 만으로 구현 가능

- `.cuda.FloatTensor`
    - Numpy의 array와 비슷하지만 GPU에서 실행 가능

> AutoGrad 적용
> 

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in), requires_grad=False)
y = Variable(torch.randn(N, D_out), requires_grad=False)
w1 = Variable(torch.randn(D_in, H), requires_grad=True)
w2 = Variable(torch.randn(H, D_out), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

    if w1.grad: w1.grad.data.zero_()
    if w2.grad: w2.grad.data.zero_()
    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
```

- `Variable()`
    
    ※ `Tensor` 와 `Variable` 은 같은 API를 지닌다.
    
    - `Variable.data` = `Tensor`
    - `Variable.grad` = `Variable` of gradients (`Variable.data` 와 같은 사이즈)
    - `Variable.grad.data` = `Tensor` of gradients
- `.grad.data.zero_()`
    - 초기에 그래디언트를 0으로 초기화
- `.backward()`
    - 자동으로 그래디언트 계산

→ 계산 그래프를 구축한 이후 실행시키는 TensorFlow와 달리, PyTorch는 그래프 구축 없이 매번 순전파를 실행

- AutoGrad 함수
    
    ```python
    class ReLU(torch.autograd.Function):
        def forward(self, x):
            self.save_for_backward(x)
            return x.clamp(min=0)
    
        def backward(self, grad_y):
            x, = self.saved_tensors
            grad_input = grad_y.clone()
            grad_input[x < 0] = 0
            return grad_input
    ```
    
    - `Tensor` 를 이용해 AutoGrad 함수를 정의할 수 있다.
        
        ```python
        ...
        for t in range(500):
            relu = ReLU()
            y_pred = relu(x.mm(w1)).mm(w2)
            loss = (y_pred - y).pow(2).sum()
            ...
        ```
        
    - 이를 계산 그래프 구축에 이용할 수 있음
        - 그러나, 대부분의 경우 이미 구현되어 있기 때문에 직접 구현할 일은 많지 않음

> 상위 레벨 Wrapper nn, optim을 이용한 구현
> 

```python
import torch
from torch.autograd import Variable

N, D_in, H, D_out = 64, 1000, 100, 10
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
loss_fn = torch.nn.MSELoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

- Model
    - `.nn.Sequential()`
        - 연속된 레이어들로 모델을 정의
    - `.nn.MSELoss()`
        - 자동으로 loss를 계산
- Optimizer
    - `.optim.Adam()`
        - `.step()` 으로 자동으로 매개변수를 업데이트

※ 순전파를 실행할 때마다 새로운 계산 그래프를 구축한다

- 새로운 모델 사용자 정의하기
    
    ```python
    import torch
    from torch.autograd import Variable
    
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            super(TwoLayerNet, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, D_out)
    
        def forward(self, x):
            h_relu = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relu)
            return y_pred
    
    N, D_in, H, D_out = 64, 1000, 100, 10
    
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)
    
    model = TwoLayerNet(D_in, H, D_out)
    
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    for t in range(500):
        y_pred = model(x)
        loss = criterion(y_pred, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ```
    
    - Module은 `Variable` 로서의 가중치와 다른 Module을 포함할 수 있다
    - 순전파를 자식 Module과 autograd 연산을 이용해서 정의하면 역전파를 정의할 필요가 없음. autograd가 자동으로 계산해줌

> DataLoader
> 

```python
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

loader = DataLoader(TensorDataset(x, y), batch_size=8)

model = TwoLayerNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

for epoch in range(10):
    for x_batch, y_batch in loader:
        x_var, y_var = Variable(x_batch), Variable(y_batch)
        y_pred = model(x_var)
        loss = criterion(y_pred, y_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

- DataLoader는 Dataset을 감싸서 미니배치 처리, 셔플링, 멀티스레딩 등을 자동으로 제공함
- 사용자 정의 데이터를 불러와야 할 때는 직접 Dataset 클래스를 작성하면 됨
- `DataLoader()`
    - `DataLoader()` 는 `Tensor` 를 내놓기 때문에 `Variable()` 로 변환해줘야 함

- 여러 도움되는 것들
    - Pretrained Models
    - Visdom

# 정적(Static) vs 동적(Dynamic) 그래프

| **정적 그래프(TensorFlow)** | **동적 그래프(PyTorch)** |
| --- | --- |
| 그래프를 한 번 구축하고, 여러 번 실행한다 | 매 순전파마다 새로운 그래프를 정의한다 |

## 정적 그래프 장점

> 프레임워크가 그래프를 최적화할 수 있다.
> 

![작성한 그래프](assets/Lecture%208/image%201.png)

작성한 그래프

![최적화된 그래프](assets/Lecture%208/image%202.png)

최적화된 그래프

- 레이어 통합, 재배치 등을 통해 그래프를 최적화할 수 있음
- 초기 최적화에는 비용이 들지만, 이후 여러 번의 반복 실행에서 높은 효율로 사용 가능

> 직렬화(serialization)가 가능하다.
> 
- 한 번 그래프를 구축하고 나면, 해당 자료구조를 디스크에 파일로 저장할 수 있음
- 이후에 그래프를 구축하는 원본 코드에 접근하지 않아도 그래프를 실행할 수 있음
    - 그래프를 배포할 때 매우 편리

## 동적 그래프 장점

> 코드를 깔끔하고 쉽게 작성 가능하다.
> 

**예시 1) 조건**

$$
y =\begin{cases}w_1 \cdot x & \text{if } z > 0 \\w_2 \cdot x & \text{otherwise}\end{cases}
$$

※ PyTorch

```python
z = 10
if z > 0:
    y = x.mm(w1)
else:
    y = x.mm(w2)
```

※ TensorFlow

```python
def f1(): return tf.matmul(x, w1)
def f2(): return tf.matmul(x, w2)
y = tf.cond(tf.less(z, 0), f1, f2)
```

- 매 실행마다 새로운 그래프를 구축하기 때문에 파이썬의 조건문으로 매번 새로운 형태의 그래프를 형성할 수 있음
    - 정적 그래프는 그래프를 실행시키기 전에 구축을 완료해야하므로 TensorFlow 버전의 조건문을 그래프 안에 구현해야 함

**예시 2) 반복**

$$
y_t = \left( y_{t-1} + x_t \right) \cdot w
$$

![image.png](assets/Lecture%208/image%203.png)

※ PyTorch

```python
...
x = Variable(torch.randn(T, D))
...

y = [y0]
for t in range(T):
    prev_y = y[-1]
    next_y = (prev_y + x[t]) * w
    y.append(next_y)
```

※ TensorFlow

```python
...
x = tf.placeholder(tf.float32, shape=(T, D))
...

def f(prev_y, cur_x):
    return (prev_y + cur_x) * w

y = tf.foldl(f, x, y0)
```

- 입력값에 따라 그래프의 형태가 변하는 경우, 동적 그래프는 쉽게 구현 가능하나 정적 그래프는 그렇지 않음
    - 정적 그래프는 모든 것을 그래프 내의 제어 흐름 연산자으로 구현해야하기 때문에 원하는 모든 기능을 구현하기는 힘듦

# Caffe

### 개요

- C++로 작성됨, Python, MATLAB 바인딩 있음
- 코드를 작성할 필요 없음
- feedforward classification model을 학습시키고 파인튜닝하는 데 좋음
- 이제 연구에서는 쓰이지 않지만, 모델을 배포하는 데 인기 있음

## 사용법

1. Convert data (run a script)
    1. LMDB 또는 HDF5에서 DataLayer를 읽어온다.
    2. convert_imageset을 사용해서 LMDB를 생성한다.
    3. 각 줄이 `[path/to/image.jpeg] [label]` 형식으로 되어 있는 텍스트 파일이 필요하다.

1. Define net (edit prototxt)
    
    ```protobuf
    name: "LogisticRegressionNet"
    layers {
      top: "data"
      top: "label"
      name: "data"
      type: HDF5_DATA
      hdf5_data_param {
        source: "examples/hdf5_classification/data/train.txt"
        batch_size: 10
      }
      include {
        phase: TRAIN
      }
    }
    layers {
      bottom: "data"
      top: "fc1"
      name: "fc1"
      type: INNER_PRODUCT
      blobs_lr: 1
      blobs_lr: 2
      weight_decay: 1
      weight_decay: 0
      inner_product_param {
        num_output: 2
        weight_filler {
          type: "gaussian"
          std: 0.01
        }
        bias_filler {
          type: "constant"
          value: 0
        }
      }
    }
    layers {
      bottom: "fc1"
      bottom: "label"
      top: "loss"
      name: "loss"
      type: SOFTMAX_LOSS
    }
    ```
    
    1. `.prototext` 텍스트 파일을 이용하여 네트워크를 구축한다.
    2.  큰 모델에서는 파일이 매우 길어짐. 중복되는 블록을 정의하고 재사용할 수 없음

1. Define solver (edit prototxt)
    
    ```protobuf
    net: "models/bvlc_alexnet/train_val.prototxt"
    test_iter: 1000
    test_interval: 1000
    base_lr: 0.01
    lr_policy: "step"
    gamma: 0.1
    stepsize: 100000
    display: 20
    max_iter: 450000
    momentum: 0.9
    weight_decay: 0.0005
    snapshot: 10000
    snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train"
    solver_mode: GPU
    ```
    
    1. 다른 `.prototext` 텍스트 파일을 이용하여 optimizer에 해당하는 solver를 정의한다.

1. Train (with pretrained weights) (run a script)
    
    ```bash
    ./build/tools/caffe train \
    	-gpu 0 \
    	-model path/to/trainval.prototxt \
    	-solver path/to/solver.prototxt \
    	-weights path/to/pretrained_weights.caffemodel
    ```
    
    1. `train` 명령어와 함께 Caffe 바이너리를 실행한다.

- 여러 도움되는 것들
    - Pretrained Models
    - Python Interface

### Caffe2

- 정적 그래프 방식, TensorFlow와 어느 정도 유사
- 코어는 C++로 작성됨, 편리한 Python 인터페이스 제공
- Python에서 모델을 학습시킨 뒤, 직렬화(serialize)하여 Python 없이 배포 가능
- iOS / Android 등 다양한 환경에서 동작