# Lecture 7 | Training Neural Networks II

## Recap

### Data Preprocessing

<aside>
💡

정규화를 하면 같은 함수에 대해서도 작은 변화에 덜 민감해진다

</aside>

![image.png](assets/Lecture%207/image.png)

### Batch Normalization

- Input
    
    $x : N \times D$
    
- Learnable params
    
    $\gamma, \beta : D$
    
- Intermediates
    
    $\mu, \sigma : D$
    
    $\hat{x} : N \times D$
    
- Output
    
    $y : N \times D$
    

$\mu_j = \frac{1}{N} \sum_{i=1}^N x_{i,j}$

$\sigma_j^2 = \frac{1}{N} \sum_{i=1}^N (x_{i,j} - \mu_j)^2$

$\hat{x}_{i,j} = \frac{x{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$

$y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j$

### Hyperparameter Search

<aside>
💡

learning rate가 가장 중요하니 제일 먼저 찾을 것

다른 하이퍼파라미터가 바뀌어도 learning rate는 덜 민감하게 반응하는 경향

</aside>

# 최적화(Optimization)

### SGD의 문제점

- 지그재그(Zig Zag, Taco shell problem)
    
    ![image.png](assets/Lecture%207/image%201.png)
    
    - 만약, loss가 한 방향으로는 빠르게 변하고 다른 방향으로는 느리게 변하면, 지그재그로 이동하며 오랜 시간이 걸림
    - 이 문제는 차원이 높을수록 커짐 → 10억개가 넘는 차원을 가진 신경망에서 큰 문제
- 극솟값(Local minima) & 안장점(Saddle point)
    
    
    ![극솟값(Local minima)](assets/Lecture%207/image%202.png)
    
    극솟값(Local minima)
    
    ![안장점(Saddle point)](assets/Lecture%207/image%203.png)
    
    안장점(Saddle point)
    
    - 그래디언트가 0이 되어 그래디언트 강하가 멈춤
    - (직관과 다르게) 차원이 높을수록, 안장점이 극솟값보다 더 자주 발생하고 큰 문제가 됨
    - 안장점은 그 점뿐만 아니라 그 주변에서도 강하 속도가 매우 느리다는 문제 존재
- 확률적인(Stochastic)
    
    ![image.png](assets/Lecture%207/image%204.png)
    
    - mini-batch에서 얻은 그래디언트는 노이즈가 있을 수 있어 최솟점에 도착하는 데 오래 걸림

## SGD + Momentum

### SGD

$$
x_{t+1} = x_t - \alpha \nabla f(x_t)
$$

```python
while True:
    dx = compute_gradient(x)
    x += learning_rate * dx
```

### SGD + Momentum

$$
v_{t+1} = \rho v_t + \nabla f(x_t) \\x_{t+1} = x_t - \alpha v_{t+1}
$$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx + dx
    x += learning_rate * vx
```

- ‘속도’를 그래디언트의 연속 평균치로 정의
- $\rho$ 는 ‘마찰’을 의미
    - 일반적으로 $\rho$ = 0.9 or 0.99

> 이전의 문제점들을 대부분 해결
> 

- Poor Conditioning
    
    ![asdf.png](assets/Lecture%207/image%205.png)
    
- Gradient Noise
    
    ![image.png](assets/Lecture%207/image%206.png)
    

- Local Minima & Saddle points
    
    ![image.png](assets/Lecture%207/image%207.png)
    
    ![image.png](assets/Lecture%207/image%208.png)
    

## Nesterov Momentum

### Momentum update

![image.png](assets/Lecture%207/image%209.png)

원점에서의 그래디언트 활용

### Nesterov Momentum

![image.png](assets/Lecture%207/image%2010.png)

속도로 이동한 점에서의 그래디언트 활용

- 이론적으로 convex optimization에서는 매우 뛰어나지만, 신경망같은 non-convex optimization에서는 크게 적용되지 않음

### 수식 표현

- 개념적
    
    $$
    \begin{aligned}v_{t+1} &= \rho v_t - \alpha \nabla f(x_t + \rho v_t) \\x_{t+1} &= x_t + v_{t+1}\end{aligned}
    $$
    
    → $x_t$ 에서 $\nabla f(x_t)$ 를 업데이트하고 싶음
    

- 조정한 수식
    
    $\tilde{x}_t = x_t + \rho v_t$ 로 치환
    
    $$
    \begin{aligned}
    v_{t+1} &= \rho v_t - \alpha \nabla f(\tilde{x}_t) \\
    \tilde{x}_{t+1} &= \tilde{x}_t - \rho v_t + (1 + \rho)v_{t+1} \\
    &= \tilde{x}_t + v_{t+1} + \rho (v_{t+1} - v_t)
    \end{aligned}
    
    $$
    
    → $\tilde{x}_{t+1}$ 는 속도의 오차를 반영하는 형태
    

### 조정된 수식을 이용한 코드

```python
dx = compute_gradient(x)
old_v = v
v = rho * v - learning_rate * dx
x += -rho * old_v + (1 + rho) * v
```

> 오차 반영 항 때문에 vanilla에 비해 덜 overshooting 됨
> 

![image.png](assets/Lecture%207/image%2011.png)

## AdaGrad & RMSProp

### AdaGrad

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared += dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

- 과거의 제곱합을 기반으로 하여 각 성분을 스케일링한다
    - 빠르게 변하는 방향에 제동을 주고 느리게 변하는 방향을 가속해 지그재그 문제가 발생하지 않도록 함
    - 이론적으로 convex optimization에서는 매우 뛰어나지만, 신경망에서는 안장점 문제 발생 가능성

### RMSProp

```python
grad_squared = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

- grad_squared 항을 감소시키며 더한다
    - decay_rate = 0.9 or 0.99
    - 계속해서 강하가 느려질 수 있는 리스크가 있음

### Momentum과의 비교

![image.png](assets/Lecture%207/image%2012.png)

- SGD Momentum과 RMSProp은 둘 다 SGD보다 효과적
- SGD Momentum은 overshooting 이후 다시 돌아오지만, RMSProp은 경로를 조정하여 모든 차원에서 거의 옳게 진행함
- AdaGrad도 초록색으로 표기되어 있으나, learning rate가 지속적으로 감소하는 문제 때문에 멈춤.
→ 신경망을 학습시킬 때는 거의 사용하지 않음

## Adam

> *“Momentum의 운동량 개념과 AdaGrad/RMSProp의 그래디언트 제곱합으로 나누는 개념 둘 다 좋은데 동시에 사용하자.”*
> 

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - beta1) * dx
    second_moment = beta2 * second_moment + (1 - beta2) * dx * dx
    first_unbias = first_moment / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```

※ Momentum  Bias correction  AdaGrad / RMSProp

<aside>
💡

**대부분의 아키텍처에서 잘 작동함**

- Adam with beta1 = **0.9**, beta2 = **0.999**, and learning_rate = **1e-3** or **5e-4**는
대부분의 모델에서 자주 좋은 시작점
</aside>

### Momentum, RMSProp과의 비교

![image.png](assets/Lecture%207/image%2013.png)

- SGD Momentum처럼 overshoot하면서도, RMSProp처럼 옳은 방향을 찾아가려고 함

## Learning rate

> Learning rate를 학습 내내 하나로 고정할 필요는 없다.
> 
- Step decay
e.g. decay learning rate by half every few epochs.
- Exponential decay
    
    $$
    \alpha = \alpha_{0} e^{-kt}
    $$
    
- 1/t decay
    
    $$
    \alpha = \frac{\alpha_{0}}{1 + kt}
    $$
    

![image.png](assets/Lecture%207/image%2014.png)

- SGD Momentum에서 흔하고 Adam에서는 덜 흔함
- Learning rate decay는 2차 하이퍼파라미터
    - 처음부터 이를 최적화하는 경우는 드뭄
    - 시작할 때는 decay 없이 최적의 learning rate를 찾고, 이후에는 loss curve를 보며 감소가 필요한지, 필요한 부분은 어딘지 관찰하는 것이 좋음

## Second-Order Optimization

### First-Order Optimization

![image.png](assets/Lecture%207/image%2015.png)

그동안 선형 근사(1차 테일러 근사)한 것을 실제 함수라 가정하고, 그 근사를 최소화하기 위한 스텝으로 이동했음

### Second-Order Optimization

![image.png](assets/Lecture%207/image%2016.png)

1차 근사뿐만 아니라 2차 근사까지 고려하여 지역 근사의 최솟점으로 이동함.

→ 그래디언트와 Hessian를 활용

- 2차 테일러 전개(second-order Taylor expansion)
    
    $$
    J(\theta) \approx J(\theta_{0}) + (\theta - \theta_{0})^{\top} \nabla_{\theta} J(\theta_{0}) + \frac{1}{2} (\theta - \theta_{0})^{\top} H (\theta - \theta_{0})
    $$
    
- 뉴턴 매개변수 업데이트(Newton parameter update)
    
    $$
    \theta^{*} = \theta_{0} - H^{-1} \nabla_{\theta} J(\theta_{0})
    $$
    
    헤세 행렬의 역행렬을 활용하여 임계점(critical point)을 바로 찾을 수 있음
    

### 특징

- 하이퍼파라미터 없음. learning rate 없음
- 딥러닝에는 좋지 않음
    - 헤세 행렬은 $O(N^2)$개의 성분을 가지고, 역행렬 변환은 $\text{O}(N^3)$의 연산
    ($N$ = (Tens or Hundreds of) Millions)

> 준-뉴턴법(Quasi-Newton methods)
> 
> - BGFS(가장 널리 쓰임)
>     - 헤세 행렬(Hessian)을 직접 역행렬로 계산하는 O(n^3) 연산 대신, 시간에 따라 랭크 1(rank-1) 업데이트를 이용해 역헤세 행렬의 근사값을 계산.
>     - 각 업데이트의 계산 복잡도는 O(n^2)
> - L-BFGS (Limited-memory BFGS):
>     - 전체 역헤세 행렬을 생성하거나 저장하지 않음
>     - 확률적(stochatic)이거나 non-convex한 상황에서 잘 작동하지 않음

<aside>
💡

**결론(In practice)**

- Adam은 대부분의 상황에서 좋은 디폴트 선택이다.
- full batch 업데이트를 할 여유가 있다면 L-BFGS를 시도해라.
</aside>

# 규제(Regularization)

![image.png](assets/Lecture%207/image%2017.png)

![image.png](assets/Lecture%207/image%2018.png)

> 최적화는 train loss만 줄일 수 있다. train loss와 validation loss 간의 차이를 줄이기 위해서는 어떡해야 하는가?
> 

## 여러 모델에 대해: 모델 앙상블(Model Ensembles)

1. 여러 개의 독립적인 모델들을 임의의 초기화 상태로 시작하여 학습시킨다.
2. Test time에는 각각의 모델들에 대해 결과를 내어 평균한다.

→ 일반적으로 2% 정도의 적은 추가 성능

<aside>
💡

높은 성능 향상을 이루어낼 수는 없지만, **일관된 향상**을 보여줌

→ 대회에서 일반적으로 대부분 사용

</aside>

### Tricks and Tips

- 스냅샷(Snapshot)
    
    ![image.png](assets/Lecture%207/image%2019.png)
    
    - 여러 개의 모델을 학습시키는 대신, 하나의 모델을 학습시키는 과정에서 스냅샷을 찍어 각각의 스냅샷을 테스트시에 사용 후 그 결과를 평균한다.
    - learning rate를 키웠다 줄였다를 반복하여 여러 지역 극솟값을 찾고 각 극솟값에 스냅샷을 평균하면 높은 성능 향상을 이뤄낼 수 있음 (최신 2017 논문)
- 폴리아크 평균(Polyak averaging)
    
    ```python
    while True:
        data_batch = dataset.sample_data_batch()
        loss = network.forward(data_batch)
        dx = network.backward()
        x += - learning_rate * dx
        x_test = 0.995 * x_test + 0.005 * x  # use for test set
    ```
    
    - 실제 파라미터 벡터를 그대로 사용하는 대신, 파라미터 벡터의 이동 평균(moving average)을 유지하고, 테스트 시에는 그 값을 사용한다.
    - In practice, 잘 쓰이지 않음

## 하나의 모델에 대해: 규제(Regularization)

### 일반적인 Regularization 패턴

1. Training: 어떤 랜덤성을 부여한다.
    
    $$
    y = f_W(x, z)
    $$
    
2. Testing: 랜덤성을 평균한다. (가끔 근사하기도 함)
    
    $$
    y = f(x) = \mathbb{E}_{z} \left[ f(x, z) \right] = \int p(z) f(x, z) \, dz
    $$
    
- 예시
    - Batch Normalization
    매번 특정 데이터포인트가 포함되기도 하고, 포함되지 않기도 함. 매번 다른 데이터포인트끼리 묶어짐
    → 확률적+노이즈 효과로 Dropout과 비슷한 효과를 낼 수 있음
    - Dropout
    매번 네트워크의 다른 부분집합으로 학습함.
    → 매개변수 p를 조정함으로써 랜덤성을 조정할 수 있음
    - Data Augmentation
    랜덤하게 입력값을 변형시켜 학습함.
    - DropConnect
    활성을 0으로 만드는 것이 아니라 가중치 행렬의 일부를 0으로 변환함.
    - Fractional Max Pooling (일반적으로 쓰이진 않음)
    Pooling 레이어에서 고정된 부분을 풀링하는 것이 아니라 임의의 영역을 풀링함.
    - Stochastic Depth (일반적으로 쓰이진 않음)
    학습시 임의로 몇몇 레이어를 drop하고, 테스트시 모든 레이어를 사용함.

### Dropout

> L2 regularization은 신경망의 맥락에서는 타당하지 않을 수 있다
> 

![image.png](assets/Lecture%207/image%2020.png)

![image.png](assets/Lecture%207/image%2021.png)

- 각 forward pass에서 해당 레이어의 임의의 일부 뉴런을 0으로 설정한다.
    - Fully-connected 네트워크의 일부로 형성된 신경망이 동작하는 것처럼 보임
    - dropping 확률은 하이퍼파라미터, 0.5가 일반적

```python
p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
    """ X contains the data """

    # forward pass for example 3-layer neural network
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < p # first dropout mask
    H1 *= U1 # drop!
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < p # second dropout mask
    H2 *= U2 # drop!
    out = np.dot(W3, H2) + b3

    # backward pass: compute gradients... (not shown)
    # perform parameter update... (not shown)
```

- 왜 효과적인가?
    - 해석 1: handwavy(중요한 세부 정보 혹은 논리적 단계가 빠진) 해석
        
        ![image.png](assets/Lecture%207/image%2022.png)
        
        네트워크가 **중복된 표현**(redundant representation)을 가지도록 강제하고, 특징들 간의 공동 적응(co-adaptation)을 방지한다.
        
    - 해석 2: 최근 해석
        
        드롭아웃은 매개변수를 공유하는 여러 모델의 거대한 결합을 학습시키는 것과 같다.
        
        → 모델 앙상블과 비슷한 맥락
        
        - 각 binary mask는 하나의 모델을 의미 = 아주 많은 경우의 수 존재

- Test time
    
    $$
    y = f_W(x, z)
    $$
    
    $y$ : 출력, $x$ : 입력, $z$ : random mask
    
    드롭아웃은 출력이 랜덤하게 형성되게 하므로, test time에는 다음과 같이 임의성을 평균하고자 함
    
    $$
    y = f(x) = \mathbb{E}_{z} \left[ f(x, z) \right] = \int p(z) f(x, z) \, dz
    $$
    
    - 근사법
        
        ![image.png](assets/Lecture%207/image%2023.png)
        
        하나의 뉴런에 대하여 기댓값은 다음과 같음
        
        $$
        \begin{align*}
        \mathbb{E}[a] &= \frac{1}{4}(w_{1}x + w_{2}y) + \frac{1}{4}(w_{1}x + 0y) + \frac{1}{4}(0x + 0y) + \frac{1}{4}(0x + w_{2}y) \\
        &= \frac{1}{2}(w_{1}x + w_{2}y)
        \end{align*}
        $$
        
        따라서, 모든 결과값에 dropout 확률(상수)를 곱해주면 됨
        
        ```python
        def predict(X):
            # ensembled forward pass
            H1 = np.maximum(0, np.dot(W1, X) + b1) * p  # NOTE: scale the activations
            H2 = np.maximum(0, np.dot(W2, H1) + b2) * p  # NOTE: scale the activations
            out = np.dot(W3, H2) + b3
        ```
        
    
    - Inverted dropout
        
        test 대신 train 과정에서 dropout 확률을 곱해줌으로써 추론 시간을 감소시키는 방식 (더 일반적)
        
        ```python
        p = 0.5 # probability of keeping a unit active. higher = less dropout
        
        def train_step(X):
            # forward pass for example 3-layer neural network
            H1 = np.maximum(0, np.dot(W1, X) + b1)
            U1 = (np.random.rand(*H1.shape) < p) / p  # first dropout mask. Notice /p!
            H1 *= U1  # drop!
            H2 = np.maximum(0, np.dot(W2, H1) + b2)
            U2 = (np.random.rand(*H2.shape) < p) / p  # second dropout mask. Notice /p!
            H2 *= U2  # drop!
            out = np.dot(W3, H2) + b3
        
            # backward pass: compute gradients... (not shown)
            # perform parameter update... (not shown)
        
        def predict(X):
            # ensembled forward pass
            H1 = np.maximum(0, np.dot(W1, X) + b1)  # no scaling necessary
            H2 = np.maximum(0, np.dot(W2, H1) + b2)
            out = np.dot(W3, H2) + b3
        ```
        

### 데이터 증강(Data Augmentation)

![image.png](assets/Lecture%207/image%2024.png)

label을 보존한 상태로, data를 임의적으로 변형하여 원본 데이터 대신 학습에 이용한다.

- 방식들
    - Horizontal Flips
        
        ![image.png](assets/Lecture%207/image%2025.png)
        
    - Random crops and scales
        
        분할된 패치를 추출하여 학습시킨 후, 추론시 고정된 크롭을 평균
        
    
    ![image.png](assets/Lecture%207/image%2026.png)
    
    ![image.png](assets/Lecture%207/image%2027.png)
    
    - Color Jitter
    임의로 대비나 밝기 등을 변형 (e.g. PCA 방향 이동)
        
        ![image.png](assets/Lecture%207/image%2028.png)
        
    - translation, rotation, stretching, shearing, lens distortions 등등

<aside>
💡

어떠한 문제에서든, label을 보존하는 방식으로 변형을 줄 수 있다면 데이터 증강을 활용할 수 있다.

</aside>

### DropConnect

![image.png](assets/Lecture%207/image%2029.png)

활성을 0으로 만드는 것이 아니라 가중치 행렬의 일부를 0으로 변환한다.

### Fractional Max Pooling

![image.png](assets/Lecture%207/image%2030.png)

Pooling 레이어에서 고정된 부분을 풀링하는 것이 아니라 임의의 영역을 풀링한다. (일반적으로 쓰이진 않음)

### Stochastic Depth

![image.png](assets/Lecture%207/image%2031.png)

학습시 임의로 몇몇 레이어를 drop하고, 테스트시 모든 레이어를 사용한다. (일반적으로 쓰이진 않음)

<aside>
💡

일반적으로 배치 정규화(Batch Normalization)으로 충분하다. 학습시에 오버피팅이 관찰될 경우에 이를 타겟하여 다른 규제 방식을 고려할 수 있다.

</aside>

# 전이 학습(Transfer Learning)

> *“CNN을 학습시키고 사용하기 위해서는 많은 데이터가 필요하다는 생각의 파괴”*
> 

1. Train on Imagenet

![image.png](assets/Lecture%207/image%2032.png)

1. Small Dataset
(C classes)

![image.png](assets/Lecture%207/image%2033.png)

1. Bigger Dataset

![image.png](assets/Lecture%207/image%2034.png)

1. 큰 데이터셋으로 신경망을 학습시킨다.
2. 적은 데이터(C개의 클래스)로 학습시킬시, 가장 마지막 레이어만 C 차원으로 바꾸고 행렬을 랜덤하게 재초기화한다.
linear classifier와 가장 마지막 레이어만 학습시키고, 나머지 레이어들은 학습된 파라미터를 그대로 이용한다.
3. 이후 더 많은 데이터가 있으면, 더 많은 레이어를 파인튜닝할 수 있다.
이때, learning rate는 원래 값의 1/10 정도가 적당하다. 기존에 큰 데이터셋으로 학습된 신경망은 꽤 보편적으로 잘 작동할 것이기 때문이다.

### 전략

※ 위 레이어일수록 구체적, 아래 레이어일수록 일반적인 특징을 학습

|  | 기존과 아주 비슷한 데이터셋 | 기존과 아주 다른 데이터셋 |
| --- | --- | --- |
| 아주 적은 데이터 | 가장 위 레이어에서 선형 분류기 사용 | (문제..) 여러 계층에서 선형 분류기 시도 |
| 꽤 많은 데이터 | 몇 개의 적은 레이어 파인튜닝 | 더 많은 레이어 파인튜닝 |

<aside>
💡

전이 학습은 예외가 아니라 대부분의 연구에 만연해있다.

1. 비슷한 데이터의 큰 데이터셋을 찾아 이를 이용해 큰 신경망을 학습시킨다. (Model Zoo 활용)
2. 내 데이터셋으로 전이 학습시킨다.
</aside>