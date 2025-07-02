# Lecture 3 | Loss Functions and Optimization

Linear Classifier = eyeballing: 대략적으로 추정하는 것

→ handwavy(중요한 세부 정보 혹은 논리적 단계가 빠진) approach

해결:

- Loss function: W가 얼마나 나쁜지 측정
- Optimization procedure: 가능한 W space를 탐색하며 가장 덜 나쁜 W를 찾는 것

# Loss Functions

$$
L = \frac{1}{N} \sum_{i} L_i\left(f(x_i, W), y_i\right)
$$

xi: 이미지 (pixel data)

yi: 라벨/타겟 (integer)

Li: loss function = 예측과 정답을 입력받아 얼마나 나쁜지 값을 출력

L: final loss

## 1) multiclass SVM loss

$$
s = f(x_i, W)
$$

$$
\begin{align*}
L_i &= \sum_{j \ne y_i} 
\begin{cases}
0 & \text{if } s_{y_i} \ge s_j + 1 \\
s_j - s_{y_i} + 1 & \text{otherwise}
\end{cases} \\
&= \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)
\end{align*}
$$

(safety margin = 1 in this case)

s_yi: 정답 class의 score

s_j: 다른 class의 score

→ 정답 class의 점수가 다른 class에 비해 높을수록 loss가 작아짐

⇒ “Hinge loss” (s_yj가 s_j + 1 이상이면 0, 그 이하는 일차함수)

1. Q1: What happens to loss if car scores change a bit?
A: Car의 정답 class의 score가 다른 것보다 월등히 높으므로 0으로 유지
2. Q2: What is the min/max possible loss?
A: min = 0, max = infinity
3. Q3: At initialization W is small so all s ≈ 0. What is the loss?
A: (# of classes) - 1 → debug에 유용 (첫 iteration에 c - 1이 아니면 버그 의심)
4. Q4: What if the sum was over all classes? (including j = y_i)
A: loss가 기존보다 1 커짐
5. Q5: What if we used mean instead of sum?
A: 답은 변하지 않음 (# of classes가 불면이기 때문)

1. Q6: What if we used
A: 다른 classification algorithm.
다른 loss function임.
크게 틀린 것에 대해 더 큰 패널티

$$
\begin{align*}
L_i &= \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)^2
\end{align*}
$$

1. E.g. Suppose that we found a W such that L = 0. Is this W unique?
A: No (반례, 2W)

## 2) Multinomial Logistic Regression (Softmax Classifier)

(현실에서 더 잘 쓰임)

multiclass SVM loss에서는 정답의 score가 다른 것에 비해 높길 바랄 뿐 각 score가 가지는 의미를 해석하지 않음

Multinomial Logistic Regression은 각 score가 의미를 가짐

= 확률 분포 over classes (Softmax function)

- Softmax function

$$
P(Y = k \mid X = x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}} \quad \text{where} \quad \mathbf{s} = f(x_i; W)
$$

- Softmax loss (cross-entropy loss)

$$
L_i = -\log P(Y = y_i \mid X = x_i) = -\log\left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)
$$

1. Q1: What is the min/max possible loss L_i?
A: min = 0, max = infinity (정확히 0이 나올 순 없음. 이론적으로)
2. Q2: Usually at initialization W is small so all s ≈ 0. What is the loss?
A: log c (= -log (1/c)) → → debug에 유용 (첫 iteration에 log c가 아니면 버그 의심)

### multiclass SVM loss

correct score가 incorrect score에 margin을 더한 것 보다 크기만 하면 됨

**→ data point가 특정 기준을 넘어가면 더이상 고려하지 않음**

### Softmax Classifier

모든 확률 분포의 합이 1이 되므로 correct score의 확률 질량을 계속 키우려고 함
= correct score 양의 무한으로, incorrect score 음의 무한으로

**→ 모든 data point에 대해 지속적으로 개선하려 함**

※ 사실 딥러닝에서는 크게 차이 없음

## Regularization

Data loss: training data에서 잘 작동하는 것이 아니라 test data에서 잘 작동하는 것을 찾아야 함

= training data에 너무 과하게 fit되면 안됨

→ Regularization의 개념을 활용하여 해결! (오컴의 면도날)

$$
L(W) = \frac{1}{N} \sum_{i=1}^{N} L_i\left(f(x_i, W), y_i\right) + \lambda R(W)
$$

R: regularization term loss function → 더 simple한 W를 pick할 수 있도록 

lambda: regularization strength (hyperparameter)

- L2 regularization

$$
R(W) = \sum_k \sum_l W_{k,l}^2
$$

- L1 regularization

$$
R(W) = \sum_k \sum_l |W_{k,l}| 
$$

- Elastic net (L1 + L2)

$$
R(W) = \sum_k \sum_l \beta W_{k,l}^2 + |W_{k,l}|
$$

- Max norm regularization
- Dropout
- Fancier: Batch normalization, stochastic depth

# Optimization

loss를 최소화하는 W를 어떻게 찾는가에 대한 주제

## Strategy #1: Random search

Very bad idea solution

## Strategy #2: Gradient Descent

Follow the slope

$$
\frac{d f(x)}{d x} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

여기서 x는 vector

- gradient = 각 차원에 대한 partial derivative (편미분)으로 구성된 벡터
    - 어떤 방향에서의 기울기는 그 방향 벡터와 그래디언트의 내적
    - 가장 가파르게 감소하는 방향은 그래디언트의 음의 방향(negative gradient)

gradient를 찾는 법

- Numerical gradient: finite difference → 근사, 느림, 작성 쉬움
⇒ Analytic gradient를 debug하는 용도로 좋음 (gradient check)
- Analytic gradient: calculus → 정확, 빠름, error-prone
⇒ always use analytic gradient!
    
    $$
    \text{find } \nabla_W L
    $$
    

### Gradient Descent

```python
while True:
	weights_grad = evaluate_gradients(loss_fun, data, weights)
	weights += -step_size * weights_grad
```

gradient의 음의 방향으로 이동하며 수렴하길 기대

여기서 step_size는 hyperparameter → 그래디언트의 음의 방향으로 얼마나 이동할 건지
= learning rate ⇒ 가장 중요한 single hyperparameter 중 하나

### Stochastic Gradient Descent (SGD)

$$
L(W) = \frac{1}{N} \sum_{i=1}^{N} L_i\left(f(x_i, W), y_i\right) + \lambda R(W)
$$

현실에서 N은 매우 큼 → 모든 loss를 계산하는 것이 expensive (N번의 evaluation)

$$
\nabla_W L(W) = \frac{1}{N} \sum_{i=1}^{N} \nabla_W L_i(x_i, y_i, W) + \lambda \nabla_W R(W)
$$

gradient도 마찬가지. linear operator이므로 N개의 term에 대한 gradient의 합이 전체 loss의 gradient

→ 전체 loss에 대한 gradient 계산도 expensive (N번의 iteration)

⇒ minibatch 활용 (예. 32, 64, 128 등) → estimate

```python
while True:
	data_batch = sample_training_data(data, 256)
	weights_grad = evaluate_gradients(loss_fun, data, data_batch)
	weights += -step_size * weights_grad
```

## Aside: Image Features

raw pixels를 linear classifier에 넣는 것은 잘 작동하지 않음

⇒ 2-stage approach

1. image에서 여러 feature representation를 계산
    1. 이미지의 모습과 관련된 여러 값들을 계산
    2. 서로 다른 feature vector들을 이어서 feature representation을 만듦
2. feature representation을 linear classifier에게 입력

→ linear classifier로 분리할 수 없는 점들을 분리할 수 있는 방법

예. Color Histogram, Histogram of Oriented Gradients (HoG), Bag of Word

⇒ 예전엔 feature를 하나하나 입력해서 학습시켰지만, Convolution Neural Network로 넘어오고 나서부터는 그 feature를 데이터로부터 학습함