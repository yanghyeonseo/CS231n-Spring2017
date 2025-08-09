# Lecture 6 | Training Neural Networks I

# One time setup

## Activation Functions

- Sigmoid
    
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
    
    - 숫자를 [0, 1] 범위로 재설정
    - 포화(saturating)하는 뉴런의 ‘발화율(firing rate)’로 깔끔하게 해석되기 때문에 역사적으로 인기가 많았음. (현재는 ReLU와 같은 다른 비선형 함수가 더 타당함)
    - 문제점
        - 포화된 뉴런은 그래디언트를 “죽인다” (∵ 입력의 절댓값이 클수록 그래디언트가 0에 가까워짐)
        - 출력이 zero-centered가 아님 → w에 대한 그래디언트가 모두 양이거나 음이므로 비효율적인 업데이트
        
        ![image.png](assets/Lecture%206/image.png)
        
        - exp()는 계산 비용이 다소 비쌈 (다른 내적 등에 비하면 큰 영향 X)
- tanh(x)
    
    $$
    f(x) = \tanh(x)
    $$
    
    - 숫자를 [-1, 1] 범위로 재설정
    - 출력이 zero-centered임
    - 문제점
        - 포화된 뉴런은 그래디언트를 “죽인다”
- ReLU(Rectified Linear Unit)
    
    $$
    f(x) = \max(0, x)
    $$
    
    ※ x = 0에서 그래디언트는 0 (in practice)
    
    - 양의 범위에서 포화하지 않음
    - 매우 효율적인 계산
    - sigmoid, tanh보다 훨씬 빠르게 수렴
    - sigmoid보다 생물학적으로 더 타당
    - 문제점
        - zero-centered가 아닌 출력
        - 양의 범위에서 포화하지 않음
        - dead ReLU
            - 결과값이 음수일 때 그래디언트가 0이므로 앞으로 절대 활성화되지 않고 업데이트도 되지 않음
            - initialization, 너무 큰 learning rate일 때 발생

- Leaky ReLU
    
    $$
    f(x) = \max(0.01x, x)
    $$
    
    - 모든 범위에서 포화하지 않음

- Parametric Rectifier (PReLU)
    
    $$
    f(x) = \max(\alpha x, x)
    $$
    
    (alpha는 하이퍼파라미터)
    
    - 모든 범위에서 포화하지 않음

- Exponential Linear Units (ELU)
    
    $$
    f(x) = \begin{cases}
    x & \ (x > 0)\\
    \alpha(\exp(x) - 1) & \ (x \le 0)
    \end{cases}
    $$
    
    - 출력의 평균이 0인 것에 가까움
    - Leaky ReLU에 비해, 음수 영역에서 포화되는 구간이 잡음에 대한 강인성을 더해줌
    - 문제점
        - exp()의 다소 비싼 계산 비용

- Maxout Neuron
    
    $$
    f(x) = \max(w_{1}^{T}x + b_{1},\; w_{2}^{T}x + b_{2})
    $$
    
    - ReLU와 Leaky ReLU의 일반화
    - 선형 구간
    - 포화하지 않음
    - ‘die’ 문제 없음
    - 문제점
        - 매개변수와 뉴런의 수가 2배

### In practice…

| ReLU | Leaky ReLU / Maxout / ELU | tanh | sigmoid |
| --- | --- | --- | --- |
| 사용해라 | 시도해라 | 시도하되 크게 기대하지 말아라 | 사용하지 마라 |

## Data Preprocessing

### Step 1: Preprocess the data

![image.png](assets/Lecture%206/image%201.png)

입력값이 양이나 음에 몰려있을 때 그래디언트 업데이트가 비효율적으로 이뤄지는 현상
→ 부호뿐만 아니라 어떠한 종류의 편향도 영향을 줌

이를 막기 위해 평균을 빼줌으로써 zero-center가 되게 만들고, 표준편차로 나누어 정규화한다

### In practice…

- (머신러닝에서) PCA나 Whitening 기법도 사용됨

![image.png](assets/Lecture%206/image%202.png)

- (이미지에서) center만 0으로 맞춤. 분산까지 조정하는 경우는 많지 않음
→ 일반적인 머신러닝과 다르게, 이미지는 각 위치에서 상대적으로 비교할만한 값의 범위와 분포를 지니기 때문
    - 평균 이미지 빼기: 픽셀 단위(H×W×C) 평균을 빼기 (평균 = [H, W, C] array)
    - 채널별 평균 빼기: 채널 단위(C) 평균을 빼기 (평균 = 숫자 C개)

## Weight Initialization

- 만약 W = 0으로 초기화한다면?
    - 모든 뉴런이 같은 동작(동일한 연산)을 할 것이다
    - 같은 출력을 내보내므로 같은 그래디언트를 가진다
    - 동일하게 업데이트되므로 결국 모든 뉴런이 같아지게 된다
- First idea: 작은 랜덤 숫자들
    
    ```python
    W = 0.01 * np.random.randn(D, H)
    ```
    
    - 작은 네트워크에서는 괜찮지만, 큰 네트워크에서는 모든 활성 지도가 0이 되는 문제 발생
    ∵ 각 레이어의 작은 입력값과 작은 W을 계속 곱하기 때문
    - W*X의 그래디언트 또한 입력값인 X이므로 0에 가까워 업데이트도 일어나지 않음
- 큰 랜덤 숫자들
    
    ```python
    W = 1.0 * np.random.randn(D, H)
    ```
    
    - 대부분의 뉴런이 포화상태가 되는 문제 발생
    - (tanh 활성함수 기준) 활성 함수의 그래디언트가 0에 가까워 업데이트도 일어나지 않음

### Xavier initialization

```python
W = np.random.randn(D, H) / np.sqrt(D)
```

- 입력의 분산이 출력의 분산과 같도록 하기 위한 가중치를 도출하여 얻은 공식
- 직관적으로, 입력의 개수가 적으면 W의 값이 커지고 이들을 내적해서 얻은 출력은 큰 분산을 갖게 됨.
마찬가지로, 입력의 개수가 많으면 W의 값이 작아지고 이들을 내적해서 얻은 출력은 작은 분산을 갖게 됨.

→ 각 레이어에서 근사적으로 unit gaussian의 효과를 냄 (tanh의 active region에 있다는 가정 하에)

### ReLU 사용 시 공식

```python
W = np.random.randn(D, H) / np.sqrt(D/2)
```

※ ReLU를 사용하는 네트워크에서 Xavier initialization을 그대로 사용하면 붕괴됨. 매번 절반 정도의 값이 0으로 바뀌고 분산 또한 매번 절반 정도로 감소하기 때문

## Batch Normalization

> *“가우시안 활성을 원한다면, 그렇게 만들면 된다.”*
> 

특정 레이어의 활성의 집단을 생각하자. 시작하기 전에 각 레이어에 대해 정규화를 진행하고 이를 유지하면 된다.

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}
$$

1. 각 차원에 대해 독립적으로 경험적인 평균과 분산을 계산한다.
    
    ![image.png](assets/Lecture%206/image%203.png)
    
2. 정규화한다.
    
    $$
    \hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\mathrm{Var}[x^{(k)}]}}
    $$
    
3. 일반적으로 Fully-Connected 레이어 또는 Convolutional 레이어 뒤에, 그리고 nonlinearity 레이어 전에 삽입한다.
    
    → 가중치(W)를 곱하면서 나쁜 scaling 효과가 발생하는데, 이것을 원상태로 돌리는 과정
    

![image.png](assets/Lecture%206/image%204.png)

> 다만, tanh 레이어에 표준정규화된 입력을 넣는 것이 필요한가?
> 
> 
> → 애초에 tanh 레이어의 역할이 linearity를 nonlinearity로 만드는 것이기 때문에 필요 없을 수 있으나,
> 정규화시 포화를 얼마나 많이 얻을지 정할 수 있다는 장점이 있음
> 

다음과 같이 신경망이 범위를 정하도록 할 수 있음.

$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$

※ 다음과 같이 베타와 감마를 정하면 원래의 매핑으로도 설정할 수 있음

→ 더 많은 학습의 유연성

$$
\gamma^{(k)} = \sqrt{\mathrm{Var}[x^{(k)}]} \\\beta^{(k)} = \mathbb{E}[x^{(k)}]
$$

### 요약

- 입력

$$
\text{Values of } x \text{ over a mini-batch: } \mathcal{B} = \{x_1, \dots, x_m\}; \\
\quad \text{Parameters to be learned: } \gamma, \beta
$$

- 출력
    
    $$
    \{ y_i = \text{BN}_{\gamma, \beta}(x_i) \}
    $$
    
    - 계산 방법
        
        $$
        \begin{aligned}
        \mu_\mathcal{B} &= \frac{1}{m} \sum_{i=1}^{m} x_i \\
        \sigma^2_\mathcal{B} &= \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2 \\
        \hat{x}_i &= \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} \\
        y_i &= \gamma \hat{x}_i + \beta \equiv \text{BN}_{\gamma, \beta}(x_i)
        \end{aligned}
        $$
        
- 참고
    - BatchNorm 레이어는 Test time에서는 다르게 동작함
        - 평균과 분산을 계산하지 않음
        - 하나의 고정된 값을 사용함. (e.g. training time에 추정한 값)

### 효과

- 네트워크 내의 그래디언트 흐름을 개선
- 강건성 상승
    - 높은 learning rate 가능
    - 초기화에 대한 의존 감소
- 일종의 regularization으로 작동
∵ 해당 input뿐만 아니라 전체 input을 반영해 평균과 분산을 계산하기 때문

# Training dynamics

## Babysitting the Learning Process

### Step 1: Preprocess the data

- zero-center (평균 빼기)
- normalization (표준편차로 나누기)

### Step 2: Choose the architecture

- 아키텍쳐 정하기
- loss가 타당한지 확인하기 (sanity check)
    - e.g. W가 0으로 초기화된 상태에서 softmax loss는 -log(1/10) (N=10, regularization 무시할 때)

### Step 3: Start try to train

- 학습 시험
    - 학습 데이터의 아주 적은 일부한 학습 시도. regularization은 0으로 설정
    - 이때 overfit이 가능해야 하므로, loss가 0에 가까워야 함
- 작은 regularization으로 시작하여 loss가 감소하는 learning rate 찾기
    - loss가 거의 변하지 않으면 → learning rate가 너무 작은 것
        - loss가 아주 작게 변했어도 train/val accuracy가 크게 증가할 수 있음
        - 그 이유는 확률이 널리 퍼져있으므로 학습과정에서 옳은 방향으로 살짝 치우치도 loss는 크게 변화하지 않는 것처럼 보일 수 있기 때문
    - loss가 폭팔하면 (대체로 NaN) → learning rate가 너무 큰 것

## Hyperparameter Optimization

stage에 따라 거친 cross-validation → 세밀한 cross-validation 순서로

1. First stage
    1. 적은 epoch으로 매개변수들의 작동하는지 대략적으로 확인
        1. 폭발(explosion)을 확인하는 조언: cost가 기존 cost의 3배 이상 커지면 반복문 탈출
2. Second stage
    1. 더 긴 실행시간 동안 더 질 좋은(finer) 탐색
    2. (필요하다면 stage 1, 2를 반복)

<aside>
💡

**Log space**에서 최적화하는 것이 좋다!

</aside>

```python
max_count = 100
for count in xrange(max_count):
    ***reg = 10**uniform(-5, 5)***
    ***lr = 10**uniform(-3, -6)***

    trainer = ClassifierTrainer()
    model = init_two_layer_model(32*32*3, 50, 10)  # input size, hidden size, number of classes
    trainer = ClassifierTrainer()
    best_model_local, stats = trainer.train(X_train, y_train, X_val, y_val,
                                            model, two_layer_net,
                                            num_epochs=5, reg=reg,
                                            update='momentum', learning_rate_decay=0.9,
                                            sample_batches=True, batch_size=100,
                                            learning_rate=lr, verbose=False)
```

결과를 보고, 다음과 같이 범위를 조정해나간다.

```python
    ***reg = 10**uniform(-4, 0)***
    ***lr = 10**uniform(-3, -4)***
```

만약, 결과에서 높은 accuracy가 경계값 주변에서 나온다면, 모든 범위를 탐색한 것이 아니므로 범위를 더 넓혀서 다시 진행한다.

<aside>
💡

현실에서, Grid Layout과 Random Layout 중 **Random Layout**이 더 좋은 구간을 탐색할 가능성이 높다!

</aside>

![image.png](assets/Lecture%206/image%205.png)

![image.png](assets/Lecture%206/image%206.png)

### Monitoring or Tracking

- learning rate
    
    ![image.png](assets/Lecture%206/image%207.png)
    
    - 처음에 가파르게 감소하고 이후에도 감소세가 유지되는 값이 최적의 값
- Initialization
    
    ![image.png](assets/Lecture%206/image%208.png)
    
    - 처음에 loss가 줄지 않다가 이후 가파르게 주는 것은 나쁜 초기화를 의심해볼 수 있음
- overfitting
    
    ![image.png](assets/Lecture%206/image%209.png)
    
    - Train과 Val 정확도의 차이가
        - 너무 큼 = 오버피팅 → regularization strength를 증가
        - 너무 작음 = 오버피팅 도달 X → 모델 capacity 증가
- ratio of (weight updates) / (weight manitudes)
    
    ```python
    # assume parameter vector W and its gradient vector dW
    param_scale = np.linalg.norm(W.ravel())
    update = -learning_rate*dW # simple SGD update
    update_scale = np.linalg.norm(update.ravel())
    W += update # the actual update
    print update_scale / param_scale # want ~1e-3
    ```
    
    - 업데이트와 값의 비율이 0.001 정도가 되어야 함