# Lecture 10 | Recurrent Neural Networks

## Recap

- ILSVRC’14는 배치 정규화(Batch Normalization)가 발명되기 전이어서 깊은 네트워크에 대한 학습이 어려웠음
    - GoogLeNet, VGG를 포함한 이전의 모델들은 깊은 네트워크에 대해 학습 어려움을 겪음
    - e.g. GoogLeNet의 보조 분류 출력(auxiliary classification outputs)은 배치 정규화를 이용하면 필요하지 않음
- 잔차 블록
    - L2 정규화(L2 Regularization)의 해석 부여
        - 일반적인 신경망에서 L2 정규화의 해석은 모호했지만, 잔차 블록은 $F(x) = 0$ 일 때 항등 매핑의 역할을 하므로 ‘필요없는 레이어는 사용하지 않는다’는 의미로 해석 가능
    - 그래디언트 흐름의 고속도로의 역할
        - 역전파에서 이전 단계 기울기(upstream gradient)가 들어오면 addition gate에서 두 개의 다른 경로로 갈라짐
        - 하나는 합성곱 레이어를 통해, 하나는 잔차 연결을 통해 직접 연결되는데, 쌓인 잔차 연결은 그래디언트 흐름의 일종의 고속도로 역할
        - 학습을 더 쉽고 빠르게 할 수 있음. 모델이 합리적으로 수렴할 수 있도록 함.

<aside>
💡

그래디언트 흐름에 대한 개념은 모든 머신러닝에서 중요하며, 순환 신경망(RNN, Recurrent Neural Network)에서 널리 퍼져있다

</aside>

- 파라미터
    - AlexNet과 VGG는 매우 많은 파라미터를 갖고 있는데, 그중 대부분은 FC 레이어에서 나옴
    - GoogLeNet과 ResNet에서는 이를 전역 평균 풀링(Global Average Pooling) 레이어로 대체하면서 비슷한 역할을 훨씬 더 적은 파라미터로 수행

# 순환 신경망(Recurrent Neural Network)

## 특징: 가변 입출력

> 평범한 신경망은 입출력의 수가 고정이지만, 순환 신경망은 입출력의 수가 가변적이다.
> 

![image.png](assets/Lecture%2010/image.png)

| 입력 및 출력 | 적용 예시 | 입출력 예시 |
| --- | --- | --- |
| 1 → N | 이미지 캡셔닝(Image Captioning) | 이미지 → 단어 시퀀스 |
| N → 1 | 감정 분류(Sentiment Classification)
비디오 분석 | 단어 시퀀스 → 단어
비디오(사진 시퀀스) → 단어 |
| N → M | 기계 번역(Machine Translation) | 단어 시퀀스 → 단어 시퀀스 |
| N → N | 프레임 수준 비디오 분류
(Video classification on frame level) | 각 비디오 프레임 → 단어 |

> 입출력의 수가 고정이어도 순환 신경망은 유용하다.
> 
- 이미지 → 정수 분류
    - 처음에 이미지를 둘러보고, 일련의 ‘흘깃 보기(glimpse)’를 통해 이미지를 분류
    
    → 입력 길이가 가변적
    
- 정수 → 이미지 생성
    - 이미지의 여러 부분을 시간에 따라 순차적으로 생성
    
    → 출력 길이가 가변적
    

## 작동 방식

![image.png](assets/Lecture%2010/image%201.png)

> RNN 내부에는 은닉 상태(hidden state)가 존재한다
> 
1. RNN 코어 블록에 입력값이 들어온다
2. 은닉 상태를 업데이트한다
3. 출력을 생성한다
4. 다음 입력값이 들어올 때 이 은닉 상태를 피드백(feedback)한다.

### 점화식(Recurrence Formula)

$$
h_{t} = f_{W}\left(h_{t-1}, x_{t}\right)
$$

- $f_W$ : 파라미터 $W$를 가진 함수
- $x_t$ : 시점 $t$에서의 입력 벡터
- $h_t$ : 새로운 상태
- $h_{t-1}$ : 이전 상태 (일반적으로 $h_0 = 0$ 초기화)

 ※ 모든 시점에서 동일한 함수와 파라미터가 사용된다

→ 만약 매시점 출력을 생성하고 싶다면, $h_t$를 입력받는 새로운 FC 레이어를 추가해 해당 결과를 출력

## 기본 순환 신경망

$$
\begin{align}h_t &= \tanh \left( W_{hh} h_{t-1} + W_{xh} x_t \right) \\y_t &= W_{hy} h_t\end{align}
$$

![image 2.png](assets/Lecture%2010/image%202.png)

- 모든 입력값 $x_t$ 를 소비할 때까지 작동
- 다른 작은 신경망에 $h_t$를 입력함으로써 $y_t$를 출력
- 매 시점에 정답 라벨이 있다면 각 시점마다 독립적인 손실값 $L_t$을 계산 가능
    - 최종 손실값 $L$은 각 시점 손실값의 합
- 역전파시 $W$의 그래디언트는 각 시점의 독립된 그래디언트의 합

> Many-to-One & One-to-Many 신경망
> 

![Many to One](assets/Lecture%2010/image%203.png)

Many to One

![One to Many](assets/Lecture%2010/image%204.png)

One to Many

> Sequence-to-Sequence 신경망
= Many-to-One + One-to-Many
> 

![image.png](assets/Lecture%2010/image%205.png)

- 인코더(Encoder) = Many-to-One : 입력 시퀀스($x$)를 하나의 벡터($h_T$)로 요약하여 출력
- 디코더(Decoder) = One-to-Many : 하나의 입력 벡터($h_T$)로 출력 시퀀스($y$)를 생성

## 시간 역전파 vs 부분 시간 역전파

### 시간 역전파(Backpropagation through time)

![image.png](assets/Lecture%2010/image%206.png)

- 전체 시퀀스를 순전파하여 손실을 계산하고 이후 전체 시퀀스를 역전파하여 그래디언트를 계산한다.
    - 매우 느리고, 많은 메모리 소요

### 부분 시간 역전파(Truncated Backpropagation through time)

![1단계](assets/Lecture%2010/image%207.png)

1단계

![2단계](assets/Lecture%2010/image%208.png)

2단계

![3단계](assets/Lecture%2010/image%209.png)

3단계

- 정해진 단계 수만큼 서브시퀀스를 순전파하여 손실값을 계산한 후 해당 서브시퀀스를 역전파한다.
- 도출된 은닉 상태를 입력한 이후의 서브시퀀스에 대해 이를 반복한다.
    - 시간 역전파와 완전히 동일한 순전파가 수행됨

→ 그래디언트를 계산하기 위해 비용이 비싼 전체가 아닌 미니 배치 사용

# 순환 신경망 모델 예시

## 문자 단위 언어 모델(Character-level Language Model)

> 주어진 문자들 뒤에 어떤 문자가 올지 예측하는 모델
> 
- “hello” 시퀀스를 학습하는 과정
    
    ![image.png](assets/Lecture%2010/image%2010.png)
    
- ‘h’를 입력했을 때 뒤따르는 문자를 합성하는 과정
    
    ![image 7.png](assets/Lecture%2010/image%2011.png)
    
- 활용 예시

![셰익스피어풍 희극](assets/Lecture%2010/image%2012.png)

셰익스피어풍 희극

![LaTeX 문서](assets/Lecture%2010/image%2013.png)

LaTeX 문서

![C 코드](assets/Lecture%2010/image%2014.png)

C 코드

셰익스피어풍의 희극을 작성하고, LaTeX 문서를 작성하며, C 코드를 코딩할 수 있다. 의미를 해석할 수 없는 부분도 많지만, 희극의 대사, LaTeX의 증명과 그래프, C 코드의 들여쓰기와 조건문 등 구조를 나타낼 수 있다.

→ 형식에 대한 설명 없이, 단순히 다음에 오는 문자를 예측하는 것만으로도 학습 시 잠재된 구조를 습득할 수 있다.

### 의미론적 해석

> Karpathy, Johnson, and Fei-Fei
은닉 벡터에서 해석할 수 있는 셀을 찾으려 노력했다
> 

![image.png](assets/Lecture%2010/image%2015.png)

일반적으로, 아래와 같이 해석할 수 없는 경우가 많았지만,

![image.png](assets/Lecture%2010/image%2016.png)

아래와 같이 해석이 가능한 경우도 있었다.

![인용문 감지 셀](assets/Lecture%2010/image%2017.png)

인용문 감지 셀

![줄 길이 추적 셀](assets/Lecture%2010/image%2018.png)

줄 길이 추적 셀

![조건문 셀](assets/Lecture%2010/image%2019.png)

조건문 셀

![주석 셀](assets/Lecture%2010/image%2020.png)

주석 셀

![코드 깊이 셀](assets/Lecture%2010/image%2021.png)

코드 깊이 셀

## 이미지 캡셔닝(Image Captioning)

> Karpathy et al., 2015
합성곱 신경망에 이미지를 입력하여 요약 벡터를 출력하고, 그것을 순환 신경망에 입력하여 캡션을 생성한다.
> 

![image.png](assets/Lecture%2010/image%2022.png)

- 학습하는 과정
    
    ![image.png](assets/Lecture%2010/image%2023.png)
    
    - 입력값과 은닉 상태만 존재하는 기존의 점화식을 그림 요약 벡터도 입력할 수 있도록 수정
        
        $$
        h = \tanh \left( W_{xh} \, x \;+\; W_{hh} \, h \;+\; W_{ih} \, v \right)
        $$
        
    - 캡션을 시작한다는 START 토큰으로 시작
    - 캡션을 끝낸다는 END 토큰이 출력되면 종료

- 결과 예시
    - 성공 사례
        
        ![image.png](assets/Lecture%2010/image%2024.png)
        
    - 실패 사례
        
        ![image.png](assets/Lecture%2010/image%2025.png)
        
        - 학습 데이터에 포함되지 않은 경우에 대해 잘못된 캡션을 출력

## 어텐션(Attention) 기반 이미지 캡셔닝

> Xu et al., 2015
순환 신경망은 각 단어를 생성할 때마다 서로 다른 공간적 위치에 주의를 집중한다
> 

![image.png](assets/Lecture%2010/image%2026.png)

![image.png](assets/Lecture%2010/image%2027.png)

- 합성곱 신경망에서 전체 이미지를 요약하는 단일 벡터가 아니라 각 벡터가 이미지의 공간적 위치를 요약한 벡터 그리드를 출력한다
- 순전파의 매 단계마다 어휘를 추출할 뿐만 아니라 보고자 하는 이미지의 위치 분포도 생성한다
    - 이 위치 분포는 학습 동안 모델이 어디를 바라봐야 하는지 나타낸 일종의 긴장도

![image.png](assets/Lecture%2010/image%2028.png)

- 학습이 완료된 이후에, 생성된 캡션의 각 단어마다 주의가 이동하는 것을 확인할 수 있음
    - Soft attention은 이미지의 전체 공간적 위치 특징의 가중치 결합
    - Hard attention은 이미지의 특정 한 위치만 본 것

- 결과
    
    ![image.png](assets/Lecture%2010/image%2029.png)
    
    - 캡션을 생성할 때 모델이 가장 중요하거나 의미론적으로 의미있는 부분에 집중하는 것을 볼 수 있음

- 응용: 시각적 질문 응답 시스템(Visual Question Answering)
    - 방식
        
        ![image.png](assets/Lecture%2010/image%2030.png)
        
        - 이미지와 자연어 질문 및 선지를 입력
        - 모델이 정답을 선택
    - 원리
        
        ![image.png](assets/Lecture%2010/image%2031.png)
        
        - RNN이 질문의 자연어 시퀀스를 입력받아 요약한 하나의 벡터를 출력
        - CNN이 이미지를 입력받아 요약한 하나의 벡터를 출력
        
        → 두 출력을 결합해 정답의 분포를 예측
        

# 다층 순환 신경망(Multilayer Recurrent Neural Network)

![image.png](assets/Lecture%2010/image%2032.png)

- 하나의 단계에서 하나의 은닉 상태가 아니라 은닉 상태 시퀀스가 만들어짐
- 일반적으로 2~3 레이어 모델이 쓰이며, 매우 깊은 모델은 RNN에서는 쓰지 않음

→ 깊은 CNN이 여러 문제에서 좋은 성능을 내듯, 깊은(2~3개의 레이어) RNN도 마찬가지임

### Multilayer RNN

$h \in \mathbb{R}^n, \quad W^l \in \mathbb{R}^n\times\mathbb{R}^{2n}$ 일 때,

$$
h_t^l = \tanh \, W^l \begin{pmatrix}h_t^{l-1} \\h_{t-1}^l\end{pmatrix}
$$

# RNN 그래디언트 흐름

### 기본 RNN 그래디언트 흐름

![image.png](assets/Lecture%2010/image%2033.png)

$$
\begin{aligned}h_t &= \tanh \left( W_{hh} h_{t-1} + W_{xh} x_t \right) \\    &= \tanh \left(         \begin{pmatrix}        W_{hh} & W_{hx}        \end{pmatrix}        \begin{pmatrix}        h_{t-1} \\        x_t        \end{pmatrix}      \right) \\    &= \tanh \left(         W        \begin{pmatrix}        h_{t-1} \\        x_t        \end{pmatrix}      \right)\end{aligned}
$$

$h_t$에서 $h_{t-1}$로의 역전파 과정에서, $W$(정확히는 $W_{hh}^\top$)를 곱하게 된다

![image.png](assets/Lecture%2010/image%2034.png)

→ $h_0$의 기울기를 계산하는 과정에는 다수의 $W$ 항(그리고 반복되는 $\tanh$)이 포함되게 됨

행렬곱이 반복되는 결과로,

- 가장 큰 특이값이 1보다 크면: 그래디언트 폭발(Exploding gradients)
    - 해결)
    그래디언트 클리핑(Gradient clipping): 그래디언트의 노름(norm)이 너무 크면 크기를 축소(스케일링)한다
        
        ```bash
        grad_norm = np.sum(grad * grad)
        if grad_norm > threshold:
            grad *= (threshold / grad_norm)
        ```
        
- 가장 큰 특이값이 1보다 작으면: 그래디언트 소실(Vanishing gradients)
    - 해결)
    RNN 아키텍처를 변경한다

## LSTM (Long Short Term Memory)

> RNN의 장기 의존성 문제(gradient vanishing/exploding)를 해결하기 위한 구조
> 

$W^l\in\mathbb{R}^{4n}\times\mathbb{R}^{2n}$ 일 때,

$$
\begin{pmatrix}i \\f \\o \\g\end{pmatrix}=\begin{pmatrix}\sigma \\\sigma \\\sigma \\\tanh\end{pmatrix}W^l\begin{pmatrix}h_t^{l-1} \\h_{t-1}^l\end{pmatrix}
$$

$$
c_t^l = f \odot c_{t-1}^l + i \odot g
$$

$$
h_t^l = o \odot \tanh(c_t^l)
$$

- 은닉 상태(Hidden state): LSTM 외부로 노출되는 값 $h_t$
- 셀 상태(Cell state): LSTM 내부에서만 사용되는 값 $c_t$

→ 네 개의 게이트($i$, $f$, $o$, $g$)로 셀 상태를 업데이트하고, 셀 상태를 이용하여 은닉 상태를 업데이트

![image.png](assets/Lecture%2010/image%2035.png)

입력값과 은닉 상태를 이어붙여서 가중치 행렬에 곱함으로써 바로 다음 은닉 상태를 얻는 기본 RNN과 다르게, 큰 가중치 행렬과의 행렬곱 결과값으로 4개의 게이트를 생성한다 (각 게이트의 사이즈는 은닉 상태와 동일)

- $f$ (Forget gate): 망각 게이트
    - 셀 정보 삭제 여부 결정
    - sigmoid 비선형 → (0, 1) 범위
    - 이전 셀 상태를 0이면 망각, 1이면 기억
- $g$ (Gate gate(?)): 후보 게이트
    - 셀에 기록할 후보 정보 제공
    - tanh 비선형 → (-1, 1) 범위
    - 셀 상태에 기록하고자 하는 값을 1만큼 증가 또는 감소시킴
- $i$ (Input gate): 입력 게이트
    - 새로운 정보 기록 여부 결정
    - sigmoid 비선형 → (0, 1) 범위
    - 0이면 $g$ 를 입력하지 않고, 1이면 $g$ 를 입력
- $o$ (Output gate): 출력 게이트
    - 셀 정보를 외부로 얼마나 내보낼지 결정
    - sigmoid 비선형 → (0, 1) 범위
    - 셀 상태를 은닉 상태로서 얼마만큼 외부에 노출시킬지 결정. 0이면 노출하지 않고, 1이면 노출

→ 셀 상태는 작은 스케일러 정수 카운터(counter) 역할을 함

→ 은닉 상태는 셀 상태의 외부 노출값의 역할을 함

![LSTM의 도식화된 연산](assets/Lecture%2010/image%2036.png)

LSTM의 도식화된 연산

### LSTM 그래디언트 흐름

![image.png](assets/Lecture%2010/image%2037.png)

$c_t$ 에서 $c_{t-1}$ 로의 역전파는 오직 $f$와의 요소별 곱(elementwise multiplication)만 포함되며, $W$ 와의 행렬곱은 없음

→ 두 가지 장점

1. 요소별 곱이 행렬곱보다 조금 더 나음
2. 기본 RNN에서는 역전파시 동일한 가중치 행렬을 계속해서 곱했기 때문에 그래디언트 폭발이나 소실이 일어날 것이 명시적이었지만, LSTM에서는 매 시점 달라지는 $f$ 를 곱하기 때문에 이러한 문제들을 피할 수 있음
    1. 특히, $f$ 는 sigmoid를 통과해서 나온 것이므로, 계속되는 곱셈에서 그 범위가 0~1로 보장됨

> 따라서, 셀 상태를 통과하는 경로를 통해 손실부터 맨 처음 셀 상태까지 가로막는 것이 없는 ‘그래디언트 흐름 고속도로’가 형성된다.
> 

![image.png](assets/Lecture%2010/image%2038.png)

> 매 시점에서의 은닉 상태와 셀 상태로부터 지역 가중치 행렬에 대한 그래디언트를 얻을 수 있다. 따라서, 셀 상태의 그래디언트가 안정적으로 유지되는 LSTM에서 가중치 행렬에 대한 그래디언트도 부드럽게 전파할 수 있다.
> 

⇒ 이는 ResNet이나 Highway Network 등의 개념과 유사

### RNN 변형

- GRU [Learning phrase representations using rnn encoder-decoder for statistical machine translation, Cho et al. 2014]
- [LSTM: A Search Space Odyssey, Greff et al., 2015]
- [An Empirical Exploration of Recurrent Network Architectures, Jozefowicz et al., 2015]

위와 같은 여러 변형이 연구되었지만, LSTM과 GRU에 비해 확연히 뛰어난 것은 없었음