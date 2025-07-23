# Lecture 5 | Convolutional Neural Networks

# History

- Frank Rosenblatt, ~1957: Perceptron
최초의 퍼셉트론(두뇌의 인지 능력을 모방하도록 만든 인위적인 네트워크) 알고리즘 구현
→ f(x) = 0 or 1, update rule 존재
- Widrow and Hoff, ~1960: Adaline/Madaline
여러 linear classifier를 연결
- Rumelhart et al., 1986: First time back-propagation became popular
우리에게 친숙한 back-propagation 등장
- Hinton and Salakhutdinov 2006: Reinvigorated research in Deep Learning
initialize 과정이 복잡하지만 backprop을 이용한 최초의 파인튜닝
- Acoustic Modeling using Deep Belief Networks, Abdel-rahman Mohamed, George Dahl, Geoffrey Hinton, 2010
Context-Dependent Pre-trained Deep Neural Networks for Large Vocabulary Speech Recognition, George Dahl, Dong Yu, Li Deng, Alex Acero, 2012
Imagenet classification with deep convolutional
neural networks, Alex Krizhevsky, Ilya Sutskever, Geoffrey E Hinton, 2012
음향, 이미지에 대한 딥러닝을 이용한 강력한 결과

- Hubel & Wiesel
1959: RECEPTIVE FIELDS OF SINGLE NEURONES IN THE CAT'S STRIATE CORTEX
1962: RECEPTIVE FIELDS, BINOCULAR INTERACTION AND FUNCTIONAL ARCHITECTURE IN THE CAT'S VISUAL CORTEX
1968…
고양이가 시각적 자극을 받았을 때 피질의 뉴런의 반응을 관찰
- Topographical mapping in the cortex: nearby cells in cortex represent nearby regions in the visual field
피질의 인접한 세포가 시야에서 인접한 구역에 해당한다
- Hierarchical organization
망막 절세포(Retinal ganglion cell)는 수용야, 단순 세포(Simple cell)는 빛의 방향, 복합 세포(Complex cell)는 빛의 방향 및 움직임, 과복합 세포(Hypercomplex cell)는 움직임과 종점에 반응한다
- Neocognitron [Fukushima 1980]
“sandwich” architecture (SCSCSC…)
simple cells: modifiable parameters
complex cells: perform pooling
단순 세포와 복합 세포를 샌드위치처럼 배치하여 단순 세포에서의 작은 변화에 대해 invariant
- Gradient-based learning applied to document recognition [LeCun, Bottou, Bengio, Haffner 1998]
backpropagation과 gradient 기반 학습을 Convolutional Neural Network를 학습시키는 데에 적용, 숫자를 인식하는 데 잘 동작
- ImageNet Classification with Deep Convolutional Neural Networks [Krizhevsky, Sutskever, Hinton, 2012]
”AlexNet” = 현대의 Convolutional Neural Network
LeCun의 것과 크게 다르지 않지만 데이터와 병렬 컴퓨팅(GPU)의 발전 활용
- 현재: ConvNets are everywhere

# Convolutional Neural Networks (w/o the brain stuff)

## Convolutional Layer

Fully Connected Layer와 다르게 공간 구조(spatial structure)를 유지함

![image.png](assets/Lecture%205/image.png)

**가중치 = 작은 filter (height와 width는 줄어들지만 depth는 유지)**

→ 필터를 이미지와 합성곱한다 = 모든 공간 위치에 대해 dot product를 계산 = 필더를 이미지에 겹치고 각 성분끼리 곱한 것의 합을 계산

![image.png](assets/Lecture%205/image%201.png)

![image.png](assets/Lecture%205/image%202.png)

**Activation map**: 필터를 각 위치마다 슬라이드하며 곱한 결과들

→ 해당 레이어에서 사용하는 필터 개수만큼 activation map의 depth이 됨

![image.png](assets/Lecture%205/image%203.png)

**ConvNet**은 activation function이 사이에 배치된 일련의 Convolutional Layer들의 연속

![image.png](assets/Lecture%205/image%204.png)

학습이 끝나면, 각 레이어의 여러 필터들은 계층적인 학습 결과를 가짐

- 앞 레이어: Low-level 특징 (e.g. edges)
- 중간 레이어: Middle-level 특징 (e.g. corners, blobs)
- 뒷 레이어: High-level 특징 (e.g. resemble concepts)

<aside>
💡

**이 결과는 ConvNet에게 명시적으로 특징 학습을 강제하지 않았음에도
실제 생물 시각의 simple cell → complex cell의 역할에 대한 Hubel & Wiesel의 연구결과와 일치함**

</aside>

> **Convolution (합성곱)**
> 
> 
> Signal processing에서 합성곱은 다음과 같이 정의됨
> 
> $$
> f[x,y]\ast g[x,y]=\sum_{n_1=-\infty}^{\infty}\sum_{n_2=-\infty}^{\infty}f[n_1,n_2]\;\cdot\;g[x-n_1,\,y-n_2]
> $$
> 
> 위의 과정이 합성곱과 본질이 같기 때문에 Convolutional Neural Network로 불리는 것
> 

**전체적인 Convolutional Neural Network의 생김새**

image → CONV → RELU → CONV → RELU → POOL → … → FC

- POOL: Activation map의 크기를 downsample하는 레이어
- FC: 마지막 레이어의 모든 출력과 연결된 fully connected layer를 활용한 최종 score 함수

## Spatial dimensions

Stride: 필터를 이동시키는 간격

- 7x7 input (spatially), assume 3x3 filter
    
    ![image.png](assets/Lecture%205/image%205.png)
    
    - stride = 1 → 5x5 output
    - stride = 2 → 3x3 output
    - stride = 3 → doesn’t fit
    → 이렇게 합성곱하지 않음. asymmetric ouput으로 이어지기 때문

### Output size 공식

$$
\text{Output size}=\frac{N - F}{\text{stride}}+ 1
$$

### Zero pad

input size를 유지하기 위해 가장자리를 0으로 두르기도 함

※ 관련 없는 특징이 생길 가능성 있음. 굳이 0이 아니더라도 다른 패딩 방식이 있음.

일반적으로 CONV layers는 stride 1, FxF 사이즈의 filter들 , 그리고 (F-1)/2의 zero-padding으로 이뤄짐.

→ 공간적으로 사이즈를 유지 (shrink 방지)

### Summary

- **입력 볼륨 크기:** W1 x H1 x D1

- **필요한 하이퍼파라미터 (4가지):**
    - 필터 개수: K
    - 필터 공간적 크기(spatial extent): F
    - Stride: S
    - Zero-padding: P

- 일반적 설정
    - K = 2의 거듭제곱
    - F = 1, 3, 5
    - S = 1, 2
    - P = 0, 1, 2, whatever fits
- **출력 볼륨 크기:** W2 x H2 x D2
    
    $$
    \begin{align*}
    & W_2 \;=\; \frac{W_1 - F + 2P}{S} \;+\, 1 &&\\
    & H_2 \;=\; \frac{H_1 - F + 2P}{S} \;+\, 1 &&\\
    & D_2 \;=\; K && \end{align*}
    $$
    
- **파라미터 공유 시 가중치 수:**
    - 필터당:
        
        $$
        F \times F \times D_1
        $$
        
    - 전체(필터 개수, 바이어스 고려):
        
        $$
        (F \times F \times D_1)\times K + K
        $$
        
- 출력 볼륨에서, **d-th depth silce**(W2 x H2)는 d번째 필터와 입력 볼륨에 stride S로 유효한 **합성곱**을 수행하고 d번째 바이어스를 더한 결과임

## Brain/Neuron view of CONV Layer

![image.png](assets/Lecture%205/image%206.png)

- 각 공간적 위치의 필터는 해당 위치의 뉴런이 얼마나 활성화되었는지를 의미
    - **수용 영역(Receptive Field)**: 각 필터가 바라보는 사이즈
    (e.g. 5x5 filter = 5x5 receptive field for each neuron)
- 여러 입력에 대해 하나의 값을 출력하는 것은 Fully-connected layer와 같지만, 모든 입력과 연결되지 않아 local connectivity를 가진다는 점이 다름
    - Fully connected layer에서는 각 뉴런이 모든 입력 볼륨을 바라봄
- Activation map의 동일한 위치(W, H), 다른 깊이(D)의 값은 다른 것을 찾는 여러 개의 다른 뉴련이 같은 영역(region)을 바라보는 것을 의미
(이미지의 같은 영역에 대해 다른 필터를 적용)

## Pooling Layer

![image.png](assets/Lecture%205/image%207.png)

더 작고 관리하기 쉬운 표현(representation)을 만듦

→ 주어진 영역에서 불변성을 수행하고 매개변수 개수를 줄임

= 공간적으로(spatially) 다운샘플(downsample) ⇒ depth는 유지

### Max Pooling

![image.png](assets/Lecture%205/image%208.png)

필터를 슬라이드하며 최댓값을 구함

### Summary

- **입력 볼륨 크기:** W1 x H1 x D1

- **필요한 하이퍼파라미터 (4가지):**
    - 필터 공간적 크기(spatial extent): F
    - Stride: S

- 일반적 설정
    - F = 2, 3
    - S = 2
- **출력 볼륨 크기:** W2 x H2 x D2
    
    $$
    \begin{align*}
    & W_2 \;=\; \frac{W_1 - F}{S} \;+\, 1 &&\\
    & H_2 \;=\; \frac{H_1 - F}{S} \;+\, 1 &&\\
    & D_2 \;=\; D_1 && \end{align*}
    $$
    
- 가중치 수 = 0: 입력에 대해 고정된 함수를 계산하기 때문
- 일반적으로 zero-padding 사용 안함: 직접적인 다운샘플링을 하고자 하기 때문

## Fully Connected Layer (FC layer)

Convolutional network output을 받은 후, 일반적인 신경망처럼 볼륨을 펼쳐(stretch) 모든 성분과 연결

이전 레이어에서는 공간적 구조를 유지했지만, 마지막 레이어에서는 이들을 모두 통합해 추론하여 점수를 출력

# Summary

- ConvNets는 CONV, POOL, FC 레이어들을 포갠 것
- 필터를 작게, 아키텍처를 깊게 하는 추세
- POOL, FC 레이어를 없애고 CONV 레이어만 사용하는 추세
- 전형적인 아키텍처는 다음과 같음
    
    $$
    
    [(\text{CONV} - \text{RELU}) \times N - \text{POOL?}] \times M - (\text{FC}-\text{RELU}) \times K \text{, SOFTMAX}
    $$
    
    N은 일반적으로 ~5 까지, M은 큼, 0 ≤ K ≤ 2.
    
- 그러나, ResNet/GoogLeNet과 같은 최근의 발전은 이러한 패더다임을 변화시킴