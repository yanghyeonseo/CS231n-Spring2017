# Lecture 9 | CNN Architectures

## Review: LeNet-5

> LeCun et al., 1998
> 

![image.png](assets/Lecture%209/image.png)

- 실제로 성공적으로 사용된 합성곱신경망
- CONV-POOL-CONV-POOL-FC-FC
    - CONV: 5x5, stride 1
    - POOL: 2x2, stride 2
- 정확한 숫자 인식 가능

## 개요

![image.png](assets/Lecture%209/image%201.png)

1. ‘12의 AlexNet은 최초의 CNN 기반 우승자로 오차를 의미 있게 감소시켰음
(’13)의 ZFNet은 본질적으로 AlexNet과 거의 동일함
2. ‘14, ‘14의 VGGNet과 GoogLeNet은 더 깊은 네트워크
3. ‘15의 ResNet은 엄청나게 깊은 네트워크

# AlexNet

> Krizhevsky et al. 2012
최초의 큰 합성곱신경망, 딥러닝 연구가 활발해진 계기
> 

![image.png](assets/Lecture%209/image%202.png)

- ImageNet 분류(classification) 성능 크게 향상

- 구조

CONV1
MAX POOL 1
NORM1
CONV2
MAX POOL 2
NORM2

→

CONV3
CONV4
CONV5
Max POOL 3
FC6
FC7
FC8

### Volume

1. Input: $227 \times 227 \times 3$ images
2. First layer (CONV1): 96 $11\times11$ filters applied at stride 4
$(227-11)/4+1 = 55$
→ Output volume: $55 \times 55 \times 96$
→ Parameters: $(11 \times 11 \times 3) \times 96 = 35K$
3. Second layer (POOL1): $3 \times 3$ filters applied at stride 2
$(55-3)/2+1 = 27$
→ Output volume: $27 \times 27 \times 96$
→ Parameters: $0$
4. [27x27x96] NORM1: Normalization layer
[27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
[13x13x256] MAX POOL2: 3x3 filters at stride 2
[13x13x256] NORM2: Normalization layer
[13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
[13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
[13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
[6x6x256] MAX POOL3: 3x3 filters at stride 2
[4096] FC6: 4096 neurons
[4096] FC7: 4096 neurons
[1000] FC8: 1000 neurons (class scores)
- 당시 3GB 메모리의 GTX 580로 학습했기 때문에 네트워크를 2개의 GPU로 나누어 각 GPU에 절반씩의 뉴런(특징 맵)를 분배한 레이어도 있음
    - 같은 GPU의 특징 맵에만 연결: CONV1, CONV2, CONV4, CONV5
    - 두 GPU의 전체 특징 맵에 연결: CONV3, FC6, FC7, FC8

### Details

- 최초의 ReLU 사용
- Norm 레이어 사용 (현재는 일반적이지 않음)
- 여러 데이터 증강(data augmentation) 적용
- 드롭아웃(dropout) 0.5
- 배치 크기(batch size) 128
- SGD Momentum 0.9
- Learning rate 1e-2, val accuracy가 정체기일 때 수동으로 10배 감소
- L2 weight decay 5e-4
- 7 CNN 앙상블(ensemble): 18.2% -> 15.4%

## ZFNet

> Zeiler and Fergus, 2013
AlexNet과 동일하지만 개선된 하이퍼파라미터
> 

![image.png](assets/Lecture%209/image%203.png)

- 같은 레이어 수, 일반적인 구조
- 다른 스트라이드(stride) 사이즈, 필터 개수

# VGGNet & GoogLeNet

## VGGNet

> Simonyan and Zisserman, 2014
더 작은 필터, 더 깊은 네트워크
> 

![image.png](assets/Lecture%209/image%204.png)

![image.png](assets/Lecture%209/image%205.png)

- 분류(classification)에서 2등, 위치 추정(localization)에서 1등
- 8 레이어 → 16 ~ 19 레이어
- 오직 3x3 CONV stride 1, pad 1와 2x2 MAX POOL stride 2

### Volume

- 더 작은 필터를 사용
    - 이유: 3×3 합성곱 레이어를 3개 쌓으면, 하나의 7×7 합성곱 레이어와 같은 유효 수용영역(effective receptive field)을 가진다 ( ∵ 레이어마다 3×3 → 5×5 → 7×7 로 변화)
    - 그러나, 더 깊어지고, 더 많은 비선형을 가지며, 더 적은 하이퍼파라미터(27C^2 < 49C^2)를 가짐
- 총 메모리: 24M × 4 bytes ≈ 96MB / image (오직 순전파, 역전파까지 고려하면 두 배)
    - 더 큰 공간적 차원을 가진 앞의 레이어에서 많은 메모리 사용
- 총 파라미터: 138M 개 (↔ AlexNet 60M 개)
    - 뒤의 더 빽빽한 연결을 가진 FC 레이어에 많은 파라미터 분포

### Details

- AlexNet(2012)과 유사한 학습 절차
- 지역 반응 정규화(Local Response Normalisation, LRN) 사용 안 함
- VGG16 또는 VGG19 사용 (VGG19가 약간 더 성능이 좋지만 메모리 소모가 큼)
- 최상의 결과를 위해 앙상블(ensembles) 사용
- FC7 레이어의 특징은 (ImageNet 이외의) 다른 과제에도 잘 일반화됨

## GoogLeNet

> Szegedy et al., 2014
계산적으로 효율적인 더 깊은 네트워크
> 

![image.png](assets/Lecture%209/image%206.png)

- 분류(classification)에서 1등
- 22 레이어
- 적은 계산량
- 효율적인 인셉션 모듈(Inception module)
- FC 레이어 없음
- 오직 5M 개의 파라미터 (AlexNet보다 12배 적음)

### 인셉션 모듈(Inception module)

> 잘 설계된 국소 네트워크 토폴로지(네트워크 안의 네트워크)를 만들고, 이러한 모듈들을 차곡차곡 쌓아 올리자.
> 

![단순한 인셉션 모듈](assets/Lecture%209/image%207.png)

단순한 인셉션 모듈

![차원 감소를 적용한 인셉션 모듈](assets/Lecture%209/image%208.png)

차원 감소를 적용한 인셉션 모듈

- 입력에 대해 여러 필터 연산을 병렬적으로 수행하고 출력을 깊이 방향으로 이어 붙인다
    - 합성곱(convolution)에 대해 **여러 크기의 수용 영역** 사용 (1×1, 3×3, 5×5)
    - 풀링 연산 (3×3)

> 문제) 연산량이 너무 많다.
> 

![image.png](assets/Lecture%209/image%209.png)

- 연산량이 너무 많음
    - 합성곱 연산:
    [1×1 conv, 128] $(28 \times 28 \times 128) \times (1 \times 1 \times 256)$
    [3×3 conv, 192] $(28 \times 28 \times 192) \times (3  \times 3  \times 256)$
    [5×5 conv, 96] $(28  \times 28 \times 96) \times (5 \times 5 \times 256)$
    총 845M 회 연산
    - 풀링 연산:
    특징 깊이를 유지하기 때문에, 출력 깊이가 항상 증가함

> 해결) ‘병목(bottleneck)’ 레이어를 사용하여 1×1 합성곱으로 특징 맵의 깊이를 줄인다.
> 

![image.png](assets/Lecture%209/image%2010.png)

- 1×1 합성곱 레이어를 이용하면 공간적 차원을 유지하면서 깊이를 줄일 수 있음
    
    ![image.png](assets/Lecture%209/image%2011.png)
    
    - 특징 맵들을 조합하여 깊이를 더 낮은 차원으로 사상(project)한다. ( ≈ linear combination)
- 연산량의 감소
    - 합성곱 연산:
    [1x1 conv, 64] $(28 \times 28 \times 64) \times (1 \times 1 \times 256)$
    [1x1 conv, 64] $(28 \times 28 \times 64) \times (1 \times 1 \times 256)$
    [1x1 conv, 128] $(28 \times 28 \times 128) \times (1 \times 1 \times 256)$
    [3x3 conv, 192] $(28 \times 28 \times 192) \times (3 \times 3 \times 64)$
    [5x5 conv, 96] $(28 \times 28 \times 96) \times (5 \times 5 \times 64)$
    [1x1 conv, 64] $(28 \times 28 \times 64) \times (1 \times 1 \times 256)$
    총 358M 회 연산

### 전체 GoogLeNet 아키텍처

![image.png](assets/Lecture%209/image%2012.png)

- 입력부 네트워크(Stem Network)
    - Conv-Pool-2×Conv-Pool
- 인셉션 모듈 적층 구조
- 분류기 출력(Classifier output)
    - 계산 비용이 크로 파라미터가 많은 FC 레이어들을 제거
- 보조 분류 출력(auxiliary classification outputs)
    - 출력과 loss를 계산
    - 깊은 네트워크에서 하위 계층에 추가적인 그래디언트를 주입함으로써 신호 흐름(signal flow)에 도움
    - AvgPool-1×1Conv-FC-FC-Softmax
    
    → 총 세 곳(출력 + 2 보조 출력)에서 분류 loss를 계산
    

→ 총 22개의 가중치를 가지는 레이어 (Inception 모듈 안의 각 병렬 층 포함)

# ResNet

> He et al., 2015
잔차 연결(residual connections)을 활용한 매우 깊은 신경망
> 

![image.png](assets/Lecture%209/image%2013.png)

- 두드러진 오차 감소로 모든 분류(classification) 및 검출(detection) 대회 1등
- 사람보다 낮은 오차율
- 152 레이어

## 잔차 연결(Residual Connection)

> 보통의 합성곱 신경망에 계속해서 레이어를 쌓아 깊게 만들면 성능이 좋아질까?
→ No!
> 

![image.png](assets/Lecture%209/image%2014.png)

![image.png](assets/Lecture%209/image%2015.png)

- 더 깊은 모델이 training과 test 오차 둘 다에서 더 안 좋은 성능을 낸다.
    - 그러나, test뿐만 아니라 training에서도 오차가 큰 것으로 보아 과적합(overfitting) 때문은 아님

> 가설) 문제는 최적화이다. 더 깊은 모델은 최적화하기가 더 어렵다.
> 
- 위와 같은 가설을 설정한 이유
    - 깊은 모델은 적어도 얕은 모델만큼은 성능을 낼 수 있어야 함
    - 얕은 모델에서 학습된 층을 복사하고, 추가된 층들을 항등 매핑(identity mapping)으로 설정하면 같은 성능을 낼 수 있기 때문

> 해결) 목표 매핑을 직접 학습하는 것이 아니라, 대신 **잔차 매핑(residual mapping)**을 학습하는 데 네트워크 층을 사용한다.
> 

![image.png](assets/Lecture%209/image%2016.png)

- $H(x)$ : 목표함수 → Plain layer가 학습하는 것
- $F(x)$ : 잔차 → Residual block이 학습하는 것
    - $F(x) = H(x) - x$ 를 학습하고 최종 출력은 $F(x) + x$ 로 얻는다
        - 기존 출력 $H(x)$ 를 얻을 수도 있고
        - 항등 매핑 $x$ 를 $F(x) = 0$ 으로 학습함으로써 쉽게 얻을 수 있음

### 전체 ResNet 아키텍처

![image.png](assets/Lecture%209/image%2017.png)

- 입력부
    - 추가적인 합성곱 층
- 잔차 블록(Residual block) 적층 구조
    - 각 잔차 블록은 두 개의 3×3 합성곱 레이어로 구성
- 일정 단계마다 필터 수 두 배, 공간 차원 절반 다운샘플링
    - 스트라이드(stride) 2 이용
- 분류기 출력
    - 추가적인 FC 레이어 없음
- 깊은 네트워크의 경우, 병목 레이어 사용
    
    ![image.png](assets/Lecture%209/image%2018.png)
    
    - 계산 효율 개선

### Details

- 깊은 레이어에 대해서도 좋은 그래디언트 흐름 유지, 낮은 학습 오차
- 모든 합성곱(CONV) 레이어 뒤에 배치 정규화(Batch Normalization) 적용
- Xavier/2 초기화 방식 사용
- SGD + Momentum 0.9
- Learning rate 1e-2, val accuracy가 정체기일 때 수동으로 10배 감소
- 미니배치 크기(mini-batch size) 256
- weight decay 1e-5
- 드롭아웃(dropout)은 사용하지 않음

# 복잡도(Complexity) 비교

- 정확도(accuracy)

![image.png](assets/Lecture%209/image%2019.png)

- 계산량 & 메모리 사용량 & 정확도

![image.png](assets/Lecture%209/image%2020.png)

※ Inception은 ResNet에 Inception을 적용해 발전시킨 것

|  | 정확도 | 메모리 | 계산량 |
| --- | --- | --- | --- |
| AlexNet | **낮음** | 보통 | **적음** |
| VGG | 보통 | **많음** | **많음** |
| GoogLeNet | 보통 | **적음** | **적음** |
| ResNet | **높음** | 보통 | 보통 |

# 이외의 아키텍처들

## 역사적으로 의미가 있는 네트워크

### Network in Network (NiN)

> Lin et al. 2014
GoogLeNet의 철학적 영감(philosophical inspiration)
> 

![image.png](assets/Lecture%209/image%2021.png)

- 각 합성곱 레이어 안에 마이크로 네트워크(micronetwork)를 두어, 로컬 패치(local patches)에 대해 더 추상적인 특징을 계산하는 Mlpconv 층
    - 이 마이크로 네트워크는 다층 퍼셉트론(FC, 즉 1×1 합성곱 층)을 사용함
- GoogLeNet과 ResNet의 병목(bottleneck) 레이어의 전신

## ResNet을 개선한 네트워크

### Identity Mappings in Deep Residual Networks

> He et al. 2016
ResNet 제작자의 개선된 ResNet 블록 설계
> 

![image.png](assets/Lecture%209/image%2022.png)

- 네트워크 전반에 걸쳐 정보를 전파하기 위한 더 직접적인 경로를 형성
    - 활성화 함수의 위치를 조정하여 잔차(residual) 경로 쪽으로 이동
    
    → 더 나은 성능을 제공
    

### **Wide Residual Networks**

> Zagoruyko et al. 2016
깊이가 아니라 잔차 자체가 중요
> 

![image.png](assets/Lecture%209/image%2023.png)

- 더 넓은 잔차 블록
    - 각 레이어에서 F개의 필터 대신 F × k개의 필터 사용
- 50 레이어 Wide ResNet이 기존 152 레이어 ResNet보다 더 좋은 성능을 보임
- 깊이를 늘리는 대신 너비를 늘리는 것이 계산 효율이 더 좋음
    - 병렬화(parallelization) 가능

### **Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)**

> Xie et al. 2016
ResNet 제작자의 또다른 ResNet 블록 설계
인셉션 모듈(Inception module)과 비슷한 철학
> 

![image.png](assets/Lecture%209/image%2024.png)

- 여러 개의 병렬 경로(pathway)를 사용하여 잔차 블록의 너비(width)를 증가
    - 카디널리티(cardinality)

### **Deep Networks with Stochastic Depth**

> Huang et al. 2016
학습 시 짧은 네트워크 사용
> 

![image.png](assets/Lecture%209/image%2025.png)

- 학습 과정마다 레이어의 부분 집합을 무작위로 드롭(drop)
    - 제거된 층은 항등 함수로 우회 처리
    
    → 그래디언트 소실을 줄이고 학습 시간을 단축
    
- 테스트 시에는 전체 깊은 네트워크 사용

## ResNet 이후의 네트워크

### **FractalNet: Ultra-Deep Neural Networks without Residuals**

> Larsson et al. 2017
잔차는 필요하지 않으며, 얕은 깊이에서 깊은 깊이로의 전환이 핵심
> 

![image.png](assets/Lecture%209/image%2026.png)

- 출력까지 얕은 경로와 깊은 경로가 모두 존재하는 프랙탈(fractal) 아키텍처
- 부분 경로(sub-path)를 무작위로 드롭(drop)하며 학습
- 테스트 시에는 전체 네트워크를 사용

→ 좋은 성능을 보임

### **Densely Connected Convolutional Networks (DenseNet)**

> Huang et al. 2017
각 레이어가 모든 이전 레이어와 연결
> 

![image.png](assets/Lecture%209/image%2027.png)

- 각 레이어가 피드포워드(feedforward) 방식으로 모든 이전 레이어 연결된 Dense block

→ 그래디언트 소실(vanishing gradient) 완화, 특징 전파(feature propagation) 강화, 특징 재사용(feature reuse) 촉진

## 효율적인 네트워크

### SqueezeNet: AlexNet-level Accuracy With 50x Fewer Parameters and <0.5Mb Model Size

> Iandola et al. 2017
정확도를 유지하면서 압축된 네트워크
> 

![image.png](assets/Lecture%209/image%2028.png)

- 1×1 필터로 구성된 ‘squeeze’ 층과, 그 출력을 입력으로 받아 1×1 및 3×3 필터를 사용하는 ‘expand’ 층으로 이루어진 Fire module
- 50배 적은 파라미터로 ImageNet에서 AlexNet 수준의 정확도
- AlexNet보다 최대 510배까지 모델 압축 가능 (0.5MB)

# **요약**

- VGG, GoogLeNet, ResNet은 모두 널리 이용됨
    - 현재 기본 선택지로는 ResNet이 가장 우수
- 극도로 깊은 네트워크 추세
- 레이어나 스킵 연결(skip connection) 설계와 그래디언트 흐름 개선 연구 중심
    - 더 최근에는, 깊이 vs 너비 및 잔차 연결의 필요성 연구 추세