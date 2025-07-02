# Lecture 2 | Image Classification pipeline

Image Classification 문제:

- Semantic Gap
- Viewpoint variation
- Illumination
- Deformation
- Occlusion
- Background Clutter
- Intraclass variation

등등… 에 대하여 모두 robust 해야 함

edge는 중요하다 - Hubel & Wiesel의 논문

**단계:** Find edges → Find corners → explicit rules ⇒ 잘 동작 X (brittle, 일반화X)

# Data-Driven Approach

1. 이미지와 라벨의 데이터셋 수집
2. classifier를 학습시키기 위해 머신러닝 활용 (ingest, summarize)
3. 새로운 이미지에 대해 classifier 평가

* 함수로 구현

- train(이미지와 라벨 입력 → 모델 출력)
- predict(모델과 새로운 이미지 입력 → 라벨 출력)

## K-Nearest Neighbors (kNN)

### First classifier: Nearest Neighbor

1. train: Memorize all data and labels
2. predict: Predict the label of the most similar training image

비교는 어떻게 하는가?

L1 distance - pixel마다 차이를 다 더한다 (simple, easy)

N개의 example → Train O(1), Predict O(N)

⇒ training은 느려도 되지만 prediction은 빨라야 함. Not good

### K-Nearest Neighbors

가장 가까운 이웃의 라벨을 복사하는 것이 아니라, K개의 가장 가까운 점에서 다수결로 결정

→ boundary를 smooth하고 더 좋은 결과를 얻을 수 있음

⇒ First(K=1)일 때는 다른 영역 중간에 작은 island가 있거나 경계에 finger가 있는 문제가 있었는데, K를 늘릴수록 해당 문제가 완화됨

### L1 (Manhattan) distance

$$
d_1(I_1, I_2) = \sum_{p} \left| I_1^p - I_2^p \right|
$$

circle이 square의 모습 → 축을 회전시킬시 distance가 바뀜

벡터 내의 각 성분이 특별한 의미를 가질 경우 L1 distance가 더 자연스럽게 fit할 수 있음

### L2 (Euclidean) distance

$$
d_2(I_1, I_2) = \sqrt{\sum_{p} \left (I_1^p - I_2^p \right)^2}
$$

circle이 circle의 모습 → 축을 회전시켜도 distance는 불변

벡터가 일반적이고 각 성분의 의미를 모를 경우 L2 distance가 조금 더 fit할 수 있음

<aside>
💡

K-Nearest Neighbors는 generally 사용 가능함.
distance function, metric만 결정하면 텍스트 등에도 바로 적용 가능

</aside>

## Hyperparameter

**“알고리즘에 대해 미리 설정하는 선택사항들”**

예) k, distance, …

→ 매우 problem-dependent ⇒ 모두 시도해본 뒤 가장 잘 작동하는 것을 확인

- Idea #1 : data에서 가장 잘 작동하는 hyperparameter 선택
- Idea #2 : data를 train과 test로 나누고, test data에서 가장 잘 작동하는 hyperparameter 선택
- Idea #3 : data를 train, validation, test로 나누고, validation에서 가장 잘 작동하는 hyperparameter 선택, test에서 이를 평가

→ Idea #3으로 해야 함!

- Idea #4(Cross-Validation) : data 중 test를 미리 빼놓고, 나머지를 fold들로 나눠 각 fold를 validation으로 설정해가며 학습, 그 결과들을 평균하여 그중에서 가장 잘 작동하는 hyperparameter 선택.
→ 작은 dataset에서는 유용하지만, 딥러닝에는 잘 안 쓰임.

<aside>
💡

**K-Nearest Neighbor은 이미지에서 절대 쓰이지 않는다.**

1. test time이 너무 느림
2. pixel 간 distance 측정은 유용한 정보를 주지 않음
3. Curse of dimensionality: 차원을 늘릴수록 더 촘촘하게 point를 배치해야하므로 필요로 하는 test example이 기하학적으로 증가
</aside>

## Linear Classification

이미지 CNN + 언어 CNN을 결합하여 한 번에 학습시키는 것. 이런 것이 Linear Classification (레고 조각)

### Parametric Approach

$$
f(x, W)
$$

x: 이미지 input

W: 가중치 parameters = train 데이터를 요약한 정보 ⇒ test time에 W만 있으면 됨

→ f: 각 class에 대한 점수 (점수가 높을수록 x가 해당 class일 확률이 커짐)

### Linear Classifier

$$
f(x, W) = Wx + b
$$

f(x, W): 10 x 1

W: 10 x 3072

x: 3072 x 1

b(bias term): 10 x 1

⇒ template matching approach (이미지로 시각화 가능)

W의 각 행 = 각 class의 template 

→ 각 class에 대해 하나의 template만을 학습하는 것이 문제

high dimensional space에 있는 이미지를 하나의 line을 기준으로 해당 class와 그것이 아닌 것으로 나누는 것

→ parity problem, multimodal situation 등 문제 존재