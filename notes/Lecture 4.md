# Lecture 4 | Backpropagation and Neural Networks

# Computational graphs

임의의 복잡한 함수에 대해 analytic gradient를 계산하는 방법 → backpropagation 사용 가능

backpropagation: chain rule을 재귀적으로 이용해서 computational graph 내 모든 변수에 대한 gradient 계산

⇒ convolutional neural networks와 같은 복잡한 함수에서 유용 (+ 다른 딥러닝 모델인 Neural Turing Machine에서도 사용)

1. function을 computational graph로 표현
2. 각 변수에 값을 대입하고, 각 노드의 gradient와 intermediate value를 계산
3. 마지막 노드부터 재귀적으로 f의 편미분을 계산. (직접적으로 f와 변수가 연관되어 있지 않다면 chain rule 활용)

<aside>
💡

각 노드는 local gradient를 가지고 있음.

**backpropagation을 통해 upstream gradient 값을 받으면 이를 local gradient와 곱하여 연결된 뒷 노드에게 전달한다**

(두 개 이상의 노드에서 upstream gradient 값을 받으면 이들을 모두 더하여 total upstream gradient로 계산)

</aside>

※ computational node는 아무 단위로나 설정할 수 있음
⇒ 덧셈, 곱셈 등 가장 단순한 연산부터 여러 연산을 동시에 진행하는 것까지 (예, sigmoid function)

- Patterns in backward flow
    - add gate: gradient distributor
    - max gate: gradient router
    - mul gate: gradient switcher (and scaler)

→ 값을 모든 뒷 노드로

→ 값을 더 큰 노드로만 (나머지 0)

→ 값을 다른 branch의 값으로 scale

<aside>
💡

변수들이 vector라면? 모든 gradient들을 Jacobian matrix로 표현

</aside>

### Modularized implementation

- forward API → node의 output 계산
- backward API → gradient 계산

```python
class ComputationalGraph(object):
    #...

    def forward(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss  # the final gate in the graph outputs the loss

    def backward():
        for gate in reversed(self.graph.nodes_topologically_sorted()):
            gate.backward()  # little piece of backprop (chain rule applied)
        return inputs_gradients
```

- gate
    - input 값을 cache 해야 함

```python
class MultiplyGate(object):
    def forward(self, x, y):
        z = x * y
        self.x = x  # must keep these around!
        self.y = y
        return z

    def backward(self, dz):
        dx = self.y * dz  # [dz/dx * dL/dz]
        dy = self.x * dz  # [dz/dy * dL/dz]
        return [dx, dy]
```

→ framework에 layer로 구현되어 있음

## Summary

- neural nets는 매우 크기 때문에 gradient formula를 직접 작성하는 것은 불가능
- backpropagation: computational graph를 따라 chain rule을 재귀적으로 적용함으로써 모든 입력/매개변수/중간값의 gradient를 계산할 수 있음
    - forward: 연산의 결과를 계산하고 중간값을 저장
    - backward: chain rule 적용하여 input에 대한 loss function의 gradient 계산

# Neural Networks

- (Before) Linear score function
    
    $$
    f = Wx
    $$
    
- (Now) 2-layer Neural Network
    - 첫 번째 layer = non-linear, 두 번째 layer = linear
    - 하나의 template 밖에 없던 문제
    multiple layer → W1은 template, W2는 weighted sum of templates
    ~~각 intermediate variable가 template을 가질 수 있음~~
    
    $$
    f = W_2\max\left(0, W_1x\right)
    $$
    
- (…) 3-layer Neural Network
    
    $$
    f = W_3\max\left(0, W_2\max\left(0, W_1x\right)\right)
    $$
    

→ Neural network는 복잡한 non-linear function을 형성하기 위해 hierarchically 쌓인 class of functions

### Neuron

Neuron과 computational graph의 node는 작동 방식이 비슷함

| 뉴런 | 노드 | 차이 |
| --- | --- | --- |
| 가지돌기 | 앞선 노드로부터의 가지 | 가지돌기는 복잡한 비선형 연산 가능 |
| 세포체 | 연산 + activation function | 시냅스는 하나의 가중치가 아닌 복잡한 비선형 동적 시스템 |
| 축색돌기 | 뒤 노드로의 가지 | rate code 충분히 설명 못함 |
- 뉴런은 가지돌기(dendrite)로 자극(impulse)을 전달받고, 노드도 앞선 노드로부터 입력을 전달받음
- 뉴런은 이 자극을 세포체(cell body)에서 통합하고, 노드도 연산과 activation function을 통해 계산함
    - 뉴런의 자극은 급등(spike)의 모양을 띄는데 activation function이 정확히 그 역할을 함 (예, sigmoid activation function)
- 뉴런은 축색돌기(axon)으로 자극을 전달하고, 노드도 출력을 뒤의 노드로 전달

### Neural Networks

- 각 layer는 matrix multiply
- layer(또는 hidden layer) 수에 따라 ‘[n]-layer Neural Net’ 또는 ‘[n-1]-hidden-layer Neural Net’

```python
def neuron_tick(inputs):
    # assume inputs and weights are 1-D numpy arrays and bias is a number
    cell_body_sum = np.sum(inputs * self.weights) + self.bias
    firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))  # sigmoid activation
    return firing_rate
```