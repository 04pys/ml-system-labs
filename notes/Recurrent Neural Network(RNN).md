# Recurrent Neural Network(RNN) 복습 트리거

## 1. RNN을 배우는 이유

기존 MLP는 기본적으로 입력 하나를 받아 출력 하나를 만드는 구조이다.

예를 들어 이미지 분류에서는 이미지 하나를 입력으로 넣고, 숫자 0~9 중 하나를 예측한다.

하지만 문장, 음성, 주가, 센서 데이터처럼 순서가 중요한 데이터는 입력 하나하나가 서로 독립적이지 않다.

예시:
나는 / 오늘 / 밥을 / 먹었다

이 문장에서 “먹었다”를 이해하려면 앞에서 “나는”, “오늘”, “밥을”이 나왔다는 흐름이 중요하다.

RNN은 이런 순서가 있는 데이터, 즉 sequence를 처리하기 위해 만들어진 신경망이다.

핵심:
현재 입력만 보는 것이 아니라, 이전까지의 정보를 요약한 hidden state를 함께 사용한다.

---

## 2. Sequence와 time step

sequence는 순서가 있는 데이터 묶음이다.

예시:
문장: 단어 또는 토큰의 나열
음성: 시간 순서대로 들어오는 음성 프레임
주가: 날짜별 가격 데이터
센서 데이터: 시간 순서대로 측정된 값

time step은 sequence 안의 각 위치이다.

예를 들어 문장 길이가 4라면:

x_1 = "나는"
x_2 = "오늘"
x_3 = "밥을"
x_4 = "먹었다"

이때 time steps의 수는 4이다.

RNN은 입력을 한 번에 모두 보는 것이 아니라, x_1부터 x_4까지 순서대로 처리한다.

---

## 3. RNN의 기본 아이디어

RNN은 매 time step마다 다음 두 가지를 함께 사용한다.

1. 현재 시점의 입력 x_t
2. 이전 시점의 hidden state h_{t-1}

그리고 새로운 hidden state h_t를 만든다.

기본 식:

h_t = tanh(W_x x_t + W_h h_{t-1} + b)

의미:
x_t = 현재 시점의 입력
h_{t-1} = 이전 시점까지의 정보를 담은 hidden state
W_x = 현재 입력에 곱해지는 가중치
W_h = 이전 hidden state에 곱해지는 가중치
b = bias
h_t = 현재 시점까지의 정보를 반영한 새로운 hidden state

즉 RNN cell은 매 시점마다 이렇게 작동한다.

현재 입력 + 이전 기억 -> 새로운 기억

---

## 4. hidden state란 무엇인가?

hidden state는 이전까지의 입력 정보를 요약한 벡터이다.

엄밀히 말하면 문장을 그대로 저장하는 메모리는 아니다.
대신 지금까지 본 입력의 흐름을 숫자 벡터로 압축해둔 것이다.

예를 들어:

이 영화는 초반에는 지루했지만 후반부가 정말 좋았다

이 문장을 RNN이 읽는다면,
처음에는 “이 영화는”, “초반에는”, “지루했지만”을 반영한 hidden state를 만들고,
뒤로 갈수록 “후반부가”, “정말 좋았다”까지 반영한 hidden state를 만든다.

마지막 hidden state h_T에는 문장 전체의 흐름이 어느 정도 요약되어 있다고 볼 수 있다.

---

## 5. RNN을 펼쳐서 이해하기

RNN은 그림으로 보면 자기 자신으로 돌아가는 화살표가 있어서 복잡해 보인다.
하지만 시간 방향으로 펼치면 이해하기 쉽다.

h_0 -> RNN cell -> h_1 -> RNN cell -> h_2 -> RNN cell -> h_3
          ↑                  ↑                  ↑
         x_1                x_2                x_3

시점별 계산:

h_1 = tanh(W_x x_1 + W_h h_0 + b)
h_2 = tanh(W_x x_2 + W_h h_1 + b)
h_3 = tanh(W_x x_3 + W_h h_2 + b)

여기서 중요한 점은 W_x와 W_h가 모든 time step에서 공유된다는 것이다.

즉,
시점 1에서도 W_x, W_h 사용
시점 2에서도 W_x, W_h 사용
시점 3에서도 W_x, W_h 사용

RNN은 time step마다 다른 가중치를 쓰는 것이 아니라, 같은 RNN cell을 반복해서 사용하는 구조이다.

---

## 6. MLP와 RNN의 차이

### MLP

은닉층이 3개인 MLP는 보통 이렇게 계산된다.

h_1 = f(W_1 x + b_1)
h_2 = f(W_2 h_1 + b_2)
h_3 = f(W_3 h_2 + b_3)
y_hat = g(W_4 h_3 + b_4)

특징:
입력 하나를 여러 층에 통과시킨다.
층마다 서로 다른 가중치를 사용한다.
W_1, W_2, W_3는 서로 다른 파라미터이다.

즉 MLP는:
입력 x -> 1층 변환 -> 2층 변환 -> 3층 변환 -> 출력

---

### RNN

RNN은 입력이 하나가 아니라 순서가 있는 입력이다.

x_1, x_2, x_3, ..., x_T

계산은 다음과 같다.

h_1 = f(W_x x_1 + W_h h_0 + b)
h_2 = f(W_x x_2 + W_h h_1 + b)
h_3 = f(W_x x_3 + W_h h_2 + b)

특징:
sequence를 time step별로 읽는다.
모든 time step에서 같은 W_x, W_h를 사용한다.
현재 입력과 이전 hidden state를 함께 사용한다.

즉 RNN은:
첫 번째 입력 읽고 기억 갱신
두 번째 입력 읽고 기억 갱신
세 번째 입력 읽고 기억 갱신

---

## 7. 가중치 공유의 의미

RNN에서 가중치가 공유된다는 말은,
모든 time step에서 같은 W_x, W_h를 사용한다는 뜻이다.

하지만 역전파 중에 h_3 시점에서 가중치를 먼저 수정하고,
그 수정된 가중치가 h_2 계산에 바로 반영되는 것은 아니다.

학습 과정은 다음과 같다.

1. 현재 가중치로 전체 sequence에 대해 forward 수행
2. loss 계산
3. 시간 방향으로 거슬러 올라가며 gradient 계산
4. 각 time step에서 같은 가중치에 대한 gradient가 누적됨
5. optimizer.step()에서 가중치를 한 번 업데이트

즉, 같은 학습 step 안에서는 기존 가중치로 모든 forward와 backward를 끝낸 뒤,
모든 time step에서 나온 gradient를 합쳐서 한 번에 업데이트한다.

---

## 8. RNN의 출력 구조

RNN은 문제 유형에 따라 출력을 다르게 사용한다.

### 1) many-to-one

입력 sequence 전체에 대해 하나의 출력만 필요한 경우이다.

예시:
감성 분석
스팸 메일 분류
문장 분류

구조:
x_1, x_2, ..., x_T -> h_T -> output

마지막 hidden state h_T를 문장 전체의 요약 표현으로 보고, 이를 출력층에 넣는다.

예시:
y_hat = sigmoid(W_y h_T + b)
또는
y_hat = softmax(W_y h_T + b)

---

### 2) many-to-many

각 time step마다 출력이 필요한 경우이다.

예시:
품사 태깅
개체명 인식
다음 단어 예측

구조:
h_1 -> y_1
h_2 -> y_2
h_3 -> y_3
...
h_T -> y_T

각 시점의 hidden state를 이용해 각 시점의 출력을 만든다.

---

### 3) one-to-many

하나의 입력으로 sequence를 생성하는 경우이다.

예시:
이미지 캡셔닝

이미지 하나를 입력으로 받고, 여러 단어로 이루어진 문장을 출력한다.

---

## 9. time steps, input size, hidden size

RNN에서 자주 헷갈리는 세 가지 개념이다.

### time steps

sequence의 길이이다.
RNN cell이 시간 방향으로 몇 번 반복되는지를 의미한다.

예:
문장에 단어가 10개 있으면 time steps = 10

---

### input size

각 time step에서 들어오는 입력 벡터의 차원이다.

예:
각 단어가 4차원 벡터로 표현된다면 input size = 4

---

### hidden size

각 time step마다 만들어지는 hidden state 벡터의 차원이다.

예:
hidden size = 8이면,
h_1, h_2, ..., h_T 각각은 8차원 벡터이다.

정리:
time steps = hidden state가 몇 개 생기는가
hidden size = hidden state 하나가 몇 차원인가

예:
time steps = 10
hidden size = 8

의미:
RNN cell이 10번 반복된다.
각 time step마다 8차원 hidden state가 생성된다.
따라서 h_1부터 h_10까지 총 10개의 hidden state가 생기고, 각각은 8차원이다.

---

## 10. PyTorch에서의 입력 shape

RNN 구현 예제에서는 이해를 쉽게 하기 위해 2D 입력을 사용할 수 있다.

예:
inputs.shape = (timesteps, input_size)

하지만 실제 PyTorch RNN에서는 보통 batch까지 포함한 3D 텐서를 사용한다.

batch_first=True 기준:

input.shape = (batch_size, time_steps, input_size)

예:
batch_size = 4
time_steps = 10
input_size = 5
hidden_size = 8

입력 shape:
(4, 10, 5)

의미:
4 = 한 번에 처리하는 데이터 개수
10 = 각 데이터의 sequence 길이
5 = 각 time step의 입력 벡터 차원

RNN output shape:
(4, 10, 8)

의미:
4 = 데이터 개수
10 = 각 time step마다 hidden state가 나옴
8 = 각 hidden state의 크기

---

## 11. output과 h_n의 차이

PyTorch RNN에서는 보통 output과 h_n이 나온다.

### output

모든 time step의 hidden state를 모아둔 값이다.

예:
output = [h_1, h_2, h_3, ..., h_T]

batch_first=True 기준:
output.shape = (batch_size, time_steps, hidden_size)

---

### h_n

마지막 hidden state이다.

단층 단방향 RNN이라면:
h_n = h_T

shape:
h_n.shape = (num_layers * num_directions, batch_size, hidden_size)

단층 단방향이면:
(1, batch_size, hidden_size)

---

## 12. 양방향 RNN

일반 RNN은 sequence를 앞에서 뒤로만 읽는다.

x_1 -> x_2 -> x_3 -> ... -> x_T

양방향 RNN은 두 방향으로 읽는다.

순방향:
x_1 -> x_2 -> x_3 -> ... -> x_T

역방향:
x_T -> x_{T-1} -> ... -> x_2 -> x_1

각 time step마다 두 개의 hidden state가 생긴다.

1. forward hidden state
2. backward hidden state

예:
h_t_forward = 앞에서부터 x_t까지 읽은 정보
h_t_backward = 뒤에서부터 x_t까지 읽은 정보

최종적으로 보통 둘을 이어 붙인다.

h_t_output = [h_t_forward ; h_t_backward]

hidden_size = 8인 양방향 RNN이라면:

h_t_forward = 8차원
h_t_backward = 8차원
h_t_output = 16차원

따라서 양방향 RNN의 output 마지막 차원은 보통 2 * hidden_size가 된다.

정리:
단방향 RNN output 마지막 차원 = hidden_size
역방향-only RNN output 마지막 차원 = hidden_size
양방향 RNN output 마지막 차원 = 2 * hidden_size

---

## 13. 역전파: BPTT

RNN의 역전파는 시간 방향으로 펼친 뒤 진행된다.
이를 BPTT, Backpropagation Through Time이라고 한다.

예:
x_1 -> h_1 -> h_2 -> h_3 -> loss

역전파 흐름:
loss -> h_3 -> h_2 -> h_1

중요한 점:
모든 time step에서 같은 W_x, W_h를 사용했기 때문에,
각 time step에서 같은 가중치에 대한 gradient가 나온다.

그리고 이 gradient들이 합쳐져서 최종적으로 W_x, W_h가 업데이트된다.

즉:
W_x에 대한 gradient = 시점 1의 영향 + 시점 2의 영향 + ... + 시점 T의 영향
W_h에 대한 gradient = 시점 1의 영향 + 시점 2의 영향 + ... + 시점 T의 영향

---

## 14. RNN의 한계: 기울기 소실과 폭주

RNN은 시간 방향으로 같은 가중치 행렬을 반복해서 사용한다.

그래서 sequence가 길어질수록 역전파 과정에서 gradient가 계속 곱해진다.

이때 gradient가 너무 작아지면 기울기 소실이 발생한다.
gradient가 너무 커지면 기울기 폭주가 발생한다.

기울기 소실이 발생하면 오래전 입력의 정보가 제대로 학습되지 않는다.

예:
문장 앞부분의 중요한 단서가 뒤쪽 출력에 영향을 주어야 하는데,
역전파가 앞부분까지 제대로 전달되지 못함

그래서 기본 RNN은 긴 문맥을 잘 기억하기 어렵다.

이 문제를 완화하기 위해 LSTM, GRU 같은 구조가 등장했다.

---

## 15. RNN, LSTM, GRU, Transformer의 위치

기본 RNN은 sequence modeling의 출발점이다.

하지만 긴 문맥을 다룰 때는 기울기 소실 문제가 크다.

그래서 이후에는 다음 모델들이 많이 사용된다.

LSTM:
gate 구조를 사용해 장기 기억을 더 잘 유지하도록 만든 RNN 계열 모델

GRU:
LSTM보다 구조가 단순한 gated RNN 계열 모델

Transformer:
RNN처럼 순서대로 하나씩 처리하지 않고 attention을 이용해 sequence 전체 관계를 병렬적으로 계산하는 모델

최신 NLP에서는 Transformer 계열이 주류이지만,
RNN은 sequence modeling, hidden state, time step, BPTT를 이해하기 위한 중요한 기초이다.

---

## 16. 핵심 요약

RNN은 sequence 데이터를 처리하기 위한 신경망이다.

sequence는 순서가 있는 데이터이고, 각 위치를 time step이라고 한다.

RNN은 현재 입력 x_t와 이전 hidden state h_{t-1}를 함께 사용해 현재 hidden state h_t를 만든다.

기본 식:
h_t = tanh(W_x x_t + W_h h_{t-1} + b)

hidden state는 이전까지의 입력 정보를 요약한 벡터이다.

time steps는 RNN이 시간 방향으로 몇 번 반복되는지를 의미한다.

hidden size는 각 hidden state 벡터의 크기이다.

RNN은 모든 time step에서 같은 가중치 W_x, W_h를 공유한다.

가중치 업데이트는 각 time step의 gradient를 모두 합친 뒤 한 번에 수행된다.

문장 분류처럼 전체 sequence에 대해 하나의 출력만 필요하면 마지막 hidden state를 사용한다.

품사 태깅처럼 각 time step마다 출력이 필요하면 모든 hidden state에서 각각 출력을 만든다.

양방향 RNN은 forward hidden state와 backward hidden state를 각각 만들고, 보통 둘을 이어 붙여 사용한다.

hidden_size가 8인 양방향 RNN의 output 마지막 차원은 16이다.

기본 RNN은 긴 sequence에서 기울기 소실/폭주 문제가 생기기 쉬우며, 이를 보완하기 위해 LSTM과 GRU가 등장했다.
