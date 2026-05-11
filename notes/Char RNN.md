# Char RNN

## 1. Char RNN이란?

Char RNN은 문자 단위 RNN이다.

즉, RNN의 입력과 출력 단위를 단어가 아니라 문자로 둔다.

RNN 구조 자체가 바뀐 것은 아니다.

차이는 다음과 같다.

단어 단위 RNN:
- 입력 단위: 단어
- 예: "I", "love", "you"

문자 단위 RNN:
- 입력 단위: 문자
- 예: "a", "p", "p", "l", "e"

Char RNN은 주로 다음 문자 예측 문제로 이해하면 된다.

현재 문자와 이전까지의 문맥을 보고 다음 문자를 예측한다.

---

## 2. apple -> pple! 예제의 의미

문서의 첫 번째 예제는 다음과 같다.

입력:

apple

정답:

pple!

이것은 apple 전체를 넣고 pple! 전체를 한 번에 맞히는 문제가 아니다.

각 time step마다 현재 문자를 보고 다음 문자를 예측하는 문제이다.

입력과 정답의 대응:

| time step | 입력 문자 | 예측해야 할 문자 |
|---|---|---|
| t=1 | a | p |
| t=2 | p | p |
| t=3 | p | l |
| t=4 | l | e |
| t=5 | e | ! |

즉, 구조는 다음과 같다.

a p p l e
↓ ↓ ↓ ↓ ↓
p p l e !

그래서 input_str은 "apple"이고, label_str은 "pple!"이 된다.

핵심:

Char RNN은 각 시점에서 다음 문자를 예측하도록 학습된다.

---

## 3. 문자 집합 만들기

입력과 정답에 등장하는 모든 문자를 모아 중복을 제거한다.

input_str = "apple"

label_str = "pple!"

등장 문자:

!, a, e, l, p

문자 집합의 크기:

vocab_size = 5

각 문자에 정수 인덱스를 부여한다.

예시:

! -> 0

a -> 1

e -> 2

l -> 3

p -> 4

따라서 apple은 다음처럼 정수 인코딩된다.

a p p l e

1 4 4 3 2

정답 pple!은 다음처럼 정수 인코딩된다.

p p l e !

4 4 3 2 0

---

## 4. 원-핫 인코딩

신경망은 문자 자체를 직접 입력받을 수 없다.

그래서 각 문자를 숫자 벡터로 바꾼다.

이 예제에서는 원-핫 벡터를 사용한다.

문자 집합 크기가 5이므로 각 문자는 길이 5짜리 벡터가 된다.

예시:

! = [1, 0, 0, 0, 0]

a = [0, 1, 0, 0, 0]

e = [0, 0, 1, 0, 0]

l = [0, 0, 0, 1, 0]

p = [0, 0, 0, 0, 1]

따라서 apple은 다음과 같은 원-핫 벡터 시퀀스가 된다.

a = [0, 1, 0, 0, 0]

p = [0, 0, 0, 0, 1]

p = [0, 0, 0, 0, 1]

l = [0, 0, 0, 1, 0]

e = [0, 0, 1, 0, 0]

---

## 5. 입력 텐서 X의 크기

PyTorch의 nn.RNN은 기본적으로 3차원 텐서를 입력받는다.

batch_first=True인 경우 입력 크기는 다음과 같다.

[batch_size, time_steps, input_size]

apple 예제에서는 다음과 같다.

X.shape = [1, 5, 5]

각 차원의 의미:

| 차원 | 값 | 의미 |
|---|---:|---|
| batch_size | 1 | 학습 샘플이 1개 |
| time_steps | 5 | apple의 문자 수 5개 |
| input_size | 5 | 각 문자의 원-핫 벡터 크기 |

즉, [1, 5, 5]는 다음을 의미한다.

하나의 샘플 안에 5개의 time step이 있고, 각 time step마다 길이 5짜리 원-핫 벡터가 들어간다.

---

## 6. 레이블 Y의 크기

정답 pple!은 문자 인덱스 시퀀스이다.

Y = [[4, 4, 3, 2, 0]]

Y.shape = [1, 5]

각 차원의 의미:

| 차원 | 값 | 의미 |
|---|---:|---|
| batch_size | 1 | 학습 샘플이 1개 |
| time_steps | 5 | 각 시점마다 정답 문자 1개 |

주의:

X는 원-핫 벡터이므로 [1, 5, 5]이다.

Y는 정답 클래스 인덱스이므로 [1, 5]이다.

CrossEntropyLoss는 정답을 원-핫 벡터가 아니라 클래스 인덱스로 받는다.

---

## 7. 모델 구조

문서의 모델 구조는 다음과 같다.

RNN + Linear

구조:

입력 X
-> RNN
-> 각 time step의 hidden state
-> Linear
-> 각 time step의 문자 후보 점수

코드 구조:

```
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
```

여기서 중요한 부분:

self.rnn = torch.nn.RNN(...)

이 부분이 RNN 셀을 만든다.

x, _status = self.rnn(x)

이 부분에서 RNN이 시퀀스를 time step 순서대로 처리한다.

x = self.fc(x)

이 부분에서 hidden state를 문자 후보 점수로 바꾼다.

---

## 8. self.rnn(x) 내부에서 일어나는 일

RNN은 각 시점마다 현재 입력 x_t와 이전 hidden state h_(t-1)을 이용해 현재 hidden state h_t를 만든다.

기본 식:

h_t = tanh(W_x x_t + W_h h_(t-1) + b)

apple 예제에서는 다음처럼 진행된다.

h_1 = RNN(a, h_0)

h_2 = RNN(p, h_1)

h_3 = RNN(p, h_2)

h_4 = RNN(l, h_3)

h_5 = RNN(e, h_4)

초기 hidden state h_0은 따로 넣지 않으면 보통 0으로 시작한다.

즉, self.rnn(x) 한 줄 안에서 RNN은 time step을 따라가며 이전 hidden state를 다음 시점으로 넘긴다.

---

## 9. hidden_state가 의미하는 것

hidden state h_t는 현재 입력 문자 자체가 아니다.

h_t는 1번부터 t번까지 읽은 결과를 압축한 내부 문맥 표현이다.

apple 예제에서는 다음처럼 이해하면 된다.

h_1 = "a까지 읽은 상태"

h_2 = "ap까지 읽은 상태"

h_3 = "app까지 읽은 상태"

h_4 = "appl까지 읽은 상태"

h_5 = "apple까지 읽은 상태"

예를 들어 h_2는 다음 식으로 볼 수 있다.

h_2 = tanh(W_x x_2 + W_h h_1 + b)

여기서 의미는 다음과 같다.

W_x x_2 = 현재 입력 p가 현재 상태에 주는 영향

W_h h_1 = 이전까지의 문맥, 즉 a까지 읽은 상태가 현재 상태에 주는 영향

따라서 h_2는 단순히 p의 벡터가 아니다.

h_2는 "a를 본 뒤 p를 본 상태", 즉 "ap까지 읽은 문맥 상태"이다.

---

## 10. 같은 p인데 왜 다른 예측을 할 수 있는가?

apple에는 p가 두 번 나온다.

입력과 정답을 보면 다음과 같다.

t=2: 입력 p -> 정답 p

t=3: 입력 p -> 정답 l

현재 입력 문자는 둘 다 p이다.

그런데 정답은 다르다.

이것이 가능한 이유는 이전 hidden state가 다르기 때문이다.

t=2의 계산:

h_2 = RNN(p, h_1)

h_1 = a까지 읽은 상태

따라서 h_2 = ap까지 읽은 상태

예측 목표 = p

t=3의 계산:

h_3 = RNN(p, h_2)

h_2 = ap까지 읽은 상태

따라서 h_3 = app까지 읽은 상태

예측 목표 = l

즉, RNN은 현재 입력만 보는 것이 아니라 이전까지의 문맥을 hidden state로 함께 본다.

그래서 같은 p가 들어와도 다음 문자를 다르게 예측할 수 있다.

핵심:

h_t는 현재 문자 x_t의 표현이 아니라, x_1부터 x_t까지 읽은 결과를 담은 문맥 표현이다.

---

## 11. RNN의 output과 _status

코드에서 다음 부분이 있다.

```
x, _status = self.rnn(x)
```

여기서 x는 모든 time step의 hidden state를 모은 것이다.

x = [h_1, h_2, h_3, h_4, h_5]

_status는 마지막 hidden state이다.

_status = h_5

이 예제에서는 각 시점마다 다음 문자를 예측해야 한다.

따라서 마지막 hidden state h_5만 쓰면 안 된다.

모든 시점의 hidden state가 필요하다.

그래서 x = [h_1, h_2, h_3, h_4, h_5] 전체를 fc에 넣는다.

각 시점의 예측:

h_1 -> fc -> a 다음 문자 예측 = p

h_2 -> fc -> ap 다음 문자 예측 = p

h_3 -> fc -> app 다음 문자 예측 = l

h_4 -> fc -> appl 다음 문자 예측 = e

h_5 -> fc -> apple 다음 문자 예측 = !

---

## 12. Linear 출력층의 역할

RNN이 만든 h_t는 정답 문자 그 자체가 아니다.

h_t는 내부 문맥 벡터이다.

이 벡터를 문자 후보에 대한 점수로 바꾸는 층이 Linear 층이다.

흐름:

h_t
-> Linear
-> output_size개의 점수

apple 예제에서는 output_size = 5이다.

따라서 각 time step마다 5개 문자에 대한 점수가 나온다.

가능한 문자:

!, a, e, l, p

예를 들어 h_2를 Linear에 넣으면 다음과 같은 의미의 점수가 나온다.

fc(h_2) = 다음 문자가 !, a, e, l, p일 점수

가장 점수가 높은 인덱스를 고르면 예측 문자가 된다.

주의:

hidden_size가 5라고 해서 h_t의 각 차원이 !, a, e, l, p를 직접 의미하는 것은 아니다.

문자 후보 점수로 해석되는 것은 Linear를 통과한 뒤이다.

---

## 13. 출력 outputs의 크기

모델 출력의 크기는 다음과 같다.

outputs.shape = [1, 5, 5]

각 차원의 의미:

| 차원 | 값 | 의미 |
|---|---:|---|
| batch_size | 1 | 샘플 1개 |
| time_steps | 5 | 각 문자 위치마다 예측 |
| output_size | 5 | 가능한 문자 후보 5개에 대한 점수 |

즉, 모델은 apple 전체에 대해 하나의 결과만 내는 것이 아니다.

각 time step마다 문자 후보 점수 벡터를 낸다.

구조:

입력: a     p     p     l     e

출력: [5]   [5]   [5]   [5]   [5]

정답: p     p     l     e     !

---

## 14. 왜 outputs.view(-1, input_size)를 하는가?

outputs의 원래 크기는 다음과 같다.

[1, 5, 5]

CrossEntropyLoss에 넣기 위해 배치 차원과 시점 차원을 합친다.

outputs.view(-1, input_size)

결과:

[5, 5]

의미:

5개의 time step 각각에 대해, 5개 문자 후보 점수를 가진다.

Y도 펼친다.

Y.shape = [1, 5]

Y.view(-1).shape = [5]

즉, 손실 계산은 다음 구조로 이루어진다.

예측 점수 [5, 5] vs 정답 인덱스 [5]

각 행은 한 time step의 예측이다.

| time step | 예측 점수 크기 | 정답 |
|---|---|---|
| t=1 | 5개 문자 점수 | p |
| t=2 | 5개 문자 점수 | p |
| t=3 | 5개 문자 점수 | l |
| t=4 | 5개 문자 점수 | e |
| t=5 | 5개 문자 점수 | ! |

---

## 15. CrossEntropyLoss와 학습

손실 함수:

CrossEntropyLoss

옵티마이저:

Adam

학습 루프의 흐름:

1. optimizer.zero_grad()
   - 이전 gradient 초기화

2. outputs = net(X)
   - RNN과 Linear를 거쳐 각 time step의 문자 후보 점수 계산

3. loss = criterion(outputs.view(-1, input_size), Y.view(-1))
   - 각 time step의 예측과 정답 비교

4. loss.backward()
   - 역전파로 gradient 계산

5. optimizer.step()
   - 파라미터 업데이트

학습 목표:

각 time step에서 정답 다음 문자의 점수가 높아지도록 RNN과 Linear의 가중치를 조정한다.

---

## 16. 예측 결과 확인

outputs는 각 time step마다 문자 후보 점수를 가진다.

따라서 가장 높은 점수의 인덱스를 고르면 예측 문자 인덱스가 된다.

argmax(axis=2)의 의미:

각 time step의 output_size 차원에서 가장 큰 값을 가진 문자 인덱스를 고른다.

예시:

초기 예측:

pp!p!

정답:

pple!

학습 후 예측:

pple!

처음에는 랜덤에 가까운 예측을 하지만, 학습이 반복되면서 정답 시퀀스 pple!에 가까워진다.

---

## 17. 더 긴 문장으로 학습하는 Char RNN

두 번째 실습에서는 더 긴 문장을 사용한다.

문장 예시:

if you want to build a ship, don't drum up people together to collect wood ...

이 문장에서 문자 집합을 만든다.

공백, 쉼표, 마침표, 아포스트로피도 모두 하나의 문자로 취급된다.

예를 들어 문자 집합 크기가 25라면, 각 문자의 원-핫 벡터 크기도 25가 된다.

input_size = dic_size

hidden_size = dic_size

여기서 hidden_size를 dic_size와 같게 둔 것은 선택일 뿐이다.

hidden_size는 반드시 문자 집합 크기와 같을 필요는 없다.

---

## 18. sequence_length = 10의 의미

긴 문장 전체를 한 번에 넣지 않고, 길이 10짜리 시퀀스로 잘라서 학습 샘플을 만든다.

sequence_length = 10

샘플 생성 방식:

입력 x_str = 현재 위치부터 10글자

정답 y_str = x_str을 한 칸 오른쪽으로 민 10글자

예시:

if you wan -> f you want

f you want ->  you want

 you want  -> you want t

즉, 각 입력 시퀀스에 대해 정답은 한 글자 뒤의 시퀀스이다.

이것도 결국 다음 문자 예측이다.

첫 번째 샘플:

입력:

if you wan

정답:

f you want

각 time step 대응:

i -> f

f -> 공백

공백 -> y

y -> o

o -> u

...

---

## 19. 긴 문장 실습의 텐서 크기

문서의 두 번째 실습에서는 총 170개의 샘플이 생성된다.

각 샘플의 길이는 10이다.

문자 집합 크기는 25이다.

따라서 입력 X의 크기는 다음과 같다.

X.shape = [170, 10, 25]

각 차원의 의미:

| 차원 | 값 | 의미 |
|---|---:|---|
| batch_size | 170 | 길이 10짜리 학습 샘플 170개 |
| time_steps | 10 | 각 샘플의 문자 길이 |
| input_size | 25 | 각 문자의 원-핫 벡터 크기 |

레이블 Y의 크기:

Y.shape = [170, 10]

각 time step마다 정답 문자 인덱스 하나가 있기 때문이다.

---

## 20. 긴 문장 실습의 모델

두 번째 실습에서는 RNN 층을 2개 쌓는다.

```
self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True)
self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
```

num_layers=2의 의미:

RNN 은닉층을 두 개 쌓는다.

첫 번째 RNN 층의 hidden state 시퀀스가 두 번째 RNN 층의 입력처럼 사용된다.

출력 크기:

outputs.shape = [170, 10, 25]

이를 손실 계산을 위해 펼친다.

outputs.view(-1, dic_size).shape = [1700, 25]

Y.view(-1).shape = [1700]

즉, 총 170 * 10 = 1700개의 문자 예측 문제로 손실을 계산하는 것이다.

---

## 21. 긴 문장에서 예측 문자열을 복원하는 방식

모델은 각 샘플마다 길이 10의 예측 결과를 낸다.

results.shape = [170, 10]

하지만 샘플들은 서로 한 글자씩 겹친다.

예시:

샘플 1 예측: 길이 10

샘플 2 예측: 앞 9글자가 이전 샘플과 겹침

그래서 예측 문자열을 만들 때는 다음 방식으로 이어 붙인다.

첫 번째 샘플은 예측 결과 10글자를 전부 사용한다.

그 다음 샘플부터는 마지막 글자만 추가한다.

이렇게 하면 겹치는 부분을 중복해서 붙이지 않고 전체 문장 형태로 복원할 수 있다.

---

## 22. 전체 흐름 요약

Char RNN 학습 과정:

1. 문자열 데이터를 준비한다.

2. 문자 집합을 만든다.

3. 각 문자에 정수 인덱스를 부여한다.

4. 입력 문자를 원-핫 벡터로 바꾼다.

5. 정답 문자는 한 칸 뒤의 문자 인덱스로 만든다.

6. RNN이 각 time step의 hidden state를 만든다.

7. Linear 층이 hidden state를 문자 후보 점수로 바꾼다.

8. CrossEntropyLoss로 각 time step의 예측과 정답을 비교한다.

9. 역전파로 RNN과 Linear의 파라미터를 업데이트한다.

10. 학습이 진행되면 다음 문자를 점점 더 정확히 예측한다.

---

## 23. 가장 중요한 직관

Char RNN은 현재 문자만 보고 다음 문자를 예측하는 모델이 아니다.

현재 문자 x_t와 이전까지의 문맥 h_(t-1)을 함께 보고 다음 문자를 예측한다.

그래서 hidden state h_t가 중요하다.

h_t는 다음을 의미한다.

h_t = x_1부터 x_t까지 읽은 결과를 압축한 내부 문맥 상태

apple 예제에서는 다음과 같다.

h_1 = a까지 읽은 상태

h_2 = ap까지 읽은 상태

h_3 = app까지 읽은 상태

h_4 = appl까지 읽은 상태

h_5 = apple까지 읽은 상태

이 hidden state 덕분에 같은 문자 p가 입력되어도 위치와 앞 문맥에 따라 다른 예측을 할 수 있다.

---

## 24. 최종 한 줄 정리

Char RNN은 문자열을 문자 단위 시퀀스로 보고, 각 time step에서 현재 문자와 이전 hidden state에 담긴 문맥을 이용해 다음 문자를 예측하도록 학습하는 RNN이다.

apple -> pple! 예제는 다음 문자 예측 구조를 가장 단순하게 보여주는 예제이다.

h_t는 t번째 문자 자체가 아니라, x_1부터 x_t까지 읽은 내용을 압축한 문맥 상태이다.
