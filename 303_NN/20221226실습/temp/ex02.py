import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 다중 선형 회귀 실습
# 앞서 배운 x가 1개인 선형 회귀 -> 단순 선형 이라고합니다.
# 다수 x 로부터 y를 예측하는 다중 선형 회귀
x1_train = torch.FloatTensor([[73], [93], [83], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 가중치 w 와 편향 b를 선언 필요하고 w -> 3개 b -> 1개
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 1e-04 - 0.0001 0.00001
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

# 학습 몇번 진행할래
epoch_num = 10000
for epoch in range(epoch_num + 1):

    # 가설 xw + xw .... + b
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # loss
    loss = torch.mean((hypothesis - y_train) ** 2)

    # loss H(x) 개선
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            "Epoch {:4d}/{} w1 {:.3f} w2 {:.3f} w3 {:.3f} b {:.3f} loss {:.6f}".format(
                epoch, epoch_num, w1.item(), w2.item(), w3.item(), b.item(), loss.item()
            ))

"""
Epoch    0/1000 w1 28.969 w2 29.360 w3 29.738 b 0.342 loss 29661.800781
Epoch  100/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  200/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  300/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  400/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  500/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  600/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  700/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  800/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch  900/1000 w1 nan w2 nan w3 nan b nan loss nan
Epoch 1000/1000 w1 nan w2 nan w3 nan b nan loss nan


Epoch    0/1000 w1 0.000 w2 0.000 w3 0.000 b 0.000 loss 29661.800781
Epoch  100/1000 w1 0.003 w2 0.003 w3 0.003 b 0.000 loss 29404.349609
Epoch  200/1000 w1 0.006 w2 0.006 w3 0.006 b 0.000 loss 29149.134766
Epoch  300/1000 w1 0.009 w2 0.009 w3 0.009 b 0.000 loss 28896.134766
Epoch  400/1000 w1 0.012 w2 0.012 w3 0.012 b 0.000 loss 28645.328125
Epoch  500/1000 w1 0.014 w2 0.015 w3 0.015 b 0.000 loss 28396.703125
Epoch  600/1000 w1 0.017 w2 0.017 w3 0.018 b 0.000 loss 28150.230469
Epoch  700/1000 w1 0.020 w2 0.020 w3 0.021 b 0.000 loss 27905.902344
Epoch  800/1000 w1 0.023 w2 0.023 w3 0.023 b 0.000 loss 27663.693359
Epoch  900/1000 w1 0.026 w2 0.026 w3 0.026 b 0.000 loss 27423.587891
Epoch 1000/1000 w1 0.028 w2 0.029 w3 0.029 b 0.000 loss 27185.566406
"""
