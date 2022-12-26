import numpy
import matplotlib.pyplot as plt
from collections import OrderedDict
from optimizer import *  # 직접 제작한 optimizer 모듈 로드


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0 * y


init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
print(params)  # {'x': -7.0, 'y': 2.0}
grads = {}
grads['x'], grads['y'] = 0, 0

optimizer = OrderedDict()
optimizer['SGD'] = SGD(lr=0.95)
optimizer['Momentum'] = Momentum(lr=0.1)
optimizer['AdaGrad'] = AdaGrad(lr=1.5)
optimizer['Adam'] = Adam(lr=0.3)
