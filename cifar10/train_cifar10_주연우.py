# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.cifar10 import load_cifar10
from model.new_deeplearning import deepdeep
import matplotlib.pyplot as plt
from common.trainer import Trainer
from common.optimizer import *

# for reproducibility
np.random.seed(0)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=True, flatten=False, one_hot_label=True)

network = deepdeep()
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=30, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 파라미터 저장
path_dir = './ckpt'
file_name = "cifar10_params.pkl"
if not os.path.isdir(path_dir):
    os.mkdir(path_dir)

network.save_params(os.path.join(path_dir, file_name))
print("Parameter Save Comp lete!")