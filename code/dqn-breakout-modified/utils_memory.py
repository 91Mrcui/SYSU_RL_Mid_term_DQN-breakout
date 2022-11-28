from typing import (
    Tuple,
)

import torch
import numpy as np

ALPHA = 0.6                  #将TD误差转换为优先度
BETA_INIT = 0.4              #beta值得初始值，用于调整重要性权重在梯度下降的过程中的作用
BETA_INC = 1e-4              #beta随着训练步数的增长速率
EPSILON = 1e-3              #用于防止出现TD误差为0的情况
ABS_ERR_UPPER = 1.           #用于在保存优先度时限制TD误差的最大值，避免优先度差距过大

from utils_types import (
    BatchIndex,
    BatchISweight,
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    BatchErr,
    TensorStack5,
    TorchDevice,
)

'''prioritized experience replay采用proportional prioritization的方法实现
   需要实现Sum Tree数据结构'''
class Sumtree(object) :
    def __init__(
        self, 
        capacity : int
    ) -> None:
        self.__arr = np.zeros(2 * capacity)
        self.__capacity = capacity
    
    #更新sumtree结点
    def update(
        self, 
        val : float, 
        position : int
    ) -> None  :
        index = position + self.__capacity
        change = val - self.__arr[index]
        self.__arr[index] = val
        while True :
            parent = index // 2
            self.__arr[parent] += change
            if parent == 1 :
                break
            index = parent
    
    #获得根节点值，即所有经验优先度的和
    def get_root(self) -> float :
        return self.__arr[1]
    
    #根据输入的值，获得一个叶结点的优先度值和序号
    def get_leaf(
        self, 
        val : float
    ) -> Tuple[int, float] :
        index = 1
        while True :
            lchild = index * 2
            rchild = index * 2 + 1
            if index >= self.__capacity :
                break
            else :
                if self.__arr[lchild] >= val :
                    index = lchild
                else :
                    val -= self.__arr[lchild]
                    index = rchild
        return index - self.__capacity, self.__arr[index]
    
    #获取最大的优先度值
    def get_p_max(self) -> float :
        return np.max(self.__arr[-self.__capacity:])
    
    #获取最小的优先度值
    def get_p_min(self) -> float :
        return np.min(self.__arr[-self.__capacity:])
    
    def get_p_all(self):
        return np.copy(self.__arr[-self.__capacity:])

class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            full_sink: bool = True,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0
        self.__tree = Sumtree(capacity = self.__capacity)
        self.__beta = BETA_INIT

        sink = lambda x: x.to(device) if full_sink else x
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))
    
    #插入记忆队列
    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        
        p = self.__tree.get_p_max()
        if p == 0 :
            p = ABS_ERR_UPPER    #由论文第一个p为1
        self.__tree.update(p, self.__pos)
        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity
    
    #采样
    def sample(self, batch_size: int) -> Tuple[
            BatchIndex,
            BatchISweight,
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        b_index = [0] * batch_size
        seg = self.__tree.get_root() / batch_size        #分段大小
        probs = np.power(self.__tree.get_p_all(), ALPHA)
        sumProb = np.sum(probs)
        probs = probs / sumProb
        isweights = torch.Tensor(np.power(self.__capacity * probs, -self.__beta)).to(self.__device)
        self.__beta = np.min([1.0, self.__beta + BETA_INC])    #beta值更新
        #分段，采样并计算importance weight
        for j in range(batch_size) :
            a = j * seg
            b = (j + 1) * seg
            val = np.random.uniform(a, b)                                    #在段内随机一个值
            index, p = self.__tree.get_leaf(val)                             #采集一个样本的序号
            b_index[j] = index
            
        
        #根据采集样本序号获得样本
        b_index = torch.LongTensor(tuple(b_index))
        b_isweight = isweights[b_index]
        b_isweight /= torch.max(b_isweight)
        b_isweight = b_isweight.to(self.__device).float()
        b_state = self.__m_states[b_index, :4].to(self.__device).float()   #state:[1,2,3,4]
        b_next = self.__m_states[b_index, 1:].to(self.__device).float()    #next_state:[2,3,4,5]
        b_action = self.__m_actions[b_index].to(self.__device)
        b_reward = self.__m_rewards[b_index].to(self.__device).float()
        b_done = self.__m_dones[b_index].to(self.__device).float()
        #返回
        return b_index, b_isweight, b_state, b_action, b_reward, b_next, b_done
    
    #更新被采样经验的p值
    def batch_update(
        self,
        batch_index : BatchIndex,
        abs_err : BatchErr,
    ) -> None :
        batch_index = batch_index.cpu().data.numpy()
        abs_err = abs_err.cpu().data.numpy()
        abs_err += EPSILON                               #防止出现TD误差为0
        #clip_err = np.minimum(abs_err, ABS_ERR_UPPER)    #防止出现TD误差超过1.
        for i in range(len(batch_index)) :
            self.__tree.update(abs_err[i], batch_index[i])
    
    def __len__(self) -> int:
        return self.__size
