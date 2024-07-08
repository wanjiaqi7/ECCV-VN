from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput


class BaseModel(torch.nn.Module):
    def __init__(self, args):                 # args包含了模型的超参数，如动作空间的大小、类别数量、隐藏状态的大小等。
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()
                                                                        # 定义卷积层和全连接层
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)                # 1x1卷积层，将输入特征通道数从resnet_embedding_sz变为64
        self.maxp1 = nn.MaxPool2d(2, 2)                                   # 2x2最大池化层
        self.detection_feature = nn.Linear(518, 49)                       # 全连接层，将检测特征的维度从518变为49
        self.embed_action = nn.Linear(action_space, 10)                   # 全连接层，将动作嵌入从动作空间大小变为10
        pointwise_in_channels = 64 + self.num_cate + 10
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)        # 1x1卷积层，输入通道数为64 + 类别数量 + 动作嵌入的通道数

        self.lstm_input_sz = 7 * 7 * 64                                         # LSTM输入大小
        self.hidden_state_sz = hidden_state_sz                                  # LSTM隐藏状态大小
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)             # 两层LSTM
        # self.lstm_1 = nn.LSTMCell(self.lstm_input_sz, hidden_state_sz)
        # self.lstm_2 = nn.LSTMCell(hidden_state_sz, hidden_state_sz)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)                   # 用于值网络的两层全连接层
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)             # 用于策略网络的全连接层
                                                                                # 初始化各层的权重和偏置。weights_init是自定义的初始化函数，norm_col_init用于列初始化。
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)

        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_embedding_input):                                # embedding函数将状态、目标和动作输入嵌入成特征向量
        target = torch.cat((target['appear'], target['info'], target['indicator']), dim=1)

        target = F.relu(self.detection_feature(target))
        target_embedding = target.reshape(1, self.num_cate, 7, 7)

        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

        image_embedding = F.relu(self.conv1(state))
        x = self.dropout(image_embedding)

        x = torch.cat((x, target_embedding, action_reshaped), dim=1)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):                  # a3clstm函数将嵌入特征输入LSTM，输出策略和价值

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))
        x = output.reshape([1, self.hidden_state_sz])

        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear_1(x)
        critic_out = self.critic_linear_2(critic_out)

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs

        x, image_embedding = self.embedding(state, target, action_probs)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
