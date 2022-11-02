import numpy as np


class MLP:
    '''
    用于多分类的MLP
    '''
    def predict(self, feature, parameter_dict):
        feature = np.mat(feature)
        feature = np.mat(self.normalize(feature))
        re_list = []
        sample_num = feature.shape[0]
        for m in range(sample_num):
            current_sample = feature[m]
            for layer_index in range(len(parameter_dict.keys())):
                current_sample = np.insert(current_sample, 0, values=1, axis=1)
                # print(current_sample)
                # print("===================")
                # print(parameter_dict[layer_index + 1])
                current_sample = current_sample * parameter_dict[layer_index + 1]
                current_sample = self.sigmoid(current_sample)
            # print(current_sample)
            re_list.append(np.argmax(np.array(current_sample)))
        print("*****************")
        return re_list

    def train(self, feature, label, hidden, learning_rate, iteration_num):
        '''

        :param feature: 装有 m行 * n列 数据的特征矩阵，样本数为m，特征数为n
        :param label: 装有 m行 * 1列 标签的矩阵，样本数为m
        :param hidden: 装有隐藏层信息的字典，格式为{层数: 神经元个数}，层数从1开始
        :param learning_rate: 学习率
        :param iteration_num: 梯度下降迭代次数
        :return: parameter_dict: 各层之间的参数矩阵
        '''

        feature = np.mat(feature)
        # feature = np.mat(self.normalize(feature))
        feature = np.mat(self.normalize(feature))
        label = np.mat(label)

        # 初始化参数矩阵
        feature_num = feature.shape[1]
        hidden_layer_num = len(hidden.keys())
        label_set = set()
        for i in np.array(label)[0]:
            label_set.add(i)
        label_categories_num = len(label_set)
        parameter_dict = {}
        parameter_dict[1] = np.mat(np.random.rand(feature_num + 1, int(hidden[1])))                          # 初始化输入层到隐藏层之间的参数矩阵
        if hidden_layer_num > 1:                                                                                    # 初始化隐藏层之间的参数矩阵
            for layer_index in range(1, hidden_layer_num):
                parameter_dict[layer_index+1] = np.mat(np.random.rand(hidden[layer_index] + 1, hidden[layer_index + 1]))
        parameter_dict[hidden_layer_num + 1] = np.mat(np.random.rand(hidden[hidden_layer_num] + 1, label_categories_num))  # 初始化最后一个隐藏层到输出层之间的参数矩阵

        # 初始化标签矩阵
        sample_num = feature.shape[0]
        label_matrix = np.mat(np.zeros((sample_num, label_categories_num)))
        for m in range(sample_num):
            label_matrix[m, label[0, m]] = 1

        # 返回训练出来每一层间的参数矩阵
        parameter_dict = self.gradient_descent(feature, label_matrix, parameter_dict, learning_rate, iteration_num)
        return parameter_dict

    # 梯度下降更新参数矩阵
    def gradient_descent(self, feature, label, parameter_dict, learning_rate, iteration_num):
        # 梯度下降更新参数矩阵
        for _ in range(iteration_num):
            sample_num = feature.shape[0]
            parameter_num = len(parameter_dict.keys())
            # 对每一个样本使用反向传播算法
            for m in range(sample_num):
                current_sample = feature[m]
                current_label = label[m]
                forward_input_value = {0: current_sample}
                activation_value = {0: current_sample}
                deviation = {}
                # 前向传播算每一层的前向输入值和激活输出值
                for layer_index_fp, parameter in parameter_dict.items():
                    activation_value[layer_index_fp - 1] = np.insert(activation_value[layer_index_fp - 1], 0, values=1, axis=1)  # 增加偏置项
                    forward_input_value[layer_index_fp] = activation_value[layer_index_fp - 1] * parameter_dict[layer_index_fp]
                    activation_value[layer_index_fp] = self.sigmoid(forward_input_value[layer_index_fp])
                # 反向传播求误差值
                deviation[parameter_num] = activation_value[parameter_num] - current_label  # 交叉熵损失函数下求输出层误差
                for layer_index_bp in range(parameter_num - 1, 0, -1):
                    # 前向输入增加偏置参数
                    forward_input_value[layer_index_bp] = np.insert(forward_input_value[layer_index_bp], 0, values=1, axis=1)
                    # 求隐藏层误差
                    ones = np.mat(np.ones((1, forward_input_value[layer_index_bp].shape[1])))
                    deviation[layer_index_bp] = np.multiply( (deviation[layer_index_bp + 1] * parameter_dict[layer_index_bp + 1].T), ( np.multiply( self.sigmoid(forward_input_value[layer_index_bp]), (ones - self.sigmoid(forward_input_value[layer_index_bp]))) ) )
                    # 误差去除偏置参数
                    deviation[layer_index_bp] = np.delete(deviation[layer_index_bp], 0, axis=1)
                # 更新参数
                for parameter_index in range(parameter_num, 0, -1):
                    parameter_dict[parameter_index] -= learning_rate * activation_value[parameter_index - 1].T * deviation[parameter_index]
        return parameter_dict

    # sigmoid函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 标准化
    def normalize(self, feature):
        feature_normalized = np.copy(feature).astype(float)
        feature_mean = np.mean(feature, 0)
        feature_deviation = np.std(feature, 0)
        if feature.shape[0] > 1:
            feature_normalized -= feature_mean
        feature_deviation[feature_deviation == 0] = 1
        feature_normalized /= feature_deviation
        return feature_normalized

