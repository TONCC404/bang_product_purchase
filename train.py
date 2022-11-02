import pandas as pd
from sklearn.preprocessing import Normalizer
from MLP import *
import matplotlib.pyplot as plt
import data_preparation



if __name__ == '__main__':

    # data测试
    train_data = data_preparation.train_data_preparation()
    test_data = data_preparation.tes_data_preparation()
    train_data = np.array(train_data)


    feature_train = train_data[:, 1:13]
    label_train = np.array(train_data[:, [13]].T)


    test_data = np.array(test_data)
    feature_test = test_data[:, 1:13]
    label_test=pd.read_csv('./submission.csv')
    label_test=np.array(label_test)
    test_label_list=[]

    for i in label_test[:,1]:
        label_value=data_preparation.judge(i)
        test_label_list.append(label_value)





    # 多层感知机
    MLP_test = MLP()
    parameter_dict = MLP_test.train(feature=feature_train, label=label_train, hidden={1: 5}, learning_rate=0.001, iteration_num=1)
    # print(parameter_dict)
    result1 = MLP_test.predict(feature_test, parameter_dict)
    # result2 = MLP_test.predict(feature_test2, parameter_dict)

    # print(result1)
    # print(result2)
    count = 0
    sum=0
    for i in range(len(result1)):
        sum=sum+abs(result1[i]-test_label_list[i])
    error_rate = sum / len(result1)
    accuracy_rate=1-error_rate

    print("error_rate:",error_rate)  # 用以上参数，测试中准确率约为85%
    print("accuracy_rate:",accuracy_rate)