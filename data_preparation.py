import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



'''
1、离散特征与连续特征的问题
   无序离散通常用onehot编码,对于有序离散通常用labelhot编码
2、unknown数据的处理: background与economic取平均值
'''

def judge(str):
    if str=='yes':
        result=0
    else:
        result=1
    return result

def transpose(matrix):
    new_matrix = []
    for i in range(len(matrix[0])):
        matrix1 = []
        for j in range(len(matrix)):
            matrix1.append(matrix[j][i])
        new_matrix.append(matrix1)
    return new_matrix

# 缺失数据过大
def drop_bias_data(dataset):
    dataset=dataset[~(dataset['job'].isin(['unknown'])|
                      dataset['marital'].isin(['unknown'])|
                      dataset['education'].isin(['unknown'])|
                      dataset['default'].isin(['unknown'])|
                      dataset['housing'].isin(['unknown'])|
                      dataset['loan'].isin(['unknown'])|
                      dataset['poutcome'].isin(['nonexistent']))]

    return dataset

def background_combination(job,education):

    job_dic={'admin.':4,'blue-collar':3,'entrepreneur':6,
             'housemaid':3,'management':5,'retired':3,
             'self-employed':4,'services':2,'student':1,
             'technician':3,'unemployed':1,'unknown':3}

    education_dic={'basic.4y':1,'basic.6y':1,'basic.9y':2,
                   'high.school':3,'illiterate':0,'professional.course':4,
                   'university.degree':4,'unknown':2}
    job_score=job_dic[job]
    education_score=education_dic[education]
    background_score=job_score+education_score
    return background_score

def economic_level(default,housing,loan):
    default_dic={'yes':1,'no':3,'unknown':2}
    housing_dic={'yes':3,'no':1,'unknown':2}
    loan_dic={'yes':1,'no':3,'unknown':2}
    economic_score=default_dic[default]+housing_dic[housing]+loan_dic[loan]
    return economic_score
def one_hot_data_preparation(input):
    marital_dic={'divorced':1,'single':2,'married':3,'unknown':2}
    contact_dic={'cellular':1,'telephone':2}
    poutcome_dic={'failure':2,'nonexistent':1,'success':3}
    if input in marital_dic:
        return marital_dic[input]
    if input in contact_dic:
        return contact_dic[input]
    if input in poutcome_dic:
        return poutcome_dic[input]

# 整合数据，将数据变成Dataframe格式，
# 并且给需要one-hot编码的marital,contact和poutcome进行数值化

def data_transfer_process(dataset):
    background_score_list=[]
    economic_score_list=[]
    marital_list=[]
    contact_list=[]
    poutcome_list=[]
    label_list=[]
    for i in dataset.values:
        # print(i[0],i[1],type(i[0]))
        background_score=background_combination(i[0],i[1])
        economic_score=economic_level(i[2],i[3],i[4])
        marital_value=one_hot_data_preparation(i[5])
        contact_value=one_hot_data_preparation(i[6])
        poutcome_value=one_hot_data_preparation(i[7])

        background_score_list.append(background_score)
        economic_score_list.append(economic_score)
        marital_list.append(marital_value)
        contact_list.append(contact_value)
        poutcome_list.append(poutcome_value)

    # return background_score_list,economic_score_list,marital_list,contact_list,poutcome_list
    list=[background_score_list,economic_score_list,marital_list,contact_list,poutcome_list]
    list=transpose(list)
    name=['background','economic_level','marital','contact','poutcome']
    data=pd.DataFrame(columns=name,data=list)
    return data



def train_data_preparation():

    # data induction
    train_data = pd.DataFrame(pd.read_csv('./train.csv'))
    label=train_data[['subscribe']]

    label_list=[]
    train_data=train_data[['job','education',
                           'default','housing','loan',
                           'marital','contact','poutcome']]

    processed_train_data=data_transfer_process(train_data)

    for i in label.values:
        label_value=judge(i)
        label_list.append(label_value)
    label_name=['label']
    label_train=pd.DataFrame(columns=label_name,data=label_list)

    one_hot_marital=OneHotEncoder(sparse=False).fit_transform(processed_train_data[['marital']])
    train_marital_list=['marital_1','marital_2','marital_3']
    one_hot_marital=pd.DataFrame(one_hot_marital).astype(int)
    one_hot_marital.columns=train_marital_list


    one_hot_contact=OneHotEncoder(sparse=False).fit_transform(processed_train_data[['contact']])
    train_contact_list=['contact_1','contact_2']
    one_hot_contact=pd.DataFrame(one_hot_contact).astype(int)
    one_hot_contact.columns=train_contact_list

    one_hot_poutcome=OneHotEncoder(sparse=False).fit_transform(processed_train_data[['poutcome']])
    train_poutcome_list=['poutcome_1','poutcome_2','poutcome_3']
    one_hot_poutcome=pd.DataFrame(one_hot_poutcome).astype(int)
    one_hot_poutcome.columns=train_poutcome_list


    train_data=processed_train_data.drop('marital',axis=1)
    train_data=processed_train_data.drop('contact',axis=1)
    train_data=processed_train_data.drop('poutcome',axis=1)

    frames=[processed_train_data,one_hot_marital,one_hot_contact,one_hot_poutcome,label_train]
    train_result=pd.concat(frames,axis=1)

    return train_result

def tes_data_preparation():

    # data induction
    test_data = pd.DataFrame(pd.read_csv('./test.csv'))
    test_data=test_data[['job','education',
                         'default','housing','loan',
                         'marital','contact','poutcome']]
    processed_test_data=data_transfer_process(test_data)

    one_hot_marital=OneHotEncoder(sparse=False).fit_transform(processed_test_data[['marital']])
    train_marital_list=['marital_1','marital_2','marital_3']
    one_hot_marital=pd.DataFrame(one_hot_marital).astype(int)
    one_hot_marital.columns=train_marital_list

    one_hot_contact=OneHotEncoder(sparse=False).fit_transform(processed_test_data[['contact']])
    train_contact_list=['contact_1','contact_2']
    one_hot_contact=pd.DataFrame(one_hot_contact).astype(int)
    one_hot_contact.columns=train_contact_list

    one_hot_poutcome=OneHotEncoder(sparse=False).fit_transform(processed_test_data[['poutcome']])
    train_poutcome_list=['poutcome_1','poutcome_2','poutcome_3']
    one_hot_poutcome=pd.DataFrame(one_hot_poutcome).astype(int)
    one_hot_poutcome.columns=train_poutcome_list


    test_data=processed_test_data.drop('marital',axis=1)
    test_data=processed_test_data.drop('contact',axis=1)
    test_data=processed_test_data.drop('poutcome',axis=1)
    frames=[processed_test_data,one_hot_marital,one_hot_contact,one_hot_poutcome]
    test_result=pd.concat(frames,axis=1)

    return test_result



# a=train_data_preparation()
# print(a)

