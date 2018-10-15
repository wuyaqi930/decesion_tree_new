from math import log
import operator
import numpy as np,pandas as pd

#定义相关变量
subp_square_sum = 0 


def read_newdataset(filename):
    dataset=pd.read_csv(filename)
    return dataset

def split_dataset_by_feature_value(Dataset,feature_name,value):
    subdataset_for_each_value=Dataset[Dataset[feature_name]==value]
    del subdataset_for_each_value[feature_name]
    return subdataset_for_each_value

def dataset_of_next_stage(featvalue,Dataset,feature_name):
    #featvalue = [0, 1, 2, 3]
    subdataset = {}    #用一个空字典来存储分出来的数据块
    for value in featvalue:
        # print(split_dataset_to_next_stage(dataset,feature_name,v))
        temp_subdataset = split_dataset_by_feature_value(dataset,feature_name, value)
        subdataset[value] = temp_subdataset
    return subdataset


#找出基尼指数最小的特征
def CART_chooseBestFeatureToSplit(dataset):
    feature_names = dataset.columns

    print("\nfeature_names")
    print(feature_names)

    # 将结果那一行删除，显示结果
    feature_names_list=list(feature_names)
    del feature_names_list[-1]

    print("\nfeature_names_list")
    print(feature_names_list) 

    # 巫亚奇定义的相关变量
    gini_value_list = 0 
    gini_gain = 0
    gini_gain_min = 9999 

    # 计算每一个特征的基尼指数
    for feature in feature_names_list:
        print("\nfeature")
        print(feature)

        feat_value = set(dataset[feature]) # 每个特征下的独特的value 
        print("\nfeature下独特的取值为")
        print(feat_value)

        gini = 0.0
        sub_dataset=dataset_of_next_stage(feat_value, dataset, feature) #每一类下，根据独特的数值，将该类下面的数据进行分组

        print("\nfeature对应的分组")
        print(sub_dataset)

        # 计算一个特征的基尼指数
        for v in feat_value:

            print("\n此时feature特定取值为：")
            print(v)

            print("\nfeature特定取值对应的子分组")
            print(sub_dataset[v]) #对某个特征，按照数值，进行分组

            #feature特定取值对应的子分组，对应的结果
            print("\nfeature特定取值对应的子分组对应的结果")
            sub_dataset[v]['结果']
            print(sub_dataset[v]['结果'])


            #feature特定取值对应的子分组，对应的结果的独立取值显示出来
            value_of_answer = set(sub_dataset[v]['结果']) #对某个特征的独立取值显示出来
            print(value_of_answer)

            subp_square_sum = 0

            # 计算value of answer 对应的每个value的概率
            for value in value_of_answer:
                print("\nsub_dataset[v]['结果'].value_counts(value)")
                subp = sub_dataset[v]['结果'].value_counts(value)           

            print("\nsubp")
            print(subp)

            # 计算概率的平方            
            subp_square = subp*subp
          
            print("\nsubp_square：")
            print(subp_square)

            # 计算概率的平方和
            subp_square.sum()

            print("\nsubp_square.sum()")
            print(subp_square.sum())
            
            # 计算基尼值       
            gini_value = 1 - subp_square.sum() #求出基尼值

            print("\ngini_value")
            print(gini_value)

            print("\ntype of gini_value")
            print(type(gini_value))

            # 计算基尼值对应的概率 
            p = dataset[feature].value_counts(normalize=True)

            print("\np")
            print(p)                   

            # 筛选该v in feat_value对应的基尼值概率 
            p_v = p.at[v]

            print("\np.at[v]")
            print(p.at[v])   
            
            # 计算基尼指数各个部分
            gini_gain_part = p_v*gini_value

            print("\ngini_gain_part")
            print(gini_gain_part)

            # 计算基尼指数总和
            gini_gain = gini_gain + gini_gain_part 

            print("\ngini_gain")
            print(gini_gain)
        
        # 将计算结果进行排序，筛选最小的基尼指数
        if (gini_gain < gini_gain_min):
            gini_gain_min = gini_gain
            gini_gain_min_feature = feature
           
    # 将计算结果进行排序，筛选最小的基尼指数
    print("\n最小基尼指数为：")
    print(gini_gain_min)

    print("\n最小基尼指数对应的feature为：")
    print(gini_gain_min_feature)

    #返回的最合适的feature,对应基尼指数最小的feature
    return gini_gain_min_feature


def CART_createTree(dataset):#还处于阉割状态
    classList=dataset['结果']
    if set(classList) == 1:
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataset.columns) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(dataset)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    #print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    CARTTree = {bestFeatLabel:{}}#字典声明
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    #在这里判断深度是否达到预设最大值(<=特征个数)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitdataset(dataset, bestFeat, value), subLabels)
        print(CARTTree)
    return CARTTree


def majorityCnt(data_last):#阉割状态
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    leafs=set(data_last)
    count={}
    for leaf in leafs:
        count[leaf]=data_last['结果'].value_counts(leaf)
    return count.index(max(count))



    classCont={}
    for vote in classList:
        if vote not in classCont.keys():
            classCont[vote]=0
        classCont[vote]+=1
    sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCont[0][0]




if __name__ == '__main__':
    filename = 'dataset.txt'
    testfile = 'testset.txt'
    dataset= read_newdataset(filename)
    # dataset,features=createDataSet()
    # print('dataset is')
    #print(dataset)
    #print(dataset.loc[[0,1,2,84]])  # 查询指定的行
    #print(dataset[['颜值','信贷情况','结果']].head(8))  # 查询指定的列
    #print("---------------------------------------------")
    #print(u"数据集长度", len(dataset))
    #print(type(dataset))
    #print (dataset['颜值'])
    #print(dataset.columns)
   
    #print('===========================')

    #CART_createTree(dataset)


    tree = CART_chooseBestFeatureToSplit(dataset)

    #featvalue = [0, 1, 2, 3]
    #feature_name='颜值'
    #print(dataset_of_next_stage(featvalue, dataset, feature_name))
