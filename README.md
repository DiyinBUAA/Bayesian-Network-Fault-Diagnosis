>该算法为基于贝叶斯网络的故障诊断算法，出自北京航空航天大学ATE实验室

# 基于贝叶斯网络的故障诊断算法

## 算法简介

传统的故障诊断算法如TEAMS-RT、专家系统等只能依据测试结果，定性地分析出哪些部件可能出现故障。如果可疑的故障部件非常多，依然会给现场测试人员的排故和维修更换造成很大的困扰。这里，我们基于FMECA表格中部件的故障概率，利用不同部件、故障与测试之间的联系，通过贝叶斯网络的方法实现了一种定量故障诊断算法。该算法能够利用测试的结果作为输入，自动推理出不同部件的故障概率，为人工进一步确定具体的故障部件提供了测试顺序，大幅减少了测试人员的工作量，提升了维修排故的效率。

## 算法运行环境

`python 3.8.16  pgmpy贝叶斯网络包`
见`requirements.txt`

## 调用方式

### 1 文件格式

建立贝叶斯网络需要‘部件-测试’等层与层之间的连接关系，这里算法规定每两层之间的关系均存储至一个excel表格文件中，支持`'xlsx'、'xls'、'xlr'`格式的表格文件。按照贝叶斯网络从顶至下的顺序，表格名称分别命名为`1.xlsx`、`2.xlsx`...，最多支持10层贝叶斯网络结构的建立。每个文件中信息存储的形式如下：

| 当前层名称 | 下一层名称 |
| :-----: | :-----: |
| 节点1 | 节点a |
| 节点2 | 节点b |
| 节点3 | 节点a |

由于顶层贝叶斯网络中部件节点需要先验概率，因此`1.xlsx`文件信息储存形式如下：

| 部件名称 | 部件故障概率 | 下一层名称 |
| :-----: | :-----: | :-----: |
| 节点1 | 0.01 | 节点a |
| 节点2 | 0.02 | 节点b |
| 节点3 | 0.03 | 节点a |
| 节点1 | 0.01 | 节点d |

### 2 算法输入输出说明

```python
BayDiagnosis.setmodel(file_name, observe_num, save_path, prob_rate=1)
"""生成贝叶斯网络模型

Args:
    file_name (_type_): 贝叶斯网络关系表格存储的文件夹路径
    observe_num (_type_): 蒙特卡洛仿真的部件个数
    save_path (_type_): 模型存储的位置
    prob_rate (_type_): 表格中部件故障先验概率的小数单位

Returns:
    node_data: 仿真数据的dataframe
"""

BayDiagnosis.diagnosis(model_path, unit_path, test_result)
"""进行贝叶斯网络故障诊断

Args:
    model_path (_type_): 模型路径
    unit_path (_type_): _包含最顶层部件名称的excel路径
    test_result (dict): 测试结果
    测试结果中1为测试正常，0为测试异常
Returns:
    prob (dict): 各个部件的推断故障概率
"""
```

### 3. 算法使用举例

```python
baydiag = BayDiagnosis()
# 建立模型
# excel文件的父文件夹名称
file = r'COM'
# 仿真部件的个数
observe_num = int(1e4)
prob_rate = 1e-3
# 模型存储路径
save_path = 'COMbay.xmlbif'
baydiag.setmodel(file, observe_num, save_path, prob_rate)

# 进行定量故障诊断
# 测试结果(字典)：键：测试名称；值：测试结果
test_result = {'test01': '1', 'test02': '1', 'test03': '1',
                'test04': '1', 'test05': '1', 'test06': '1',
                'test11': '1', 'test12': '0', 'test13': '1', 
                'test14': '1', 'test15': '1', 'test16': '1', 
                'test17': '0', 'test18': '1', 'test19': '1', 
                'test20': '1', 'test21': '1', 'test22': '1', 
                'test25': '1', 'test26': '1', 'test27': '1',
                'test29': '1', 'test30': '1'
                }
# 部件文件存储位置
unit_path = r'COM\1.xlsx'
# 贝叶斯网络模型位置
model_path = r'COMbay.xmlbif'
prob = baydiag.diagnosis(model_path, unit_path, test_result)
print(sorted(prob.items(),key=lambda x:x[1], reverse=True))
```

## 其他说明

贝叶斯网络不同节点边缘概率的计算采用依据数据进行极大似然估计的方法，推理采用的是变量消除法。
