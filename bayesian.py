import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import os

class BayDiagnosis():
    def __init__(self) -> None:
        """基于贝叶斯网络的故障诊断，通过部件的先验故障概率，利用蒙特卡洛方法获得
        不同层所有节点的数据，再使用极大似然估计获得所有节点的边缘概率，最后采用变量消除法
        获取在已知测试条件下不同部件的故障概率

        将层与层之间贝叶斯网络联系以excel来表示，每层网络从上至下的excel
        文件依次命名为1.xlsx、2.xlsx、3.xlsx...，其中数字代表从上至下的层数编号，e.g.
        ——————————————————————————————
        |本层节点名称  |下一层节点名称 |
        |fault1       |test1         |
        |fault2       |test1         |
        |............................|
        ——————————————————————————————
        贝叶斯网络顶部的部件层需要给出故障概率，即1.xlsx表格的组织形式为
        —————————————————————————————————————————————
        |本层节点名称  |故障概率       |下一层节点名称 |
        |fault1       |0.001         |test1         |
        |fault2       |0.003         |test1         |
        |...........................................|
        ————————————————————————————————————————————

        """
        pass

    def get_excel_name(self, file_name):
        file_name = os.listdir(file_name)
        excel_name = []
        for name in file_name:
            if 'xlsx' or 'xls' or 'xlr' in name:
                excel_name.append(name)
        return excel_name

    def setmodel(self, file_name, observe_num, save_path, prob_rate=1):
        """生成贝叶斯网络模型

        Args:
            file_name (_type_): 贝叶斯网络关系表格存储的文件夹路径
            observe_num (_type_): 蒙特卡洛仿真的部件个数
            save_path (_type_): 模型存储的位置
            prob_rate (_type_): 表格中部件故障先验概率的小数单位

        Returns:
            node_data: 仿真数据的dataframe
        """
        # 读入贝叶斯网络不同层的名称
        excel_list = self.get_excel_name(file_name)
        self.layer = len(excel_list)
        # 每一层所有节点和每一层所有仿真数据
        data_list = []
        num = 0
        for excel in excel_list:
            if excel.split('.')[0] == '1':
                excel_data = pd.read_excel(os.path.join(file_name, excel))
                # 部件名称与其故障概率
                unit_prob = excel_data[excel_data.columns[0:2]]
                unit_prob = unit_prob.drop_duplicates().reset_index(drop=True)
                unit = excel_data[excel_data.columns[0]].drop_duplicates().reset_index(drop=True)
                fault = excel_data[excel_data.columns[2]].drop_duplicates().reset_index(drop=True)
                # 使用蒙特卡洛方法产生仿真数据
                # 发生故障为0，未发生故障为1
                unit_data = pd.DataFrame(columns=unit)
                fault_data = pd.DataFrame(columns=fault, index=np.arange(observe_num))
                for i in range(len(unit_prob)):
                    prob = unit_prob[unit_prob.columns[1]][i] * prob_rate
                    observe = np.random.choice([0, 1], size=observe_num, p=[prob, 1-prob])
                    unit_data[unit[i]] = observe
                    for j in range(len(excel_data)):
                        if excel_data[excel_data.columns[0]][j] == unit[i]:
                            for k in range(observe_num):
                                if observe[k] == 0:
                                    fault_data[excel_data[excel_data.columns[2]][j]][k] = 0
                fault_data.fillna(1, inplace=True)
                                    
                data_list.append(unit_data)
                data_list.append(fault_data)

                num += 1
            else:
                excel_data = pd.read_excel(os.path.join(file_name, excel))
                # 获取下一层名称
                layer_name = excel_data[excel_data.columns[1]].drop_duplicates().reset_index(drop=True)
                # 使用蒙特卡洛方法产生仿真数据
                layer_data = pd.DataFrame(columns=layer_name, index=np.arange(observe_num))
                for i in data_list[num]:
                    for j in range(len(excel_data)):
                        if excel_data[excel_data.columns[0]][j] == i:
                            for k in range(observe_num):
                                if data_list[num][i][k] == 0:
                                    layer_data[excel_data[excel_data.columns[1]][j]][k] = 0
                layer_data.fillna(1, inplace=True)
                data_list.append(layer_data)
                # 所有节点的仿真数据
        node_data = pd.concat(data_list, axis=1)

        # 创建模型
        model = BayesianNetwork()
        nodes = []
        for node in node_data:
            model.add_node(node)
            nodes.append(node)

        for excel in excel_list:
            if excel.split('.')[0] == '1':
                excel_data = pd.read_excel(os.path.join(file_name, excel))
                for i in range(len(excel_data)):
                    if excel_data[excel_data.columns[0]][i]  in nodes and excel_data[excel_data.columns[2]][i] in nodes:
                        model.add_edge(excel_data[excel_data.columns[0]][i],
                                       excel_data[excel_data.columns[2]][i])
            else:
                excel_data = pd.read_excel(os.path.join(file_name, excel))
                for i in range(len(excel_data)):
                    if excel_data[excel_data.columns[0]][i] in nodes and excel_data[excel_data.columns[1]][i] in nodes:
                        model.add_edge(excel_data[excel_data.columns[0]][i],
                                       excel_data[excel_data.columns[1]][i])

        model.fit(node_data)
        model.save(save_path, filetype='xmlbif')
        return node_data

    def diagnosis(self, model_path, unit_path, test_result):
        """进行贝叶斯网络故障诊断

        Args:
            model_path (_type_): 模型路径
            unit_path (_type_): _包含最顶层部件名称的excel路径
            test_result (dict): 测试结果

        Returns:
            prob (dict): 各个部件的推断故障概率
        """
        model = BayesianNetwork.load(model_path)
        # for i in model.get_cpds():
        #    print(i)
        excel_data = pd.read_excel(unit_path)
        unit = excel_data[excel_data.columns[0]].drop_duplicates().reset_index(drop=True)
        model_infer = VariableElimination(model)
        prob = {}
        for u in unit:
            p = model_infer.query(variables=[u], evidence=test_result)
            # 防止部件故障为0的情况
            if len(p.values) == 1:
                prob[u] = 0
            else:    
                prob[u] = p.values[0]
        return prob

if __name__ == '__main__':
    
    # 测试1：3层贝叶斯网络
    # 建立模型
    baydiag1 = BayDiagnosis()
    file = r'COM'
    observe_num = int(1e4)
    prob_rate = 1e-3
    save_path = 'COMbay.xmlbif'
    # baydiag1.setmodel(file, observe_num, save_path, prob_rate)
    # 进行定量故障诊断
    test_result = {'test01': '1', 'test02': '0', 'test03': '0',
                    'test04': '1', 'test05': '1', 'test06': '1',
                    'test11': '1', 'test12': '0', 'test13': '1', 
                    'test14': '1', 'test15': '1', 'test16': '0', 
                    'test17': '1', 'test18': '1', 'test19': '0', 
                    'test20': '1', 'test21': '1', 'test22': '1', 
                    'test25': '1', 'test26': '1', 'test27': '1',
                    'test29': '1', 'test30': '1'
                    }

    unit_path = r'COM\1.xlsx'
    model_path = r'COMbay.xmlbif'
    prob = baydiag1.diagnosis(model_path, unit_path, test_result)
    print(sorted(prob.items(),key=lambda x:x[1], reverse=True)[0:10])

    # 测试2：2层贝叶斯公式
    baydiag2 = BayDiagnosis()
    # 建立模型
    file = r'IMU'
    observe_num = int(1e4)
    prob_rate = 1
    save_path = 'IMUbay.xmlbif'
    baydiag2.setmodel(file, observe_num, save_path, prob_rate)
    # 进行定量故障诊断
    test_result = {'test1': '0', 'test2': '0', 'test3': '0', 'test4': '0','test5': '0'}
    unit_path = r'IMU\1.xlsx'
    model_path = r'IMUbay.xmlbif'
    prob = baydiag2.diagnosis(model_path, unit_path, test_result)
    print(sorted(prob.items(),key=lambda x:x[1], reverse=True))


