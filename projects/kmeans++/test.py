"""
#kmeans++聚类算法，对小狗数据集进行聚类，返回最终各个聚类的结果
"""

import random
import numpy as np
from math import sqrt


class KmeanppCluster():
    def __init__(self, k, filename, types='kmeans', iters=10):
        self.k = k
        self.data_dict = {}
        self.data = []
        self.center_dict = {}
        self.iters = iters
        self.types = types
        self.new_center = [[0.0, 0.0] for k in range(self.k)]
        # 加载数据
        self.loaddata(filename)
        # random center
        self.center = self.init_center(k)
        # 数据格式初始化
        for i in range(k):
            self.center_dict.setdefault(i, {})
            self.center_dict[i].setdefault("center", self.center[i])
            self.center_dict[i].setdefault("items_dist", {})
            self.center_dict[i].setdefault("classify", [])

    def init_center(self, k):
        if self.types == 'kmeans':
            # kmeans算法是初始化随机k个中心点
            random.seed(1)
            center = [[self.data[i][r] for i in range(1, len((self.data)))]
                      for r in random.sample(range(len(self.data)), k)]
        else:
            # Kmeans ++ 算法基于距离概率选择k个中心点
            # 1.随机选择一个点
            center = []
            center.append(random.choice(range(len(self.data[0]))))
            # 2.根据距离的概率选择其他中心点
            for i in range(self.k - 1):
                weights = [self.distance_closest(self.data[0][x], center)
                           for x in range(len(self.data[0])) if x not in center]
                dp = [x for x in range(len(self.data[0])) if x not in center]
                total = sum(weights)
                # 基于距离设定权重
                weights = [weight / total for weight in weights]
                num = random.random()
                x = -1
                i = 0
                while i < num:
                    x += 1
                    i += weights[x]
                center.append(dp[x])
            center = [self.data_dict[self.data[0][center[k]]] for k in range(len(center))]
        print("初始化center", center)
        return center

    # 计算个点与中心的最小距离
    def distance_closest(self, x, center):
        min = 99999
        for centroid in center:
            distance = 0
            cent = self.data_dict[self.data[0][centroid]]
            for i in range(len(cent)):
                distance += abs(self.data_dict[x][i] - cent[i])  # 计算曼哈顿距离
            if distance < min:
                min = distance
        return min

    def get_median(self, alist):
        """get median of list"""
        tmp = list(alist)
        tmp.sort()
        alen = len(tmp)
        if (alen % 2) == 1:
            return tmp[alen // 2]
        else:
            return (tmp[alen // 2] + tmp[(alen // 2) - 1]) / 2

            # 标准化

    def standlize(self, column):
        median = self.get_median(column)
        asd = sum([abs(x - median) for x in column]) / len(column)
        result = [(x - median) / asd for x in column]
        return result

    # 加载数据并标准化
    def loaddata(self, filename):
        lista = []
        with open(filename, "r") as fileobject:
            lines = fileobject.readlines()
        header = lines[0].split(",")
        self.data = [[] for i in range(len(header))]
        for line in lines[1:]:
            line = line.split(",")
            for i in range(len(header)):
                if i == 0:
                    self.data[i].append(line[i])
                else:
                    self.data[i].append(float(line[i]))
        for col in range(1, len(header)):
            self.data[col] = self.standlize(self.data[col])
        # data_dict  data对应Key
        for i in range(0, len(self.data[0])):
            for col in range(1, len(self.data)):
                self.data_dict.setdefault(self.data[0][i], [])
                self.data_dict[self.data[0][i]].append(self.data[col][i])

    def kcluster(self):
        # 最大迭代次数iters
        for i in range(self.iters):
            class_dict = self.count_distance()  # 计算距离，比较个样本到各个中心的的出最小值，并划分到相应的类
            self.locate_center(class_dict)  # 重新计算中心点
            # print(self.data_dict)
            print("----------------迭代%d次----------------" % i)
            print(self.center_dict)  # 聚类结果{k:{{center:[]},{distance：{item：0.0}，{classify:[]}}}}
            if sorted(self.center) == sorted(self.new_center):
                break
            else:
                self.center = self.new_center
            if i < self.iters - 1:
                for j in range(self.k):
                    self.center_dict[j]["center"] = self.center[j]
                    self.center_dict[j]["items_dist"] = {}
                    self.center_dict[j]["classify"] = []

                    # 距离并分类

    def count_distance(self):
        min_list = []
        class_dict = {}
        for i in range(len(self.data[0])):
            class_dict[self.data[0][i]] = 0
            min_list.append(99999)
        # 计算距离
        for k in self.center_dict:
            # 遍历row,manhadon计算距离
            for i in range(len(self.data[0])):
                # 遍历column
                for col in range(1, len(self.data)):
                    self.center_dict[k]["items_dist"].setdefault(self.data[0][i], 0.0)
                    self.center_dict[k]["items_dist"][self.data[0][i]] += abs(
                        self.data[col][i] - self.center_dict[k]["center"][col - 1])
            # 分类 {item:class}
            for i in range(len(self.data[0])):
                if self.center_dict[k]["items_dist"][self.data[0][i]] < min_list[i]:
                    min_list[i] = self.center_dict[k]["items_dist"][self.data[0][i]]
                    class_dict[self.data[0][i]] = k
        return class_dict

    # 计算新的中心点
    def locate_center(self, class_dict):
        # class_dict {'Boston Terrier': 0, 'Brittany Spaniel': 1,
        # 加入分类的列表
        for item_name, k in class_dict.items():
            self.center_dict[k]["classify"].append(item_name)
        # print(class_dict)
        # print(self.center_dict)
        self.new_center = [[0.0 for j in range(1, len(self.data))] for i in range(self.k)]
        for k in self.center_dict:
            for i in range(len(self.data) - 1):
                for cls_item in self.center_dict[k]["classify"]:
                    self.new_center[k][i] += self.data_dict[cls_item][i]
                self.new_center[k][i] /= len(self.center_dict[k]["classify"])


filename = "./dogs.csv.txt"
model = KmeanppCluster(3, filename, types='kmeans++', iters=10)
model.kcluster()

