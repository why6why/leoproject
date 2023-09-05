#!/usr/bin/env python37
# -*- coding: utf-8 -*-
# Python version: 3.7

import numpy as np
from prettytable import PrettyTable
# pip install PTable


class ConfusionMatrix(object):

    def __init__(self, num_classes: int, classes_label_name: list, args_py: str):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  
        self.classes_label_name = classes_label_name  # 类别标签
        self.py = args_py
        # self.is_png = is_png

    def update(self, preds, labels): 
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1
        return

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = 100.00 * sum_TP / n  # 准确率

        Precision, Recall, f1 = 0.0, 0.0, 0.0  # 暂时赋值为0

        # if self.is_png == 'Yes':
        # precision, recall, specificity, f1_score
        table = PrettyTable()  # 创建一个表格
        # table.field_names = ["", "Precision", "Recall", "Specificity"]  # 精确度、召回率、特异度的计算
        table.field_names = ["", "Precision", "Recall", "f1"]
        for i in range(self.num_classes):  # 每一类
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            # TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 5) if TP + FP != 0 else 0.   # round（x，n）方法将返回x的值，该值四舍五入到小数点后的n位数字。
            Recall = round(TP / (TP + FN), 5) if TP + FN != 0 else 0.
            f1 = round((2 * Precision * Recall) / (Precision + Recall), 5) if Precision + Recall != 0 else 0.
            # Specificity = round(TN / (TN + FP), 5) if TN + FP != 0 else 0.

            table.add_row([self.classes_label_name[i], Precision, Recall, f1])
            # print(table)

            # with open('./result/'+ self.py + str('_table.txt'), 'a', encoding='utf-8') as f:
            #     f.write(str(table))
            # f.close()

        return acc, Recall, Precision, f1

