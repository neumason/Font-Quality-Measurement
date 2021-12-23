import xlwt
# import xlrd
import os
import inspect
import re


def mat2excel(data, name):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet
    [h, l] = data.shape  # h为行数，l为列数
    for i in range(h):
        for j in range(l):
            sheet1.write(i, j, float(data[i, j]))
    path_folder = './excel'
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    path = path_folder + '/' + name + '.csv'
    f.save(path)
