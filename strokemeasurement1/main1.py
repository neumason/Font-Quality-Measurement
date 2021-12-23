import cv2
from pic_show import rgb_show
from pic_show import gray_show
from pic_show import bw_show
from mat2excel import mat2excel
import numpy as np
import os
import openpyxl
import PySimpleGUI as sg


# 把二维列表存入excel中
def write2excel(file_path, new_list):
    # total_list = [['A', 'B', 'C', 'D', 'E'], [1, 2, 4, 6, 8], [4, 6, 7, 9, 0], [2, 6, 4, 5, 8]]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '明细'
    for r in range(len(new_list)):
        for c in range(len(new_list[0])):
            ws.cell(r + 1, c + 1).value = new_list[r][c]
            # excel中的行和列是从1开始计数的，所以需要+1
    wb.save(file_path)  # 注意，写入后一定要保存
    # print("成功写入文件: " + file_path + " !")
    # return 1


def shu_dingwei(image_bw_each, num_peak_list, mm):
    [image_bw_each_m, image_bw_each_n] = image_bw_each.shape
    pix_col1 = num_peak_list[mm][3]
    # print(pix_col1)
    pix_start = []
    pix_end = []
    pix_re = []
    bb = 0
    while bb < image_bw_each_m:
        bb = bb + 1
        if image_bw_each[bb - 1][pix_col1] == 1:
            pix_row1_start = bb
            # print(pix_row1_start)
            for aa in range(pix_row1_start, image_bw_each_m):
                if image_bw_each[aa][pix_col1] != 1:
                    pix_row1_end = aa
                    pix_start.append(pix_row1_start)
                    pix_end.append(pix_row1_end)
                    pix_re.append(abs(pix_row1_start - pix_row1_end))
                    bb = pix_row1_end
                    break
    if pix_re != [] and max(pix_re) > 30:
        max_where = pix_re.index(max(pix_re))
        pix_row1_start0 = pix_start[max_where]
        pix_row1_end0 = pix_end[max_where]
        # 开始位置x坐标，y坐标，结束位置x坐标，y坐标，竖宽度，竖长度
        return [pix_col1, pix_row1_start0 + 3, pix_col1, pix_row1_end0 - 3, num_peak_list[mm][2], abs(aa - bb)]



def shu_dingwei2(image_bw_each, num_peak_list, mm):
    [image_bw_each_m, image_bw_each_n] = image_bw_each.shape
    pix_col1 = num_peak_list[mm][3]
    # print(pix_col1)
    pix_start = []
    pix_end = []
    pix_re = []
    bb = 0
    while bb < image_bw_each_n:
        bb = bb + 1
        if image_bw_each[pix_col1][bb - 1] == 1:
            pix_row1_start = bb
            # print(pix_row1_start)
            for aa in range(pix_row1_start, image_bw_each_n):
                if image_bw_each[pix_col1][aa] != 1:
                    pix_row1_end = aa
                    pix_start.append(pix_row1_start)
                    pix_end.append(pix_row1_end)
                    pix_re.append(abs(pix_row1_start - pix_row1_end))
                    bb = pix_row1_end
                    break
    # 开始位置x坐标，y坐标，结束位置x坐标，y坐标，竖宽度，竖长度
    if pix_re!= [] and max(pix_re) > 30:
        max_where = pix_re.index(max(pix_re))
        pix_row1_start0 = pix_start[max_where]
        pix_row1_end0 = pix_end[max_where]
        return [pix_row1_start0 + 3, pix_col1, pix_row1_end0 - 3, pix_col1, num_peak_list[mm][2], abs(aa - bb)]


# rgb_show(image_origin)
def rgb2result(image_rgb):
    """
    对输入图片矩阵中的横竖进行识别
    @param image_rgb: 输入的三通道彩色图像矩阵
    @return: 带横竖的图片，列信息，横信息
    """
    image_rgb_c = image_rgb.copy()
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)  # 图像灰度化
    # gray_show(image_gray)
    # image_gray_cut = image_gray
    # mat2excel(image_gray, 'image_gray')

    #m*n是像素长和宽
    [image_gray_m, image_gray_n] = image_gray.shape
    # print(image_gray_m)
    # print(image_gray_n)
    image_gray_cut = image_gray[1:image_gray_m - 1, 1:image_gray_n - 1]   # 裁剪图片，去除黑框
    image_gray_cut_c = image_gray_cut.copy()
    image_bw_ret, image_bw = cv2.threshold(image_gray_cut, 100, 255, cv2.THRESH_BINARY_INV)   # 图像二值化
    # bw_show(image_bw)
    # 字体是白色1，其他是黑色0
    cnts, hierarchy = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # 图像外轮廓
    # print(cnts)
    cols_last = []
    rows_last = []
    # print(len(cnts))
    for a in range(len(cnts)):
        mask_roi = np.zeros(image_bw.shape[:2], dtype="uint8")
        cnts_temp = cnts[a].astype(np.int32)
        cv2.polylines(mask_roi, cnts_temp, 1, 255)
        cv2.fillPoly(mask_roi, [cnts_temp], 255)
        # cv2.imshow("mask_roi", mask_roi)
        # cv2.waitKey()
        image_bw_each = cv2.add(image_bw, np.zeros(np.shape(image_bw), dtype=np.uint8), mask=mask_roi)
        # bw_show(image_bw_each)
        image_bw_each = image_bw_each.astype(np.float32) / 255   # 每一块的掩膜图像
        # mat2excel(image_bw_each, "image_bw_each")
        # print(image_bw)
        [image_bw_each_m, image_bw_each_n] = image_bw_each.shape
        # print(image_bw_m)
        # print(image_bw_n)
        # 对竖进行统计
        num_col1 = np.zeros(shape=(1, image_bw_each_n))
        for i in range(image_bw_each_n):  # 对每一列进行循环
            # 统计每一列中所有黑色像素值的值
            num_row1 = 0
            for j in range(image_bw_each_m):
                num_row1 = num_row1 + image_bw_each[j][i]
            num_col1[0][i] = num_row1
        # print(num_col1)
        num_reduce1 = np.abs(num_col1[0][1:] - num_col1[0][0:-1])
        # print("num_reduce1")
        # print(num_reduce1)
        # print(num_reduce.shape[0])
        num_peak_where = []
        for n in range(num_reduce1.shape[0]):
            if num_reduce1[n] > 20:
                num_peak_where.append(n + 2)
        # print("num_peak_where")
        # print(num_peak_where)
        # num_peak_where_z = []
        # num_peak_where_z_temp = []   # 用于存放临近元素
        # for gg in range(len(num_peak_where)):
        #     for ff in range(gg + 1, len(num_peak_where)):
        #         # print(ff)
        #         if abs(num_peak_where[gg] - num_peak_where[ff]) < 20:
        #             num_peak_where_z_temp.append(num_peak_where[gg])
        #             break
        # for hh in num_peak_where:
        #     if hh not in num_peak_where_z_temp:
        #         num_peak_where_z.append(hh)
        # print("num_peak_where_z")
        # print(num_peak_where_z)
        num_peak_where_z = num_peak_where
        num_peak_list = []
        for m in range(1, len(num_peak_where_z)):
            if 2 < num_peak_where_z[m] - num_peak_where_z[m - 1] < 18:
                num_peak_list.append([num_peak_where_z[m - 1], num_peak_where_z[m], (num_peak_where_z[m] - num_peak_where_z[m - 1]), int((num_peak_where_z[m] + num_peak_where_z[m - 1]) / 2)])
                # 每一竖左侧坐标，右侧坐标，宽度，中间坐标
        # print("num_peak_list")
        # print(num_peak_list)
        shu = []
        if len(num_peak_list) != 0:
            for mm in range(len(num_peak_list)):
                shu.append(shu_dingwei(image_bw_each, num_peak_list, mm))
                cols_last.append(shu_dingwei(image_bw_each, num_peak_list, mm))
        # print(shu)
        shu = [dd for dd in shu if dd is not None]
        for cc in range(len(shu)):
            shu_each = shu[cc]
            cv2.line(image_rgb_c, (shu_each[0], shu_each[1]), (shu_each[2], shu_each[3]), (255, 0, 0), 5)

        # 对行进行统计
        num_row2 = np.zeros(shape=(1, image_bw_each_m))
        for i in range(image_bw_each_m):  # 对每一行进行循环
            # 统计每一行中所有黑色像素值的值
            num_col2 = 0
            for j in range(image_bw_each_n):
                num_col2 = num_col2 + image_bw_each[i][j]
            num_row2[0][i] = num_col2
        # print(num_row2)
        num_reduce2 = np.abs(num_row2[0][1:] - num_row2[0][0:-1])
        # print("num_reduce2")
        # print(num_reduce2)
        # print(num_reduce.shape[0])
        num_peak_where2 = []
        for n in range(num_reduce2.shape[0]):
            if num_reduce2[n] > 20:
                num_peak_where2.append(n + 2)
        # print("num_peak_where2")
        # print(num_peak_where2)
        num_peak_where_z2 = []
        # num_peak_where_z2_temp = []
        # for gg in range(len(num_peak_where2)):
        #     for ff in range(gg + 1, len(num_peak_where2)):
        #         if abs(num_peak_where2[gg] - num_peak_where2[ff]) < 15:
        #             num_peak_where_z2_temp.append(num_peak_where2[gg])
        #             break
        # for hh in num_peak_where2:
        #     if hh not in num_peak_where_z2_temp:
        #         num_peak_where_z2.append(hh)
        num_peak_list2 = []
        num_peak_where_z2 = num_peak_where2
        for m in range(1, len(num_peak_where_z2)):
            if 2 < num_peak_where_z2[m] - num_peak_where_z2[m - 1] < 15:
                num_peak_list2.append(
                    [num_peak_where_z2[m - 1], num_peak_where_z2[m], (num_peak_where_z2[m] - num_peak_where_z2[m - 1]),
                     int((num_peak_where_z2[m] + num_peak_where_z2[m - 1]) / 2)])
                # 每一竖左侧坐标，右侧坐标，宽度，中间坐标
        # print("num_peak_list2")
        # print(num_peak_list2)
        shu2 = []
        if len(num_peak_list2) != 0:
            for mm in range(len(num_peak_list2)):
                shu2.append(shu_dingwei2(image_bw_each, num_peak_list2, mm))
                rows_last.append(shu_dingwei2(image_bw_each, num_peak_list2, mm))
        # print("shu2")
        # print(shu2)
        shu2 = [dd for dd in shu2 if dd is not None]
        if shu2 != []:
            for cc in range(len(shu2)):
                shu_each = shu2[cc]
                cv2.line(image_rgb_c, (shu_each[0], shu_each[1]), (shu_each[2], shu_each[3]), (0, 255, 0), 5)
    # cv2.imshow("123", image_rgb_c)
    cv2.waitKey()
    print("竖信息")
    print(cols_last)
    print("行信息")
    print(rows_last)
    return image_rgb_c, cols_last, rows_last


if __name__ == '__main__':
    # image_path = "./fonts/2/1270.png"
    # image_origin = cv2.imread(image_path)
    # # rgb_show(image_origin)
    # image_bw = rgb2result(image_origin)
    file_path = './fonts/2/'
    file_name = os.listdir(file_path)
    # print(file_name)
    for i in range(len(file_name)):
        sg.one_line_progress_meter('This is my progress meter!', i + 1, len(file_name), '-key-')
        print(file_name[i])
        image_origin = cv2.imread('./fonts/2/' + file_name[i])
        # rgb_show(image_origin)
        image_shu_heng, shu_xinxi, heng_xinxi = rgb2result(image_origin)
        shu_xinxi = [dd for dd in shu_xinxi if dd is not None]
        shu_xinxi = [dd for dd in shu_xinxi if dd != []]
        heng_xinxi = [dd for dd in heng_xinxi if dd is not None]
        heng_xinxi = [dd for dd in heng_xinxi if dd != []]
        cv2.imwrite('./lzy_result/image/' + file_name[i], image_shu_heng)
        with open('./lzy_result/竖信息.csv', 'a+') as f1:
            f1.write(file_name[i])
            f1.write(',')
            f1.write(str(len(shu_xinxi)))
            f1.write(',')
            for a in shu_xinxi:
                for b in a:
                    f1.write(str(b))
                    f1.write(',')
            f1.write('\n')
        with open('./lzy_result/横信息.csv', 'a+') as f2:
            f2.write(file_name[i])
            f2.write(',')
            f2.write(str(len(heng_xinxi)))
            f2.write(',')
            for a in heng_xinxi:
                for b in a:
                    f2.write(str(b))
                    f2.write(',')
            f2.write('\n')








