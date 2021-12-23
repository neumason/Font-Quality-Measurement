import cv2
from matplotlib import pyplot as plt


def rgb_show(img_rgb):
    """
    显示彩色图片
    @param img_rgb:输入的rgb三通道图片
    @return:none
    """
    b, g, r = cv2.split(img_rgb)
    img_rgb = cv2.merge([r, g, b])
    plt.imshow(img_rgb)
    plt.show()
    # 显示几秒后自动关闭，使用时打开下面三行，注释上面两行
    # plt.ion()
    # plt.pause(0.1)
    # plt.close()


def gray_show(img_gray):
    """
    显示灰度图片
    @param img_gray: 输入的灰度图片
    @return:
    """
    cv2.split(img_gray)
    plt.imshow(img_gray, cmap='gray')
    plt.show()
    # 显示几秒后自动关闭，使用时打开下面三行，注释上面两行
    # plt.ion()
    # plt.pause(0.1)
    # plt.close()


def bw_show(img_bw):
    cv2.imshow('二值化图片', img_bw)
    cv2.waitKey()
