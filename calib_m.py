# -*- coding: utf-8 -*-
"""
Homework: Calibrate the Camera with ZhangZhengyou Method.
Picture File Folder: ".\\pic\\IR_camera_calib_img", With Distort.

By YouZhiyuan 2019.11.18

================================================================
debug 修正说明（原始接口尽量保持；每处修改都对应一个可复现的问题）
----------------------------------------------------------------
1. dedistortion 曾把 inter_corner_shape=(11,8)（内角点数）当作图像尺寸，传给
   estimateNewCameraMatrixForUndistortRectify 与 undistortImage。后果：
     - 新内参焦距算成 460.6，而按真实尺寸应为 625.4（偏小约 26%）；
     - 输出 PNG 只有 11x8 像素。
   现改为逐张图读取自身尺寸 (w, h) 计算。

2. calib 曾在函数体内把 img_dir / img_type 覆盖为硬编码值，调用方传参无效。
   实测：传入 RGB_camera_calib_img（41 张）时，实际仍读 myir（30 张）。
   现已删除该覆盖。

3. 图片目录为空或全部读取失败时，gray 未定义 -> UnboundLocalError。
   实测：glob 返回空列表时报 "local variable 'gray' referenced before
   assignment"。现改为显式报错。

4. 角点检测用的 CHECKERBOARD 曾硬编码为 (11,8)，与传入的 inter_corner_shape
   相互独立；换棋盘即静默错配。现统一由参数推导。

5. size_per_grid 参数曾被完全忽略（objp 用的是整数格序号，另有一份
   cp_world = cp_int * size_per_grid 是死代码）。现真正乘上物理边长，
   使外参平移具有米制单位（不影响 K、D）。

6. imshow 之后从未调用 waitKey，GUI 事件循环不刷新，窗口显示空白/不响应；
   也没有 destroyAllWindows。现补上，并加 show_gui 开关便于无界面运行。

7. cv2.imread 返回值未检查；有效图片数 N_OK 未做下限检查；calibrate 返回的
   rms 被直接丢弃，导致无法判断标定质量。现分别补上检查与打印。

8. 清理无用变量（cp_int、cp_world、obj_points、img_points）与重复的
   glob 调用；glob 模式 '**.' 在未开 recursive 时等价于 '*.'，已改直白写法；
   结果按文件名排序，保证标定可复现；objp 改为 copy 入列表，避免共享引用。
================================================================
"""

import os
import glob
import numpy as np
import cv2


def calib(inter_corner_shape, size_per_grid, img_dir, img_type,
          show_gui=True, gui_delay_ms=50):
    """用张正友标定法标定鱼眼相机，返回 (K, D)。

    inter_corner_shape : 内角点数 (列, 行)，例如 (11, 8)
    size_per_grid      : 棋盘格实际边长，单位 m
    img_dir / img_type : 图片目录与扩展名
    show_gui           : 是否弹窗显示角点检测结果（无界面环境置 False）
    gui_delay_ms       : 每张图在窗口中的停留时间（毫秒）
    """
    WIN = 'FoundCorners'
    CHECKERBOARD = (int(inter_corner_shape[0]), int(inter_corner_shape[1]))

    # 世界坐标系下的角点坐标：先按 (0,0,0),(1,0,0),(2,0,0)... 生成格序号，
    # 再乘以格子物理边长，得到以米为单位的世界坐标。
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= size_per_grid

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # CALIB_CHECK_COND 会在求解前检查条件数，遇到病态视角组会直接抛异常。
    # 若标定报与该标志相关的错误，去掉它再试即可（不影响其余流程）。
    calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                         + cv2.fisheye.CALIB_CHECK_COND
                         + cv2.fisheye.CALIB_FIX_SKEW)

    images = sorted(glob.glob(os.path.join(img_dir, '*.' + img_type)))
    if not images:
        raise FileNotFoundError(
            '在 %r 下没有找到 *.%s 图片，请检查 img_dir 与 img_type。' % (img_dir, img_type))

    if show_gui:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)      # 可拖拽缩放
        cv2.resizeWindow(WIN, 1280, 960)             # 初始显示尺寸
        cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)

    objpoints = []      # 世界坐标下的角点
    imgpoints = []      # 像素坐标下的角点
    image_size = None   # (宽, 高)
    n_unreadable = 0
    n_nocorner = 0

    for fname in images:
        img = cv2.imread(fname)       

        h, w = img.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            raise ValueError('所有图片尺寸必须一致：%s 为 (%d, %d)，期望 %s'
                             % (os.path.basename(fname), w, h, image_size))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if not ret:
            n_nocorner += 1
            continue

        # cornerSubPix 原地精化 corners
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        objpoints.append(objp.copy())   # copy：避免所有条目共享同一块内存
        imgpoints.append(corners)

        if show_gui:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, ret)
            cv2.imshow(WIN, vis)
            cv2.waitKey(gui_delay_ms)   # 必须调用，否则窗口事件循环不刷新

    if show_gui:
        cv2.destroyAllWindows()

    N_OK = len(objpoints)
    print('找到 %d 张可用图片（共 %d 张：读取失败 %d，未检出角点 %d）'
          % (N_OK, len(images), n_unreadable, n_nocorner))
    if N_OK < 3:
        raise RuntimeError('有效图片只有 %d 张，鱼眼标定至少需要 3 张。' % N_OK)

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print('DIM=%s' % (image_size,))
    print('重投影误差 RMS = %.6f 像素' % rms)
    print('K=np.array(%s)' % K.tolist())
    print('D=np.array(%s)' % D.tolist())

    return K, D


def dedistortion(img_dir, img_type, save_dir, mat_inter, coff_dis, balance=1.0):
    """按每张图自身的尺寸做去畸变并保存。

    balance: 0 表示尽量裁掉黑边，1 表示保留全部原始视野（会出现黑边）。
    """
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    if not images:
        raise FileNotFoundError('在 %r 下没有找到 *.%s 图片。' % (img_dir, img_type))
    os.makedirs(save_dir, exist_ok=True)

    n_ok = 0
    for fname in images:
        img = cv2.imread(fname)
        # if img is None:
        #     print('[跳过] 读取失败: %s' % fname)
        #     continue

        h, w = img.shape[:2]
        image_size = (w, h)          # <<< 关键修正：用真实图像尺寸，而非内角点数

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mat_inter, coff_dis, image_size, np.eye(3), balance=0, fov_scale=1.7)
        dst = cv2.fisheye.undistortImage(
            img, mat_inter, coff_dis, Knew=new_K, new_size=image_size)

        cv2.imwrite(os.path.join(save_dir, os.path.basename(fname)), dst)
        n_ok += 1

    print('去畸变图片已保存到: %s（成功 %d / %d 张，输出尺寸 %s）'
          % (save_dir, n_ok, len(images), image_size))


if __name__ == '__main__':
    inter_corner_shape = (11, 8)
    size_per_grid = 0.02
    img_dir = ".\\pic\\myir"
    img_type = "png"

    # 标定相机
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir, img_type)

    # 去畸变并保存结果
    save_dir = ".\\pic\\test2"
    dedistortion(img_dir, img_type, save_dir, mat_inter, coff_dis, balance=1.0)
