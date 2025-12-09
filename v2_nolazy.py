import os
import sys
import json
import traceback
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib

from PyQt5 import QtCore, QtGui, QtWidgets


# ======================
# 数据加载部分
# ======================

def is_dicom_file(path: str) -> bool:
    """简单判定是否是 DICOM 文件。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return True
    try:
        _ = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def index_dicom_series(root_dir: str):
    """
    只遍历 root_dir，按 SeriesInstanceUID 组织 DICOM 文件（不读像素），
    返回: { series_uid: {"paths": [slice_path,...], "meta": {...}} }
    """
    series_files = defaultdict(list)

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if not os.path.isfile(fpath):
                continue
            if not is_dicom_file(fpath):
                continue
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                series_uid = getattr(ds, "SeriesInstanceUID", None)
                if series_uid:
                    series_files[series_uid].append(fpath)
            except Exception:
                continue

    series_dict = {}

    for series_uid, file_list in series_files.items():
        header_infos = []
        for f in file_list:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                header_infos.append((ds, f))
            except Exception:
                continue

        if not header_infos:
            continue

        # 用 header 排序（不读像素）
        def sort_key(info):
            ds, path = info
            inst = getattr(ds, "InstanceNumber", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if inst is not None:
                return float(inst)
            if ipp is not None and len(ipp) == 3:
                return float(ipp[2])
            return 0.0

        header_infos.sort(key=sort_key)
        paths_sorted = [hi[1] for hi in header_infos]
        ds0 = header_infos[0][0]

        def _first(v):
            try:
                return float(v[0])
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        meta = {
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": getattr(ds0, "StudyInstanceUID", None),
            "Modality": getattr(ds0, "Modality", None),
            "PatientID": getattr(ds0, "PatientID", None),
            "PixelSpacing": getattr(ds0, "PixelSpacing", None),
            "SliceThickness": getattr(ds0, "SliceThickness", None),
            "WindowCenter": _first(getattr(ds0, "WindowCenter", None)),
            "WindowWidth": _first(getattr(ds0, "WindowWidth", None)),
            "SeriesFolder": os.path.dirname(paths_sorted[0]) if paths_sorted else None,
            "NumSlices": len(paths_sorted),
        }

        series_dict[series_uid] = {
            "meta": meta,
            "paths": paths_sorted,
        }

    return series_dict


def index_nifti_files(root_dir: str):
    """
    只遍历 root_dir 中的 NIfTI 文件 (.nii / .nii.gz) 做索引，
    返回: { filename: {"meta": {...}} }
    实际像素数据在需要时再通过 file_path 懒加载。
    """
    nifti_dict = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            lower = fname.lower()
            if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                img = nib.load(fpath)  # nibabel 默认使用 mmap，比较省内存
                shape = img.shape
                affine = img.affine

                # 只允许 3D 或 4D（时间维度），其它直接跳过
                if len(shape) == 4:
                    num_slices = shape[2]
                elif len(shape) == 3:
                    num_slices = shape[2]
                else:
                    print(f"[WARN] NIfTI 维度不是 3D/4D，跳过: {fpath}")
                    continue

                meta = {
                    "file_path": fpath,
                    "shape": shape,
                    "affine": affine.tolist(),
                    "NumSlices": int(num_slices),
                }

                nifti_dict[fname] = {
                    "meta": meta
                }

            except Exception as e:
                print(f"[WARN] 读取 NIfTI 头失败，已跳过: {fpath}, error={e}")
                continue

    return nifti_dict


class VolumeWrapper:
    """
    统一封装 DICOM/NIfTI 体数据，支持懒加载：

    - DICOM: 通过 slice_paths 按需读取 .dcm 像素
    - NIfTI: 通过 file_path 按需读取 .nii/.nii.gz 像素
    """
    def __init__(self, name, meta: dict,
                 source_type: str,
                 slice_paths=None,
                 file_path=None):
        self.name = name
        self.meta = meta
        self.source_type = source_type

        # 路径信息
        self.slice_paths = slice_paths or []
        self.file_path = file_path or meta.get("file_path")

        # 实际像素 volume：(Z,H,W)，一开始不加载
        self._volume = None  # type: np.ndarray | None
        self.loaded = False

        # WL/WW 设置：先用 meta 里的，如果没有就等加载完后再算
        self.window_center = meta.get("WindowCenter", None)
        self.window_width = meta.get("WindowWidth", None)

    @property
    def volume(self) -> np.ndarray:
        """
        外部统一通过 .volume 访问；第一次访问时触发 load_volume()
        """
        if not self.loaded:
            self.load_volume()
        return self._volume

    def load_volume(self):
        """
        实际加载像素数据，并做必要的 shape 检查等。
        """
        if self.loaded:
            return

        if self.source_type == "dicom":
            self._load_dicom_volume()
        elif self.source_type == "nifti":
            self._load_nifti_volume()
        else:
            raise ValueError(f"未知 source_type: {self.source_type}")

        # 设置默认窗宽窗位（如果 meta 里没有）
        vmin = float(np.min(self._volume))
        vmax = float(np.max(self._volume))

        if self.window_center is None or self.window_width is None or self.window_width <= 0:
            self.window_center = (vmin + vmax) / 2.0
            self.window_width = max(vmax - vmin, 1.0)

        self.loaded = True

    def _load_dicom_volume(self):
        if not self.slice_paths:
            raise RuntimeError(f"DICOM {self.name} 没有任何 slice 路径。")

        valid_imgs = []
        valid_paths = []
        first_shape = None

        for path in self.slice_paths:
            try:
                ds = pydicom.dcmread(path, force=True)
                arr = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", 1.0)
                intercept = getattr(ds, "RescaleIntercept", 0.0)
                arr = arr * slope + intercept
            except Exception as e:
                print(f"[WARN] 读取 DICOM 像素失败，已跳过: {path}, error={e}")
                continue

            if first_shape is None:
                first_shape = arr.shape

            if arr.shape != first_shape:
                print(f"[WARN] DICOM series {self.meta.get('SeriesInstanceUID')} 中存在 shape 不一致的 slice，"
                      f"预期 {first_shape}，实际 {arr.shape}，已跳过: {path}")
                continue

            valid_imgs.append(arr)
            valid_paths.append(path)

        if not valid_imgs:
            raise RuntimeError(f"DICOM series {self.meta.get('SeriesInstanceUID')} 无有效 slice。")

        self._volume = np.stack(valid_imgs, axis=0)  # (Z,H,W)
        # 如果有 shape 不一致而被过滤，这里更新 slice_paths
        self.slice_paths = valid_paths

    def _load_nifti_volume(self):
        if not self.file_path:
            raise RuntimeError(f"NIfTI {self.name} 缺少 file_path。")

        img = nib.load(self.file_path)
        data = img.get_fdata()

        if data.ndim == 4:
            data = data[..., 0]

        # 变成 (Z,H,W)，你之前的转换逻辑如果有特别要求可以自行调整
        if data.ndim == 3:
            # 假设原本是 (X,Y,Z) -> (Z,Y,X)
            volume = np.transpose(data, (2, 1, 0))
        else:
            raise RuntimeError(f"NIfTI 维度异常: {self.file_path}, shape={data.shape}")

        self._volume = volume.astype(np.float32)
        # 更新 meta 里的 shape
        self.meta["shape"] = self._volume.shape


def build_volumes_from_folder(root_dir: str):
    """
    从 root_dir 只构建 VolumeWrapper 索引（懒加载像素）。
    - 每个 DICOM SeriesInstanceUID 对应一个 VolumeWrapper（带 slice_paths）
    - 每个 NIfTI 文件对应一个 VolumeWrapper（带 file_path）
    """
    volumes = []

    # DICOM series
    dicom_series = index_dicom_series(root_dir)
    for series_uid, v in dicom_series.items():
        meta = v["meta"]
        paths = v["paths"]
        name = f"DICOM {series_uid}"
        volumes.append(
            VolumeWrapper(
                name=name,
                meta=meta,
                source_type="dicom",
                slice_paths=paths,
                file_path=None,
            )
        )

    # NIfTI files
    nifti_files = index_nifti_files(root_dir)
    for fname, v in nifti_files.items():
        meta = v["meta"]
        fpath = meta.get("file_path")
        name = f"NIfTI {fname}"
        volumes.append(
            VolumeWrapper(
                name=name,
                meta=meta,
                source_type="nifti",
                slice_paths=None,
                file_path=fpath,
            )
        )

    return volumes


# ======================
# GUI 部分
# ======================

class ImageLabel(QtWidgets.QLabel):
    """
    负责接收鼠标事件的图像显示控件：
    - “点模式”：左键单击发出 pointAdded 信号
    - “框选模式”：左键拖拽，释放时发出 bboxFinished 信号
    - 鼠标滚轮：发出 wheelScrolled 信号，由主窗口控制 slice 切换
    """
    pointAdded = QtCore.pyqtSignal(float, float)
    bboxFinished = QtCore.pyqtSignal(float, float, float, float)
    wheelScrolled = QtCore.pyqtSignal(int)  # 正数 / 负数 表示滚动方向

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "point"

        # 用于 bbox 橡皮筋效果
        self._origin = None
        self._rubber_band = QtWidgets.QRubberBand(
            QtWidgets.QRubberBand.Rectangle, self
        )

        # 用于“丝滑”滚轮：累积小的 delta
        self._wheel_accum = 0

        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                           QtWidgets.QSizePolicy.Fixed)

    def setMode(self, mode: str):
        self.mode = mode

    # ---------- 鼠标点击/拖动，用于点和 bbox ----------

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.pixmap() is None:
            return

        x = event.x()
        y = event.y()
        if x < 0 or y < 0 or x >= self.width() or y >= self.height():
            return

        if self.mode == "point":
            # 点标注：直接发送坐标
            self.pointAdded.emit(float(x), float(y))
        elif self.mode == "bbox":
            # 框选模式：记录起点，并显示橡皮筋
            self._origin = QtCore.QPoint(x, y)
            self._rubber_band.setGeometry(
                QtCore.QRect(self._origin, QtCore.QSize())
            )
            self._rubber_band.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # 仅在 bbox 模式且已按下左键的情况下更新橡皮筋矩形
        if self.mode != "bbox":
            return
        if self._origin is None:
            return
        if self.pixmap() is None:
            return

        x = event.x()
        y = event.y()
        rect = QtCore.QRect(self._origin, QtCore.QPoint(x, y)).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self.mode != "bbox":
            return
        if self._origin is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.pixmap() is None:
            return

        # 结束橡皮筋显示
        self._rubber_band.hide()

        start = self._origin
        end = event.pos()
        self._origin = None

        x1 = float(start.x())
        y1 = float(start.y())
        x2 = float(end.x())
        y2 = float(end.y())

        self.bboxFinished.emit(x1, y1, x2, y2)

    # ---------- 鼠标滚轮，用于切换 slice（带累积，适配触控板） ----------

    def wheelEvent(self, event: QtGui.QWheelEvent):
        """
        将滚轮滚动转换为“步数”，并做累积：
        - 普通鼠标：通常每次 120，steps=1/−1
        - 触控板：会产生很多小于 120 的 delta，通过 self._wheel_accum 累积，
          达到一档（120）才发出一次 wheelScrolled，从而感觉更丝滑。
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return

        self._wheel_accum += delta
        step_unit = 120  # 一般 120 是一“格”

        steps = int(self._wheel_accum / step_unit)
        if steps != 0:
            # 剩余的碎片保留在累积里，下一次滚动继续加
            self._wheel_accum -= steps * step_unit
            self.wheelScrolled.emit(steps)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("医学影像标注工具（DICOM / NIfTI）")
        self.resize(1100, 700)

        self.volumes = []
        self.current_volume_index = -1
        self.current_slice_index = 0

        # 全局标注记录
        self.point_annotations = []  # dict(volume_idx, slice_idx, x, y)
        self.bbox_annotations = []   # dict(volume_idx, slice_idx, x1,y1,x2,y2)

        self._build_ui()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # 左侧：序列列表
        left_panel = QtWidgets.QVBoxLayout()
        self.series_list = QtWidgets.QListWidget()
        self.series_list.currentRowChanged.connect(self.on_series_changed)
        left_panel.addWidget(QtWidgets.QLabel("序列 / Volume 列表"))
        left_panel.addWidget(self.series_list)

        btn_open = QtWidgets.QPushButton("打开文件夹...")
        btn_open.clicked.connect(self.on_open_folder)
        left_panel.addWidget(btn_open)

        btn_save = QtWidgets.QPushButton("保存标注...")
        btn_save.clicked.connect(self.on_save_annotations)
        left_panel.addWidget(btn_save)

        left_panel.addStretch(1)


        # 中间：图像显示
        center_panel = QtWidgets.QVBoxLayout()
        self.image_label = ImageLabel()
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.pointAdded.connect(self.on_point_added)
        self.image_label.bboxFinished.connect(self.on_bbox_finished)
        self.image_label.wheelScrolled.connect(self.on_wheel_scrolled)  # 新增：滚轮 -> 切 slice
        center_panel.addWidget(self.image_label, alignment=QtCore.Qt.AlignCenter)

        center_panel.addSpacing(10)

        # slice 控件
        slice_layout = QtWidgets.QHBoxLayout()
        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_label = QtWidgets.QLabel("Slice: 0/0")

        slice_layout.addWidget(QtWidgets.QLabel("层面："))
        slice_layout.addWidget(self.slice_slider)
        slice_layout.addWidget(self.slice_label)

        center_panel.addLayout(slice_layout)

        # 右侧：参数和模式
        right_panel = QtWidgets.QVBoxLayout()

        # WL/WW
        wl_layout = QtWidgets.QHBoxLayout()
        self.spin_wl = QtWidgets.QDoubleSpinBox()
        self.spin_wl.setDecimals(1)
        self.spin_wl.setRange(-5000, 5000)
        self.spin_wl.valueChanged.connect(self.on_wl_changed)
        wl_layout.addWidget(QtWidgets.QLabel("WL:"))
        wl_layout.addWidget(self.spin_wl)

        ww_layout = QtWidgets.QHBoxLayout()
        self.spin_ww = QtWidgets.QDoubleSpinBox()
        self.spin_ww.setDecimals(1)
        self.spin_ww.setRange(1, 10000)
        self.spin_ww.valueChanged.connect(self.on_ww_changed)
        ww_layout.addWidget(QtWidgets.QLabel("WW:"))
        ww_layout.addWidget(self.spin_ww)

        btn_lung = QtWidgets.QPushButton("肺窗预设 (WL=-500, WW=1500)")
        btn_lung.clicked.connect(self.on_lung_preset)

        right_panel.addWidget(QtWidgets.QLabel("窗宽 / 窗位"))
        right_panel.addLayout(wl_layout)
        right_panel.addLayout(ww_layout)
        right_panel.addWidget(btn_lung)

        right_panel.addSpacing(20)

        # 标注模式
        right_panel.addWidget(QtWidgets.QLabel("标注模式"))

        self.radio_point = QtWidgets.QRadioButton("点标记")
        self.radio_bbox = QtWidgets.QRadioButton("Bounding Box")
        self.radio_point.setChecked(True)
        self.radio_point.toggled.connect(self.on_mode_changed)

        right_panel.addWidget(self.radio_point)
        right_panel.addWidget(self.radio_bbox)

        right_panel.addSpacing(20)

        # 清除标注按钮
        btn_clear_current = QtWidgets.QPushButton("清除当前层面标记")
        btn_clear_current.clicked.connect(self.on_clear_current_slice)

        btn_clear_all = QtWidgets.QPushButton("清除所有标记")
        btn_clear_all.clicked.connect(self.on_clear_all_annotations)

        right_panel.addWidget(btn_clear_current)
        right_panel.addWidget(btn_clear_all)

        right_panel.addSpacing(20)

        # 当前序列信息
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setReadOnly(True)
        right_panel.addWidget(QtWidgets.QLabel("当前序列信息"))
        right_panel.addWidget(self.info_text, stretch=1)

        # 布局组合
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addLayout(center_panel, stretch=5)
        main_layout.addLayout(right_panel, stretch=3)

    # ---------- 窗宽窗位应用 ----------

    def apply_window(self, img2d: np.ndarray, volume: VolumeWrapper):
        wc = volume.window_center
        ww = max(volume.window_width, 1e-3)
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
        img = img2d.astype(np.float32)
        img = np.clip(img, lo, hi)
        img = (img - lo) / ww
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        return img

    def update_image_display(self):
        if not self.volumes or self.current_volume_index < 0:
            self.image_label.clear()
            return

        volume = self.volumes[self.current_volume_index]

        # 确保已加载（一般在 on_series_changed 已经加载过，这里是安全兜底）
        try:
            vol_data = volume.volume
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            self.show_error_console("刷新图像时出错", tb_str)
            return

        z, h, w = vol_data.shape
        self.current_slice_index = max(0, min(self.current_slice_index, z - 1))

        img2d = vol_data[self.current_slice_index, :, :]
        img_norm = self.apply_window(img2d, volume)

        # 构造灰度 QImage
        qimage = QtGui.QImage(img_norm.data, w, h, w,
                              QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qimage.copy())  # copy 防止引用问题

        # 在当前 slice 上叠加标注
        painter = QtGui.QPainter(pixmap)
        pen_point = QtGui.QPen(QtCore.Qt.red)
        pen_point.setWidth(4)
        pen_box = QtGui.QPen(QtCore.Qt.green)
        pen_box.setWidth(2)

        # 当前 slice 的点和框
        for p in self.point_annotations:
            if p["volume_index"] == self.current_volume_index and p["slice_index"] == self.current_slice_index:
                painter.setPen(pen_point)
                x = p["x"]
                y = p["y"]
                painter.drawLine(int(x) - 4, int(y),
                                 int(x) + 4, int(y))
                painter.drawLine(int(x), int(y) - 4,
                                 int(x), int(y) + 4)

        for b in self.bbox_annotations:
            if b["volume_index"] == self.current_volume_index and b["slice_index"] == self.current_slice_index:
                painter.setPen(pen_box)
                rect = QtCore.QRectF(b["x1"], b["y1"],
                                     b["x2"] - b["x1"],
                                     b["y2"] - b["y1"])
                painter.drawRect(rect)

        painter.end()

        # 调整 label 尺寸为图像尺寸，保证坐标一一对应
        self.image_label.setFixedSize(w, h)
        self.image_label.setPixmap(pixmap)

        # 更新 slice label
        self.slice_label.setText(f"Slice: {self.current_slice_index + 1}/{z}")

        # 更新 WL/WW 控件显示
        self.spin_wl.blockSignals(True)
        self.spin_ww.blockSignals(True)
        self.spin_wl.setValue(volume.window_center)
        self.spin_ww.setValue(volume.window_width)
        self.spin_wl.blockSignals(False)
        self.spin_ww.blockSignals(False)

    def show_error_console(self, title: str, error_text: str):
        """
        简单的“console”对话框，把错误/traceback 显示出来，不让程序崩溃。
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 400)

        layout = QtWidgets.QVBoxLayout(dialog)

        text_edit = QtWidgets.QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(error_text)
        layout.addWidget(text_edit)

        btn_close = QtWidgets.QPushButton("关闭", dialog)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)

        dialog.exec_()


    # ---------- 事件处理：序列 / slice / WL/WW ----------

    def on_series_changed(self, row: int):
        if row < 0 or row >= len(self.volumes):
            self.current_volume_index = -1
            self.current_slice_index = 0
            self.update_image_display()
            self.info_text.clear()
            return

        self.current_volume_index = row
        self.current_slice_index = 0

        volume = self.volumes[row]

        # 第一次切到这个 volume 时，可能需要实际加载像素数据，显示等待光标
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            vol_data = volume.volume  # 触发懒加载
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            tb_str = traceback.format_exc()
            print(tb_str)
            self.show_error_console("加载影像数据时出错", tb_str)
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        z, h, w = vol_data.shape
        self.slice_slider.blockSignals(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(z - 1)
        self.slice_slider.setValue(0)
        self.slice_slider.blockSignals(False)

        # 显示 meta 信息
        info_lines = [f"名称: {volume.name}",
                      f"形状: Z={z}, H={h}, W={w}",
                      f"来源: {volume.source_type}"]
        for k, v in volume.meta.items():
            info_lines.append(f"{k}: {v}")
        self.info_text.setPlainText("\n".join(info_lines))

        self.update_image_display()

    def on_slice_changed(self, value: int):
        self.current_slice_index = value
        self.update_image_display()

    def on_wl_changed(self, value: float):
        if self.current_volume_index < 0:
            return
        volume = self.volumes[self.current_volume_index]
        volume.window_center = float(value)
        self.update_image_display()

    def on_ww_changed(self, value: float):
        if self.current_volume_index < 0:
            return
        volume = self.volumes[self.current_volume_index]
        volume.window_width = float(value)
        self.update_image_display()

    def on_lung_preset(self):
        """肺窗预设，按需可调整参数"""
        if self.current_volume_index < 0:
            return
        volume = self.volumes[self.current_volume_index]
        volume.window_center = -500.0
        volume.window_width = 1500.0
        self.update_image_display()

    def on_wheel_scrolled(self, steps: int):
        """
        鼠标滚轮切换当前 volume 的 slice。
        steps > 0 一般表示向上滚；你可以按习惯决定是上一层还是下一层。
        这里约定：
          - 向上滚：slice_index 减小（往“前”翻）
          - 向下滚：slice_index 增大（往“后”翻）
        """
        if self.current_volume_index < 0 or not self.volumes:
            return

        volume = self.volumes[self.current_volume_index]
        try:
            vol_data = volume.volume
        except Exception as e:
            tb_str = traceback.format_exc()
            print(tb_str)
            self.show_error_console("滚动切换层面时出错", tb_str)
            return

        z = vol_data.shape[0]

        new_slice = self.current_slice_index - steps
        new_slice = max(0, min(z - 1, new_slice))
        self.slice_slider.setValue(new_slice)


    # ---------- 标注模式 / 鼠标标注 ----------

    def on_mode_changed(self):
        if self.radio_point.isChecked():
            self.image_label.setMode("point")
        else:
            self.image_label.setMode("bbox")

    def on_point_added(self, x: float, y: float):
        if self.current_volume_index < 0:
            return
        record = {
            "volume_index": self.current_volume_index,
            "slice_index": self.current_slice_index,
            "x": x,
            "y": y
        }
        self.point_annotations.append(record)
        print("记录点：", record)
        self.update_image_display()

    def on_bbox_finished(self, x1: float, y1: float, x2: float, y2: float):
        if self.current_volume_index < 0:
            return
        # 规范化到左上-右下
        xmin = float(min(x1, x2))
        xmax = float(max(x1, x2))
        ymin = float(min(y1, y2))
        ymax = float(max(y1, y2))

        record = {
            "volume_index": self.current_volume_index,
            "slice_index": self.current_slice_index,
            "x1": xmin,
            "y1": ymin,
            "x2": xmax,
            "y2": ymax
        }
        self.bbox_annotations.append(record)
        print("记录 bounding box：", record)
        self.update_image_display()

    def on_clear_current_slice(self):
        """
        清除当前 volume + 当前 slice 上的所有点和 bbox 标记
        """
        if self.current_volume_index < 0:
            return

        v_idx = self.current_volume_index
        s_idx = self.current_slice_index

        # 只保留不在当前层面的标记
        self.point_annotations = [
            p for p in self.point_annotations
            if not (p["volume_index"] == v_idx and p["slice_index"] == s_idx)
        ]
        self.bbox_annotations = [
            b for b in self.bbox_annotations
            if not (b["volume_index"] == v_idx and b["slice_index"] == s_idx)
        ]

        print(f"已清除 volume={v_idx}, slice={s_idx} 上的所有标记")
        self.update_image_display()


    def on_clear_all_annotations(self):
        """
        清除所有 volume 的所有点和 bbox 标记
        """
        if not (self.point_annotations or self.bbox_annotations):
            QtWidgets.QMessageBox.information(self, "提示", "当前没有任何标记。")
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "确认",
            "确定要清除所有标记吗？该操作无法撤销。",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self.point_annotations.clear()
        self.bbox_annotations.clear()
        print("已清除所有标记")
        self.update_image_display()


    # ---------- 打开文件夹 / 保存标注 ----------

    def on_open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择包含 DICOM 或 NIfTI 的文件夹"
        )
        if not folder:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            try:
                # 这里如果内部抛异常，会被下面的大 except 接住
                volumes = build_volumes_from_folder(folder)
            finally:
                # 无论成败都要恢复光标
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            # 捕获所有未处理异常，打印到终端，并弹出错误“console”
            tb_str = traceback.format_exc()
            print(tb_str)
            self.show_error_console("打开文件夹时发生错误", tb_str)
            return

        if not volumes:
            QtWidgets.QMessageBox.warning(
                self, "提示", "该文件夹中未找到 DICOM 或 NIfTI 数据。"
            )
            return

        self.volumes = volumes
        self.series_list.clear()
        for v in self.volumes:
            self.series_list.addItem(v.name)

        # 清空标注
        self.point_annotations.clear()
        self.bbox_annotations.clear()

        # 自动选中第一个
        self.series_list.setCurrentRow(0)


    def on_save_annotations(self):
        if not (self.point_annotations or self.bbox_annotations):
            QtWidgets.QMessageBox.information(self, "提示", "当前没有任何标记。")
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存标注为 JSON", "annotations.json", "JSON files (*.json)"
        )
        if not path:
            return

        def volume_name(idx):
            if 0 <= idx < len(self.volumes):
                return self.volumes[idx].name
            return ""

        def volume_file_info(idx, slice_idx):
            """
            根据 volume_index 和 slice_index 计算对应的 file_path 和 file_name
            - NIfTI: 直接使用 volume.file_path
            - DICOM: 使用 volume.slice_paths[slice_idx]
            """
            if idx < 0 or idx >= len(self.volumes):
                return "", ""
            v = self.volumes[idx]

            # NIfTI：每个 volume 对应一个文件
            if v.source_type == "nifti":
                fp = v.file_path or v.meta.get("file_path", "")
                fn = os.path.basename(fp) if fp else ""
                return fp, fn

            # DICOM：每个 slice 对应一张 .dcm
            if v.source_type == "dicom":
                if v.slice_paths and 0 <= slice_idx < len(v.slice_paths):
                    fp = v.slice_paths[slice_idx]
                    fn = os.path.basename(fp)
                    return fp, fn
                # 找不到对应 slice 时，退而求其次给 series 目录
                series_folder = v.meta.get("SeriesFolder", "")
                return series_folder, os.path.basename(series_folder) if series_folder else ""

            # 其它类型默认空
            return "", ""

        data_points = []
        for p in self.point_annotations:
            fp, fn = volume_file_info(p["volume_index"], p["slice_index"])
            data_points.append(
                {
                    "volume_name": volume_name(p["volume_index"]),
                    "slice_index": p["slice_index"],
                    "x": p["x"],
                    "y": p["y"],
                    "file_path": fp,
                    "file_name": fn,
                }
            )

        data_bboxes = []
        for b in self.bbox_annotations:
            fp, fn = volume_file_info(b["volume_index"], b["slice_index"])
            data_bboxes.append(
                {
                    "volume_name": volume_name(b["volume_index"]),
                    "slice_index": b["slice_index"],
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "file_path": fp,
                    "file_name": fn,
                }
            )

        data = {
            "points": data_points,
            "bboxes": data_bboxes,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "成功", f"标注已保存到：\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存失败：\n{e}")



def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
