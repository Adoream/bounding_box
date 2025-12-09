# main_window.py
import sys
import os
import numpy as np
import json
import traceback
from PyQt5 import QtCore, QtGui, QtWidgets

from data_io import build_volumes_from_folder, VolumeWrapper
from widgets import ImageLabel
from annotations import AnnotationStore

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.volumes = []
        self.current_volume_index = -1
        self.current_slice_index = 0

        self.annotations = AnnotationStore()

        self._build_ui()

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
        z, h, w = volume.volume.shape
        self.current_slice_index = max(0, min(self.current_slice_index, z - 1))

        img2d = volume.volume[self.current_slice_index, :, :]
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

        # 当前 slice 的点和框（通过 AnnotationStore 过滤）
        pts, bboxes = self.annotations.iter_slice(
            self.current_volume_index,
            self.current_slice_index,
        )
        
        for p in pts:
            painter.setPen(pen_point)
            x = p.x
            y = p.y
            painter.drawLine(int(x) - 4, int(y), int(x) + 4, int(y))
            painter.drawLine(int(x), int(y) - 4, int(x), int(y) + 4)

        for b in bboxes:
            painter.setPen(pen_box)
            rect = QtCore.QRectF(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1,)
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
        z, h, w = volume.volume.shape
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
        z = volume.volume.shape[0]

        # 这里使用 -steps，是因为 angleDelta().y() > 0 表示向上滚
        new_slice = self.current_slice_index - steps
        new_slice = max(0, min(z - 1, new_slice))

        # 统一通过 slider 来驱动更新逻辑
        self.slice_slider.setValue(new_slice)


    def on_mode_changed(self):
        if self.radio_point.isChecked():
            self.image_label.setMode("point")
        else:
            self.image_label.setMode("bbox")

    def on_point_added(self, x: float, y: float):
        if self.current_volume_index < 0:
            return

        self.annotations.add_point(
            self.current_volume_index,
            self.current_slice_index,
            x,
            y,
        )
        print("记录点：", self.current_volume_index, self.current_slice_index, x, y)
        self.update_image_display()

    def on_bbox_finished(self, x1: float, y1: float, x2: float, y2: float):
        if self.current_volume_index < 0:
            return
        # 规范化到左上-右下
        xmin = float(min(x1, x2))
        xmax = float(max(x1, x2))
        ymin = float(min(y1, y2))
        ymax = float(max(y1, y2))

        self.annotations.add_bbox(
            self.current_volume_index,
            self.current_slice_index,
            xmin,
            ymin,
            xmax,
            ymax,
        )
        print("记录 bounding box：", self.current_volume_index, self.current_slice_index, xmin, ymin, xmax, ymax)
        self.update_image_display()


    def on_clear_current_slice(self):
        """
        清除当前 volume + 当前 slice 上的所有点和 bbox 标记
        """
        if self.current_volume_index < 0:
            return

        v_idx = self.current_volume_index
        s_idx = self.current_slice_index

        self.annotations.clear_slice(v_idx, s_idx)

        print(f"已清除 volume={v_idx}, slice={s_idx} 上的所有标记")
        self.update_image_display()


    def on_clear_all_annotations(self):
        """
        清除所有 volume 的所有点和 bbox 标记
        """
        if not self.annotations.has_any():
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

        self.annotations.clear_all()
        print("已清除所有标记")
        self.update_image_display()


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
        self.annotations.clear_all()

        # 自动选中第一个
        self.series_list.setCurrentRow(0)

    def on_save_annotations(self):
        if not self.annotations.has_any():
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
            根据 volume_index 和 slice_index 计算
            file_path / file_name，兼容 DICOM / NIfTI。
            """
            if idx < 0 or idx >= len(self.volumes):
                return "", ""

            v = self.volumes[idx]

            # NIfTI：整卷对应一个文件
            if v.source_type == "nifti":
                fp = getattr(v, "file_path", None) or v.meta.get("file_path", "")
                fn = os.path.basename(fp) if fp else ""
                return fp, fn

            # DICOM：一个 slice 对应一个 DICOM 文件
            if v.source_type == "dicom":
                if getattr(v, "slice_paths", None) and 0 <= slice_idx < len(v.slice_paths):
                    fp = v.slice_paths[slice_idx]
                    fn = os.path.basename(fp)
                    return fp, fn
                # 找不到对应 slice 时，退而求其次给 series 目录
                series_folder = v.meta.get("SeriesFolder", "")
                return series_folder, os.path.basename(series_folder) if series_folder else ""

            # 其它类型默认空
            return "", ""

        # 交给 AnnotationStore 生成最终 JSON 结构
        data = self.annotations.to_json_serializable(
            volume_name_func=volume_name,
            volume_file_info_func=volume_file_info,
        )

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QtWidgets.QMessageBox.information(self, "成功", f"标注已保存到：\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"保存失败：\n{e}")
