import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


# ======================
# 数据加载部分
# ======================

def is_dicom_file(path: str) -> bool:
    """简单判定是否是 DICOM 文件：优先用后缀，其次尝试读取标签。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return True
    # 不带后缀的 DICOM。这里简单粗暴地尝试读取头部。
    try:
        _ = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def load_dicom_series(root_dir: str):
    """
    遍历 root_dir，按 SeriesInstanceUID 组织 DICOM 文件，
    返回: { series_uid: {"volume": ndarray (Z,H,W), "meta": {...}} }
    """
    series_files = defaultdict(list)

    # 1. 遍历目录，找到所有 DICOM 文件，按 SeriesInstanceUID 归类
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
                # 非 DICOM 或损坏的文件直接跳过
                continue

    series_dict = {}

    # 2. 对每个 series，按 InstanceNumber / ImagePositionPatient 排序并堆叠
    for series_uid, file_list in series_files.items():
        slices = []
        for f in file_list:
            try:
                ds = pydicom.dcmread(f, force=True)
                slices.append(ds)
            except Exception:
                continue

        if not slices:
            continue

        # 排序：优先用 InstanceNumber，其次用 ImagePositionPatient[2]
        def sort_key(ds):
            inst = getattr(ds, "InstanceNumber", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if inst is not None:
                return float(inst)
            if ipp is not None and len(ipp) == 3:
                return float(ipp[2])
            return 0.0

        slices.sort(key=sort_key)

        # 读取像素并做 RescaleSlope/Intercept
        img_list = []
        for ds in slices:
            arr = ds.pixel_array.astype(np.float32)
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            arr = arr * slope + intercept
            img_list.append(arr)

        volume = np.stack(img_list, axis=0)  # (Z, H, W)

        # 取第一个 slice 的元数据
        ds0 = slices[0]
        meta = {
            "SeriesInstanceUID": series_uid,
            "StudyInstanceUID": getattr(ds0, "StudyInstanceUID", None),
            "Modality": getattr(ds0, "Modality", None),
            "PatientID": getattr(ds0, "PatientID", None),
            "PixelSpacing": getattr(ds0, "PixelSpacing", None),
            "SliceThickness": getattr(ds0, "SliceThickness", None),
        }

        # WindowCenter/WindowWidth 可能是 MultiValue
        wc = getattr(ds0, "WindowCenter", None)
        ww = getattr(ds0, "WindowWidth", None)

        def _first(v):
            try:
                # pydicom 的 MultiValue 可用索引
                return float(v[0])
            except Exception:
                try:
                    return float(v)
                except Exception:
                    return None

        meta["WindowCenter"] = _first(wc)
        meta["WindowWidth"] = _first(ww)

        series_dict[series_uid] = {
            "volume": volume,
            "meta": meta
        }

    return series_dict


def load_nifti_files(root_dir: str):
    """
    遍历 root_dir 中的 NIfTI 文件 (.nii / .nii.gz)，
    返回: { filename: {"volume": ndarray (Z,H,W), "meta": {...}} }
    """
    nifti_dict = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            lower = fname.lower()
            if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                img = nib.load(fpath)
                data = img.get_fdata()  # numpy ndarray, float64
                # 只处理 3D 的情况（Z,H,W），若是 4D，则取第一个时间帧
                if data.ndim == 4:
                    data = data[..., 0]
                # 将轴顺序调整为 (Z,H,W)
                # nibabel 默认是 (X,Y,Z) 或 (i,j,k)，这里简单假定最后一维是 Z
                if data.shape[2] < min(data.shape[0], data.shape[1]):
                    volume = np.transpose(data, (2, 0, 1))
                else:
                    # 如果不确定，就当第一维是 Z
                    volume = data
                volume = volume.astype(np.float32)

                affine = img.affine
                meta = {
                    "file_path": fpath,
                    "affine": affine.tolist()
                }

                nifti_dict[fname] = {
                    "volume": volume,
                    "meta": meta
                }

            except Exception:
                continue

    return nifti_dict


# ======================
# 交互式查看 + 标注
# ======================

class VolumeWrapper:
    """
    统一封装 DICOM series / NIfTI volume
    """
    def __init__(self, name, volume: np.ndarray, meta: dict, source_type: str):
        """
        volume: (Z, H, W)
        source_type: "dicom" or "nifti"
        """
        self.name = name
        self.volume = volume
        self.meta = meta
        self.source_type = source_type

        # 默认 WL/WW
        # 若有 DICOM 的 WindowCenter/Width，则用它；否则用数据范围
        vmin = float(np.min(volume))
        vmax = float(np.max(volume))

        wc = meta.get("WindowCenter", None)
        ww = meta.get("WindowWidth", None)

        if wc is None or ww is None:
            self.window_center = (vmin + vmax) / 2.0
            self.window_width = vmax - vmin if vmax > vmin else 1.0
        else:
            self.window_center = float(wc)
            self.window_width = float(ww)


class MedicalImageViewer:
    """
    简易医学影像查看/标注器：
    - 方向键 ↑/↓：切换当前 volume 的 slice
    - 方向键 ←/→：切换不同的 volume（不同序列/不同 NIfTI）
    - W/S：增大/减小 Window Width
    - E/D：增大/减小 Window Level
    - 左键单击：记录点坐标
    - 右键拖拽：RectangleSelector，记录 bounding box (x1,y1,x2,y2)
    - P：将当前所有标注保存到 annotations.json
    - Q/Esc：退出
    """

    def __init__(self, volumes):
        if not volumes:
            raise ValueError("未找到任何 DICOM 序列或 NIfTI volume。")

        self.volumes = volumes
        self.current_volume_idx = 0
        self.current_slice_idx = 0

        # 标注记录
        self.points = []   # 每个元素: dict(volume_name, slice_index, x, y)
        self.bboxes = []   # 每个元素: dict(volume_name, slice_index, x1, y1, x2, y2)

        # 初始化 figure
        self.fig, self.ax = plt.subplots(1, 1)
        self.img_handle = None

        # RectangleSelector（右键绘制矩形）
        self.rect_selector = RectangleSelector(
            self.ax,
            self.on_select_rectangle,
            useblit=True,
            button=[3],  # 右键
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=False
        )

        # 连接键盘和鼠标事件
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # 显示第一张图像
        self._show_current_slice()

    # ---------- Windowing ----------

    def _apply_window(self, img2d: np.ndarray, volume: VolumeWrapper):
        """
        将原始像素应用 window level / window width 后归一化到 [0,1]
        """
        wc = volume.window_center
        ww = max(volume.window_width, 1e-3)  # 避免除零
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0

        img = img2d.astype(np.float32)
        img = np.clip(img, lo, hi)
        img = (img - lo) / ww
        img = np.clip(img, 0.0, 1.0)
        return img

    # ---------- 显示 ----------

    def _get_current_volume(self) -> VolumeWrapper:
        return self.volumes[self.current_volume_idx]

    def _show_current_slice(self):
        volume = self._get_current_volume()

        z, h, w = volume.volume.shape
        self.current_slice_idx = max(0, min(self.current_slice_idx, z - 1))

        img2d = volume.volume[self.current_slice_idx, :, :]
        img_display = self._apply_window(img2d, volume)

        if self.img_handle is None:
            self.img_handle = self.ax.imshow(img_display, cmap='gray', origin='lower')
            self.ax.set_title(self._make_title())
        else:
            self.img_handle.set_data(img_display)
            self.ax.set_title(self._make_title())

        self.ax.set_xlim(0, img_display.shape[1])
        self.ax.set_ylim(0, img_display.shape[0])

        self.fig.canvas.draw_idle()

    def _make_title(self):
        volume = self._get_current_volume()
        z, h, w = volume.volume.shape
        return (
            f"{volume.name} | volume {self.current_volume_idx + 1}/{len(self.volumes)} | "
            f"slice {self.current_slice_idx + 1}/{z} | "
            f"WL={volume.window_center:.1f}, WW={volume.window_width:.1f}"
        )

    # ---------- 事件处理 ----------

    def on_key(self, event):
        key = event.key
        volume = self._get_current_volume()

        if key in ['up']:
            # 下一张 slice
            self.current_slice_idx += 1
            self._show_current_slice()
        elif key in ['down']:
            # 上一张 slice
            self.current_slice_idx -= 1
            self._show_current_slice()
        elif key in ['right']:
            # 下一 volume
            self.current_volume_idx = (self.current_volume_idx + 1) % len(self.volumes)
            self.current_slice_idx = 0
            self._show_current_slice()
        elif key in ['left']:
            # 上一 volume
            self.current_volume_idx = (self.current_volume_idx - 1) % len(self.volumes)
            self.current_slice_idx = 0
            self._show_current_slice()
        elif key in ['w', 'W']:
            # 增大 WW
            volume.window_width *= 1.1
            self._show_current_slice()
        elif key in ['s', 'S']:
            # 减小 WW
            volume.window_width /= 1.1
            self._show_current_slice()
        elif key in ['e', 'E']:
            # 增大 WL
            volume.window_center += volume.window_width * 0.05
            self._show_current_slice()
        elif key in ['d', 'D']:
            # 减小 WL
            volume.window_center -= volume.window_width * 0.05
            self._show_current_slice()
        elif key in ['p', 'P']:
            self.save_annotations()
        elif key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def on_click(self, event):
        """
        左键单击记录点坐标；右键由 RectangleSelector 接管绘制 bbox
        """
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # 非左键
            return

        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return

        volume = self._get_current_volume()
        record = {
            "volume_name": volume.name,
            "slice_index": int(self.current_slice_idx),
            "x": float(x),
            "y": float(y)
        }
        self.points.append(record)
        print("记录点：", record)

        # 在图上画一个小十字用于可视化（简单示例）
        self.ax.plot(x, y, 'r+', markersize=8)
        self.fig.canvas.draw_idle()

    def on_select_rectangle(self, eclick, erelease):
        """
        RectangleSelector 回调，记录 bounding box 的左上与右下坐标。
        """
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # 统一为 (x_min, y_min) 到 (x_max, y_max)
        xmin = float(min(x1, x2))
        xmax = float(max(x1, x2))
        ymin = float(min(y1, y2))
        ymax = float(max(y1, y2))

        volume = self._get_current_volume()
        record = {
            "volume_name": volume.name,
            "slice_index": int(self.current_slice_idx),
            "x1": xmin,
            "y1": ymin,
            "x2": xmax,
            "y2": ymax
        }
        self.bboxes.append(record)
        print("记录 bounding box：", record)

    # ---------- 标注保存 ----------

    def save_annotations(self, out_path="annotations.json"):
        data = {
            "points": self.points,
            "bboxes": self.bboxes
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"标注已保存到 {out_path}")


# ======================
# 主程序入口
# ======================

def build_volumes_from_folder(root_dir: str):
    """
    从 root_dir 中构建所有的 VolumeWrapper 列表：
    - 每个 DICOM SeriesInstanceUID 对应一个 volume
    - 每个 NIfTI 文件对应一个 volume
    """
    volumes = []

    # DICOM series
    dicom_series = load_dicom_series(root_dir)
    for series_uid, v in dicom_series.items():
        volume = v["volume"]
        meta = v["meta"]
        name = f"DICOM Series {series_uid}"
        volumes.append(VolumeWrapper(name=name, volume=volume, meta=meta, source_type="dicom"))

    # NIfTI files
    nifti_files = load_nifti_files(root_dir)
    for fname, v in nifti_files.items():
        volume = v["volume"]
        meta = v["meta"]
        name = f"NIfTI {fname}"
        volumes.append(VolumeWrapper(name=name, volume=volume, meta=meta, source_type="nifti"))

    return volumes


def main():
    parser = argparse.ArgumentParser(description="简易 DICOM/NIfTI 医学影像查看与标注工具")
    parser.add_argument("root", type=str, help="包含 DICOM 序列或 NIfTI 的根目录")
    parser.add_argument(
        "--annotations",
        type=str,
        default="annotations.json",
        help="标注保存路径（默认 annotations.json）"
    )
    args = parser.parse_args()

    root_dir = args.root
    out_path = args.annotations

    volumes = build_volumes_from_folder(root_dir)
    if not volumes:
        print("未在指定目录中找到任何 DICOM 或 NIfTI 数据。")
        return

    viewer = MedicalImageViewer(volumes)
    # 将输出文件路径记下来（也可以在按 P 时传参）
    viewer.annotations_path = out_path

    # 覆写保存函数以使用命令行参数路径
    def save_with_custom_path():
        viewer.save_annotations(out_path)

    viewer.save_annotations = save_with_custom_path

    print("操作说明：")
    print("  ↑ / ↓ : 切换当前 volume 的 slice")
    print("  ← / → : 切换不同 volume（不同 DICOM 序列或 NIfTI 文件）")
    print("  W / S : 增大 / 减小 Window Width")
    print("  E / D : 增大 / 减小 Window Level")
    print("  左键单击 : 记录点坐标")
    print("  右键拖拽 : 绘制 bounding box（记录左上与右下坐标）")
    print("  P : 保存标注到", out_path)
    print("  Q / Esc : 退出")

    plt.show()


if __name__ == "__main__":
    main()
