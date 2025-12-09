import os
import csv
import glob
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import nibabel as nib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle


# -------------------- 数据结构 --------------------
@dataclass
class Box:
    slice_idx: int
    x: float
    y: float
    w: float
    h: float


# -------------------- 工具函数 --------------------
def _try_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    amin, amax = float(np.min(arr)), float(np.max(arr))
    return (arr - amin) / (amax - amin) if amax > amin else np.zeros_like(arr, dtype=np.float32)


def apply_window(img: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    在 [0,1] 归一化空间里做一个简单的窗宽/窗位:
    center, width 都在 0~1 左右，裁剪到 [center-width/2, center+width/2] 再拉伸回 [0,1]
    """
    img = img.astype(np.float32, copy=False)
    if width <= 0:
        return img

    low = center - width / 2.0
    high = center + width / 2.0
    if low >= high:
        return img

    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    return img


def normalize_and_maybe_invert_dicom(arr: np.ndarray, ds: pydicom.dataset.FileDataset) -> Tuple[np.ndarray, bool]:
    """
    只做 RescaleSlope/Intercept（得到 HU 或原始灰度）和 MONOCHROME1 反转，
    不再使用 VOI LUT 和全局归一化。
    """
    arr = arr.astype(np.float32, copy=False)

    # 转成 HU（或线性灰度）
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    inverted = False
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        # 简单反转灰度，保留线性关系
        arr = -arr
        inverted = True

    # 注意：不再 normalize 到 [0,1]，直接返回真实数值
    return arr, inverted


# -------------------- DICOM --------------------
def stack_dicom_series(datasets: List[pydicom.dataset.FileDataset]) -> Tuple[np.ndarray, Optional[Tuple[float, float]], bool, List[str]]:
    """把同一 series 的切片堆成 (N,H,W)。返回 (体数据, (x_mm,y_mm)或None, 是否反转过, 每slice的SOPUID)"""
    def sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        z = _try_float(ipp[2], 0.0) if isinstance(ipp, (list, tuple)) and len(ipp) == 3 else _try_float(getattr(ds, "SliceLocation", 0.0), 0.0)
        inst = _try_float(getattr(ds, "InstanceNumber", 0), 0)
        # 某些序列 z 递减，这里按 z 再按 InstanceNumber 排
        return (z, inst)

    datasets_sorted = sorted(datasets, key=sort_key)

    imgs, inverted_any = [], False
    sop_uids = []
    for ds in datasets_sorted:
        try:
            arr = ds.pixel_array
        except Exception:
            continue
        img, inv = normalize_and_maybe_invert_dicom(arr, ds)
        inverted_any |= inv
        if img.ndim == 3:  # 多帧
            imgs.extend([img[k] for k in range(img.shape[0])])
            sop_uids.extend([getattr(ds, "SOPInstanceUID", "")] * img.shape[0])
        else:
            imgs.append(img)
            sop_uids.append(getattr(ds, "SOPInstanceUID", ""))

    if not imgs:
        raise ValueError("该 Series 无可用像素数据。")

    # PixelSpacing: [row_mm, col_mm] -> (x_mm, y_mm) = (col, row)
    px_spacing = None
    first = datasets_sorted[0]
    if hasattr(first, "PixelSpacing"):
        try:
            ps = [float(x) for x in first.PixelSpacing]
            if len(ps) == 2:
                px_spacing = (ps[1], ps[0])
        except Exception:
            pass

    # 对齐尺寸
    H = min(im.shape[0] for im in imgs)
    W = min(im.shape[1] for im in imgs)
    imgs = [im[:H, :W] for im in imgs]
    vol = np.stack(imgs, axis=0)

    if len(sop_uids) != vol.shape[0]:
        sop_uids = (sop_uids + [""] * vol.shape[0])[:vol.shape[0]]

    return vol, px_spacing, inverted_any, sop_uids


# -------------------- NIfTI --------------------
def load_nifti_volume(path: str) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float, float]]:
    """载入 NIfTI，返回 (vol(N,H,W), (x_mm,y_mm), (dx,dy,dz))"""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim >= 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"NIfTI 维度不支持: shape={data.shape}")
    # (X,Y,Z) -> (Z,Y,X)
    data = np.transpose(data, (2, 1, 0))
    # data = normalize01(data)

    zooms = img.header.get_zooms()[:3]
    if len(zooms) != 3:
        raise ValueError("NIfTI header 缺少体素尺寸信息。")
    dx, dy, dz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    return data, (dx, dy), (dx, dy, dz)


# -------------------- GUI --------------------
class BoundingBoxGUI:
    def __init__(self, root):
        self.root = root
        root.title("Medical Image Bounding Box (DICOM & NIfTI)")
        root.geometry("1100x820")
        root.resizable(False, False)

        # 状态
        self.mode: Optional[str] = None  # 'dicom' or 'nifti'
        self.series_map: Dict[str, List[pydicom.dataset.FileDataset]] = {}
        self.study_uid: str = ""
        self.series_uid: Optional[str] = None
        self.sop_uids_by_slice: List[str] = []
        self.nifti_path: Optional[str] = None
        self.nifti_voxel_size: Optional[Tuple[float, float, float]] = None

        self.vol: Optional[np.ndarray] = None
        self.pixel_spacing: Optional[Tuple[float, float]] = None
        self.current_slice: int = 0
        self.boxes_by_slice: Dict[int, List[Box]] = {}

        self.vol: Optional[np.ndarray] = None
        self.pixel_spacing: Optional[Tuple[float, float]] = None
        self.current_slice: int = 0
        self.boxes_by_slice: Dict[int, List[Box]] = {}

        # Window level / width（在 0~1 归一化空间）
        self.window_center: float = 0.5
        self.window_width: float = 1.0

        # 保存当前体数据的强度范围（HU 或原始灰度）
        self.hu_min: float = 0.0
        self.hu_max: float = 1.0

        # ---- 布局：左侧按钮栏 + 右侧画布 ----
        center = tk.Frame(root); center.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(center, width=220, bd=2, relief=tk.GROOVE)
        sidebar.pack(side=tk.LEFT, fill=tk.Y); sidebar.pack_propagate(False)

        canvas_wrap = tk.Frame(center, bd=2, relief=tk.GROOVE)
        canvas_wrap.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧栏控件
        tk.Label(sidebar, text="Selection").pack(padx=12, pady=(12, 6), anchor="w")

        tk.Button(sidebar, text="打开影像…", command=self.open_image).pack(padx=16, pady=6, fill=tk.X)

        tk.Label(sidebar, text="Series (DICOM)").pack(padx=12, pady=(14, 2), anchor="w")
        self.series_combo = ttk.Combobox(sidebar, state="readonly")
        self.series_combo.bind("<<ComboboxSelected>>", self.on_series_change)
        self.series_combo.pack(padx=16, pady=(0, 8), fill=tk.X)

        tk.Label(sidebar, text="Slice").pack(padx=12, pady=(10, 2), anchor="w")
        self.slice_scale = tk.Scale(sidebar, from_=0, to=0, orient=tk.HORIZONTAL,
                                    command=self.on_slice_change, length=180, state=tk.DISABLED)
        self.slice_scale.pack(padx=16, pady=(0, 10))

        tk.Label(sidebar, text="Window Level").pack(padx=12, pady=(6, 2), anchor="w")
        self.win_level_scale = tk.Scale(
            sidebar,
            from_=-1000,   # 典型 CT 最低 HU
            to=3000,       # 典型 CT 最高 HU
            orient=tk.HORIZONTAL,
            command=self.on_window_change,
            length=180
        )
        self.win_level_scale.set(40)   # 一个常见的默认窗位（例如脑窗 40/80 之类）
        self.win_level_scale.pack(padx=16, pady=(0, 6))

        tk.Label(sidebar, text="Window Width").pack(padx=12, pady=(0, 2), anchor="w")
        self.win_width_scale = tk.Scale(
            sidebar,
            from_=1,       # 宽度最小 1
            to=4000,       # 常见上限（比 CT 动态范围略大一点）
            orient=tk.HORIZONTAL,
            command=self.on_window_change,
            length=180
        )
        self.win_width_scale.set(400)  # 常见的初始窗宽（比如软组织/脑窗）
        self.win_width_scale.pack(padx=16, pady=(0, 10))


        self.btn_undo = tk.Button(sidebar, text="撤销 (Ctrl+Z)", command=self.undo_last, state=tk.DISABLED)
        self.btn_undo.pack(padx=16, pady=4, fill=tk.X)
        self.btn_clear = tk.Button(sidebar, text="清空本 slice", command=self.clear_slice_boxes, state=tk.DISABLED)
        self.btn_clear.pack(padx=16, pady=4, fill=tk.X)
        self.btn_save = tk.Button(sidebar, text="导出 CSV", command=self.export_csv, state=tk.DISABLED)
        self.btn_save.pack(padx=16, pady=(12, 8), fill=tk.X)

        # 右侧：Matplotlib 画布（更大）
        self.fig, self.ax = plt.subplots(figsize=(8.8, 8.8))
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_wrap)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, canvas_wrap).update()

        # 滚轮切片
        # 如果想用 Tk 的原生滚轮事件来切 slice：
        canvas_widget.bind("<MouseWheel>", self.on_mousewheel)   # Windows / 大部分 macOS
        canvas_widget.bind("<Button-4>", self.on_mousewheel)     # Linux 上滚
        canvas_widget.bind("<Button-5>", self.on_mousewheel)     # Linux 下滚
        # 如果不需要 matplotlib 自己的 scroll_event，可以注释后一行：
        # self.canvas.mpl_connect("scroll_event", self.on_scroll)


        # 底部信息（横向滚动）
        bottom = tk.Frame(root); bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_text = tk.Text(bottom, height=2, wrap=tk.NONE)
        self.info_text.configure(state=tk.DISABLED)
        xscroll = tk.Scrollbar(bottom, orient=tk.HORIZONTAL, command=self.info_text.xview)
        self.info_text.configure(xscrollcommand=xscroll.set)
        self.info_text.pack(side=tk.TOP, fill=tk.X)
        xscroll.pack(side=tk.TOP, fill=tk.X)

        # 交互组件
        self.im = None
        self.rect_selector: Optional[RectangleSelector] = None
        self.patch_layer: List[Rectangle] = []

        # 快捷键
        root.bind("<Control-s>", lambda e: self.export_csv())
        root.bind("<Control-z>", lambda e: self.undo_last())

    def on_mousewheel(self, event):
        if self.vol is None:
            return

        # Windows / macOS: event.delta 正负
        if hasattr(event, "delta") and event.delta != 0:
            step = 1 if event.delta > 0 else -1
        else:
            # Linux: Button-4 / Button-5
            if getattr(event, "num", None) == 4:
                step = 1
            else:
                step = -1

        self.step_slice(step)


    # ---------- 信息条 ----------
    def set_info(self, text: str):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state=tk.DISABLED)

    # ---------- 打开影像（自动判断 NIfTI / DICOM） ----------
    def open_image(self):
        """
        选一个文件：
        - 若扩展名为 .nii/.nii.gz -> NIfTI
        - 否则尝试以 DICOM 读取该文件的元数据，并加载其所在文件夹里同一 Series 的整套切片
        macOS：不传 filetypes 以规避 NSOpenPanel 崩溃；其它平台提供过滤。
        """
        if sys.platform == "darwin":
            path = filedialog.askopenfilename(title="选择影像文件（NIfTI 或任意 DICOM 文件）")
        else:
            path = filedialog.askopenfilename(
                title="选择影像文件（NIfTI 或任意 DICOM 文件）",
                filetypes=[("Images", ("*.nii", "*.nii.gz", "*.dcm", "*.dicom")), ("All files", "*")]
            )
        if not path:
            return

        low = path.lower()
        if low.endswith(".nii") or low.endswith(".nii.gz"):
            self._load_nifti(path)
            return

        # 尝试按 DICOM 处理：以所选文件为“种子”，自动聚合所在目录的同 series
        try:
            ds_seed = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if not hasattr(ds_seed, "SeriesInstanceUID"):
                raise ValueError("不是有效的 DICOM 文件。")
            series_uid = str(getattr(ds_seed, "SeriesInstanceUID"))
            folder = os.path.dirname(path)

            # 在该目录扫描所有同 series 的 DICOM
            dsets = []
            for p in glob.glob(os.path.join(folder, "*")):
                try:
                    ds = pydicom.dcmread(p, stop_before_pixels=False, force=True)
                except Exception:
                    continue
                if getattr(ds, "SeriesInstanceUID", None) == series_uid and hasattr(ds, "SOPInstanceUID"):
                    dsets.append(ds)

            if not dsets:
                raise ValueError("未在该目录找到同一 Series 的 DICOM 切片。")

            self._load_dicom_series(dsets, study_uid=str(getattr(ds_seed, "StudyInstanceUID", "")), series_uid=series_uid)
        except Exception as e:
            messagebox.showerror("错误", f"无法作为 DICOM 加载：{e}")

    # ---------- 内部加载实现 ----------
    def _load_nifti(self, path: str):
        try:
            vol, inplane_spacing, voxel_size = load_nifti_volume(path)
        except Exception as e:
            messagebox.showerror("错误", f"NIfTI 读取失败：\n{e}")
            return

        self.mode = "nifti"
        self.nifti_path = path
        self.nifti_voxel_size = voxel_size
        self.vol = vol
        self.pixel_spacing = inplane_spacing
        self.sop_uids_by_slice = []
        self.study_uid = ""
        self.series_uid = ""
        self.boxes_by_slice.clear()
        self.current_slice = 0

        # ---- 根据数据范围初始化窗位/窗宽和滑条 ----
        vmin = float(np.min(self.vol))
        vmax = float(np.max(self.vol))
        self.hu_min, self.hu_max = vmin, vmax

        self.window_center = 0.5 * (vmin + vmax)
        self.window_width = max(vmax - vmin, 1.0)

        self.win_level_scale.configure(from_=int(vmin), to=int(vmax))
        self.win_width_scale.configure(from_=1, to=max(int(vmax - vmin), 1))

        self.win_level_scale.set(int(self.window_center))
        self.win_width_scale.set(int(self.window_width))

        self.slice_scale.configure(state=tk.NORMAL, from_=0, to=vol.shape[0]-1)
        self.slice_scale.set(0)
        self.btn_save.configure(state=tk.NORMAL)
        self.btn_undo.configure(state=tk.NORMAL)
        self.btn_clear.configure(state=tk.NORMAL)

        self.series_combo.set("")
        self.series_combo["values"] = []

        H, W = vol.shape[1], vol.shape[2]
        dx, dy, dz = voxel_size
        spacing = f"{self.pixel_spacing[0]:.3f}×{self.pixel_spacing[1]:.3f} mm (dx×dy), dz={dz:.3f} mm"
        self.set_info(f"NIfTI: {os.path.basename(path)} | 尺寸: {W}×{H}×{vol.shape[0]} | 体素: {spacing}")

        self.init_rectangle_selector()
        self.render_slice()

    def _load_dicom_series(self, ds_list: List[pydicom.dataset.FileDataset], study_uid: str, series_uid: str):
        try:
            vol, px_spacing, inverted_any, sop_uids = stack_dicom_series(ds_list)
        except Exception as e:
            messagebox.showerror("错误", f"DICOM Series 读取失败：\n{e}")
            return

        self.mode = "dicom"
        self.nifti_path = None
        self.nifti_voxel_size = None

        self.vol = vol
        self.pixel_spacing = px_spacing
        self.sop_uids_by_slice = sop_uids
        self.study_uid = study_uid
        self.series_uid = series_uid
        self.boxes_by_slice.clear()
        self.current_slice = 0

        self.slice_scale.configure(state=tk.NORMAL, from_=0, to=vol.shape[0]-1)
        self.slice_scale.set(0)
        self.btn_save.configure(state=tk.NORMAL)
        self.btn_undo.configure(state=tk.NORMAL)
        self.btn_clear.configure(state=tk.NORMAL)

        # 也把同目录所有 series 列出来，便于切换
        folder = os.path.dirname(getattr(ds_list[0], "filename", "") or ".")
        series_map: Dict[str, List[pydicom.dataset.FileDataset]] = {}
        for p in glob.glob(os.path.join(folder, "*")):
            try:
                ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
                if hasattr(ds, "SeriesInstanceUID") and hasattr(ds, "SOPInstanceUID"):
                    series_map.setdefault(str(getattr(ds, "SeriesInstanceUID")), []).append(ds)
            except Exception:
                continue
        self.series_map = series_map

        values, ids = [], []
        for suid, lst in series_map.items():
            desc = str(getattr(lst[0], "SeriesDescription", "") or "NoDescription")
            snum = str(getattr(lst[0], "SeriesNumber", "NA"))
            mod = str(getattr(lst[0], "Modality", "NA"))
            values.append(f"{mod} | #{len(lst):03d} | {snum} | {desc} | {suid}")
            ids.append(suid)
        self.series_combo["values"] = values
        self.series_combo_ids = ids
        if series_uid in ids:
            self.series_combo.current(ids.index(series_uid))
        elif ids:
            self.series_combo.current(0)

        H, W = vol.shape[1], vol.shape[2]
        spacing = f"{px_spacing[0]:.3f}×{px_spacing[1]:.3f} mm" if px_spacing else "未知"
        inv = "是" if inverted_any else "否"
        mod = str(getattr(ds_list[0], "Modality", "NA"))
        self.set_info(
            f"DICOM StudyUID: {self.study_uid} | SeriesUID: {self.series_uid} | 模态: {mod} | "
            f"尺寸: {W}×{H} | 像素间距: {spacing} | MONOCHROME1反转: {inv}"
        )

        self.init_rectangle_selector()
        self.render_slice()

    # 切换 series（下拉）
    def on_series_change(self, event):
        if not getattr(self, "series_combo_ids", None):
            return
        idx = self.series_combo.current()
        if idx < 0:
            return
        suid = self.series_combo_ids[idx]
        # 重新完整加载该 series（包括像素）
        folder_series = []
        if self.series_map.get(suid):
            for ds_meta in self.series_map[suid]:
                src = getattr(ds_meta, "filename", None)
                if src and os.path.exists(src):
                    try:
                        folder_series.append(pydicom.dcmread(src, force=True))
                    except Exception:
                        pass
        if folder_series:
            self._load_dicom_series(folder_series, str(getattr(folder_series[0], "StudyInstanceUID", "")), suid)

    # ---------- 交互 ----------
    def init_rectangle_selector(self):
        if self.rect_selector is not None:
            self.rect_selector.set_active(False)
            self.rect_selector.disconnect_events()
            self.rect_selector = None

        def on_select(eclick, erelease):
            if any(v is None for v in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)):
                return
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            x, y = float(min(x1, x2)), float(min(y1, y2))
            w, h = float(abs(x2 - x1)), float(abs(y2 - y1))

            H, W = self.vol.shape[1], self.vol.shape[2]
            x = max(0.0, min(x, W)); y = max(0.0, min(y, H))
            w = max(0.0, min(w, W - x)); h = max(0.0, min(h, H))
            if w <= 1 or h <= 1:
                return

            self.boxes_by_slice.setdefault(self.current_slice, []).append(Box(self.current_slice, x, y, w, h))
            self.draw_overlays()
            self.canvas.draw_idle()

        self.rect_selector = RectangleSelector(
            self.ax, onselect=on_select,
            useblit=True, button=[1],
            minspanx=2, minspany=2, spancoords='pixels', interactive=False
        )

    def step_slice(self, step: int):
        if self.vol is None:
            return
        new_idx = self.current_slice - int(step)  # 与原来的逻辑一致
        new_idx = max(0, min(new_idx, self.vol.shape[0] - 1))
        if new_idx != self.current_slice:
            self.current_slice = new_idx
            self.slice_scale.configure(state=tk.NORMAL)
            self.slice_scale.set(self.current_slice)
            self.render_slice()

    def on_scroll(self, event):
        if self.vol is None:
            return
        step = getattr(event, "step", None)
        if step is None:
            step = 1 if getattr(event, "button", "") == "up" else -1
        self.step_slice(step)

    def render_slice(self):
        if self.vol is None:
            return
        self.ax.clear()
        self.ax.axis("off")

        # 当前切片原始强度（HU 或 NIfTI intensity）
        img = self.vol[self.current_slice].astype(np.float32)

        # 按窗位/窗宽计算上下限
        wl = float(self.window_center)
        ww = max(float(self.window_width), 1.0)

        low = wl - ww / 2.0
        high = wl + ww / 2.0

        # 裁剪到 [low, high]
        img = np.clip(img, low, high)

        # 线性映射到 [0,1] 供 imshow 使用
        img_disp = (img - low) / (high - low)

        self.ax.imshow(img_disp, cmap="gray", interpolation="nearest")
        self.draw_overlays()
        self.canvas.draw_idle()

    def draw_overlays(self):
        for p in getattr(self, "patch_layer", []):
            try:
                p.remove()
            except Exception:
                pass
        self.patch_layer = []
        for b in self.boxes_by_slice.get(self.current_slice, []):
            rect = Rectangle((b.x, b.y), b.w, b.h, fill=False, linewidth=1.5)
            self.ax.add_patch(rect)
            self.patch_layer.append(rect)
        cnt = len(self.boxes_by_slice.get(self.current_slice, []))
        self.ax.text(5, 15, f"Slice {self.current_slice} | Boxes: {cnt}",
                     fontsize=9, color="white",
                     bbox=dict(facecolor="black", alpha=0.4, pad=2))

    def on_slice_change(self, val):
        if self.vol is None:
            return
        try:
            idx = int(float(val))
        except Exception:
            idx = self.current_slice
        idx = max(0, min(idx, self.vol.shape[0] - 1))
        if idx != self.current_slice:
            self.current_slice = idx
            self.render_slice()

    def on_window_change(self, val=None):
        if self.vol is None:
            return
        # 直接使用滑条当前值作为窗位/窗宽（单位：HU 或原始强度）
        self.window_center = float(self.win_level_scale.get())
        self.window_width = max(float(self.win_width_scale.get()), 1.0)
        self.render_slice()

    def undo_last(self):
        if self.vol is None:
            return
        lst = self.boxes_by_slice.get(self.current_slice, [])
        if lst:
            lst.pop()
            if not lst:
                self.boxes_by_slice.pop(self.current_slice, None)
            self.render_slice()

    def clear_slice_boxes(self):
        if self.vol is None:
            return
        if self.current_slice in self.boxes_by_slice:
            self.boxes_by_slice.pop(self.current_slice)
            self.render_slice()

    # ---------- 导出 ----------
    def export_csv(self):
        if not self.boxes_by_slice:
            messagebox.showwarning("提示", "没有任何标注可导出。")
            return
        save_path = filedialog.asksaveasfilename(title="导出 CSV", defaultextension=".csv",
                                                 filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        rows = []
        if self.mode == "dicom":
            for sidx, boxes in self.boxes_by_slice.items():
                sop_uid = self.sop_uids_by_slice[sidx] if 0 <= sidx < len(self.sop_uids_by_slice) else ""
                for b in boxes:
                    r = {
                        "modality": "DICOM",
                        "dicom_study_uid": self.study_uid,
                        "dicom_series_uid": self.series_uid or "",
                        "dicom_sop_uid": sop_uid,
                        "slice_index": sidx,
                        "x": round(b.x, 2), "y": round(b.y, 2),
                        "w": round(b.w, 2), "h": round(b.h, 2),
                    }
                    if self.pixel_spacing:
                        px, py = self.pixel_spacing
                        r.update({
                            "x_mm": round(b.x * px, 3),
                            "y_mm": round(b.y * py, 3),
                            "w_mm": round(b.w * px, 3),
                            "h_mm": round(b.h * py, 3),
                        })
                    rows.append(r)
        elif self.mode == "nifti":
            for sidx, boxes in self.boxes_by_slice.items():
                for b in boxes:
                    r = {
                        "modality": "NIFTI",
                        "nifti_path": self.nifti_path or "",
                        "slice_index": sidx,
                        "x": round(b.x, 2), "y": round(b.y, 2),
                        "w": round(b.w, 2), "h": round(b.h, 2),
                    }
                    if self.pixel_spacing:
                        px, py = self.pixel_spacing
                        r.update({
                            "x_mm": round(b.x * px, 3),
                            "y_mm": round(b.y * py, 3),
                            "w_mm": round(b.w * px, 3),
                            "h_mm": round(b.h * py, 3),
                        })
                    rows.append(r)
        else:
            messagebox.showwarning("提示", "尚未载入影像。")
            return

        # 字段顺序
        base_cols = ["modality", "slice_index", "x", "y", "w", "h"]
        extra_cols = ["dicom_study_uid", "dicom_series_uid", "dicom_sop_uid"] if self.mode == "dicom" else ["nifti_path"]
        mm_cols = ["x_mm", "y_mm", "w_mm", "h_mm"] if self.pixel_spacing else []
        fieldnames = extra_cols + base_cols + mm_cols

        try:
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader(); writer.writerows(rows)
            messagebox.showinfo("成功", f"已导出：\n{save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：\n{e}")


def main():
    root = tk.Tk()
    app = BoundingBoxGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()