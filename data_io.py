import os
import numpy as np
import pydicom
import nibabel as nib
from collections import defaultdict


class VolumeWrapper:
    """
    统一封装 DICOM/NIfTI 体数据
    """
    def __init__(self, name, volume: np.ndarray, meta: dict,
                 source_type: str,
                 slice_paths=None,
                 file_path=None):
        """
        slice_paths: 对于 DICOM，长度为 Z 的列表，每个 slice 对应一个 .dcm 路径
        file_path: 对于 NIfTI，原始 .nii/.nii.gz 文件路径
        """
        self.name = name
        self.volume = volume  # (Z,H,W)
        self.meta = meta
        self.source_type = source_type

        # 额外保存路径信息
        self.slice_paths = slice_paths  # list or None
        # 如果没显式传入 file_path，就从 meta 里找（NIfTI 的 meta 里有）
        self.file_path = file_path or meta.get("file_path")

        vmin = float(np.min(volume))
        vmax = float(np.max(volume))

        wc = meta.get("WindowCenter", None)
        ww = meta.get("WindowWidth", None)

        if wc is None or ww is None or ww <= 0:
            self.window_center = (vmin + vmax) / 2.0
            self.window_width = max(vmax - vmin, 1.0)
        else:
            self.window_center = float(wc)
            self.window_width = float(ww)


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


def load_dicom_series(root_dir: str):
    """
    遍历 root_dir，按 SeriesInstanceUID 组织 DICOM 文件，
    返回: { series_uid: {"volume": ndarray (Z,H,W), "meta": {...}, "paths": [slice_path,...]} }
    """
    series_files = defaultdict(list)

    # 先按 series 分组
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
        # 同时保存 ds 和 file_path
        slice_infos = []
        for f in file_list:
            try:
                ds = pydicom.dcmread(f, force=True)
                slice_infos.append((ds, f))
            except Exception:
                continue

        if not slice_infos:
            continue

        # 排序（和之前一样）
        def sort_key(info):
            ds, path = info
            inst = getattr(ds, "InstanceNumber", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if inst is not None:
                return float(inst)
            if ipp is not None and len(ipp) == 3:
                return float(ipp[2])
            return 0.0

        slice_infos.sort(key=sort_key)

        # 逐个 slice 取像素，同时保证 shape 一致
        valid_imgs = []
        valid_paths = []
        first_shape = None

        for ds, path in slice_infos:
            try:
                arr = ds.pixel_array.astype(np.float32)
                slope = getattr(ds, "RescaleSlope", 1.0)
                intercept = getattr(ds, "RescaleIntercept", 0.0)
                arr = arr * slope + intercept
            except Exception:
                # 这个 slice 本身就读不了，跳过
                print(f"[WARN] 读取 DICOM 像素失败，已跳过: {path}")
                continue

            if first_shape is None:
                first_shape = arr.shape

            if arr.shape != first_shape:
                # 形状不一致的 slice 直接跳过，避免 np.stack 报错
                print(f"[WARN] DICOM series {series_uid} 中存在 shape 不一致的 slice，"
                      f"预期 {first_shape}，实际 {arr.shape}，已跳过: {path}")
                continue

            valid_imgs.append(arr)
            valid_paths.append(path)

        if not valid_imgs:
            # 整个 series 没有任何可用 slice，跳过
            print(f"[WARN] DICOM series {series_uid} 无有效 slice，已跳过。")
            continue

        # 这里就不会再触发 "all input arrays must have the same shape" 了
        volume = np.stack(valid_imgs, axis=0)  # (Z,H,W)

        # 用第一个有效 slice 取 meta
        ds0 = slice_infos[0][0]

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
            "SeriesFolder": os.path.dirname(valid_paths[0]) if valid_paths else None,
        }

        series_dict[series_uid] = {
            "volume": volume,
            "meta": meta,
            "paths": valid_paths,  # 与 volume 的 Z 方向一一对应
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
                data = img.get_fdata()
                if data.ndim == 4:
                    data = data[..., 0]
                # 尝试变成 (Z,H,W)
                if data.ndim == 3:
                    # 假设原顺序为 (X,Y,Z) -> (Z,Y,X)
                    volume = np.transpose(data, (2, 1, 0))
                else:
                    continue
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


def build_volumes_from_folder(root_dir: str):
    """
    从文件夹构建所有 VolumeWrapper 列表
    """
    volumes = []

    dicom_series = load_dicom_series(root_dir)
    for series_uid, v in dicom_series.items():
        volume = v["volume"]
        meta = v["meta"]
        paths = v.get("paths")  # 新增
        name = f"DICOM {series_uid}"
        volumes.append(
            VolumeWrapper(
                name=name,
                volume=volume,
                meta=meta,
                source_type="dicom",
                slice_paths=paths,
                file_path=None,  # DICOM 情况下一般不需要单一 file_path
            )
        )

    nifti_files = load_nifti_files(root_dir)
    for fname, v in nifti_files.items():
        volume = v["volume"]
        meta = v["meta"]
        fpath = meta.get("file_path")
        name = f"NIfTI {fname}"
        volumes.append(
            VolumeWrapper(
                name=name,
                volume=volume,
                meta=meta,
                source_type="nifti",
                slice_paths=None,
                file_path=fpath,
            )
        )


    return volumes

