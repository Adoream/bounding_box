from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Callable


@dataclass
class PointAnnotation:
    volume_index: int
    slice_index: int
    x: float
    y: float


@dataclass
class BBoxAnnotation:
    volume_index: int
    slice_index: int
    x1: float
    y1: float
    x2: float
    y2: float

# Type alias, for convenience in passing callbacks in the main program
VolumeNameFunc = Callable[[int], str]
VolumeFileInfoFunc = Callable[[int, int], Tuple[str, str]] 


class AnnotationStore:
    def __init__(self) -> None:
        self.points: List[PointAnnotation] = []
        self.bboxes: List[BBoxAnnotation] = []

    # ---------- 基本操作 ----------

    def add_point(self, volume_index: int, slice_index: int, x: float, y: float) -> None:
        """添加一个点标注"""
        self.points.append(
            PointAnnotation(
                volume_index=int(volume_index),
                slice_index=int(slice_index),
                x=float(x),
                y=float(y),
            )
        )

    def add_bbox(
        self,
        volume_index: int,
        slice_index: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        """添加一个 bbox 标注（不在这里做坐标归一化，保持调用方原样）"""
        self.bboxes.append(
            BBoxAnnotation(
                volume_index=int(volume_index),
                slice_index=int(slice_index),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )

    def clear_slice(self, volume_index: int, slice_index: int) -> None:
        """清除指定 volume + slice 上的所有点和 bbox 标记"""
        v_idx = int(volume_index)
        s_idx = int(slice_index)

        self.points = [
            p
            for p in self.points
            if not (p.volume_index == v_idx and p.slice_index == s_idx)
        ]
        self.bboxes = [
            b
            for b in self.bboxes
            if not (b.volume_index == v_idx and b.slice_index == s_idx)
        ]

    def clear_all(self) -> None:
        """清除所有 volume 的所有点和 bbox 标记"""
        self.points.clear()
        self.bboxes.clear()

    def has_any(self) -> bool:
        """是否存在任意标注"""
        return bool(self.points or self.bboxes)


    def iter_slice(
        self, volume_index: int, slice_index: int
    ) -> Tuple[List[PointAnnotation], List[BBoxAnnotation]]:
        """
        返回某个 volume + slice 上的所有点和 bbox 标注，
        常用于当前层面绘制时的过滤。
        """
        v_idx = int(volume_index)
        s_idx = int(slice_index)

        pts = [
            p
            for p in self.points
            if p.volume_index == v_idx and p.slice_index == s_idx
        ]
        bbs = [
            b
            for b in self.bboxes
            if b.volume_index == v_idx and b.slice_index == s_idx
        ]
        return pts, bbs


    def to_json_serializable(
        self,
        volume_name_func: VolumeNameFunc,
        volume_file_info_func: VolumeFileInfoFunc,
    ) -> Dict[str, Any]:
        data_points: List[Dict[str, Any]] = []
        for p in self.points:
            fp, fn = volume_file_info_func(p.volume_index, p.slice_index)
            data_points.append(
                {
                    "volume_name": volume_name_func(p.volume_index) or "",
                    "slice_index": int(p.slice_index),
                    "x": float(p.x),
                    "y": float(p.y),
                    "file_path": fp,
                    "file_name": fn,
                }
            )

        data_bboxes: List[Dict[str, Any]] = []
        for b in self.bboxes:
            fp, fn = volume_file_info_func(b.volume_index, b.slice_index)
            data_bboxes.append(
                {
                    "volume_name": volume_name_func(b.volume_index) or "",
                    "slice_index": int(b.slice_index),
                    "x1": float(b.x1),
                    "y1": float(b.y1),
                    "x2": float(b.x2),
                    "y2": float(b.y2),
                    "file_path": fp,
                    "file_name": fn,
                }
            )

        return {
            "points": data_points,
            "bboxes": data_bboxes,
        }
