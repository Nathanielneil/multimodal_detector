# -*- coding: utf-8 -*-
"""
命令历史落盘记录器

检测结果不再实时展示在界面里（表格在小窗口下会截断/挤压导致显示故障），
改为持续追加写入项目根目录下的 JSON 文件，供后续查看/排查使用。
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from detectors.base_detector import DetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "logs" / "command_history.json"
_MAX_RECORDS = 1000


class CommandHistoryLogger:
    """将检测结果追加写入单个 JSON 文件，超过上限后丢弃最旧记录。"""

    def __init__(self, path: Optional[Path] = None, max_records: int = _MAX_RECORDS):
        self._path = Path(path) if path else _DEFAULT_PATH
        self._max_records = max_records
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = data.get("records", [])
        except Exception as exc:
            logger.warning("命令历史文件读取失败，将从空记录开始: %s (%s)", self._path, exc)
            self._records = []

    def add_result(self, result: DetectionResult):
        """追加一条检测结果并立即落盘。"""
        with self._lock:
            self._records.append(result.to_dict())
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]
            self._flush()

    def clear(self):
        """清空历史记录（对应重置统计操作）。"""
        with self._lock:
            self._records = []
            self._flush()

    def _flush(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"total_records": len(self._records), "records": self._records}
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("命令历史写入失败: %s (%s)", self._path, exc)

    @property
    def path(self) -> Path:
        return self._path
