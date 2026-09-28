"""ONNX 完整性分类器的惰性适配层。"""

from __future__ import annotations

import asyncio
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .state import SemanticState


MODEL_REPOSITORIES = {
    "small": "advent259141/astrbot_debouncer_small",
    "normal": "advent259141/astrbot_debouncer_normal",
}


class ClassifierUnavailable(RuntimeError):
    """模型或依赖不可用。"""


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    probability: float
    complete: bool
    semantic_state: SemanticState | None = None


def send_probability(logits: Any) -> float:
    """计算 Label 1（SEND）的 softmax 概率，不导入 numpy。"""

    row = logits[0] if hasattr(logits, "__getitem__") else logits
    values = [float(value) for value in row]
    if len(values) < 2:
        raise ValueError("完整性模型至少需要两个 logits")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    if total <= 0:
        raise ValueError("完整性模型 logits 无法归一化")
    return exponentials[1] / total


class SentenceClassifier:
    """同步 ONNX 推理对象，只在首次使用时构造。"""

    def __init__(self, model_path: Path, tokenizer_path: Path, package_dir: Path) -> None:
        try:
            from nekro_agent.api.plugin import dynamic_import_pkg

            self._numpy = dynamic_import_pkg("numpy>=1.21.0", "numpy", repo_dir=package_dir)
            ort = dynamic_import_pkg("onnxruntime>=1.15.0", "onnxruntime", repo_dir=package_dir)
            transformers = dynamic_import_pkg("transformers>=4.30.0", "transformers", repo_dir=package_dir)
        except Exception as exc:
            raise ClassifierUnavailable(f"加载 ONNX 依赖失败: {exc}") from exc

        if not model_path.is_file() or not tokenizer_path.is_dir():
            raise ClassifierUnavailable(f"模型文件或 tokenizer 不存在: {model_path}")
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
            )
            self._session = ort.InferenceSession(str(model_path))
        except Exception as exc:
            raise ClassifierUnavailable(f"加载完整性模型失败: {exc}") from exc

    def _predict_sync(self, text: str) -> float:
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=64,
        )
        outputs = self._session.run(
            output_names=["logits"],
            input_feed={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
        return send_probability(outputs[0])

    async def predict(self, text: str) -> float:
        return await asyncio.to_thread(self._predict_sync, text)


class ClassifierAdapter:
    """管理模型目录、惰性依赖导入和线程池推理。"""

    def __init__(self, model_type: str, data_dir: Path, logger: Any = None, debug_logging: bool = False) -> None:
        self.model_type = model_type if model_type in MODEL_REPOSITORIES else "small"
        self.data_dir = data_dir
        self.package_dir = data_dir / "packages"
        self.logger = logger
        self.debug_logging = debug_logging
        self._classifier: Optional[SentenceClassifier] = None
        self._load_lock = asyncio.Lock()

    @property
    def model_dir(self) -> Path:
        return self.data_dir / "models" / self.model_type

    async def _ensure_loaded(self) -> SentenceClassifier:
        if self._classifier is not None:
            return self._classifier
        async with self._load_lock:
            if self._classifier is None:
                self._classifier = await asyncio.to_thread(self._load_sync)
        return self._classifier

    def _load_sync(self) -> SentenceClassifier:
        model_dir = self.model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer"
        if not model_path.exists() or not tokenizer_path.exists():
            self._download_model_sync(model_dir)
        return SentenceClassifier(model_path, tokenizer_path, self.package_dir)

    def _download_model_sync(self, target_dir: Path) -> None:
        """首次使用时才允许的可选模型下载。"""

        try:
            from nekro_agent.api.plugin import dynamic_import_pkg

            snapshot_module = dynamic_import_pkg(
                "modelscope>=1.9.0",
                "modelscope.hub.snapshot_download",
                repo_dir=self.package_dir,
            )
            cache_dir = self.data_dir / ".cache"
            cache_path = snapshot_module.snapshot_download(
                MODEL_REPOSITORIES[self.model_type],
                cache_dir=str(cache_dir),
            )
            source_model = Path(cache_path) / "model" / "model.onnx"
            source_tokenizer = Path(cache_path) / "tokenizer"
            if not source_model.is_file() or not source_tokenizer.is_dir():
                raise FileNotFoundError("ModelScope 模型缺少 model.onnx 或 tokenizer")
            shutil.copy2(source_model, target_dir / "model.onnx")
            shutil.copytree(source_tokenizer, target_dir / "tokenizer", dirs_exist_ok=True)
        except Exception as exc:
            raise ClassifierUnavailable(f"模型不存在且下载失败: {exc}") from exc

    async def is_complete(self, text: str, threshold: float) -> bool:
        return (await self.classify(text, threshold, threshold)).complete

    async def classify(self, text: str, threshold: float, high_threshold: float | None = None) -> ClassificationResult:
        classifier = await self._ensure_loaded()
        score = await classifier.predict(text)
        high_threshold = threshold if high_threshold is None else high_threshold
        if score < threshold:
            state = SemanticState.INCOMPLETE
        elif score >= high_threshold:
            state = SemanticState.COMPLETE_HIGH
        else:
            state = SemanticState.COMPLETE_NORMAL
        return ClassificationResult(
            probability=score,
            complete=score >= threshold,
            semantic_state=state,
        )
