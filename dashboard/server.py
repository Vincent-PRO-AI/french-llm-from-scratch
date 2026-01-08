#!/usr/bin/env python3
"""Flask API that orchestrates data cleaning, training, and inference for the dashboard."""
from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flask import Flask, jsonify, request

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = ROOT / "trained_models" / "runs"
DEFAULT_CHECKPOINT = ROOT / "trained_models" / "tiny_subtitles_transformer.pt"
DATA_ROOT = ROOT / "data"
DATA_CLEAN_ROOT = ROOT / "data_clean"
DATA_HISTORY_DIR = DATA_CLEAN_ROOT / "_history"


def add_root_to_syspath() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))


def to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def find_checkpoint_for_run(run_dir: Path) -> Optional[Path]:
    candidates: List[Path] = [run_dir / "checkpoint.pt"]
    meta = load_json(run_dir / "metadata.json")
    if meta:
        for key in ("run_checkpoint", "final_checkpoint"):
            value = meta.get(key)
            if value:
                candidates.append(Path(value))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_run_entries(root: Path = RUNS_ROOT) -> List[Dict[str, Any]]:
    if not root.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        meta = load_json(item / "metadata.json") or {}
        started_at = meta.get("started_at")
        timestamp: Optional[float] = None
        if started_at:
            try:
                timestamp = datetime.fromisoformat(started_at).timestamp()
            except ValueError:
                timestamp = None
        if timestamp is None:
            timestamp = item.stat().st_mtime
            started_at = datetime.fromtimestamp(timestamp).isoformat()
        entries.append(
            {
                "run_name": meta.get("run_name") or item.name,
                "path": str(item),
                "started_at": started_at,
                "timestamp": timestamp,
                "status": meta.get("status"),
                "has_checkpoint": find_checkpoint_for_run(item) is not None,
            }
        )
    entries.sort(key=lambda entry: entry.get("timestamp", 0.0), reverse=True)
    return entries


def record_data_history(task: str, payload: Dict[str, Any]) -> None:
    try:
        DATA_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    body = dict(payload)
    body.setdefault("task", task)
    completed = body.get("ended_at") or body.get("completed_at")
    if isinstance(completed, (int, float)):
        body["completed_at"] = to_iso(float(completed))
    elif isinstance(completed, str):
        body["completed_at"] = completed
    else:
        body["completed_at"] = to_iso(time.time())
    path = DATA_HISTORY_DIR / f"{task}.json"
    try:
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


def load_data_history() -> Dict[str, Any]:
    history: Dict[str, Any] = {}
    if not DATA_HISTORY_DIR.exists():
        return history
    for entry in DATA_HISTORY_DIR.glob("*.json"):
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        key = entry.stem
        history[key] = payload
    return history


def collect_tokenizer_entries(root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = root or TOKENIZER_DEFAULT_OUTPUT
    if not root.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        meta = load_json(item / "metadata.json") or {}
        trained_at = meta.get("trained_at")
        timestamp: Optional[float] = None
        if trained_at:
            try:
                timestamp = datetime.fromisoformat(trained_at).timestamp()
            except ValueError:
                timestamp = None
        if timestamp is None:
            timestamp = item.stat().st_mtime
            trained_at = datetime.fromtimestamp(timestamp).isoformat()
        entries.append(
            {
                "name": meta.get("name") or item.name,
                "path": str(item),
                "trained_at": trained_at,
                "timestamp": timestamp,
                "vocab_size": meta.get("vocab_size"),
                "tokenizer_path": meta.get("tokenizer_path") or str(item / "tokenizer.json"),
            }
        )
    entries.sort(key=lambda entry: entry.get("timestamp", 0.0), reverse=True)
    return entries


add_root_to_syspath()

from scripts.clean_vtt import SOURCE_DIR, TARGET_DIR, convert_file  # noqa: E402
from scripts.prepare_linkedin import prepare_linkedin  # noqa: E402
from scripts.prepare_instagram import prepare_instagram  # noqa: E402
from scripts.clean_wikipedia import (  # noqa: E402
    DEFAULT_INPUT_DIR as WIKI_DEFAULT_INPUT_DIR,
    clean_wikipedia_corpus,
)
from scripts.download_fineweb import FineWebConfig, download_fineweb_fr  # noqa: E402
from scripts.download_oscar import OscarConfig, download_oscar  # noqa: E402
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from scripts.train_subtitles_transformer import (  # noqa: E402
    Config,
    SubtitleTrainer,
    TinyTransformerLM,
    auto_device,
    config_from_dict,
    MODEL_ARCH_PRESETS,
    summarise_model,
)
from tokenizers import Tokenizer  # noqa: E402

from scripts.train_tokenizer import (  # noqa: E402
    DEFAULT_INPUT_DIR as TOKENIZER_DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR as TOKENIZER_DEFAULT_OUTPUT,
    TokenizerConfig,
    train_tokenizer,
)


class ModelRunner:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.checkpoint_path = checkpoint
        self.device = auto_device(device)
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        cfg_dict = payload.get("config", {})
        self.cfg = config_from_dict(cfg_dict, base=Config())
        self.model = TinyTransformerLM(self.cfg).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self.summary = summarise_model(self.model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        context: str = "",
    ) -> Dict[str, Any]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        prefix = context + prompt
        input_bytes = prefix.encode("utf-8")
        if not input_bytes:
            input_bytes = b"\n"

        tokens = torch.tensor(list(input_bytes), dtype=torch.long, device=self.device).unsqueeze(0)
        new_tokens: List[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                window = tokens[:, -self.cfg.block_size :]
                logits = self.model(window)
                logits = logits[:, -1, :] / temperature
                if top_k > 0 and top_k < logits.size(-1):
                    values, _ = torch.topk(logits, top_k, dim=-1)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)
                new_tokens.append(int(next_token.item()))

        new_text = bytes(new_tokens).decode("utf-8", errors="ignore")
        full_text = (input_bytes + bytes(new_tokens)).decode("utf-8", errors="ignore")
        return {
            "prompt": prompt,
            "context": context,
            "generated_text": new_text,
            "full_text": full_text,
            "tokens_generated": len(new_tokens),
        }


class JobBase:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.status = "idle"
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.message: Optional[str] = None
        self.details: Dict[str, Any] = {}

    def is_running(self) -> bool:
        with self._lock:
            return self.status == "running"

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "started_at": to_iso(self.started_at),
                "ended_at": to_iso(self.ended_at),
                "message": self.message,
                "details": json.loads(json.dumps(self.details)),
            }


class DataPreparationJob(JobBase):
    def __init__(self, task: str) -> None:
        super().__init__()
        self.task = task

    def start(self, **kwargs: Any) -> None:
        with self._lock:
            if self.status == "running":
                raise RuntimeError(f"Data preparation '{self.task}' already running")
            self.status = "running"
            self.started_at = time.time()
            self.ended_at = None
            self.message = None
            self.details = {
                "task": self.task,
                "processed": 0,
                "total": 0,
                "total_files": 0,
            }

        def _run_subtitles() -> None:
            source = Path(kwargs.get("source_dir") or SOURCE_DIR)
            target = Path(kwargs.get("target_dir") or TARGET_DIR)
            status = "completed"
            message: Optional[str] = None
            try:
                vtt_files = sorted(source.glob("*.vtt"))
                total = len(vtt_files)
                with self._lock:
                    self.details.update(
                        {
                            "source_dir": str(source),
                            "target_dir": str(target),
                            "total": total,
                        }
                    )
                if total == 0:
                    return
                for idx, vtt_path in enumerate(vtt_files, 1):
                    output_path = convert_file(vtt_path, target_dir=target)
                    with self._lock:
                        self.details["processed"] = idx
                        self.details["last_file"] = vtt_path.name
                        self.details["last_output"] = str(output_path)
                        self.details["updated_at"] = time.time()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if message:
                        self.message = message
                self._record_history()

        def _run_linkedin() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Dict[str, int] | None = None
            try:
                summary = prepare_linkedin()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        self.details["processed"] = 1
                        self.details["total"] = 1
                    if message:
                        self.message = message
                self._record_history()

        def _run_instagram() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Dict[str, Any] | None = None
            try:
                summary = prepare_instagram()
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        self.details["processed"] = 1
                        self.details["total"] = 1
                    if message:
                        self.message = message
                self._record_history()

        def _run_wikipedia() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Dict[str, Any] | None = None
            try:
                input_dir = Path(kwargs.get("input_dir") or WIKI_DEFAULT_INPUT_DIR)
                output_dir = Path(kwargs.get("output_dir") or input_dir)
                max_files = kwargs.get("max_files")
                try:
                    max_files_int = int(max_files) if max_files is not None else None
                except (TypeError, ValueError):
                    max_files_int = None

                def _progress(update: Dict[str, Any]) -> None:
                    with self._lock:
                        self.details.update(update)
                        self.details["updated_at"] = time.time()

                summary = clean_wikipedia_corpus(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    max_files=max_files_int,
                    progress_cb=_progress,
                )
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        self.details["processed"] = summary.get("processed_files", 0)
                        self.details["total"] = summary.get("total_files", 0)
                    if message:
                        self.message = message
                self._record_history()

        def _run_fineweb() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Dict[str, Any] | None = None

            dataset_path = str(kwargs.get("dataset") or FineWebConfig.dataset_path)
            config_name = str(kwargs.get("config") or FineWebConfig.config_name)
            split = str(kwargs.get("split") or FineWebConfig.split)
            output_dir = Path(kwargs.get("output_dir") or DATA_CLEAN_ROOT / "fineweb_fr")
            max_docs = kwargs.get("max_docs")
            max_bytes = kwargs.get("max_bytes")
            lang_threshold = kwargs.get("lang_threshold")
            min_chars = kwargs.get("min_chars")

            max_docs_int = _as_int(max_docs, FineWebConfig.max_docs)
            max_bytes_int = (
                _as_int(max_bytes, FineWebConfig.max_bytes)
                if max_bytes is not None
                else FineWebConfig.max_bytes
            )
            lang_threshold_float = _as_float(lang_threshold, FineWebConfig.lang_threshold)
            min_chars_int = _as_int(min_chars, FineWebConfig.min_chars)

            with self._lock:
                self.details.update(
                    {
                        "dataset": dataset_path,
                        "config": config_name,
                        "split": split,
                        "output_dir": str(output_dir),
                        "target_docs": max_docs_int,
                        "processed": 0,
                        "total": max_docs_int if max_docs_int else 0,
                    }
                )

            def _progress(update: Dict[str, Any]) -> None:
                with self._lock:
                    self.details.update(update)
                    self.details["updated_at"] = time.time()

            try:
                summary = download_fineweb_fr(
                    FineWebConfig(
                        dataset_path=dataset_path,
                        config_name=config_name,
                        split=split,
                        output_dir=output_dir,
                        max_docs=max_docs_int or 0,
                        max_bytes=max_bytes_int,
                        lang_threshold=lang_threshold_float,
                        min_chars=min_chars_int,
                    ),
                    progress_cb=_progress,
                )
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        processed_docs = summary.get("processed", 0)
                        kept_docs = summary.get("kept_docs", 0)
                        self.details["processed"] = kept_docs
                        self.details["total"] = summary.get("max_docs") or processed_docs
                        self.details["kept_docs"] = kept_docs
                        self.details["bytes_written"] = summary.get("bytes_written", 0)
                        self.details["output_path"] = summary.get("output_path")
                    if message:
                        self.message = message
                self._record_history()

        def _run_oscar() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Dict[str, Any] | None = None

            dataset = str(kwargs.get("dataset") or OscarConfig.dataset)
            language = str(kwargs.get("language") or OscarConfig.language)
            split = str(kwargs.get("split") or OscarConfig.split)
            output_dir = Path(kwargs.get("output_dir") or DATA_CLEAN_ROOT / "oscar_fr")
            max_docs = _as_int(kwargs.get("max_docs"), OscarConfig.max_docs)
            max_bytes = _as_int(kwargs.get("max_bytes")) if kwargs.get("max_bytes") is not None else None
            min_chars = _as_int(kwargs.get("min_chars"), OscarConfig.min_chars)

            with self._lock:
                self.details.update(
                    {
                        "dataset": dataset,
                        "language": language,
                        "split": split,
                        "output_dir": str(output_dir),
                        "processed": 0,
                        "total": max_docs or 0,
                    }
                )

            def _progress(update: Dict[str, Any]) -> None:
                with self._lock:
                    self.details.update(update)
                    self.details["updated_at"] = time.time()

            try:
                summary = download_oscar(
                    OscarConfig(
                        dataset=dataset,
                        language=language,
                        split=split,
                        output_dir=output_dir,
                        max_docs=max_docs,
                        max_bytes=max_bytes,
                        min_chars=min_chars or OscarConfig.min_chars,
                        progress_interval=max(1, _as_int(kwargs.get("progress_interval"), 1000)),
                    ),
                    progress_cb=_progress,
                )
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        self.details["processed"] = summary.get("kept_docs", 0)
                        self.details["total"] = summary.get("max_docs") or summary.get("processed", 0)
                    if message:
                        self.message = message
                self._record_history()

        task_runner = {
            "subtitles": _run_subtitles,
            "linkedin": _run_linkedin,
            "instagram": _run_instagram,
            "wikipedia": _run_wikipedia,
            "fineweb": _run_fineweb,
            "oscar": _run_oscar,
        }.get(self.task)

        if task_runner is None:
            with self._lock:
                self.status = "failed"
                self.message = f"Unknown data preparation task: {self.task}"
            raise RuntimeError(self.message)

        self._thread = threading.Thread(target=task_runner, daemon=True)
        self._thread.start()

    def _record_history(self) -> None:
        snapshot = self.snapshot()
        if snapshot.get("status") == "running":
            return
        record_data_history(
            self.task,
            {
                "task": self.task,
                "status": snapshot.get("status"),
                "started_at": snapshot.get("started_at"),
                "ended_at": snapshot.get("ended_at"),
                "details": snapshot.get("details"),
            },
        )


class DataPreparationManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, DataPreparationJob] = {}
        self._lock = threading.Lock()

    def start(self, task: str, **kwargs: Any) -> DataPreparationJob:
        with self._lock:
            job = self._jobs.get(task)
            if job is None or not job.is_running():
                job = DataPreparationJob(task)
                self._jobs[task] = job
            else:
                raise RuntimeError(f"Data preparation '{task}' déjà en cours")
        job.start(**kwargs)
        return job

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {task: job.snapshot() for task, job in self._jobs.items()}


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def compute_data_catalog() -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    history = load_data_history()

    youtube_dir = DATA_CLEAN_ROOT / "youtube"
    if youtube_dir.exists():
        files = list(youtube_dir.glob("*.txt"))
        size = sum(file.stat().st_size for file in files)
        catalog["subtitles"] = {
            "files": len(files),
            "size_bytes": size,
        }
        entry = history.get("subtitles")
        if entry:
            catalog["subtitles"]["last_cleaned_at"] = entry.get("completed_at")

    linkedin_dir = DATA_CLEAN_ROOT / "linkedin"
    if linkedin_dir.exists():
        counts: Dict[str, int] = {}
        size = 0
        for name in ("comments.txt", "shares.txt", "messages.txt"):
            path = linkedin_dir / name
            if not path.exists():
                continue
            size += path.stat().st_size
            counts[name.split(".")[0]] = _count_lines(path)
        catalog["linkedin"] = {
            "size_bytes": size,
            "counts": counts,
        }
        entry = history.get("linkedin")
        if entry:
            catalog["linkedin"]["last_cleaned_at"] = entry.get("completed_at")

    instagram_dir = DATA_CLEAN_ROOT / "instagram"
    if instagram_dir.exists():
        counts: Dict[str, int] = {}
        size = 0
        for name in ("messages.txt", "comments.txt"):
            path = instagram_dir / name
            if not path.exists():
                continue
            size += path.stat().st_size
            counts[name.split(".")[0]] = _count_lines(path)
        catalog["instagram"] = {
            "size_bytes": size,
            "counts": counts,
        }
        entry = history.get("instagram")
        if entry:
            catalog["instagram"]["last_cleaned_at"] = entry.get("completed_at")

    wikipedia_dir = DATA_CLEAN_ROOT / "wikipedia"
    if wikipedia_dir.exists():
        files = list(wikipedia_dir.glob("*.txt"))
        size = sum(file.stat().st_size for file in files)
        catalog["wikipedia"] = {
            "size_bytes": size,
            "counts": {"articles": len(files)},
        }
        entry = history.get("wikipedia")
        if entry:
            catalog["wikipedia"]["last_cleaned_at"] = entry.get("completed_at")

    fineweb_dir = DATA_CLEAN_ROOT / "fineweb_fr"
    if fineweb_dir.exists():
        text_files = list(fineweb_dir.glob("fineweb_fr_*.txt"))
        meta_files = list(fineweb_dir.glob("fineweb_fr_*.meta.json"))
        size = sum(path.stat().st_size for path in text_files)
        total_docs = 0
        entries: list[dict[str, Any]] = []
        for meta_path in meta_files:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            total_docs += int(meta.get("kept_docs", 0) or 0)
            entries.append(
                {
                    "config": meta.get("config_name"),
                    "kept_docs": meta.get("kept_docs"),
                    "bytes": meta.get("bytes_written"),
                }
            )
        catalog["fineweb"] = {
            "size_bytes": size,
            "counts": {"documents": total_docs},
            "entries": entries,
        }
        entry = history.get("fineweb")
        if entry:
            catalog["fineweb"]["last_cleaned_at"] = entry.get("completed_at")

    oscar_dir = DATA_CLEAN_ROOT / "oscar_fr"
    if oscar_dir.exists():
        text_files = list(oscar_dir.glob("oscar_*.txt"))
        meta_files = list(oscar_dir.glob("oscar_*.meta.json"))
        size = sum(path.stat().st_size for path in text_files)
        total_docs = 0
        entries: list[dict[str, Any]] = []
        for meta_path in meta_files:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            kept = int(meta.get("kept_docs", 0) or 0)
            total_docs += kept
            entries.append(
                {
                    "dataset": meta.get("dataset"),
                    "language": meta.get("language"),
                    "kept_docs": kept,
                    "bytes": meta.get("bytes_written"),
                }
            )
        catalog["oscar"] = {
            "size_bytes": size,
            "counts": {"documents": total_docs},
            "entries": entries,
        }
        entry = history.get("oscar")
        if entry:
            catalog["oscar"]["last_cleaned_at"] = entry.get("completed_at")

    return catalog


class TrainingJob(JobBase):
    def __init__(self, progress_handler: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        super().__init__()
        self._progress_handler = progress_handler
        self._stop_event = threading.Event()

    def start(self, config_payload: Dict[str, Any]) -> None:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("Training job already running")
            self.status = "running"
            self.started_at = time.time()
            self.ended_at = None
            self.message = None
            self.details = {
                "config": config_payload,
                "step": 0,
            }
            if config_payload.get("resume_checkpoint"):
                self.details["resume_checkpoint"] = config_payload.get("resume_checkpoint")
                self.details["resume_run_dir"] = config_payload.get("resume_run_dir")
            if config_payload.get("max_steps") is not None:
                self.details["max_steps"] = config_payload.get("max_steps")
            self._stop_event.clear()

        def _progress(update: Dict[str, Any]) -> None:
            update = dict(update)
            timestamp = time.time()
            event = update.pop("event", None)
            with self._lock:
                if event is not None:
                    self.details["last_event"] = event
                self.details.update(update)
                self.details["updated_at"] = timestamp
            if self._progress_handler is not None:
                try:
                    payload = dict(update)
                    if event is not None:
                        payload["event"] = event
                    payload["timestamp"] = timestamp
                    self._progress_handler(payload)
                except Exception:
                    pass

        def _run() -> None:
            status = "completed"
            message: Optional[str] = None
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cfg = config_from_dict(config_payload, base=Config())
                trainer = SubtitleTrainer(
                    cfg,
                    progress_callback=_progress,
                    stop_event=self._stop_event,
                )
                trainer.run()
                status = trainer.status
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if message:
                        self.message = message
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        with self._lock:
            if self.status != "running":
                return False
            self._stop_event.set()
            self.message = "Arrêt demandé par l'utilisateur"
            return True


class TokenizerJob(JobBase):
    def start(self, config_payload: Dict[str, Any]) -> None:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("Tokenizer job already running")
            self.status = "running"
            self.started_at = time.time()
            self.ended_at = None
            self.message = None
            self.details = {
                "config": dict(config_payload),
                "processed": 0,
                "total_files": 0,
            }

        def _progress(update: Dict[str, Any]) -> None:
            with self._lock:
                self.details.update(update)
                self.details["updated_at"] = time.time()

        def _run() -> None:
            status = "completed"
            message: Optional[str] = None
            summary: Optional[Dict[str, Any]] = None
            try:
                cfg = _build_tokenizer_config(config_payload)
                summary = train_tokenizer(cfg, progress_cb=_progress)
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
            finally:
                with self._lock:
                    self.status = status
                    self.ended_at = time.time()
                    if summary:
                        self.details.update(summary)
                        files_used = summary.get("files_used")
                        if isinstance(files_used, int):
                            self.details["processed"] = files_used
                            self.details["processed_files"] = files_used
                            self.details.setdefault("total_files", files_used)
                    if message:
                        self.message = message

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _build_tokenizer_config(payload: Dict[str, Any]) -> TokenizerConfig:
    name = str(payload.get("name", "wiki-bpe")).strip() or "tokenizer"
    input_dir = Path(payload.get("input_dir") or TOKENIZER_DEFAULT_INPUT)
    output_dir = Path(payload.get("output_dir") or TOKENIZER_DEFAULT_OUTPUT)
    vocab_size = _as_int(payload.get("vocab_size"), 32_000) or 32_000
    min_frequency = _as_int(payload.get("min_frequency"), 2) or 2
    limit_files = _as_int(payload.get("limit_files"))
    limit_mb = _as_float(payload.get("limit_mb"))
    limit_bytes = None if limit_mb is None else int(limit_mb * 1024 * 1024)
    lowercase = _as_bool(payload.get("lowercase"), False)

    return TokenizerConfig(
        name=name,
        input_dir=input_dir,
        output_dir=output_dir,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        lowercase=lowercase,
        limit_files=limit_files,
        limit_bytes=limit_bytes,
    )


def _resolve_tokenizer_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = TOKENIZER_DEFAULT_OUTPUT / candidate
    if candidate.is_dir():
        candidate = candidate / "tokenizer.json"
    if candidate.exists():
        return candidate
    return None


def _normalise_device(value: Optional[str]) -> str:
    if value is None:
        target = "auto"
    else:
        target = str(value).strip().lower()
    if target in {"", "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if target.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(value)


@dataclass
class ServerConfig:
    host: str
    port: int
    run_dir: Optional[Path]
    checkpoint: Optional[Path]
    device: str
    max_new_tokens: int
    temperature: float
    top_k: int


def create_app(config: ServerConfig) -> Flask:
    app = Flask(__name__)
    default_device = _normalise_device(config.device)
    state_lock = threading.Lock()
    app_state: Dict[str, Any] = {
        "selected_run_dir": config.run_dir,
        "runner": None,
        "runner_checkpoint": str(config.checkpoint) if config.checkpoint else None,
        "defaults": {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "device": default_device,
        },
        "latest_run_dir": None,
        "latest_run_name": None,
    }

    @app.get("/")
    def root() -> Any:
        # Provide a friendly landing response instead of 404 when hitting the API root
        return jsonify(
            {
                "message": "MiniGPT Dashboard API",
                "hint": "Use the frontend at http://127.0.0.1:5173 or query the endpoints below",
                "endpoints": [
                    "/metadata",
                    "/metrics",
                    "/visuals",
                    "/runs",
                    "/select",
                    "/chat",
                    "/data/prepare",
                    "/data/status",
                    "/train",
                    "/train/status",
                    "/train/resume",
                    "/train/stop",
                    "/tokenizer/train",
                    "/tokenizer/status",
                    "/tokenizer/test",
                ],
            }
        )

    def progress_handler(update: Dict[str, Any]) -> None:
        run_dir = update.get("run_dir")
        if not run_dir:
            return
        event = update.get("event")
        target_dir = Path(run_dir)
        with state_lock:
            app_state["latest_run_dir"] = target_dir
            app_state["latest_run_name"] = target_dir.name
            if event in {"run_created", "run_resumed"} or app_state.get("selected_run_dir") is None:
                app_state["selected_run_dir"] = target_dir
            default_device_snapshot = app_state["defaults"].get("device", config.device)
        if event == "finished":
            checkpoint = find_checkpoint_for_run(target_dir)
            if checkpoint is not None:
                try:
                    load_runner(checkpoint, default_device_snapshot)
                except Exception:
                    pass

    data_manager = DataPreparationManager()
    training_job = TrainingJob(progress_handler=progress_handler)
    tokenizer_job = TokenizerJob()

    def load_runner(checkpoint_path: Path, device: str) -> ModelRunner:
        runner = ModelRunner(checkpoint_path, _normalise_device(device))
        with state_lock:
            app_state["runner"] = runner
            app_state["runner_checkpoint"] = str(checkpoint_path)
            app_state["defaults"]["device"] = str(runner.device)
        return runner

    if config.checkpoint and config.checkpoint.exists():
        try:
            load_runner(config.checkpoint, config.device)
        except Exception:
            pass

    def current_runner() -> Optional[ModelRunner]:
        with state_lock:
            return app_state.get("runner")

    def selected_run_dir() -> Optional[Path]:
        with state_lock:
            run_dir = app_state.get("selected_run_dir")
        if run_dir is None:
            return None
        return Path(run_dir)

    def current_defaults() -> Dict[str, Any]:
        with state_lock:
            return dict(app_state["defaults"])

    @app.get("/metadata")
    def get_metadata() -> Any:
        run_dir = selected_run_dir()
        run_meta = load_json(run_dir / "metadata.json") if run_dir else None
        checkpoint_path = None
        if run_dir is not None:
            checkpoint = find_checkpoint_for_run(run_dir)
            checkpoint_path = str(checkpoint) if checkpoint is not None else None

        runner = current_runner()
        model_meta = None
        if runner is not None:
            summary_cfg = runner.summary.get("config", {}) if isinstance(runner.summary, dict) else {}
            model_meta = {
                "checkpoint": str(runner.checkpoint_path),
                "device": str(runner.device),
                "block_size": runner.cfg.block_size,
                "embed_dim": runner.cfg.embed_dim,
                "num_layers": runner.cfg.num_layers,
                "num_heads": runner.cfg.num_heads,
                "layernorm_dim": summary_cfg.get("layernorm_dim"),
                "head_dim": summary_cfg.get("head_dim"),
                "summary": runner.summary,
            }

        runs_list = collect_run_entries()
        summary_payload = None
        if runner is not None:
            summary_payload = runner.summary
        elif run_meta:
            summary_payload = run_meta.get("model_summary")

        with state_lock:
            defaults_snapshot = dict(app_state["defaults"])
            latest_run_dir = app_state.get("latest_run_dir")
            latest_run_name = app_state.get("latest_run_name")

        payload = {
            "run": run_meta,
            "model": model_meta,
            "defaults": defaults_snapshot,
            "runs": runs_list,
            "current_run": run_dir.name if run_dir else None,
            "current_run_path": str(run_dir) if run_dir else None,
            "checkpoint_path": checkpoint_path,
            "checkpoint_loaded": runner is not None,
            "model_summary": summary_payload,
            "jobs": {
                "data": data_manager.snapshot(),
                "training": training_job.snapshot(),
                "tokenizer": tokenizer_job.snapshot(),
            },
            "latest_run_path": str(latest_run_dir) if latest_run_dir else None,
            "latest_run_name": latest_run_name,
            "data_catalog": compute_data_catalog(),
            "data_history": load_data_history(),
            "training_presets": MODEL_ARCH_PRESETS,
            "tokenizers": collect_tokenizer_entries(),
            "tokenizer_defaults": {
                "input_dir": str(TOKENIZER_DEFAULT_INPUT),
                "output_dir": str(TOKENIZER_DEFAULT_OUTPUT),
                "name": "wiki-bpe",
                "vocab_size": 32_000,
                "min_frequency": 2,
                "lowercase": False,
            },
        }
        return jsonify(payload)

    @app.get("/metrics")
    def get_metrics() -> Any:
        run_dir = selected_run_dir()
        if run_dir is None:
            return jsonify({"train": [], "val": []})
        metrics_path = run_dir / "metrics.jsonl"
        train: List[Dict[str, Any]] = []
        val: List[Dict[str, Any]] = []
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("split") == "train":
                        train.append(record)
                    elif record.get("split") == "val":
                        val.append(record)
        return jsonify({"train": train, "val": val})

    @app.get("/visuals")
    def get_visuals() -> Any:
        run_dir = selected_run_dir()
        if run_dir is None:
            return jsonify({"records": []})
        visuals_path = run_dir / "visuals.jsonl"
        records: List[Dict[str, Any]] = []
        if visuals_path.exists():
            with visuals_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records.append(payload)
        return jsonify({"records": records[-60:]})

    @app.get("/runs")
    def list_runs() -> Any:
        return jsonify({"runs": collect_run_entries()})

    @app.post("/select")
    def select_run() -> Any:
        data = request.get_json(silent=True) or {}
        requested = data.get("run")
        if not requested:
            return jsonify({"error": "Field 'run' is required"}), 400
        run_path = Path(requested)
        if not run_path.is_absolute():
            run_path = RUNS_ROOT / requested
        if not run_path.exists() or not run_path.is_dir():
            return jsonify({"error": f"Run directory not found: {requested}"}), 404

        checkpoint = find_checkpoint_for_run(run_path)
        runner = None
        device = data.get("device", app_state["defaults"].get("device", config.device))
        message = None
        if checkpoint is not None:
            try:
                runner = load_runner(checkpoint, device)
            except Exception as exc:  # noqa: BLE001
                message = f"Impossible de charger le checkpoint: {exc}"
        else:
            message = "Aucun checkpoint disponible pour ce run."
        with state_lock:
            app_state["selected_run_dir"] = run_path
        meta = load_json(run_path / "metadata.json") or {}
        summary = runner.summary if runner is not None else meta.get("model_summary")
        payload = {
            "selected": run_path.name,
            "path": str(run_path),
            "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
            "checkpoint_loaded": runner is not None,
            "message": message,
            "model_summary": summary,
            "run_config": meta.get("config"),
        }
        return jsonify(payload)

    @app.post("/chat")
    def chat() -> Any:
        runner = current_runner()
        if runner is None:
            return jsonify({"error": "Model not loaded"}), 503
        data = request.get_json(silent=True) or {}
        raw_prompt = str(data.get("prompt", ""))
        context = str(data.get("context", ""))
        defaults = current_defaults()
        max_new_tokens = int(data.get("max_new_tokens", defaults.get("max_new_tokens", 120)))
        temperature = float(data.get("temperature", defaults.get("temperature", 0.8)))
        top_k = int(data.get("top_k", defaults.get("top_k", 50)))

        user_message = raw_prompt.strip()
        if not user_message:
            return jsonify({"error": "Le message utilisateur est vide."}), 400

        normalized_context = context.rstrip()
        if normalized_context:
            normalized_context += "\n"

        formatted_prompt = f"Utilisateur: {user_message}\nAssistant: "

        try:
            result = runner.generate(
                formatted_prompt,
                max_new_tokens,
                temperature,
                top_k,
                context=normalized_context,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500
        return jsonify(result)

    @app.post("/data/prepare")
    def trigger_data_prep() -> Any:
        data = request.get_json(silent=True) or {}
        task = str(data.get("task", "subtitles")).strip().lower()
        params = {key: value for key, value in data.items() if key != "task"}
        try:
            job = data_manager.start(task, **params)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"status": "started", "job": job.snapshot()})

    @app.get("/data/status")
    def cleaning_status() -> Any:
        return jsonify(data_manager.snapshot())

    @app.post("/train")
    def trigger_training() -> Any:
        data = request.get_json(silent=True) or {}
        config_payload = data.get("config")
        if config_payload is None:
            config_payload = dict(data)
        config_payload = dict(config_payload)
        fallback_device = current_defaults().get("device", default_device)
        config_payload.setdefault("device", fallback_device)
        config_payload["device"] = _normalise_device(config_payload.get("device"))
        # Normaliser memmap_path en absolu si fourni
        if "memmap_path" in config_payload and config_payload["memmap_path"]:
            memmap_path = Path(config_payload["memmap_path"])
            if not memmap_path.is_absolute():
                memmap_path = ROOT / memmap_path
            config_payload["memmap_path"] = str(memmap_path)
        # Normaliser pretokenized_path en absolu si fourni
        if "pretokenized_path" in config_payload and config_payload["pretokenized_path"]:
            pretokenized_path = Path(config_payload["pretokenized_path"])
            if not pretokenized_path.is_absolute():
                pretokenized_path = ROOT / pretokenized_path
            config_payload["pretokenized_path"] = str(pretokenized_path)
        # Normaliser tokenizer_path en absolu si fourni
        if "tokenizer_path" in config_payload and config_payload["tokenizer_path"]:
            tokenizer_path = Path(config_payload["tokenizer_path"])
            if not tokenizer_path.is_absolute():
                tokenizer_path = ROOT / tokenizer_path
            config_payload["tokenizer_path"] = str(tokenizer_path)
        try:
            training_job.start(config_payload)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"status": "started", "job": training_job.snapshot()})

    @app.get("/train/status")
    def training_status() -> Any:
        return jsonify(training_job.snapshot())

    @app.post("/train/resume")
    def resume_training() -> Any:
        data = request.get_json(silent=True) or {}
        requested_run = data.get("run")
        if not requested_run:
            return jsonify({"error": "Champ 'run' requis"}), 400
        run_dir = Path(requested_run)
        if not run_dir.is_absolute():
            run_dir = RUNS_ROOT / requested_run
        if not run_dir.exists() or not run_dir.is_dir():
            return jsonify({"error": f"Run introuvable: {requested_run}"}), 404

        checkpoint_raw = data.get("checkpoint")
        if checkpoint_raw:
            checkpoint_path = Path(checkpoint_raw)
            if not checkpoint_path.is_absolute():
                checkpoint_path = run_dir / checkpoint_raw
        else:
            checkpoint_path = find_checkpoint_for_run(run_dir)
        if checkpoint_path is None or not checkpoint_path.exists():
            return jsonify({"error": "Aucun checkpoint disponible pour ce run."}), 404

        metadata = load_json(run_dir / "metadata.json") or {}
        config_payload = metadata.get("config")
        if not isinstance(config_payload, dict):
            return jsonify({"error": "Configuration introuvable pour ce run."}), 400
        config_payload = dict(config_payload)

        max_steps_raw = data.get("max_steps")
        if max_steps_raw is not None:
            try:
                config_payload["max_steps"] = int(max_steps_raw)
            except (TypeError, ValueError):
                return jsonify({"error": "max_steps doit être un entier"}), 400

        # Attempt safe load first (weights_only=True introduced in PyTorch 2.6).
        # If it fails due to restricted globals, fall back to unsafe load only for a local trusted file.
        checkpoint_data = None
        try:
            # First, allowlist pathlib.PosixPath for safe unpickling if needed
            try:
                import pathlib as _pl  # type: ignore
                if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals([_pl.PosixPath])  # type: ignore[attr-defined]
            except Exception:
                pass
            # Try safe load (weights_only=True) when supported
            try:
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
            except TypeError:
                # Older torch: no weights_only kw, fall back to default
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            except Exception:
                # If safe load fails, try unsafe load for trusted local file
                try:
                    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # type: ignore[call-arg]
                except TypeError:
                    checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Lecture du checkpoint impossible: {exc}"}), 500
        resume_step = int(checkpoint_data.get("step", 0)) if isinstance(checkpoint_data, dict) else 0

        target_steps = config_payload.get("max_steps")
        if isinstance(target_steps, int) and target_steps <= resume_step:
            return jsonify({
                "error": f"max_steps ({target_steps}) doit être supérieur à l'étape actuelle ({resume_step})."
            }), 400

        config_payload["resume_checkpoint"] = str(checkpoint_path)
        config_payload["resume_run_dir"] = str(run_dir)
        config_payload["run_name"] = metadata.get("run_name") or run_dir.name
        config_payload["log_dir"] = str(run_dir.parent)

        extras = config_payload.get("extra_data_dirs")
        if extras is None:
            config_payload["extra_data_dirs"] = []
        elif isinstance(extras, list):
            config_payload["extra_data_dirs"] = extras
        else:
            config_payload["extra_data_dirs"] = [str(extras)]

        if not config_payload.get("device"):
            config_payload["device"] = current_defaults().get("device", DEFAULT_DEVICE)

        # Ensure sample prompt defaults exist when missing (Backward compat)
        config_payload.setdefault("sample_prompt", "Bonjour je suis ")

        try:
            training_job.start(config_payload)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        snapshot = training_job.snapshot()
        snapshot["resume_step"] = resume_step
        return jsonify({"status": "started", "job": snapshot})

    @app.post("/train/stop")
    def stop_training() -> Any:
        if training_job.stop():
            return jsonify({"status": "stopping"})
        return jsonify({"status": "idle"}), 409

    @app.post("/tokenizer/train")
    def tokenizer_train() -> Any:
        config_payload = request.get_json(silent=True) or {}
        try:
            tokenizer_job.start(config_payload)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"status": "started", "job": tokenizer_job.snapshot()})

    @app.get("/tokenizer/status")
    def tokenizer_status() -> Any:
        return jsonify(tokenizer_job.snapshot())

    @app.post("/tokenizer/test")
    def tokenizer_test() -> Any:
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        if text is None or not str(text).strip():
            return jsonify({"error": "Champ 'text' requis"}), 400

        tokenizer_path = data.get("tokenizer_path")
        resolved = _resolve_tokenizer_path(tokenizer_path)
        if resolved is None:
            entries = collect_tokenizer_entries()
            if not entries:
                return jsonify({"error": "Aucun tokenizer disponible"}), 404
            resolved = _resolve_tokenizer_path(entries[0].get("tokenizer_path") or entries[0].get("path"))
        if resolved is None or not resolved.exists():
            return jsonify({"error": "Tokenizer introuvable"}), 404

        try:
            tokenizer = Tokenizer.from_file(str(resolved))
            sample_text = str(text)
            encoding = tokenizer.encode(sample_text)

            decoded_tokens: list[str] = []
            offsets = getattr(encoding, "offsets", None)
            if offsets:
                try:
                    decoded_tokens = [sample_text[start:end] for (start, end) in offsets]
                except Exception:
                    decoded_tokens = []
            if not decoded_tokens:
                decoder = getattr(tokenizer, "decoder", None)
                if decoder is not None:
                    try:
                        decoded_tokens = [decoder.decode([token]) for token in encoding.tokens]
                    except Exception:
                        decoded_tokens = []
            if not decoded_tokens:
                # Fallback: replace ByteLevel markers for readability.
                decoded_tokens = [token.replace("Ġ", " ").replace("Ċ", "\n") for token in encoding.tokens]
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Échec encodage tokenizer: {exc}"}), 500

        return jsonify(
            {
                "tokenizer_path": str(resolved),
                "tokens": encoding.tokens,
                "ids": encoding.ids,
                "attention_mask": encoding.attention_mask,
                "length": len(encoding.ids),
                "tokens_pretty": decoded_tokens,
            }
        )

    # Optionally skip startup probes (useful when visuals/metrics files are large)
    import os
    if not os.getenv("SKIP_STARTUP_PROBES"):
        perform_startup_checks(app)
    return app


def perform_startup_checks(app: Flask) -> None:
    probes = [
        ("GET", "/metadata"),
        ("GET", "/metrics"),
        ("GET", "/runs"),
        ("GET", "/visuals"),
        ("GET", "/data/status"),
        ("GET", "/train/status"),
        ("GET", "/tokenizer/status"),
    ]
    with app.test_client() as client:
        for method, endpoint in probes:
            try:
                response = client.open(endpoint, method=method)
            except Exception as exc:  # noqa: BLE001
                app.logger.error("Startup probe failed for %s %s: %s", method, endpoint, exc)
                continue
            if response.status_code >= 500:
                app.logger.error(
                    "Startup probe %s %s returned status %s", method, endpoint, response.status_code
                )
            else:
                app.logger.debug(
                    "Startup probe %s %s -> %s", method, endpoint, response.status_code
                )


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Subtitle transformer dashboard API")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing metrics.jsonl")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint to load for chat")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose the dashboard")
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device used for chat inference (default: auto-detect)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120, help="Default max tokens for chat completions")
    parser.add_argument("--temperature", type=float, default=0.8, help="Default sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Default top-k sampling value")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        latest = collect_run_entries()
        run_dir = Path(latest[0]["path"]) if latest else None
    checkpoint = args.checkpoint
    if checkpoint is None and run_dir is not None:
        candidate = find_checkpoint_for_run(run_dir)
        if candidate is not None:
            checkpoint = candidate
    if checkpoint is None and DEFAULT_CHECKPOINT.exists():
        checkpoint = DEFAULT_CHECKPOINT

    return ServerConfig(
        host=args.host,
        port=args.port,
        run_dir=run_dir,
        checkpoint=checkpoint,
        device=_normalise_device(args.device),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )


def main() -> None:
    config = parse_args()
    app = create_app(config)
    app.run(host=config.host, port=config.port, threaded=True)


if __name__ == "__main__":
    main()
