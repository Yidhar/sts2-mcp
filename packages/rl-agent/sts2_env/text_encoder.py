"""Text encoding module for STS2 RL agent.

Default model: BAAI/bge-small-zh-v1.5 (Chinese-specific, 512-d output).
Embeddings are frozen, normalized, and cached to a persistent file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from sts2_rl.artifacts import resolve_artifact_path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# bge-small-zh-v1.5 output dimension
TEXT_DIM: int = 512

_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

def _resolve_default_cache_dir() -> str:
    """Return the canonical persistent embedding-cache directory."""

    return str(resolve_artifact_path(None, default="cache/text_embeddings"))


def _candidate_hf_cache_roots() -> Iterable[Path]:
    """Yield likely Hugging Face cache roots without requiring any network access."""
    seen: set[Path] = set()

    for env_name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        raw = os.environ.get(env_name)
        if not raw:
            continue
        root = Path(raw)
        if env_name == "HF_HOME":
            root = root / "hub"
        if root not in seen:
            seen.add(root)
            yield root

    try:
        from huggingface_hub import constants

        root = Path(constants.HF_HUB_CACHE)
        if root not in seen:
            seen.add(root)
            yield root
    except Exception:
        pass

    home = Path.home()
    for root in (
        home / ".cache" / "huggingface" / "hub",
        home / "AppData" / "Local" / "huggingface" / "hub",
    ):
        if root not in seen:
            seen.add(root)
            yield root

    if os.environ.get("WSL_DISTRO_NAME"):
        windows_roots = []
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            normalized = userprofile.replace("\\", "/")
            if len(normalized) >= 2 and normalized[1] == ":":
                drive = normalized[0].lower()
                remainder = normalized[2:].lstrip("/")
                windows_roots.append(Path("/mnt") / drive / remainder / ".cache" / "huggingface" / "hub")

        windows_users = Path("/mnt/c/Users")
        if windows_users.exists():
            windows_roots.append(windows_users / home.name / ".cache" / "huggingface" / "hub")

        for root in windows_roots:
            if root not in seen:
                seen.add(root)
                yield root


def _resolve_local_model_path(model_name: str) -> str | None:
    """Return a local model path if the requested model is already cached."""
    direct = Path(model_name)
    if direct.exists():
        return str(direct)

    repo_folder = f"models--{model_name.replace('/', '--')}"
    for cache_root in _candidate_hf_cache_roots():
        snapshots_dir = cache_root / repo_folder / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "modules.json").exists():
                return str(snapshot)

    legacy = Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name.replace("/", "_")
    if (legacy / "modules.json").exists():
        return str(legacy)

    return None


class TextEncoder:
    """Encodes game text into fixed-size embedding vectors.

    Features:
    - Frozen sentence-transformers model (no gradients)
    - L2-normalized embeddings
    - Persistent file cache (survives restarts, shared across processes)
    - In-memory dict cache (fast repeated access within a run)
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: str | None = None,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model = None
        self._model_load_error: RuntimeError | None = None
        self._mem_cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._model_init_lock = threading.Lock()
        self.embed_dim: int = TEXT_DIM

        # Persistent cache directory
        self._cache_dir = resolve_artifact_path(cache_dir if cache_dir is not None else _resolve_default_cache_dir())
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._disk_cache_path = self._cache_dir / f"{model_name.replace('/', '_')}.npz"
        # Migrate legacy pickle cache if it exists
        legacy_pkl = self._cache_dir / f"{model_name.replace('/', '_')}.pkl"
        if legacy_pkl.exists() and not self._disk_cache_path.exists():
            self._migrate_legacy_pkl(legacy_pkl)
        self._load_disk_cache()

    # ---- public API -------------------------------------------------------

    @property
    def dim(self) -> int:
        return self.embed_dim

    def ensure_ready(self) -> "TextEncoder":
        """Ensure the embedding model is available or raise immediately."""
        self._ensure_model()
        return self

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns (embed_dim,) float32. Cached."""
        if not text:
            return np.zeros(self.embed_dim, dtype=np.float32)

        key = self._cache_key(text)
        with self._lock:
            if key in self._mem_cache:
                return self._mem_cache[key]

        vec = self._compute([text])[0]
        with self._lock:
            self._mem_cache[key] = vec
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts. Returns (N, embed_dim) float32. Uses cache."""
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        results = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
        uncached_keys_in_order: list[str] = []
        uncached_texts_by_key: dict[str, str] = {}
        pending_indices_by_key: dict[str, list[int]] = {}

        with self._lock:
            for i, text in enumerate(texts):
                if not text:
                    continue
                key = self._cache_key(text)
                if key in self._mem_cache:
                    results[i] = self._mem_cache[key]
                else:
                    if key not in uncached_texts_by_key:
                        uncached_keys_in_order.append(key)
                        uncached_texts_by_key[key] = text
                    pending_indices_by_key.setdefault(key, []).append(i)

        if uncached_keys_in_order:
            uncached_texts = [uncached_texts_by_key[key] for key in uncached_keys_in_order]
            embeddings = self._compute(uncached_texts)
            with self._lock:
                for key, emb in zip(uncached_keys_in_order, embeddings):
                    self._mem_cache[key] = emb
                    for idx in pending_indices_by_key.get(key, ()):
                        results[idx] = emb

        return results

    def encode_or_zero(self, text: str | None) -> np.ndarray:
        """Encode text, return zero vector if None/empty."""
        if not text:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return self.encode(text)

    def save_cache(self):
        """Persist in-memory cache to disk as .npz (atomic write via temp+rename)."""
        with self._lock:
            data = dict(self._mem_cache)
        try:
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp"
            )
            os.close(fd)
            try:
                np.savez(tmp_path, **data)
                # np.savez appends .npz if missing — handle both
                saved = tmp_path + ".npz" if not tmp_path.endswith(".npz") and os.path.exists(tmp_path + ".npz") else tmp_path
                os.replace(saved, str(self._disk_cache_path))
            except Exception:
                for p in (tmp_path, tmp_path + ".npz"):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
                raise
        except Exception:
            pass  # best effort

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._mem_cache)

    # ---- internal ---------------------------------------------------------

    def _load_disk_cache(self):
        if self._disk_cache_path.exists():
            try:
                with np.load(str(self._disk_cache_path), allow_pickle=False) as data:
                    for key in data.files:
                        self._mem_cache[key] = data[key]
            except Exception:
                pass  # corrupted cache, start fresh

    def _migrate_legacy_pkl(self, pkl_path: Path):
        """One-time migration from legacy pickle cache to npz."""
        try:
            import pickle as _pickle
            with open(pkl_path, "rb") as f:
                data = _pickle.load(f)
            if isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()):
                np.savez(str(self._disk_cache_path), **data)
        except Exception:
            pass  # migration is best-effort

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._model_load_error is not None:
            raise self._model_load_error
        with self._model_init_lock:
            if self._model is not None:
                return
            if self._model_load_error is not None:
                raise self._model_load_error
            try:
                from sentence_transformers import SentenceTransformer

                local_model_path = _resolve_local_model_path(self._model_name)
                errors: list[str] = []
                model = None

                if local_model_path is not None:
                    try:
                        model = SentenceTransformer(
                            local_model_path,
                            device=self._device,
                            local_files_only=True,
                        )
                    except Exception as local_exc:
                        errors.append(f"local cache load failed from '{local_model_path}': {local_exc}")

                if model is None:
                    try:
                        model = SentenceTransformer(self._model_name, device=self._device)
                    except Exception as remote_exc:
                        errors.append(f"remote load failed for '{self._model_name}': {remote_exc}")
                        raise RuntimeError("; ".join(errors)) from remote_exc

                for param in model.parameters():
                    param.requires_grad = False
                if hasattr(model, "get_embedding_dimension"):
                    embed_dim = int(model.get_embedding_dimension())
                else:
                    embed_dim = int(model.get_sentence_embedding_dimension())
                self._model = model
                self.embed_dim = embed_dim
            except Exception as e:
                message = (
                    f"Failed to load text model '{self._model_name}'. "
                    f"Text embeddings are enabled, so execution is aborted. "
                    f"Download the model first or provide a valid local cache/model path. "
                    f"Details: {e}"
                )
                self._model_load_error = RuntimeError(message)
                raise self._model_load_error from e

    def _compute(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---- singleton ------------------------------------------------------------

_global_encoders: dict[tuple[str, str, str | None, bool], TextEncoder] = {}
_global_lock = threading.Lock()


def get_text_encoder(
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
    cache_dir: str | None = None,
    normalize: bool = True,
) -> TextEncoder:
    """Get or create the global text encoder singleton."""
    resolved_device = str(device or "cpu").strip() or "cpu"
    if resolved_device.lower() == "auto":
        try:
            import torch

            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            resolved_device = "cpu"
    key = (model_name, resolved_device, cache_dir, bool(normalize))
    with _global_lock:
        encoder = _global_encoders.get(key)
        if encoder is None:
            encoder = TextEncoder(
                model_name=model_name,
                device=resolved_device,
                cache_dir=cache_dir,
                normalize=normalize,
            )
            _global_encoders[key] = encoder
        return encoder
