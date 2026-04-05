"""Text encoding module for STS2 RL agent.

Default model: BAAI/bge-small-zh-v1.5 (Chinese-specific, 512-d output).
Embeddings are frozen, normalized, and cached to a persistent file.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# bge-small-zh-v1.5 output dimension
TEXT_DIM: int = 512

_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

def _resolve_default_cache_dir() -> str:
    """Pick a writable cache directory. Prefer project-local, fallback to temp."""
    # Try project-local first
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".text_cache")
    try:
        os.makedirs(local, exist_ok=True)
        # Quick write test
        test = os.path.join(local, ".write_test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return local
    except OSError:
        pass
    # Fallback to temp dir
    import tempfile
    return os.path.join(tempfile.gettempdir(), "sts2_text_cache")


def _candidate_hf_cache_roots() -> Iterable[Path]:
    """Yield likely Hugging Face cache roots without requiring any network access."""
    seen: set[Path] = set()

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
        self.embed_dim: int = TEXT_DIM

        # Persistent cache directory
        resolved_dir = cache_dir or _resolve_default_cache_dir()
        self._cache_dir = Path(resolved_dir)
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._disk_cache_path = self._cache_dir / f"{model_name.replace('/', '_')}.pkl"
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
        """Persist in-memory cache to disk (atomic write via temp+rename)."""
        with self._lock:
            data = dict(self._mem_cache)
        try:
            import tempfile
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._cache_dir), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                # Atomic rename (on same filesystem)
                os.replace(tmp_path, str(self._disk_cache_path))
            except Exception:
                try:
                    os.unlink(tmp_path)
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
                with open(self._disk_cache_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    self._mem_cache.update(data)
            except Exception:
                pass  # corrupted cache, start fresh

    def _ensure_model(self):
        if self._model is not None:
            return
        if self._model_load_error is not None:
            raise self._model_load_error
        try:
            from sentence_transformers import SentenceTransformer

            local_model_path = _resolve_local_model_path(self._model_name)
            errors: list[str] = []

            if local_model_path is not None:
                try:
                    self._model = SentenceTransformer(
                        local_model_path,
                        device=self._device,
                        local_files_only=True,
                    )
                except Exception as local_exc:
                    errors.append(f"local cache load failed from '{local_model_path}': {local_exc}")

            if self._model is None:
                try:
                    self._model = SentenceTransformer(self._model_name, device=self._device)
                except Exception as remote_exc:
                    errors.append(f"remote load failed for '{self._model_name}': {remote_exc}")
                    raise RuntimeError("; ".join(errors)) from remote_exc

            for param in self._model.parameters():
                param.requires_grad = False
            self.embed_dim = self._model.get_sentence_embedding_dimension()
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

_global_encoder: TextEncoder | None = None
_global_lock = threading.Lock()


def get_text_encoder(
    model_name: str = _DEFAULT_MODEL,
    device: str = "cpu",
    cache_dir: str | None = None,
    normalize: bool = True,
) -> TextEncoder:
    """Get or create the global text encoder singleton."""
    global _global_encoder
    with _global_lock:
        if _global_encoder is None:
            _global_encoder = TextEncoder(
                model_name=model_name,
                device=device,
                cache_dir=cache_dir,
                normalize=normalize,
            )
        return _global_encoder
