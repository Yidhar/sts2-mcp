"""Compact rollout snapshots for grounded structural decisions.

The online collector already pays the cost of structural encoding before it
chooses an action. Repeating the raw JSON walk during sequence learning is
unnecessary. This module stores the exact encoded token facts as canonical
sparse CPU arrays and materializes a learner batch in one collate operation.

Snapshots contain no dispatch handles and no policy output.  They are model
inputs only, bound to both the structural encoding fingerprint and the full
encoding capacity/vocabulary configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from sts2_rl.models.grounded_candidate import (
    MACRO_ECONOMIC_SURFACE_COUNT,
    MIN_TOKEN_FEATURE_DIM,
    CandidateTokenBatch,
    GroundedCandidateBatch,
    GroundedCandidateConfig,
    WorldTokenBatch,
)

ENCODED_DECISION_SNAPSHOT_VERSION: Final = "relational-encoded-decision-v5"
_WORLD_ID_WIDTH: Final = 9
_CANDIDATE_ID_WIDTH: Final = 14
_LOCAL_ID_WIDTH: Final = 9
_INT32_ID_BOUND: Final = int(np.iinfo(np.int32).max) + 1


@dataclass(frozen=True, slots=True)
class GroundedEncodingConfig:
    """Maximum tensor capacities and categorical vocabulary contract."""

    feature_dim: int = 224
    max_world_tokens: int = 2048
    max_candidates: int = 256
    max_candidate_local_tokens: int = 64
    type_vocab_size: int = 128
    role_vocab_size: int = 64
    owner_vocab_size: int = 128
    entity_vocab_size: int = 8192
    zone_vocab_size: int = 32
    max_order_id: int = 128
    domain_count: int = 8

    @classmethod
    def from_model_config(
        cls,
        model: GroundedCandidateConfig,
        *,
        max_world_tokens: int = 2048,
        max_candidates: int = 256,
        max_candidate_local_tokens: int = 64,
    ) -> GroundedEncodingConfig:
        return cls(
            feature_dim=model.token_feature_dim,
            max_world_tokens=max_world_tokens,
            max_candidates=max_candidates,
            max_candidate_local_tokens=max_candidate_local_tokens,
            type_vocab_size=model.type_vocab_size,
            role_vocab_size=model.role_vocab_size,
            owner_vocab_size=model.owner_vocab_size,
            entity_vocab_size=model.entity_vocab_size,
            zone_vocab_size=model.zone_vocab_size,
            max_order_id=model.order_vocab_size,
            domain_count=model.domain_count,
        )

    def __post_init__(self) -> None:
        integer_fields = (
            "feature_dim",
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
            "max_order_id",
            "domain_count",
        )
        wrong_types = [
            name
            for name in integer_fields
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int)
        ]
        if wrong_types:
            raise TypeError("grounded encoding dimensions must be exact integers: " + ", ".join(wrong_types))
        if self.feature_dim < MIN_TOKEN_FEATURE_DIM:
            raise ValueError("feature_dim must be at least " f"{MIN_TOKEN_FEATURE_DIM} for the grounded feature ABI")
        if self.feature_dim > int(np.iinfo(np.uint16).max) + 1:
            raise ValueError("feature_dim exceeds the sparse snapshot index ABI")
        for name in (
            "max_world_tokens",
            "max_candidates",
            "max_candidate_local_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "type_vocab_size",
            "role_vocab_size",
            "owner_vocab_size",
            "entity_vocab_size",
            "zone_vocab_size",
        ):
            if getattr(self, name) < 4:
                raise ValueError(f"{name} must be at least 4")
            if getattr(self, name) > int(np.iinfo(np.int32).max):
                raise ValueError(f"{name} exceeds the sparse snapshot ID ABI")
        if self.max_order_id < 2:
            raise ValueError("max_order_id must be at least 2")
        if self.max_order_id > int(np.iinfo(np.int32).max):
            raise ValueError("max_order_id exceeds the sparse snapshot ID ABI")
        if self.domain_count <= 5:
            raise ValueError("domain_count is too small for the fixed domain vocabulary")
        if self.max_candidates * self.max_candidate_local_tokens > int(np.iinfo(np.uint32).max):
            raise ValueError("candidate-local capacity exceeds the snapshot offset ABI")


def _owned_array(
    value: npt.NDArray[Any],
    *,
    dtype: np.dtype[Any],
    ndim: int,
    label: str,
) -> npt.NDArray[Any]:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TypeError(f"{label} must use {dtype}, got {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{label} must have rank {ndim}, got {array.ndim}")
    owned = np.ascontiguousarray(array).copy()
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class SparseTokenTable:
    """Canonical CSR-like feature storage plus dense categorical token IDs."""

    feature_indptr: npt.NDArray[np.uint32]
    feature_indices: npt.NDArray[np.uint16]
    feature_values: npt.NDArray[np.float32]
    ids: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_indptr",
            _owned_array(
                self.feature_indptr,
                dtype=np.dtype(np.uint32),
                ndim=1,
                label="feature_indptr",
            ),
        )
        object.__setattr__(
            self,
            "feature_indices",
            _owned_array(
                self.feature_indices,
                dtype=np.dtype(np.uint16),
                ndim=1,
                label="feature_indices",
            ),
        )
        object.__setattr__(
            self,
            "feature_values",
            _owned_array(
                self.feature_values,
                dtype=np.dtype(np.float32),
                ndim=1,
                label="feature_values",
            ),
        )
        object.__setattr__(
            self,
            "ids",
            _owned_array(
                self.ids,
                dtype=np.dtype(np.int32),
                ndim=2,
                label="ids",
            ),
        )
        self.validate_structure()

    @property
    def token_count(self) -> int:
        return int(self.ids.shape[0])

    def validate_structure(self) -> None:
        arrays = (
            ("feature_indptr", self.feature_indptr, np.dtype(np.uint32), 1),
            ("feature_indices", self.feature_indices, np.dtype(np.uint16), 1),
            ("feature_values", self.feature_values, np.dtype(np.float32), 1),
            ("ids", self.ids, np.dtype(np.int32), 2),
        )
        for name, value, dtype, ndim in arrays:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array")
            if value.dtype != dtype:
                raise TypeError(f"{name} must use {dtype}, got {value.dtype}")
            if value.ndim != ndim:
                raise ValueError(f"{name} must have rank {ndim}, got {value.ndim}")
            if not value.flags.c_contiguous:
                raise ValueError(f"{name} must use canonical contiguous storage")
        if self.feature_indptr.shape != (self.token_count + 1,):
            raise ValueError("feature_indptr length must equal token_count + 1")
        if int(self.feature_indptr[0]) != 0:
            raise ValueError("feature_indptr must start at zero")
        if np.any(self.feature_indptr[1:] < self.feature_indptr[:-1]):
            raise ValueError("feature_indptr must be non-decreasing")
        nonzero_count = len(self.feature_indices)
        if len(self.feature_values) != nonzero_count:
            raise ValueError("sparse feature indices/values lengths differ")
        if int(self.feature_indptr[-1]) != nonzero_count:
            raise ValueError("feature_indptr terminal offset is inconsistent")
        if not np.all(np.isfinite(self.feature_values)):
            raise ValueError("sparse token features must be finite")
        if np.any(self.feature_values == 0.0):
            raise ValueError("sparse token features must not store explicit zeros")
        if np.any(self.ids < 0):
            raise ValueError("sparse token IDs must be non-negative")
        if nonzero_count > 1:
            same_token = np.ones(nonzero_count - 1, dtype=np.bool_)
            row_starts = self.feature_indptr[1:-1]
            valid_starts = row_starts[(row_starts > 0) & (row_starts < nonzero_count)]
            same_token[valid_starts - 1] = False
            if np.any((self.feature_indices[1:] <= self.feature_indices[:-1]) & same_token):
                raise ValueError("sparse feature indices must be strictly increasing per token")

    def validate(
        self,
        *,
        feature_dim: int,
        id_bounds: tuple[int, ...],
        label: str,
    ) -> None:
        self.validate_structure()
        if self.ids.shape[1:] != (len(id_bounds),):
            raise ValueError(f"{label}.ids width must be {len(id_bounds)}, got {self.ids.shape[1:]}")
        if len(self.feature_indices) and int(self.feature_indices.max()) >= feature_dim:
            raise ValueError(f"{label} sparse feature index exceeds feature_dim")
        for column, bound in enumerate(id_bounds):
            if self.token_count and int(self.ids[:, column].max()) >= bound:
                raise ValueError(f"{label}.ids column {column} exceeds vocabulary size {bound}")


def sparse_token_table(
    *,
    features: tuple[tuple[float, ...], ...],
    ids: tuple[tuple[int, ...], ...],
    feature_dim: int,
    id_width: int,
) -> SparseTokenTable:
    """Build canonical exact sparse arrays from structural token tuples."""

    if len(features) != len(ids):
        raise ValueError("token feature and ID row counts differ")
    if id_width <= 0 or any(len(row) != id_width for row in ids):
        raise ValueError("token ID rows differ from the declared width")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= int(np.iinfo(np.int32).max)
        for row in ids
        for value in row
    ):
        raise ValueError("token IDs must fit the non-negative int32 snapshot ABI")
    indptr = [0]
    indices: list[int] = []
    values: list[float] = []
    for row in features:
        if len(row) != feature_dim:
            raise ValueError("token feature row differs from encoding feature_dim")
        for index, raw_value in enumerate(row):
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("token feature must be finite")
            if value != 0.0:
                indices.append(index)
                values.append(value)
        indptr.append(len(indices))
    return SparseTokenTable(
        feature_indptr=np.asarray(indptr, dtype=np.uint32),
        feature_indices=np.asarray(indices, dtype=np.uint16),
        feature_values=np.asarray(values, dtype=np.float32),
        ids=np.asarray(ids, dtype=np.int32).reshape(len(ids), id_width),
    )


@dataclass(frozen=True, slots=True)
class EncodedWorldSnapshot:
    """Candidate-independent sparse world input for durable state/afterstate valuation."""

    config: GroundedEncodingConfig
    encoding_fingerprint: str
    world: SparseTokenTable
    domain_id: int
    version: str = ENCODED_DECISION_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        self.validate(
            expected_config=self.config,
            expected_fingerprint=self.encoding_fingerprint,
        )

    def validate(
        self,
        *,
        expected_config: GroundedEncodingConfig,
        expected_fingerprint: str,
    ) -> None:
        if not isinstance(self.config, GroundedEncodingConfig):
            raise TypeError("encoded world snapshot config has the wrong type")
        if not isinstance(self.world, SparseTokenTable):
            raise TypeError("encoded world snapshot table has the wrong type")
        if self.version != ENCODED_DECISION_SNAPSHOT_VERSION:
            raise ValueError(f"unsupported encoded world snapshot: {self.version!r}")
        if self.config != expected_config:
            raise ValueError("encoded world snapshot config differs from active encoder")
        if self.encoding_fingerprint != expected_fingerprint:
            raise ValueError("encoded world snapshot fingerprint differs from active encoder")
        if len(self.encoding_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.encoding_fingerprint
        ):
            raise ValueError("encoded world snapshot fingerprint must be lowercase SHA-256")
        cfg = self.config
        world_bounds = (
            cfg.type_vocab_size,
            cfg.role_vocab_size,
            cfg.owner_vocab_size,
            cfg.entity_vocab_size,
            cfg.entity_vocab_size,
            _INT32_ID_BOUND,
            _INT32_ID_BOUND,
            cfg.zone_vocab_size,
            cfg.max_order_id,
        )
        self.world.validate(
            feature_dim=cfg.feature_dim,
            id_bounds=world_bounds,
            label="world",
        )
        if not 0 < self.world.token_count <= cfg.max_world_tokens:
            raise ValueError("encoded world token count is outside configured capacity")
        if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int):
            raise TypeError("encoded world domain_id must be an integer")
        if not 0 <= self.domain_id < cfg.domain_count:
            raise ValueError("encoded world domain_id exceeds configured vocabulary")

    def storage_nbytes(self) -> int:
        arrays = (
            self.world.feature_indptr,
            self.world.feature_indices,
            self.world.feature_values,
            self.world.ids,
        )
        return sum(int(array.nbytes) for array in arrays)


@dataclass(frozen=True, slots=True)
class EncodedDecisionSnapshot:
    """Rollout-owned sparse model input for one legal-action decision."""

    config: GroundedEncodingConfig
    encoding_fingerprint: str
    world: SparseTokenTable
    candidates: SparseTokenTable
    locals: SparseTokenTable
    local_offsets: npt.NDArray[np.uint32]
    action_mask: npt.NDArray[np.bool_]
    domain_id: int
    macro_economic_surface_id: int = 0
    version: str = ENCODED_DECISION_SNAPSHOT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_offsets",
            _owned_array(
                self.local_offsets,
                dtype=np.dtype(np.uint32),
                ndim=1,
                label="local_offsets",
            ),
        )
        object.__setattr__(
            self,
            "action_mask",
            _owned_array(
                self.action_mask,
                dtype=np.dtype(np.bool_),
                ndim=1,
                label="action_mask",
            ),
        )
        self.validate(
            expected_config=self.config,
            expected_fingerprint=self.encoding_fingerprint,
        )

    @property
    def candidate_count(self) -> int:
        return self.candidates.token_count

    def validate(
        self,
        *,
        expected_config: GroundedEncodingConfig,
        expected_fingerprint: str,
    ) -> None:
        if not isinstance(self.config, GroundedEncodingConfig):
            raise TypeError("encoded decision snapshot config has the wrong type")
        if (
            not isinstance(self.world, SparseTokenTable)
            or not isinstance(self.candidates, SparseTokenTable)
            or not isinstance(self.locals, SparseTokenTable)
        ):
            raise TypeError("encoded decision snapshot token tables have the wrong type")
        for name, value, dtype in (
            ("local_offsets", self.local_offsets, np.dtype(np.uint32)),
            ("action_mask", self.action_mask, np.dtype(np.bool_)),
        ):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"encoded decision {name} must be a NumPy array")
            if value.dtype != dtype or value.ndim != 1 or not value.flags.c_contiguous:
                raise ValueError(f"encoded decision {name} must be canonical contiguous {dtype} rank-1")
        if self.version != ENCODED_DECISION_SNAPSHOT_VERSION:
            raise ValueError(f"unsupported encoded decision snapshot: {self.version!r}")
        if self.config != expected_config:
            raise ValueError("encoded decision snapshot config differs from active encoder")
        if self.encoding_fingerprint != expected_fingerprint:
            raise ValueError("encoded decision snapshot fingerprint differs from active encoder")
        if len(self.encoding_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.encoding_fingerprint
        ):
            raise ValueError("encoded decision snapshot fingerprint must be lowercase SHA-256")
        cfg = self.config
        world_bounds = (
            cfg.type_vocab_size,
            cfg.role_vocab_size,
            cfg.owner_vocab_size,
            cfg.entity_vocab_size,
            cfg.entity_vocab_size,
            _INT32_ID_BOUND,
            _INT32_ID_BOUND,
            cfg.zone_vocab_size,
            cfg.max_order_id,
        )
        candidate_bounds = (
            cfg.type_vocab_size,
            cfg.role_vocab_size,
            cfg.owner_vocab_size,
            cfg.entity_vocab_size,
            cfg.entity_vocab_size,
            _INT32_ID_BOUND,
            _INT32_ID_BOUND,
            cfg.zone_vocab_size,
            cfg.owner_vocab_size,
            cfg.entity_vocab_size,
            cfg.entity_vocab_size,
            _INT32_ID_BOUND,
            _INT32_ID_BOUND,
            self.candidate_count,
        )
        self.world.validate(
            feature_dim=cfg.feature_dim,
            id_bounds=world_bounds,
            label="world",
        )
        self.candidates.validate(
            feature_dim=cfg.feature_dim,
            id_bounds=candidate_bounds,
            label="candidates",
        )
        self.locals.validate(
            feature_dim=cfg.feature_dim,
            id_bounds=world_bounds,
            label="locals",
        )
        if not 0 < self.world.token_count <= cfg.max_world_tokens:
            raise ValueError("encoded world token count is outside configured capacity")
        if not 0 < self.candidate_count <= cfg.max_candidates:
            raise ValueError("encoded candidate count is outside configured capacity")
        if self.action_mask.shape != (self.candidate_count,):
            raise ValueError("encoded action mask length differs from candidate count")
        if not bool(self.action_mask.any()):
            raise ValueError("encoded decision snapshot has no enabled action")
        if self.local_offsets.shape != (self.candidate_count + 1,):
            raise ValueError("local_offsets length must equal candidate_count + 1")
        if int(self.local_offsets[0]) != 0:
            raise ValueError("local_offsets must start at zero")
        if np.any(self.local_offsets[1:] < self.local_offsets[:-1]):
            raise ValueError("local_offsets must be non-decreasing")
        if int(self.local_offsets[-1]) != self.locals.token_count:
            raise ValueError("local_offsets terminal offset differs from local token count")
        if np.any(np.diff(self.local_offsets) > cfg.max_candidate_local_tokens):
            raise ValueError("candidate-local token count exceeds configured capacity")
        if isinstance(self.domain_id, bool) or not isinstance(self.domain_id, int):
            raise TypeError("encoded decision domain_id must be an integer")
        if not 0 <= self.domain_id < cfg.domain_count:
            raise ValueError("encoded decision domain_id exceeds configured vocabulary")
        if (
            isinstance(self.macro_economic_surface_id, bool)
            or not isinstance(self.macro_economic_surface_id, int)
        ):
            raise TypeError(
                "encoded decision macro_economic_surface_id must be an integer"
            )
        if not 0 <= self.macro_economic_surface_id < MACRO_ECONOMIC_SURFACE_COUNT:
            raise ValueError(
                "encoded decision macro_economic_surface_id exceeds reviewed vocabulary"
            )

    def storage_nbytes(self) -> int:
        """Exact NumPy payload bytes, excluding small Python/dataclass headers."""

        arrays = (
            self.world.feature_indptr,
            self.world.feature_indices,
            self.world.feature_values,
            self.world.ids,
            self.candidates.feature_indptr,
            self.candidates.feature_indices,
            self.candidates.feature_values,
            self.candidates.ids,
            self.locals.feature_indptr,
            self.locals.feature_indices,
            self.locals.feature_values,
            self.locals.ids,
            self.local_offsets,
            self.action_mask,
        )
        return sum(int(array.nbytes) for array in arrays)


def _tensor(array: npt.NDArray[Any], *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device)


def _transfer_packed(
    packed_i32: npt.NDArray[np.int32],
    packed_f32: npt.NDArray[np.float32],
    *,
    dev: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_i32 = torch.from_numpy(packed_i32)
    if dev.type == "cuda":
        i32_gpu = t_i32.pin_memory().to(device=dev, non_blocking=True)
        if packed_f32.size:
            f32_gpu = torch.from_numpy(packed_f32).pin_memory().to(device=dev, non_blocking=True)
        else:
            f32_gpu = torch.empty((0,), device=dev, dtype=torch.float32)
        return i32_gpu, f32_gpu
    return t_i32.to(device=dev), torch.from_numpy(packed_f32).to(device=dev)


def _collate_encoded_snapshots_scatter(
    snapshots: tuple[EncodedDecisionSnapshot, ...],
    *,
    cfg: GroundedEncodingConfig,
    dev: torch.device,
) -> GroundedCandidateBatch:
    """Pack flat sparse arrays into two pinned buffers and scatter on the target device."""

    bsz = len(snapshots)
    w_counts_list = [s.world.token_count for s in snapshots]
    c_counts_list = [s.candidate_count for s in snapshots]
    w_cap = max(w_counts_list)
    c_cap = max(c_counts_list)
    total_w = sum(w_counts_list)
    total_c = sum(c_counts_list)
    total_l = sum(s.locals.token_count for s in snapshots)

    l_counts_np = (
        np.concatenate([np.diff(s.local_offsets).astype(np.int32, copy=False) for s in snapshots])
        if total_c
        else np.empty((0,), dtype=np.int32)
    )
    l_cap = max(1, int(l_counts_np.max())) if l_counts_np.size else 1

    w_ids_flat = np.concatenate([s.world.ids for s in snapshots], axis=0).reshape(-1)
    w_nnz = np.concatenate([np.diff(s.world.feature_indptr).astype(np.int32, copy=False) for s in snapshots])
    w_fidx = np.concatenate([s.world.feature_indices for s in snapshots]).astype(np.int32, copy=False)
    w_fval = np.concatenate([s.world.feature_values for s in snapshots])
    total_nnz_w = int(w_fval.size)

    if total_c:
        c_ids_flat = np.concatenate([s.candidates.ids for s in snapshots], axis=0).reshape(-1)
        c_mask_i32 = np.concatenate([s.action_mask for s in snapshots]).astype(np.int32, copy=False)
        c_nnz = np.concatenate(
            [np.diff(s.candidates.feature_indptr).astype(np.int32, copy=False) for s in snapshots]
        )
        c_fidx = np.concatenate([s.candidates.feature_indices for s in snapshots]).astype(np.int32, copy=False)
        c_fval = np.concatenate([s.candidates.feature_values for s in snapshots])
    else:
        c_ids_flat = np.empty((0,), dtype=np.int32)
        c_mask_i32 = np.empty((0,), dtype=np.int32)
        c_nnz = np.empty((0,), dtype=np.int32)
        c_fidx = np.empty((0,), dtype=np.int32)
        c_fval = np.empty((0,), dtype=np.float32)
    total_nnz_c = int(c_fval.size)

    meta_i32 = np.array(
        [
            *w_counts_list,
            *c_counts_list,
            *(s.domain_id for s in snapshots),
            *(s.macro_economic_surface_id for s in snapshots),
        ],
        dtype=np.int32,
    )
    i32_parts: list[npt.NDArray[np.int32]] = [
        meta_i32,
        w_ids_flat,
        w_nnz,
        w_fidx,
        c_ids_flat,
        c_mask_i32,
        c_nnz,
        c_fidx,
        l_counts_np,
    ]
    f32_parts: list[npt.NDArray[np.float32]] = [w_fval, c_fval]

    total_nnz_l = 0
    if total_l > 0:
        l_snaps = [s for s in snapshots if s.locals.token_count]
        l_ids_flat = np.concatenate([s.locals.ids for s in l_snaps], axis=0).reshape(-1)
        l_nnz = np.concatenate(
            [np.diff(s.locals.feature_indptr).astype(np.int32, copy=False) for s in l_snaps]
        )
        l_fidx = np.concatenate([s.locals.feature_indices for s in l_snaps]).astype(np.int32, copy=False)
        l_fval = np.concatenate([s.locals.feature_values for s in l_snaps])
        total_nnz_l = int(l_fval.size)
        i32_parts.extend((l_ids_flat, l_nnz, l_fidx))
        if total_nnz_l > 0:
            f32_parts.append(l_fval)

    packed_i32 = np.concatenate(i32_parts)
    packed_f32 = (
        np.concatenate([p for p in f32_parts if p.size])
        if any(p.size for p in f32_parts)
        else np.empty((0,), dtype=np.float32)
    )
    i32_dev, f32_dev = _transfer_packed(packed_i32, packed_f32, dev=dev)

    cursor = 0
    w_counts_t = i32_dev[cursor : cursor + bsz].long()
    cursor += bsz
    c_counts_t = i32_dev[cursor : cursor + bsz].long()
    cursor += bsz
    domain_ids_t = i32_dev[cursor : cursor + bsz].long()
    cursor += bsz
    surface_ids_t = i32_dev[cursor : cursor + bsz].long()
    cursor += bsz

    w_ids_len = total_w * _WORLD_ID_WIDTH
    w_ids_src = i32_dev[cursor : cursor + w_ids_len].reshape(total_w, _WORLD_ID_WIDTH).long()
    cursor += w_ids_len
    w_nnz_t = i32_dev[cursor : cursor + total_w].long()
    cursor += total_w
    w_fidx_t = i32_dev[cursor : cursor + total_nnz_w].long()
    cursor += total_nnz_w

    c_ids_len = total_c * _CANDIDATE_ID_WIDTH
    c_ids_src = i32_dev[cursor : cursor + c_ids_len].reshape(total_c, _CANDIDATE_ID_WIDTH).long()
    cursor += c_ids_len
    c_mask_src = i32_dev[cursor : cursor + total_c].ne(0)
    cursor += total_c
    c_nnz_t = i32_dev[cursor : cursor + total_c].long()
    cursor += total_c
    c_fidx_t = i32_dev[cursor : cursor + total_nnz_c].long()
    cursor += total_nnz_c
    l_counts_t = i32_dev[cursor : cursor + total_c].long()
    cursor += total_c

    f_cursor = 0
    w_fval_t = f32_dev[f_cursor : f_cursor + total_nnz_w]
    f_cursor += total_nnz_w
    c_fval_t = f32_dev[f_cursor : f_cursor + total_nnz_c]
    f_cursor += total_nnz_c

    arange_b = torch.arange(bsz, device=dev, dtype=torch.long)

    # Scatter World
    world_mask = torch.arange(w_cap, device=dev, dtype=torch.long).unsqueeze(0) < w_counts_t.unsqueeze(1)
    world_ids_tensor = torch.zeros((bsz, w_cap, _WORLD_ID_WIDTH), device=dev, dtype=torch.long)
    world_features = torch.zeros((bsz, w_cap, cfg.feature_dim), device=dev, dtype=torch.float32)
    if total_w > 0:
        b_of_w = torch.repeat_interleave(arange_b, w_counts_t, output_size=total_w)
        cum_w = w_counts_t.cumsum(0) - w_counts_t
        t_of_w = torch.arange(total_w, device=dev, dtype=torch.long) - cum_w[b_of_w]
        world_ids_tensor[b_of_w, t_of_w] = w_ids_src
        if total_nnz_w > 0:
            tok_w_nz = torch.repeat_interleave(
                torch.arange(total_w, device=dev, dtype=torch.long),
                w_nnz_t,
                output_size=total_nnz_w,
            )
            world_features[b_of_w[tok_w_nz], t_of_w[tok_w_nz], w_fidx_t] = w_fval_t

    # Scatter Candidates & Locals
    candidate_ids_tensor = torch.zeros((bsz, c_cap, _CANDIDATE_ID_WIDTH), device=dev, dtype=torch.long)
    action_mask = torch.zeros((bsz, c_cap), device=dev, dtype=torch.bool)
    candidate_features = torch.zeros((bsz, c_cap, cfg.feature_dim), device=dev, dtype=torch.float32)
    local_mask = torch.zeros((bsz, c_cap, l_cap), device=dev, dtype=torch.bool)
    local_ids_tensor = torch.zeros((bsz, c_cap, l_cap, _LOCAL_ID_WIDTH), device=dev, dtype=torch.long)
    local_features = torch.zeros((bsz, c_cap, l_cap, cfg.feature_dim), device=dev, dtype=torch.float32)

    if total_c > 0:
        b_of_c = torch.repeat_interleave(arange_b, c_counts_t, output_size=total_c)
        cum_c = c_counts_t.cumsum(0) - c_counts_t
        t_of_c = torch.arange(total_c, device=dev, dtype=torch.long) - cum_c[b_of_c]
        candidate_ids_tensor[b_of_c, t_of_c] = c_ids_src
        action_mask[b_of_c, t_of_c] = c_mask_src
        if total_nnz_c > 0:
            tok_c_nz = torch.repeat_interleave(
                torch.arange(total_c, device=dev, dtype=torch.long),
                c_nnz_t,
                output_size=total_nnz_c,
            )
            candidate_features[b_of_c[tok_c_nz], t_of_c[tok_c_nz], c_fidx_t] = c_fval_t

        if total_l > 0:
            l_ids_len = total_l * _LOCAL_ID_WIDTH
            l_ids_src = i32_dev[cursor : cursor + l_ids_len].reshape(total_l, _LOCAL_ID_WIDTH).long()
            cursor += l_ids_len
            l_nnz_t = i32_dev[cursor : cursor + total_l].long()
            cursor += total_l
            l_fidx_t = i32_dev[cursor : cursor + total_nnz_l].long()
            cursor += total_nnz_l

            cand_of_l = torch.repeat_interleave(
                torch.arange(total_c, device=dev, dtype=torch.long),
                l_counts_t,
                output_size=total_l,
            )
            cum_l = l_counts_t.cumsum(0) - l_counts_t
            pos_of_l = torch.arange(total_l, device=dev, dtype=torch.long) - cum_l[cand_of_l]
            b_of_l = b_of_c[cand_of_l]
            c_of_l = t_of_c[cand_of_l]
            local_mask[b_of_l, c_of_l, pos_of_l] = True
            local_ids_tensor[b_of_l, c_of_l, pos_of_l] = l_ids_src
            if total_nnz_l > 0:
                l_fval_t = f32_dev[f_cursor : f_cursor + total_nnz_l]
                tok_l_nz = torch.repeat_interleave(
                    torch.arange(total_l, device=dev, dtype=torch.long),
                    l_nnz_t,
                    output_size=total_nnz_l,
                )
                local_features[
                    b_of_l[tok_l_nz],
                    c_of_l[tok_l_nz],
                    pos_of_l[tok_l_nz],
                    l_fidx_t,
                ] = l_fval_t

    return GroundedCandidateBatch(
        world=WorldTokenBatch(
            features=world_features,
            mask=world_mask,
            type_ids=world_ids_tensor[:, :, 0],
            role_ids=world_ids_tensor[:, :, 1],
            owner_ids=world_ids_tensor[:, :, 2],
            entity_ids=world_ids_tensor[:, :, 3],
            entity_aux_ids=world_ids_tensor[:, :, 4],
            definition_binding_ids=world_ids_tensor[:, :, 5],
            relation_binding_ids=world_ids_tensor[:, :, 6],
            zone_ids=world_ids_tensor[:, :, 7],
            order_ids=world_ids_tensor[:, :, 8],
        ),
        candidates=CandidateTokenBatch(
            features=candidate_features,
            type_ids=candidate_ids_tensor[:, :, 0],
            role_ids=candidate_ids_tensor[:, :, 1],
            owner_ids=candidate_ids_tensor[:, :, 2],
            entity_ids=candidate_ids_tensor[:, :, 3],
            entity_aux_ids=candidate_ids_tensor[:, :, 4],
            definition_binding_ids=candidate_ids_tensor[:, :, 5],
            relation_binding_ids=candidate_ids_tensor[:, :, 6],
            zone_ids=candidate_ids_tensor[:, :, 7],
            target_owner_ids=candidate_ids_tensor[:, :, 8],
            target_entity_ids=candidate_ids_tensor[:, :, 9],
            target_entity_aux_ids=candidate_ids_tensor[:, :, 10],
            target_definition_binding_ids=candidate_ids_tensor[:, :, 11],
            target_relation_binding_ids=candidate_ids_tensor[:, :, 12],
            policy_branch_ids=candidate_ids_tensor[:, :, 13],
            local_features=local_features,
            local_mask=local_mask,
            local_type_ids=local_ids_tensor[:, :, :, 0],
            local_role_ids=local_ids_tensor[:, :, :, 1],
            local_owner_ids=local_ids_tensor[:, :, :, 2],
            local_entity_ids=local_ids_tensor[:, :, :, 3],
            local_entity_aux_ids=local_ids_tensor[:, :, :, 4],
            local_definition_binding_ids=local_ids_tensor[:, :, :, 5],
            local_relation_binding_ids=local_ids_tensor[:, :, :, 6],
            local_zone_ids=local_ids_tensor[:, :, :, 7],
            local_order_ids=local_ids_tensor[:, :, :, 8],
            action_mask=action_mask,
        ),
        domain_ids=domain_ids_t,
        macro_economic_surface_ids=surface_ids_t,
    )


def _collate_encoded_worlds_scatter(
    snapshots: tuple[EncodedWorldSnapshot, ...],
    *,
    cfg: GroundedEncodingConfig,
    dev: torch.device,
) -> tuple[WorldTokenBatch, torch.Tensor]:
    bsz = len(snapshots)
    w_counts_list = [s.world.token_count for s in snapshots]
    w_cap = max(w_counts_list)
    total_w = sum(w_counts_list)
    w_ids_flat = np.concatenate([s.world.ids for s in snapshots], axis=0).reshape(-1)
    w_nnz = np.concatenate([np.diff(s.world.feature_indptr).astype(np.int32, copy=False) for s in snapshots])
    w_fidx = np.concatenate([s.world.feature_indices for s in snapshots]).astype(np.int32, copy=False)
    w_fval = np.concatenate([s.world.feature_values for s in snapshots])
    total_nnz_w = int(w_fval.size)

    meta_i32 = np.array([*w_counts_list, *(s.domain_id for s in snapshots)], dtype=np.int32)
    packed_i32 = np.concatenate([meta_i32, w_ids_flat, w_nnz, w_fidx])
    i32_dev, f32_dev = _transfer_packed(packed_i32, w_fval, dev=dev)

    w_counts_t = i32_dev[:bsz].long()
    domain_ids_t = i32_dev[bsz : 2 * bsz].long()
    cursor = 2 * bsz
    w_ids_len = total_w * _WORLD_ID_WIDTH
    w_ids_src = i32_dev[cursor : cursor + w_ids_len].reshape(total_w, _WORLD_ID_WIDTH).long()
    cursor += w_ids_len
    w_nnz_t = i32_dev[cursor : cursor + total_w].long()
    cursor += total_w
    w_fidx_t = i32_dev[cursor : cursor + total_nnz_w].long()

    world_mask = torch.arange(w_cap, device=dev, dtype=torch.long).unsqueeze(0) < w_counts_t.unsqueeze(1)
    world_ids_tensor = torch.zeros((bsz, w_cap, _WORLD_ID_WIDTH), device=dev, dtype=torch.long)
    world_features = torch.zeros((bsz, w_cap, cfg.feature_dim), device=dev, dtype=torch.float32)
    if total_w > 0:
        arange_b = torch.arange(bsz, device=dev, dtype=torch.long)
        b_of_w = torch.repeat_interleave(arange_b, w_counts_t, output_size=total_w)
        cum_w = w_counts_t.cumsum(0) - w_counts_t
        t_of_w = torch.arange(total_w, device=dev, dtype=torch.long) - cum_w[b_of_w]
        world_ids_tensor[b_of_w, t_of_w] = w_ids_src
        if total_nnz_w > 0:
            tok_w_nz = torch.repeat_interleave(
                torch.arange(total_w, device=dev, dtype=torch.long),
                w_nnz_t,
                output_size=total_nnz_w,
            )
            world_features[b_of_w[tok_w_nz], t_of_w[tok_w_nz], w_fidx_t] = f32_dev[:total_nnz_w]

    world_batch = WorldTokenBatch(
        features=world_features,
        mask=world_mask,
        type_ids=world_ids_tensor[:, :, 0],
        role_ids=world_ids_tensor[:, :, 1],
        owner_ids=world_ids_tensor[:, :, 2],
        entity_ids=world_ids_tensor[:, :, 3],
        entity_aux_ids=world_ids_tensor[:, :, 4],
        definition_binding_ids=world_ids_tensor[:, :, 5],
        relation_binding_ids=world_ids_tensor[:, :, 6],
        zone_ids=world_ids_tensor[:, :, 7],
        order_ids=world_ids_tensor[:, :, 8],
    )
    return world_batch, domain_ids_t


def collate_encoded_snapshots(
    snapshots: tuple[EncodedDecisionSnapshot, ...],
    *,
    expected_config: GroundedEncodingConfig,
    expected_fingerprint: str,
    device: torch.device | str | None,
    validate: bool = True,
) -> GroundedCandidateBatch:
    """Materialize tensors trimmed to the active capacity of this batch.

    ``GroundedEncodingConfig`` defines hard acceptance limits, not a requirement
    to run every transformer invocation at those limits.  Padding every small
    decision to the configured world/candidate/local limits made one
    recurrent 8x64 learner batch retain hundreds of unnecessarily large
    attention graphs.  Use the largest active shape in the current batch while
    preserving the same feature/vocabulary ABI and masks.
    """

    if not snapshots:
        raise ValueError("at least one encoded decision snapshot is required")
    if validate:
        for snapshot in snapshots:
            snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type == "cuda" or len(snapshots) >= 8:
        return _collate_encoded_snapshots_scatter(
            snapshots,
            cfg=expected_config,
            dev=dev,
        )
    cfg = expected_config
    batch_size = len(snapshots)
    world_capacity = max(snapshot.world.token_count for snapshot in snapshots)
    candidate_capacity = max(snapshot.candidate_count for snapshot in snapshots)
    local_capacity = max(
        1,
        max(
            (int(count) for snapshot in snapshots for count in np.diff(snapshot.local_offsets)),
            default=0,
        ),
    )
    world_features = np.zeros((batch_size, world_capacity, cfg.feature_dim), dtype=np.float32)
    world_mask = np.zeros((batch_size, world_capacity), dtype=np.bool_)
    world_ids = np.zeros((batch_size, world_capacity, _WORLD_ID_WIDTH), dtype=np.int64)
    candidate_features = np.zeros((batch_size, candidate_capacity, cfg.feature_dim), dtype=np.float32)
    candidate_ids = np.zeros((batch_size, candidate_capacity, _CANDIDATE_ID_WIDTH), dtype=np.int64)
    action_mask = np.zeros((batch_size, candidate_capacity), dtype=np.bool_)
    local_features = np.zeros(
        (
            batch_size,
            candidate_capacity,
            local_capacity,
            cfg.feature_dim,
        ),
        dtype=np.float32,
    )
    local_mask = np.zeros(
        (batch_size, candidate_capacity, local_capacity),
        dtype=np.bool_,
    )
    local_ids = np.zeros(
        (
            batch_size,
            candidate_capacity,
            local_capacity,
            _LOCAL_ID_WIDTH,
        ),
        dtype=np.int64,
    )
    domain_ids = np.zeros((batch_size,), dtype=np.int64)
    macro_economic_surface_ids = np.zeros((batch_size,), dtype=np.int64)

    for batch_index, snapshot in enumerate(snapshots):
        world_count = snapshot.world.token_count
        world_mask[batch_index, :world_count] = True
        world_ids[batch_index, :world_count] = snapshot.world.ids
        world_rows = np.repeat(
            np.arange(world_count),
            np.diff(snapshot.world.feature_indptr).astype(np.int64),
        )
        world_features[
            batch_index,
            world_rows,
            snapshot.world.feature_indices,
        ] = snapshot.world.feature_values

        candidate_count = snapshot.candidate_count
        candidate_ids[batch_index, :candidate_count] = snapshot.candidates.ids
        action_mask[batch_index, :candidate_count] = snapshot.action_mask
        candidate_rows = np.repeat(
            np.arange(candidate_count),
            np.diff(snapshot.candidates.feature_indptr).astype(np.int64),
        )
        candidate_features[
            batch_index,
            candidate_rows,
            snapshot.candidates.feature_indices,
        ] = snapshot.candidates.feature_values

        local_counts = np.diff(snapshot.local_offsets).astype(np.int64)
        local_candidate_rows = np.repeat(np.arange(candidate_count), local_counts)
        local_positions = (
            np.concatenate([np.arange(count, dtype=np.int64) for count in local_counts])
            if snapshot.locals.token_count
            else np.empty((0,), dtype=np.int64)
        )
        if snapshot.locals.token_count:
            local_mask[
                batch_index,
                local_candidate_rows,
                local_positions,
            ] = True
            local_ids[
                batch_index,
                local_candidate_rows,
                local_positions,
            ] = snapshot.locals.ids
            local_token_rows = np.repeat(
                np.arange(snapshot.locals.token_count),
                np.diff(snapshot.locals.feature_indptr).astype(np.int64),
            )
            local_features[
                batch_index,
                local_candidate_rows[local_token_rows],
                local_positions[local_token_rows],
                snapshot.locals.feature_indices,
            ] = snapshot.locals.feature_values
        domain_ids[batch_index] = snapshot.domain_id
        macro_economic_surface_ids[batch_index] = (
            snapshot.macro_economic_surface_id
        )

    dev = torch.device(device) if device is not None else torch.device("cpu")
    world_ids_tensor = _tensor(world_ids, device=dev)
    candidate_ids_tensor = _tensor(candidate_ids, device=dev)
    local_ids_tensor = _tensor(local_ids, device=dev)
    return GroundedCandidateBatch(
        world=WorldTokenBatch(
            features=_tensor(world_features, device=dev),
            mask=_tensor(world_mask, device=dev),
            type_ids=world_ids_tensor[:, :, 0],
            role_ids=world_ids_tensor[:, :, 1],
            owner_ids=world_ids_tensor[:, :, 2],
            entity_ids=world_ids_tensor[:, :, 3],
            entity_aux_ids=world_ids_tensor[:, :, 4],
            definition_binding_ids=world_ids_tensor[:, :, 5],
            relation_binding_ids=world_ids_tensor[:, :, 6],
            zone_ids=world_ids_tensor[:, :, 7],
            order_ids=world_ids_tensor[:, :, 8],
        ),
        candidates=CandidateTokenBatch(
            features=_tensor(candidate_features, device=dev),
            type_ids=candidate_ids_tensor[:, :, 0],
            role_ids=candidate_ids_tensor[:, :, 1],
            owner_ids=candidate_ids_tensor[:, :, 2],
            entity_ids=candidate_ids_tensor[:, :, 3],
            entity_aux_ids=candidate_ids_tensor[:, :, 4],
            definition_binding_ids=candidate_ids_tensor[:, :, 5],
            relation_binding_ids=candidate_ids_tensor[:, :, 6],
            zone_ids=candidate_ids_tensor[:, :, 7],
            target_owner_ids=candidate_ids_tensor[:, :, 8],
            target_entity_ids=candidate_ids_tensor[:, :, 9],
            target_entity_aux_ids=candidate_ids_tensor[:, :, 10],
            target_definition_binding_ids=candidate_ids_tensor[:, :, 11],
            target_relation_binding_ids=candidate_ids_tensor[:, :, 12],
            policy_branch_ids=candidate_ids_tensor[:, :, 13],
            local_features=_tensor(local_features, device=dev),
            local_mask=_tensor(local_mask, device=dev),
            local_type_ids=local_ids_tensor[:, :, :, 0],
            local_role_ids=local_ids_tensor[:, :, :, 1],
            local_owner_ids=local_ids_tensor[:, :, :, 2],
            local_entity_ids=local_ids_tensor[:, :, :, 3],
            local_entity_aux_ids=local_ids_tensor[:, :, :, 4],
            local_definition_binding_ids=local_ids_tensor[:, :, :, 5],
            local_relation_binding_ids=local_ids_tensor[:, :, :, 6],
            local_zone_ids=local_ids_tensor[:, :, :, 7],
            local_order_ids=local_ids_tensor[:, :, :, 8],
            action_mask=_tensor(action_mask, device=dev),
        ),
        domain_ids=_tensor(domain_ids, device=dev),
        macro_economic_surface_ids=_tensor(
            macro_economic_surface_ids,
            device=dev,
        ),
    )


def collate_encoded_worlds(
    snapshots: tuple[EncodedWorldSnapshot, ...],
    *,
    expected_config: GroundedEncodingConfig,
    expected_fingerprint: str,
    device: torch.device | str | None,
    validate: bool = True,
) -> tuple[WorldTokenBatch, torch.Tensor]:
    """Materialize world tensors trimmed to the active world capacity of this batch."""

    if not snapshots:
        raise ValueError("at least one encoded world snapshot is required")
    if validate:
        for snapshot in snapshots:
            snapshot.validate(
                expected_config=expected_config,
                expected_fingerprint=expected_fingerprint,
            )
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type == "cuda" or len(snapshots) >= 8:
        return _collate_encoded_worlds_scatter(
            snapshots,
            cfg=expected_config,
            dev=dev,
        )
    cfg = expected_config
    batch_size = len(snapshots)
    world_capacity = max(snapshot.world.token_count for snapshot in snapshots)
    world_features = np.zeros((batch_size, world_capacity, cfg.feature_dim), dtype=np.float32)
    world_mask = np.zeros((batch_size, world_capacity), dtype=np.bool_)
    world_ids = np.zeros((batch_size, world_capacity, _WORLD_ID_WIDTH), dtype=np.int64)
    domain_ids = np.zeros((batch_size,), dtype=np.int64)

    for batch_index, snapshot in enumerate(snapshots):
        world_count = snapshot.world.token_count
        world_mask[batch_index, :world_count] = True
        world_ids[batch_index, :world_count] = snapshot.world.ids
        world_rows = np.repeat(
            np.arange(world_count),
            np.diff(snapshot.world.feature_indptr).astype(np.int64),
        )
        world_features[
            batch_index,
            world_rows,
            snapshot.world.feature_indices,
        ] = snapshot.world.feature_values
        domain_ids[batch_index] = snapshot.domain_id

    dev = torch.device(device) if device is not None else torch.device("cpu")
    world_ids_tensor = _tensor(world_ids, device=dev)
    world_batch = WorldTokenBatch(
        features=_tensor(world_features, device=dev),
        mask=_tensor(world_mask, device=dev),
        type_ids=world_ids_tensor[:, :, 0],
        role_ids=world_ids_tensor[:, :, 1],
        owner_ids=world_ids_tensor[:, :, 2],
        entity_ids=world_ids_tensor[:, :, 3],
        entity_aux_ids=world_ids_tensor[:, :, 4],
        definition_binding_ids=world_ids_tensor[:, :, 5],
        relation_binding_ids=world_ids_tensor[:, :, 6],
        zone_ids=world_ids_tensor[:, :, 7],
        order_ids=world_ids_tensor[:, :, 8],
    )
    return world_batch, _tensor(domain_ids, device=dev)


__all__ = [
    "ENCODED_DECISION_SNAPSHOT_VERSION",
    "EncodedDecisionSnapshot",
    "EncodedWorldSnapshot",
    "GroundedEncodingConfig",
    "SparseTokenTable",
    "collate_encoded_snapshots",
    "collate_encoded_worlds",
    "sparse_token_table",
]
