"""Compatibility facade for token-world MuZero memory modules.

Implementation classes live in focused internal modules.  Re-exporting the
original API here preserves imports and allows old full-object pickles to resolve
legacy module attributes; production checkpoints use state dictionaries.
"""

from __future__ import annotations

from muzero.sts2_env.semantic_rollout import SEMANTIC_ROLLOUT_SIZE
from sts2_env.attention_blocks import (
    CrossAttentionBlock,
    EntityPooling,
    RelationBias,
    TransformerEncoderBlock,
)
from sts2_env.objective_heads import (
    HEAD_HP_PRESERVATION,
    HEAD_SURVIVAL,
    NUM_OBJECTIVE_HEADS,
    scalarize_objective_components_torch,
)
from sts2_env.observation_v2 import MAX_ACTIONS, NUM_DOMAINS, NUM_PHASES
from sts2_env.observation_v3 import (
    ENTITY_HASH_BUCKETS,
    MAX_CANDIDATE_LOCAL_TOKENS,
    MAX_ORDER_ID,
    MAX_OWNER_ID,
    MAX_ROLE_ID,
    MAX_WORLD_TOKENS,
    MAX_ZONE_ID,
    NUM_TOKEN_TYPES,
    POWER_ID_BUCKETS,
    TOKEN_FEAT_DIM,
    TOKEN_NUMERIC_DIM,
    TOKEN_ROLE_TO_ID,
    TOKEN_TEXT_DIM,
    TOKEN_TYPE_TO_ID,
    TOKEN_ZONE_TO_ID,
)

from ._token_memory_dynamics import (
    TokenDynamicsNetwork,
    TokenLatentProjector,
    TokenTransitionSurfaceHead,
)
from ._token_memory_encoder import EntityTokenEmbedder, TokenMemoryEncoder
from ._token_memory_future import TokenFutureWorldBankHead
from ._token_memory_prediction import TokenPredictionNetwork
from ._token_memory_shared import (
    GLOBAL_MEMORY_BANK_INDEX,
    GLOBAL_MEMORY_BANK_NAME,
    MEMORY_BANK_NAMES,
    MEMORY_SLOT_LAYOUT_LEGACY,
    MEMORY_SLOT_LAYOUT_PASS_LARGE_V1,
    MEMORY_SLOT_LAYOUT_QUOTA_V1,
    RISK_OBJECTIVE_HEAD_INDICES,
    VALID_MEMORY_SLOT_LAYOUTS,
    WORLD_BANK_NAMES,
    WORLD_BANK_ROLE_IDS,
    WORLD_BANK_ZONE_IDS,
    TokenMemoryEncoderOutput,
    build_memory_slot_bank_ids,
    build_zone_transport_prior,
    infer_token_decision_domain,
    normalize_memory_slot_layout,
)

# Keep the historical symbol identity used by diagnostics and any full-object
# pickle produced outside the canonical state-dict checkpoint path.  The
# implementation modules are deliberately private; the public symbol home
# stays ``muzero.sts2_env.token_memory``.
for _public_symbol in (
    EntityTokenEmbedder,
    TokenDynamicsNetwork,
    TokenFutureWorldBankHead,
    TokenLatentProjector,
    TokenMemoryEncoder,
    TokenMemoryEncoderOutput,
    TokenPredictionNetwork,
    TokenTransitionSurfaceHead,
    build_memory_slot_bank_ids,
    build_zone_transport_prior,
    infer_token_decision_domain,
    normalize_memory_slot_layout,
):
    _public_symbol.__module__ = __name__
del _public_symbol

__all__ = [
    'ENTITY_HASH_BUCKETS',
    'GLOBAL_MEMORY_BANK_INDEX',
    'GLOBAL_MEMORY_BANK_NAME',
    'HEAD_HP_PRESERVATION',
    'HEAD_SURVIVAL',
    'MAX_ACTIONS',
    'MAX_CANDIDATE_LOCAL_TOKENS',
    'MAX_ORDER_ID',
    'MAX_OWNER_ID',
    'MAX_ROLE_ID',
    'MAX_WORLD_TOKENS',
    'MAX_ZONE_ID',
    'MEMORY_BANK_NAMES',
    'MEMORY_SLOT_LAYOUT_LEGACY',
    'MEMORY_SLOT_LAYOUT_PASS_LARGE_V1',
    'MEMORY_SLOT_LAYOUT_QUOTA_V1',
    'NUM_DOMAINS',
    'NUM_OBJECTIVE_HEADS',
    'NUM_PHASES',
    'NUM_TOKEN_TYPES',
    'POWER_ID_BUCKETS',
    'RISK_OBJECTIVE_HEAD_INDICES',
    'SEMANTIC_ROLLOUT_SIZE',
    'TOKEN_FEAT_DIM',
    'TOKEN_NUMERIC_DIM',
    'TOKEN_ROLE_TO_ID',
    'TOKEN_TEXT_DIM',
    'TOKEN_TYPE_TO_ID',
    'TOKEN_ZONE_TO_ID',
    'VALID_MEMORY_SLOT_LAYOUTS',
    'WORLD_BANK_NAMES',
    'WORLD_BANK_ROLE_IDS',
    'WORLD_BANK_ZONE_IDS',
    'CrossAttentionBlock',
    'EntityPooling',
    'EntityTokenEmbedder',
    'RelationBias',
    'TokenDynamicsNetwork',
    'TokenFutureWorldBankHead',
    'TokenLatentProjector',
    'TokenMemoryEncoder',
    'TokenMemoryEncoderOutput',
    'TokenPredictionNetwork',
    'TokenTransitionSurfaceHead',
    'TransformerEncoderBlock',
    'build_memory_slot_bank_ids',
    'build_zone_transport_prior',
    'infer_token_decision_domain',
    'normalize_memory_slot_layout',
    'scalarize_objective_components_torch',
]
