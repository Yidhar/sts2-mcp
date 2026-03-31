"""Shared encoder and task heads for STS2 offline supervised training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PooledEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embedding(ids)
        mask_f = mask.float()
        if weights is None:
            weights = mask_f
        else:
            weights = weights * mask_f
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * weights.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class OfflineStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        scalar_dim: int,
        card_vocab_size: int,
        relic_vocab_size: int,
        monster_vocab_size: int,
        room_type_vocab_size: int,
        point_type_vocab_size: int,
        room_model_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.card_pool = PooledEmbedding(card_vocab_size, embed_dim)
        self.relic_pool = PooledEmbedding(relic_vocab_size, embed_dim)
        self.monster_pool = PooledEmbedding(monster_vocab_size, embed_dim)
        self.room_type_embedding = nn.Embedding(room_type_vocab_size, embed_dim, padding_idx=0)
        self.point_type_embedding = nn.Embedding(point_type_vocab_size, embed_dim, padding_idx=0)
        self.room_model_embedding = nn.Embedding(room_model_vocab_size, embed_dim, padding_idx=0)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_dim = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        scalar_out = self.scalar_net(batch["scalars"])
        deck_weights = batch["deck_counts"] + 0.25 * batch["deck_upgraded_counts"] + batch["deck_mask"].float()
        deck_out = self.card_pool(batch["deck_ids"], batch["deck_mask"], deck_weights)
        relic_out = self.relic_pool(batch["relic_ids"], batch["relic_mask"])
        monster_out = self.monster_pool(batch["monster_ids"], batch["monster_mask"])
        room_type_out = self.room_type_embedding(batch["room_type"])
        point_type_out = self.point_type_embedding(batch["map_point_type"])
        room_model_out = self.room_model_embedding(batch["room_model_id"])
        return self.fusion(
            torch.cat(
                [scalar_out, deck_out, relic_out, monster_out, room_type_out, point_type_out, room_model_out],
                dim=-1,
            )
        )


class CandidateChoiceHead(nn.Module):
    def __init__(self, *, state_dim: int, candidate_vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.candidate_embedding = nn.Embedding(candidate_vocab_size, embed_dim, padding_idx=0)
        self.candidate_numeric = nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU())
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_emb = self.candidate_embedding(batch["candidate_ids"])
        candidate_num = self.candidate_numeric(batch["candidate_upgrade_levels"].unsqueeze(-1))
        candidate_repr = candidate_emb + candidate_num
        state_expanded = state.unsqueeze(1).expand(-1, candidate_repr.shape[1], -1)
        state_proj = self.state_proj(state_expanded)
        logits = self.scorer(torch.cat([state_expanded, candidate_repr, state_proj * candidate_repr], dim=-1)).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"], -1e9)


class ClassificationHead(nn.Module):
    def __init__(self, *, state_dim: int, num_labels: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, num_labels),
        )

    def forward(self, state: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(state)


class CardSetSelectionHead(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        card_vocab_size: int,
        slot_count: int,
        embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.slot_count = slot_count
        self.card_embedding = nn.Embedding(card_vocab_size, embed_dim, padding_idx=0)
        self.candidate_numeric = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.slot_embeddings = nn.Parameter(torch.randn(slot_count, state_dim) * 0.02)
        self.state_proj = nn.Linear(state_dim, embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(state_dim + embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_repr = self.card_embedding(batch["candidate_ids"]) + self.candidate_numeric(batch["candidate_features"])
        slot_state = state.unsqueeze(1) + self.slot_embeddings.unsqueeze(0)
        slot_state_expanded = slot_state.unsqueeze(2).expand(-1, -1, candidate_repr.shape[1], -1)
        candidate_expanded = candidate_repr.unsqueeze(1).expand(-1, self.slot_count, -1, -1)
        state_proj = self.state_proj(slot_state_expanded)
        logits = self.scorer(
            torch.cat(
                [
                    slot_state_expanded,
                    candidate_expanded,
                    state_proj * candidate_expanded,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"].unsqueeze(1), -1e9)


class MultiTaskOfflineModel(nn.Module):
    def __init__(
        self,
        *,
        state_encoder: OfflineStateEncoder,
        head_builders: dict[str, dict[str, Any]],
        embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.state_encoder = state_encoder
        self.task_families: dict[str, str] = {}
        self.heads = nn.ModuleDict()

        for task, spec in head_builders.items():
            family = spec["family"]
            self.task_families[task] = family
            if family == "candidate":
                self.heads[task] = CandidateChoiceHead(
                    state_dim=state_encoder.output_dim,
                    candidate_vocab_size=spec["candidate_vocab_size"],
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                )
            elif family == "classification":
                self.heads[task] = ClassificationHead(
                    state_dim=state_encoder.output_dim,
                    num_labels=spec["num_labels"],
                )
            elif family == "route":
                self.heads[task] = ClassificationHead(
                    state_dim=state_encoder.output_dim,
                    num_labels=spec["num_labels"],
                )
            elif family == "cardset":
                self.heads[task] = CardSetSelectionHead(
                    state_dim=state_encoder.output_dim,
                    card_vocab_size=spec["card_vocab_size"],
                    slot_count=spec["slot_count"],
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                )
            else:
                raise ValueError(f"Unsupported head family: {family}")

    def forward(self, task: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = self.state_encoder(batch)
        return self.heads[task](state, batch)
