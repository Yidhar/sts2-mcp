"""Archived AuxMaskablePPO + STS2OmniAttentionPolicy training path.

Superseded by the MuZero / token_memory search-free path under ``muzero/``
(active development). Kept for reference and as a control baseline: this
PPO path reached ~0.68 resolved win-rate / 0.375 boss win on curated
mid-run combat snapshot decks, but 0/100 on starter full-run — the same
passive-collapse failure mode as MuZero, which is why the defect is
attributed to the shared env/reward/training regimen rather than the
learning algorithm. See ``_analysis_rl_diagnosis_20260602.md``.

Note: ``sts2_env/attention_blocks.py`` and ``sts2_env/aux_targets.py``
remain in the active ``sts2_env`` package because the MuZero token-memory
encoder (``muzero/sts2_env/token_memory.py``) imports them. The modules
archived here import those shared building blocks via absolute
``sts2_env.*`` imports.
"""
