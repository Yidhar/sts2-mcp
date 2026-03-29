"""Pre-warm the text embedding cache with common game texts.

Encodes canonical text patterns so the first training run
doesn't pay the full encode cost for every new text.

Usage:
    python warm_text_cache.py
    python warm_text_cache.py --session-file path/to/session.json
"""

import argparse
import time

from sts2_env.text_encoder import get_text_encoder


# Common canonical text patterns to pre-cache
_WARMUP_TEXTS = [
    # Actions
    "动作｜结束回合",
    "动作｜继续",
    "动作｜跳过",
    "动作｜跳过卡牌奖励",
    "动作｜离开商店",
    "动作｜确认选择",
    "动作｜取消选择",
    # Phases
    "阶段：combat",
    "阶段：map",
    "阶段：event",
    "阶段：reward",
    "阶段：card_reward",
    "阶段：rest_site",
    "阶段：shop",
    "阶段：deck_upgrade",
    "阶段：treasure",
    "阶段：card_selection",
    # Decision contexts
    "营火｜选择休息或锻造",
    "奖励选择｜可领取0项奖励",
    "奖励选择｜可领取1项奖励",
    "奖励选择｜可领取2项奖励",
    "奖励选择｜可领取3项奖励",
    "卡牌奖励｜3张卡牌可选",
    # Map
    "动作｜前往｜Monster",
    "动作｜前往｜Elite",
    "动作｜前往｜Rest",
    "动作｜前往｜Event",
    "动作｜前往｜Merchant",
    "动作｜前往｜Treasure",
    "动作｜前往｜Boss",
    # Rest site
    "动作｜营火｜休息",
    "动作｜营火｜锻造",
    # Rewards
    "动作｜领取奖励｜查看卡牌",
    "动作｜领取奖励｜gold",
]


def warm_from_bridge(session_file=None):
    """Collect live canonical texts from bridge state."""
    texts = []
    try:
        from sts2_env.bridge_client import BridgeClient
        client = BridgeClient(session_path=session_file)
        state = client.get_state()

        # Relics
        for relic in (state.get("player", {}).get("relics") or []):
            if isinstance(relic, dict):
                ct = relic.get("canonical_text", "")
                if ct: texts.append(ct)

        # Available actions
        for action in state.get("available_actions", []):
            ct = action.get("canonical_text", "")
            if ct: texts.append(ct)

        print(f"[warm] Collected {len(texts)} texts from bridge")
    except Exception as e:
        print(f"[warm] Bridge not available: {e}")

    return texts


def main():
    parser = argparse.ArgumentParser(description="Warm text embedding cache")
    parser.add_argument("--session-file", type=str, default=None)
    args = parser.parse_args()

    encoder = get_text_encoder().ensure_ready()
    print(f"[warm] Model: {encoder._model_name}")
    print(f"[warm] Cache before: {encoder.cache_size} entries")

    all_texts = list(_WARMUP_TEXTS)
    all_texts.extend(warm_from_bridge(args.session_file))

    # Deduplicate
    all_texts = list(set(t for t in all_texts if t))
    print(f"[warm] Encoding {len(all_texts)} unique texts...")

    t0 = time.time()
    encoder.encode_batch(all_texts)
    elapsed = time.time() - t0

    encoder.save_cache()
    print(f"[warm] Done in {elapsed:.1f}s")
    print(f"[warm] Cache after: {encoder.cache_size} entries")


if __name__ == "__main__":
    main()
