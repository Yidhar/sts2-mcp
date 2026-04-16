from __future__ import annotations

import ast
from pathlib import Path
import unittest


RL_AGENT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = RL_AGENT_ROOT / "sts2_env" / "omni_attention_policy.py"


class OmniAttentionPolicyStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = POLICY_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_policy_defines_shared_numeric_and_text_trunks(self) -> None:
        self.assertIn("self.shared_text_trunk", self.source)
        self.assertIn("self.shared_numeric_trunk", self.source)

        build_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_mlp_extractor"
        )
        assigns_text = False
        assigns_numeric = False
        for node in ast.walk(build_fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if target.attr == "shared_text_trunk":
                    assigns_text = True
                if target.attr == "shared_numeric_trunk":
                    assigns_numeric = True
        self.assertTrue(assigns_text)
        self.assertTrue(assigns_numeric)

    def test_all_entity_embedders_disable_internal_numeric_and_text_proj(self) -> None:
        build_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_mlp_extractor"
        )
        disabled_text_calls = 0
        disabled_numeric_calls = 0
        for node in ast.walk(build_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "EntityTokenEmbedder":
                continue
            for keyword in node.keywords:
                if keyword.arg == "use_internal_text_proj" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    disabled_text_calls += 1
                if keyword.arg == "use_internal_numeric_proj" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    disabled_numeric_calls += 1
        self.assertEqual(disabled_text_calls, 3)
        self.assertEqual(disabled_numeric_calls, 3)

    def test_forward_policy_uses_shared_projection_reuse_helper_for_numeric_and_text(self) -> None:
        forward_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_policy"
        )
        helper_calls = 0
        helper_receivers: set[str] = set()
        for node in ast.walk(forward_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "_project_shared_modal_trunk_with_reuse":
                continue
            helper_calls += 1
            if node.args and isinstance(node.args[0], ast.Attribute):
                helper_receivers.add(node.args[0].attr)
        self.assertEqual(helper_calls, 2)
        self.assertEqual(helper_receivers, {"shared_numeric_trunk", "shared_text_trunk"})

    def test_forward_policy_passes_projected_numeric_and_text_into_all_three_embedders(self) -> None:
        forward_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_policy"
        )
        required_embedder_calls = {"world_embedder", "query_embedder", "local_embedder"}
        seen_text: set[str] = set()
        seen_numeric: set[str] = set()
        for node in ast.walk(forward_fn):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            embedder_name = node.func.attr
            if embedder_name not in required_embedder_calls:
                continue
            if any(keyword.arg == "projected_text" for keyword in node.keywords):
                seen_text.add(embedder_name)
            if any(keyword.arg == "projected_numeric" for keyword in node.keywords):
                seen_numeric.add(embedder_name)
        self.assertEqual(seen_text, required_embedder_calls)
        self.assertEqual(seen_numeric, required_embedder_calls)

    def test_policy_bridges_query_into_local_tokens_before_world_cross_attention(self) -> None:
        self.assertIn("self.query_local_relation_bias", self.source)
        self.assertIn("self.query_local_bridge", self.source)

        build_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_mlp_extractor"
        )
        assigned_attrs: set[str] = set()
        for node in ast.walk(build_fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    assigned_attrs.add(target.attr)
        self.assertIn("query_local_relation_bias", assigned_attrs)
        self.assertIn("query_local_bridge", assigned_attrs)

        forward_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_policy"
        )
        bridge_called = False
        local_bias_called = False
        for node in ast.walk(forward_fn):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "query_local_bridge":
                bridge_called = True
            if node.func.attr == "query_local_relation_bias":
                local_bias_called = True
        self.assertTrue(local_bias_called)
        self.assertTrue(bridge_called)

    def test_policy_groups_world_cross_attention_into_banks_with_topk_routing(self) -> None:
        self.assertIn('ATTENTION_ARCHITECTURE_VERSION = "omni_attention_v1_frozen"', self.source)
        self.assertIn("WORLD_BANK_NAMES", self.source)
        self.assertIn("self.world_bank_poolers", self.source)
        self.assertIn("self.world_bank_router_q", self.source)
        self.assertIn("self.world_bank_router_k", self.source)
        self.assertIn("self.world_bank_cross_blocks", self.source)
        self.assertIn("def _build_world_bank_masks", self.source)
        self.assertIn("def _compute_world_bank_routing", self.source)
        self.assertIn("def _apply_banked_world_cross_attention", self.source)
        self.assertIn("def forward_world_bank_routing", self.source)
        self.assertIn("def architecture_spec", self.source)

        build_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_mlp_extractor"
        )
        assigned_attrs: set[str] = set()
        for node in ast.walk(build_fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    assigned_attrs.add(target.attr)
        self.assertIn("world_bank_poolers", assigned_attrs)
        self.assertIn("world_bank_router_q", assigned_attrs)
        self.assertIn("world_bank_router_k", assigned_attrs)
        self.assertIn("world_bank_cross_blocks", assigned_attrs)

        forward_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_forward_policy"
        )
        helper_calls: set[str] = set()
        for node in ast.walk(forward_fn):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            helper_calls.add(node.func.attr)
        self.assertIn("_build_world_bank_masks", helper_calls)
        self.assertIn("_compute_world_bank_summaries", helper_calls)
        self.assertIn("_build_world_bank_biases", helper_calls)
        self.assertIn("_apply_banked_world_cross_attention", helper_calls)

        routing_fn = next(
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_world_bank_routing"
        )
        uses_topk = False
        for node in ast.walk(routing_fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "topk":
                uses_topk = True
                break
        self.assertTrue(uses_topk)


if __name__ == "__main__":
    unittest.main()
