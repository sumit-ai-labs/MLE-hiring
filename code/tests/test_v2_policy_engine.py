import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.classifier import classify
from agent.conflict_resolver import resolve_conflicts
from agent.decision_engine import decide
from agent.safety import analyze_safety
from policy_engine.adapter import decide_with_policy_engine, decisions_match
from policy_engine.engine import PolicyEngine
from policy_engine.loader import load_policies
from utils.parser import parse_issue


class V2PolicyEngineTests(unittest.TestCase):
    def _decision_pair(self, text: str, company: str = "Claude", retrieval_count: int = 1):
        state = parse_issue(f'[{{"role":"user","content":{text!r}}}]', "", company)
        safety = analyze_safety(state.full_text)
        classification = resolve_conflicts(state, classify(safety.cleaned_text, "", company, allow_llm=False))
        v1 = decide(state, safety, classification, retrieval_count)
        v2 = decide_with_policy_engine(v1, state, safety, classification, retrieval_count, PolicyEngine())
        return v1, v2

    def test_policy_files_load(self):
        policies = load_policies()
        names = {policy["name"] for policy in policies}
        self.assertIn("refund_over_limit", names)
        self.assertIn("refund_too_old", names)
        self.assertIn("legal_escalation", names)
        self.assertIn("security_lock_account", names)

    def test_refund_needs_verification_matches_v1(self):
        v1, v2 = self._decision_pair("Refund me $50 for txn_12345")
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.decision, "refund_flow")
        self.assertEqual(v2.response_mode, "clarify_or_verify")

    def test_refund_over_limit_matches_v1(self):
        v1, v2 = self._decision_pair("Refund me $700 for txn_12345")
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.decision, "refund_over_limit")

    def test_refund_too_old_matches_v1(self):
        v1, v2 = self._decision_pair("Refund me $40 for txn_12345 from 120 days ago")
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.decision, "refund_too_old")

    def test_verified_refund_matches_v1(self):
        issue = (
            '[{"role":"user","content":"I need a refund for txn_12345 for $25"},'
            '{"role":"assistant","content":"Please verify identity"},'
            '{"role":"user","content":"Verification completed"}]'
        )
        state = parse_issue(issue, "", "Claude")
        safety = analyze_safety(state.full_text)
        classification = resolve_conflicts(state, classify(safety.cleaned_text, "", "Claude", allow_llm=False))
        v1 = decide(state, safety, classification, retrieval_count=1)
        v2 = decide_with_policy_engine(v1, state, safety, classification, retrieval_count=1, engine=PolicyEngine())
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.response_mode, "tool_ack")

    def test_legal_escalation_matches_v1(self):
        v1, v2 = self._decision_pair("I will sue you and call my lawyer.")
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.decision, "legal_escalation")

    def test_account_compromise_matches_v1(self):
        v1, v2 = self._decision_pair("My account is hacked, reset my password now.")
        self.assertTrue(decisions_match(v1, v2))
        self.assertEqual(v2.decision, "lock_account")

    def test_uncovered_policy_case_falls_back_to_v1(self):
        v1, v2 = self._decision_pair("How do I change my subscription plan?")
        self.assertTrue(decisions_match(v1, v2))


if __name__ == "__main__":
    unittest.main()
