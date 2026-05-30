import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.classifier import classify
from agent.conflict_resolver import resolve_conflicts
from agent.decision_engine import decide
from agent.planner import plan_tools
from agent.safety import analyze_safety
from config import ENABLE_V2_STATE_MACHINE, ENABLE_V2_STRUCTURED_MEMORY
from memory.memory_extractor import extract_memory
from memory.memory_store import MemoryStore
from state_machine.state_machine import State, TicketState, build_shadow_lifecycle
from utils.language import detect_language
from utils.parser import parse_issue
from utils.pii import detect_pii


class V2StateMachineMemoryTests(unittest.TestCase):
    def _context(self, text: str, company: str = "Claude"):
        state = parse_issue(f'[{{"role":"user","content":{text!r}}}]', "", company)
        safety = analyze_safety(state.full_text)
        classification = resolve_conflicts(state, classify(safety.cleaned_text, "", company, allow_llm=False))
        decision = decide(state, safety, classification, retrieval_count=1)
        actions = plan_tools(decision, state)
        pii = detect_pii(state.full_text)
        language = detect_language(state.latest_user_text)
        return state, safety, classification, decision, actions, pii, language

    def test_state_transition_lifecycle(self):
        ticket = TicketState(ticket_id=1)
        self.assertTrue(ticket.transition(State.SAFETY_CHECKED, "stage:safety", "safety complete"))
        self.assertTrue(ticket.transition(State.PII_CHECKED, "stage:pii", "pii complete"))
        self.assertEqual(ticket.current_state, State.PII_CHECKED)
        self.assertEqual(ticket.history[-1].timestamp, "t0002")

    def test_invalid_transition_blocked(self):
        ticket = TicketState(ticket_id=2)
        self.assertFalse(ticket.transition(State.RESPONDED, "stage:responded", "skipped stages"))
        self.assertEqual(ticket.current_state, State.NEW)
        self.assertFalse(ticket.history[-1].allowed)

    def test_refund_before_verification_denied(self):
        ticket = build_shadow_lifecycle(
            3,
            identity_verified=False,
            security_signal=None,
            risk_level="medium",
            actions=[{"action": "issue_refund"}],
        )
        self.assertEqual(ticket.refund_eligibility, "denied")
        self.assertTrue(any(record.state_after == "REFUND_DENIED" for record in ticket.history))

    def test_account_lock_transition(self):
        ticket = build_shadow_lifecycle(
            4,
            identity_verified=False,
            security_signal="account_compromise",
            risk_level="critical",
            actions=[{"action": "lock_account"}],
        )
        self.assertIn("account_compromise", ticket.security_signals)
        self.assertTrue(any(record.state_after == "LOCKED" for record in ticket.history))

    def test_structured_memory_extraction(self):
        state, safety, classification, decision, actions, pii, language = self._context("Refund me $50 for txn_12345")
        memory = extract_memory(state, classification, safety, pii, language, decision, actions)
        self.assertTrue(memory.validate())
        self.assertEqual(memory.transaction_id, "txn_12345")
        self.assertEqual(memory.refund_signal, "requested")
        self.assertEqual(memory.request_type, "refund")
        self.assertEqual(memory.verification_status, "pending")

    def test_multilingual_memory(self):
        state, safety, classification, decision, actions, pii, language = self._context("Necesito un reembolso de $20 para txn_99999")
        memory = extract_memory(state, classification, safety, pii, language, decision, actions)
        self.assertTrue(memory.validate())
        self.assertIsNotNone(memory.language)
        self.assertIn("reembolso", memory.current_issue.lower())

    def test_missing_field_handling(self):
        state, safety, classification, decision, actions, pii, language = self._context("Can you help with this?")
        memory = extract_memory(state, classification, safety, pii, language, decision, actions)
        self.assertTrue(memory.validate())
        self.assertIsNone(memory.transaction_id)
        self.assertIsNone(memory.refund_signal)

    def test_deterministic_memory(self):
        first = self._context("Refund me $50 for txn_12345")
        second = self._context("Refund me $50 for txn_12345")
        mem1 = extract_memory(first[0], first[2], first[1], first[5], first[6], first[3], first[4]).to_dict()
        mem2 = extract_memory(second[0], second[2], second[1], second[5], second[6], second[3], second[4]).to_dict()
        self.assertEqual(mem1, mem2)

    def test_v1_vs_v2_parity_shadow_mode(self):
        state, safety, classification, decision, actions, pii, language = self._context("My account is hacked")
        memory = extract_memory(state, classification, safety, pii, language, decision, actions)
        ticket = build_shadow_lifecycle(
            9,
            identity_verified=state.identity_verified,
            security_signal=memory.security_signal,
            risk_level=decision.risk_level,
            actions=actions,
        )
        self.assertEqual(decision.decision, "lock_account")
        self.assertTrue(memory.validate())
        self.assertTrue(ticket.history)

    def test_rollback_safety_flags_default_off(self):
        self.assertFalse(ENABLE_V2_STATE_MACHINE)
        self.assertFalse(ENABLE_V2_STRUCTURED_MEMORY)

    def test_memory_store_validates_schema(self):
        state, safety, classification, decision, actions, pii, language = self._context("Refund me $50 for txn_12345")
        memory = extract_memory(state, classification, safety, pii, language, decision, actions)
        store = MemoryStore()
        store.upsert(11, memory)
        self.assertEqual(store.get(11), memory)


if __name__ == "__main__":
    unittest.main()
