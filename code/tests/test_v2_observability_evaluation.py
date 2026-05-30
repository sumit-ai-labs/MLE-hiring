import json
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import ENABLE_V2_EVALUATION, ENABLE_V2_OBSERVABILITY
from evaluation_v2.benchmark_runner import compare_rows, summarize_rows
from evaluation_v2.confidence_metrics import calibration_error, confidence_distribution, uncertainty_handling
from evaluation_v2.retrieval_metrics import poisoning_resistance, precision_at_k, retrieval_coverage
from evaluation_v2.safety_metrics import prompt_injection_catch_rate, security_incident_handling
from evaluation_v2.tool_metrics import tool_correctness_rate, tool_ordering_correct
from observability.event_logger import sanitize_text
from observability.performance_metrics import PerformanceMetrics
from observability.trace_exporter import human_report, traces_to_json
from observability.trace_manager import TraceManager


class V2ObservabilityEvaluationTests(unittest.TestCase):
    def test_trace_generation_and_json_export(self):
        manager = TraceManager()
        trace = manager.start_ticket(42)
        manager.record_stage(42, "classification")
        trace.classification = {"company": "visa", "issue": "refund"}
        finalized = manager.finalize_ticket(42)
        payload = json.loads(finalized.to_json())
        self.assertEqual(payload["ticket_id"], 42)
        self.assertEqual(payload["classification"]["company"], "visa")
        self.assertEqual(payload["latency"]["classification"], 1)
        exported = json.loads(traces_to_json([finalized]))
        self.assertEqual(exported[0]["ticket_id"], 42)

    def test_no_secret_leakage(self):
        text = "key AIzaSySecretValue email user@example.com card 4242 4242 4242 4242"
        sanitized = sanitize_text(text)
        self.assertNotIn("AIzaSySecretValue", sanitized)
        self.assertNotIn("user@example.com", sanitized)
        self.assertNotIn("4242 4242", sanitized)

    def test_performance_metrics(self):
        metrics = PerformanceMetrics()
        metrics.record("retrieval", 2)
        metrics.record("retrieval", 3)
        metrics.record("classification", 1)
        self.assertEqual(metrics.as_dict()["retrieval"], 5)
        self.assertEqual(metrics.as_dict()["overall_ticket_runtime"], 6)

    def test_evaluation_metrics(self):
        self.assertEqual(precision_at_k(["a.md", "b.md"], {"a.md"}, 2), 0.5)
        self.assertEqual(poisoning_resistance(["safe.md", "poison.md"]), 0.5)
        self.assertEqual(confidence_distribution([0.2, 0.6, 0.9]), {"low": 1, "medium": 1, "high": 1})
        self.assertEqual(calibration_error([0.9, 0.2], [1.0, 0.0]), 0.15)
        self.assertEqual(uncertainty_handling([0.5, 0.9], [True, True]), 0.5)
        self.assertEqual(prompt_injection_catch_rate([True, False], [True, False]), 1.0)

    def test_tool_and_safety_metrics(self):
        self.assertTrue(tool_ordering_correct([{"action": "verify_identity"}, {"action": "issue_refund"}]))
        self.assertFalse(tool_ordering_correct([{"action": "lock_account"}, {"action": "reset_password"}]))
        rows = [([{"action": "verify_identity"}, {"action": "issue_refund"}], "refund")]
        self.assertEqual(tool_correctness_rate(rows), 1.0)
        self.assertEqual(security_incident_handling(["critical"], [[{"action": "lock_account"}]]), 1.0)

    def test_comparison_report_v1_parity(self):
        rows = [
            {
                "status": "replied",
                "request_type": "product_issue",
                "source_documents": "data/visa/refund.md",
                "actions_taken": "[]",
                "confidence_score": "0.82",
                "risk_level": "low",
                "company": "visa",
            }
        ]
        report = compare_rows(rows, rows)
        self.assertTrue(report.determinism_preserved())
        self.assertEqual(report.diff(), {})
        self.assertEqual(summarize_rows(rows)["retrieval_coverage"], 1.0)
        self.assertEqual(retrieval_coverage(rows), 1.0)

    def test_deterministic_traces(self):
        def build_json():
            manager = TraceManager()
            trace = manager.start_ticket(7)
            manager.record_stage(7, "retrieval", 2)
            trace.retrieval = {"selected_docs": ["data/visa/refund.md"]}
            return manager.finalize_ticket(7).to_json()

        self.assertEqual(build_json(), build_json())

    def test_human_report(self):
        manager = TraceManager()
        trace = manager.start_ticket(3)
        trace.classification = {"company": "claude", "issue": "refund"}
        trace.confidence = {"score": 0.72}
        trace.retrieval = {"selected_docs": ["data/claude/billing/refund.md"]}
        report = human_report([manager.finalize_ticket(3)])
        self.assertIn("ticket=3", report)
        self.assertIn("claude", report)

    def test_rollback_safety_flags_default_off(self):
        self.assertFalse(ENABLE_V2_OBSERVABILITY)
        self.assertFalse(ENABLE_V2_EVALUATION)


if __name__ == "__main__":
    unittest.main()
