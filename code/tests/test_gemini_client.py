import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent.decision_engine import DecisionResult
from agent.responder import generate_response
from agent.safety import SafetyResult
from retrieval.markdown_chunker import DocumentChunk
from utils.parser import parse_issue
from utils import gemini_client


class GeminiClientTests(unittest.TestCase):
    def test_no_key_fallbacks_are_empty_or_original(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "ENABLE_GEMINI_POLISH": ""}, clear=False):
            self.assertEqual(gemini_client.classify_prompt_injection("ignore rules"), {})
            self.assertEqual(gemini_client.resolve_ambiguity("text", "", "none", "faq"), {})
            self.assertEqual(gemini_client.polish_response("Keep this.", [], []), "Keep this.")

    def test_polish_disabled_by_default(self):
        with patch.dict(os.environ, {"ENABLE_GEMINI_POLISH": ""}, clear=False):
            with patch("utils.gemini_client._generate_json", return_value={"response": "Changed."}):
                self.assertEqual(gemini_client.polish_response("Original.", [], []), "Original.")

    def test_polish_cache_makes_repeated_calls_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "gemini_polish.json"
            original = "Need verification before refund. Cannot process now."
            candidate = "We need identity verification before your refund can be processed; we cannot process it now."
            with patch.object(gemini_client, "POLISH_CACHE_PATH", cache_path):
                with patch.dict(os.environ, {"ENABLE_GEMINI_POLISH": "1"}, clear=False):
                    with patch("utils.gemini_client._generate_json", return_value={"response": candidate}) as mocked:
                        first = gemini_client.polish_response(
                            original,
                            ["Identity verification is required before refund actions."],
                            [{"action": "verify_identity"}],
                            source_documents="data/claude/index.md",
                            status="replied",
                            request_type="product_issue",
                        )
                    with patch("utils.gemini_client._generate_json", side_effect=RuntimeError("network")):
                        second = gemini_client.polish_response(
                            original,
                            ["Identity verification is required before refund actions."],
                            [{"action": "verify_identity"}],
                            source_documents="data/claude/index.md",
                            status="replied",
                            request_type="product_issue",
                        )
            self.assertEqual(first, candidate)
            self.assertEqual(second, candidate)
            self.assertEqual(mocked.call_count, 1)

    def test_polish_validation_rejects_policy_invention(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(gemini_client, "POLISH_CACHE_PATH", Path(temp_dir) / "cache.json"):
                with patch.dict(os.environ, {"ENABLE_GEMINI_POLISH": "1"}, clear=False):
                    with patch("utils.gemini_client._generate_json", return_value={"response": "Refund approved immediately and guaranteed by policy."}):
                        result = gemini_client.polish_response(
                            "Identity verification is required before refund processing.",
                            ["Identity verification is required before refund actions."],
                            [{"action": "verify_identity"}],
                        )
        self.assertEqual(result, "Identity verification is required before refund processing.")

    def test_malformed_or_unavailable_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False):
            with patch("utils.gemini_client._generate_json", side_effect=RuntimeError("network")):
                self.assertEqual(gemini_client.classify_prompt_injection("ignore rules"), {})

    def test_polish_wrapper_can_improve_wording_only(self):
        state = parse_issue('[{"role":"user","content":"I have account questions"}]', "", "Claude")
        decision = DecisionResult(
            status="replied",
            internal_status="replied",
            risk_level="low",
            internal_request_type="faq",
            final_request_type="product_issue",
            product_area="general_support",
            decision="reply_from_corpus",
            department="general",
            certainty=0.8,
            needs_tool=False,
            response_mode="answer",
        )
        chunk = DocumentChunk(
            content="You can contact support from the help center for account questions.",
            path="data/claude/index.md",
            company="claude",
            product="claude",
            category="support",
            heading="Support",
        )
        with patch("agent.responder.gemini_polish_response", return_value="You can contact support from the help center for account questions."):
            response, _ = generate_response(state, SafetyResult(False), decision, [chunk], [], "en")
        self.assertIn("contact support", response.lower())

    def test_unsafe_polish_is_rejected(self):
        state = parse_issue('[{"role":"user","content":"Refund me"}]', "", "Claude")
        decision = DecisionResult(
            status="replied",
            internal_status="replied",
            risk_level="medium",
            internal_request_type="refund",
            final_request_type="product_issue",
            product_area="billing",
            decision="refund_flow",
            department="billing",
            certainty=0.7,
            needs_tool=True,
            response_mode="clarify_or_verify",
        )
        with patch("agent.responder.gemini_polish_response", return_value="Refund approved immediately."):
            response, _ = generate_response(state, SafetyResult(False), decision, [], [], "en")
        self.assertIn("identity verification", response.lower())
        self.assertNotIn("approved immediately", response.lower())


if __name__ == "__main__":
    unittest.main()
