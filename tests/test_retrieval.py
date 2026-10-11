from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from eval.run import metrics, read_cases, model_location
from eval.analyze_routing import fit_gate, accepted
from backend.app.rag.programs import resolve_program_mention, resolve_program_mentions
from backend.app.rag.retriever import retrieve_candidates, rank_candidates
from backend.app.providers.embeddings.sentence_transformer import encode_query, encode_documents


def row(key, text="текст", **metadata):
    return {"id": key, "payload": {"text": text, "source": f"{key}.md", **metadata}}


class RetrievalTests(unittest.TestCase):
    def test_offline_model_uses_local_snapshot_without_hub(self):
        workspace = Path(__file__).resolve().parents[1]
        temp_root = (workspace / "tmp").resolve()
        self.assertTrue(temp_root.is_relative_to(workspace))
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root, prefix="model-location-") as directory:
            cache = Path(directory)
            with self.assertRaises(FileNotFoundError):
                model_location("example/model", cache, False)
            repo = cache / "models--example--model"
            (repo / "refs").mkdir(parents=True)
            (repo / "refs/main").write_text("revision", encoding="utf-8")
            snapshot = repo / "snapshots/revision"
            snapshot.mkdir(parents=True)
            self.assertEqual(Path(model_location("example/model", cache, False)), snapshot)

    def test_embedding_does_not_require_undefined_named_prompts(self):
        model = Mock(prompts={})
        encode_query(model, "вопрос")
        model.encode.assert_called_once_with("вопрос", normalize_embeddings=True)
        model.reset_mock()
        model.prompts = {"document": "passage: "}
        encode_documents(model, ["текст"], batch_size=8)
        model.encode.assert_called_once_with(["текст"], normalize_embeddings=True,
                                            prompt_name="document", batch_size=8)

    def retrieve(self, corpus, scores, **kwargs):
        return retrieve_candidates("условия обучения", k=kwargs.pop("k", 5),
            mode=kwargs.pop("mode", "bm25"), corpus=corpus,
            bm25=Mock(get_scores=Mock(return_value=scores)), **kwargs)

    def test_bm25_does_not_call_dense_or_encoder(self):
        encoder, dense = Mock(), Mock()
        batch = self.retrieve([row(1)], [2], encode_query=encoder, dense_search=dense)
        self.assertEqual(len(batch.candidates), 1)
        encoder.assert_not_called()
        dense.assert_not_called()

    def test_empty_dense_still_returns_bm25(self):
        batch = self.retrieve([row(1)], [2], mode="hybrid",
            encode_query=lambda _: [1], dense_search=lambda *_: [])
        self.assertEqual(batch.candidates[0]["source"], "1.md")

    def test_filter_applied_before_bm25_limit(self):
        corpus = [row(i, program_code="other") for i in range(20)] + [row(21, program_code="target")]
        batch = self.retrieve(corpus, list(range(21, 0, -1)), k=1, program_code="target")
        self.assertEqual(batch.candidates[0]["source"], "21.md")

    def test_no_lexical_overlap_returns_empty(self):
        self.assertEqual(self.retrieve([row(1), row(2)], [0, 0]).candidates, ())

    def test_normalize_ids_before_fusion(self):
        batch = self.retrieve([row("12")], [1], mode="hybrid", encode_query=lambda _: [1],
            dense_search=lambda *_: [{**row(12), "score": 0.7}])
        self.assertEqual(len(batch.candidates), 1)
        scores = batch.candidates[0]["_retrieval"]
        self.assertAlmostEqual(scores["rrf_score"], 2 / 61)
        self.assertEqual(scores["bm25_score"], 1)
        self.assertEqual(scores["dense_score"], 0.7)

    def test_rerank_reuses_candidates_without_mutating_them(self):
        encoder = Mock(return_value=[1])
        dense = Mock(return_value=[])
        batch = self.retrieve([row(1), row(2)], [2, 1], mode="hybrid",
                              encode_query=encoder, dense_search=dense)
        predictor = Mock(return_value=[-3, 2])
        ranked, _ = rank_candidates(batch, top_k=1, predict=predictor)
        self.assertEqual(ranked[0]["source"], "2.md")
        encoder.assert_called_once()
        dense.assert_called_once()
        self.assertNotIn("rerank_score", batch.candidates[0]["_retrieval"])
        plain, _ = rank_candidates(batch, top_k=2)
        self.assertEqual(plain[0]["source"], "1.md")

    def test_dense_only_does_not_score_bm25(self):
        bm25 = Mock()
        retrieve_candidates("question", k=5, mode="dense", corpus=[], bm25=bm25,
            encode_query=lambda _: [1], dense_search=lambda *_: [])
        bm25.get_scores.assert_not_called()

    def test_program_names_codes_and_punctuation(self):
        self.assertEqual(resolve_program_mention("Что изучают на BDA?")["program_code"], "6B06103")
        self.assertEqual(resolve_program_mention("Код 6B06102.")["program_code"], "6B06102")
        self.assertEqual(resolve_program_mention("IT Management")["program_code"], "6B04101")
        self.assertEqual(resolve_program_mention("Сравни BDA и SE"), {})
        self.assertEqual(len(resolve_program_mentions("BDA и Big Data Analysis")), 1)

    def test_comparison_keeps_both_programs(self):
        corpus = [row(1, program_code="6B06103"), row(2, program_code="6B06102")]
        batch = retrieve_candidates("Сравни BDA и SE", k=5, mode="bm25", corpus=corpus,
                                    bm25=Mock(get_scores=Mock(return_value=[3, 2])))
        self.assertEqual(len(batch.candidates), 2)

    def test_metrics_require_both_comparison_sources(self):
        case = {"answerable": True, "evidence_groups": [["a.md"], ["b.md"]]}
        self.assertEqual(metrics(case, [{"source": "a.md"}])["all_evidence_at_5"], 0)
        self.assertEqual(metrics(case, [{"source": "a.md"}, {"source": "b.md"}])["all_evidence_at_5"], 1)
        self.assertIsNone(metrics({"answerable": False}, [{"source": "a.md"}])["hit_at_5"])

    def test_right_document_wrong_section_is_not_evidence(self):
        case = {"answerable": True, "evidence_groups": [["a.md"]],
                "section_evidence_groups": [[{"source": "a.md", "section": "Стоимость"}]]}
        wrong = metrics(case, [{"source": "a.md", "section": "Карьера"}])
        self.assertEqual(wrong["hit_at_5"], 1)
        self.assertEqual(wrong["section_evidence_at_5"], 0)
        self.assertEqual(metrics(case, [{"source": "a.md", "section": "Стоимость"}])["section_evidence_at_5"], 1)

    def test_dataset_is_grouped_and_has_required_categories(self):
        path = Path(__file__).resolve().parents[1] / "eval/golden.jsonl"
        cases = read_cases(path)
        self.assertGreaterEqual(len(cases), 120)
        families = {}
        for case in cases:
            families.setdefault(case["family"], set()).add(case["split"])
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertTrue({"exact", "numeric", "semantic", "comparison", "followup_resolved", "unanswerable"}
                        <= {case["category"] for case in cases})

    def test_gate_cannot_fit_on_holdout(self):
        with self.assertRaises(ValueError):
            fit_gate([{"split": "holdout"}])

    def test_high_scoring_unanswerable_prevents_confident_gate(self):
        features = {"candidate_count": 5, "bm25_top_score": 100, "bm25_gap": 0.9, "term_coverage": 1.0}
        positive = {"split": "calibration", "answerable": True, "features": features,
                    "metrics": {"section_evidence_at_5": 1}}
        negative = {**positive, "answerable": False}
        self.assertIsNone(fit_gate([positive, negative], min_accepts=1))
        gate = fit_gate([positive], min_accepts=1)
        self.assertTrue(accepted(features, gate))


if __name__ == "__main__":
    unittest.main()
