import unittest
from unittest.mock import Mock
from backend.app.rag.structured import make_card, table_fields, program_queries, balanced_results, source_urls, evidence_context
from eval.structured_compare import measure
from pathlib import Path


class StructuredTests(unittest.TestCase):
    def test_card_preserves_conditions_and_does_not_invent_validity(self):
        text = '| Поле | Значение |\n|---|---|\n| Пороговый балл на грант | 100 |\n\nТолько для указанного конкурса.'
        card = make_card({"text": text, "section": "Поступление", "updated_at": "2026-06-29"}, ["https://example.org"])
        self.assertEqual(card["original_text"], text)
        self.assertIn(text, card["text"])
        self.assertNotIn("Период действия", card["text"])
        self.assertEqual(card["fields"][0]["value"], 100)
        self.assertEqual(card["source_urls"], ["https://example.org"])

    def test_conditional_numbers_are_not_parsed_as_unconditional_facts(self):
        fields = table_fields('| Поле | Значение |\n|---|---|\n| Пороговый балл на грант | 100, кроме особых случаев |\n| Код программы | 6B06102 |\n| Срок обучения | 3 года |')
        self.assertNotIn("key", fields[0])
        self.assertNotIn("value", fields[1])
        self.assertEqual(fields[2]["value"], 3)
        self.assertEqual(table_fields('| Предмет | Кредиты |\n| Математика | 5 |'), [])

    def test_fanout_handles_overlapping_aliases_and_retains_topic(self):
        queries = program_queries('Сравни IT Management и Software Engineering: какие требования к поступлению?')
        self.assertEqual(len(queries), 2)
        for code, query in queries:
            self.assertIn(code, query)
            self.assertIn('требования к поступлению', query)
            self.assertEqual(len(program_queries(query)), 0)
        self.assertEqual(program_queries('Какие предметы на Software Engineering?'), [])

    def test_balancing_keeps_multiple_sections_and_deduplicates_ids(self):
        def row(key):
            return {"source": "same.md", "_retrieval": {"id": key}}
        groups = [(row('a'), row('b'), row('c')), (row('x'), row('a'), row('z'))]
        result = balanced_results(groups)
        self.assertEqual([r['_retrieval']['id'] for r in result], ['a', 'x', 'b', 'c', 'z'])
        self.assertEqual(balanced_results([]), ())

    def test_source_paths_cannot_escape_corpus_root(self):
        self.assertEqual(source_urls(Path(__file__).parent, '../../README.md'), [])

    def test_context_retains_two_sections_from_one_document(self):
        rows = [{"source": "program.md", "program_name": "Example", "program_code": "X",
                 "section": section, "text": text, "updated_at": "2026"}
                for section, text in (("Карьера", "Разработчик"), ("Приём", "Особые условия"))]
        context = evidence_context(rows + [rows[0]])
        self.assertEqual(context.count("Разработчик"), 1)
        self.assertIn("Особые условия", context)
        self.assertIn("Период действия: не указан", context)

    def test_fanout_recovers_second_program_with_same_total_budget(self):
        rows = [{"id": f"{code}-{i}", "score": 1 - i / 100,
                 "payload": {"source": f"{code}.md", "program_code": code, "text": "навыки"}}
                for code in ("6B06102", "6B04101") for i in range(6)]
        corpus = Mock(rows=rows)
        corpus.bm25.get_scores.return_value = list(range(12, 0, -1))
        corpus.dense_search.side_effect = lambda vector, scope, k: [
            r for r in rows if not scope.get("program_code") or r["payload"]["program_code"] == scope["program_code"]][:k]
        encoder = Mock(prompts={})
        encoder.encode.return_value = [1.0]
        case = {"id": "test", "question": "Сравни Software Engineering и IT Management: какие навыки?",
                "category": "arbitrary", "split": "holdout", "answerable": True,
                "evidence_groups": [["6B06102.md"], ["6B04101.md"]]}
        baseline = measure(case, corpus, encoder, False, 8)
        encoder.reset_mock()
        fanout = measure(case, corpus, encoder, True, 8)
        self.assertEqual(baseline["metrics"]["all_evidence_at_5"], 0)
        self.assertEqual(fanout["metrics"]["all_evidence_at_5"], 1)
        self.assertEqual(len(fanout["results"]), 5)
        self.assertEqual(encoder.encode.call_count, 2)


if __name__ == '__main__':
    unittest.main()
