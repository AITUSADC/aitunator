import unittest

from backend.app.rag.structured import prefer_sections, requested_sections


class SectionSelectionTests(unittest.TestCase):
    def rows(self):
        return [{"id": str(i), "doc_type": "program_profile", "section_type": section}
                for i, section in enumerate(("overview", "career", "skills", "curriculum", "general"))]

    def test_learning_promotes_evidence_and_retains_fallback(self):
        rows = self.rows()
        result = prefer_sections(rows, "Чему учат на этой программе?")
        self.assertEqual([r["section_type"] for r in result],
                         ["skills", "curriculum", "overview", "career", "general"])
        self.assertEqual(rows[0]["section_type"], "overview")
        self.assertEqual({r["id"] for r in result}, {r["id"] for r in rows})

    def test_subjects_and_career_are_distinguished(self):
        self.assertEqual(requested_sections("Какие предметы изучают?"), {"curriculum"})
        self.assertEqual(requested_sections("Кем могу работать после выпуска?"), {"career"})
        self.assertEqual(prefer_sections(self.rows(), "Кем работать?")[0]["section_type"], "career")

    def test_multiple_topics_are_kept(self):
        self.assertEqual(requested_sections("Какие навыки получу и кем смогу стать? Карьера?"),
                         {"skills", "curriculum", "career"})

    def test_no_evidence_or_unknown_topic_does_not_drop_candidates(self):
        rows = self.rows()
        for question in ("Какой пороговый балл?", "Сколько стоит обучение?", "Расскажи о программе"):
            self.assertEqual(prefer_sections(rows, question), tuple(rows))
        self.assertEqual(prefer_sections(rows[:1], "Какие навыки?"), tuple(rows[:1]))
        self.assertEqual(prefer_sections([], "Чему учат?"), ())

    def test_general_policy_is_not_reclassified_as_program_evidence(self):
        rows = [{"doc_type": "policy", "section_type": "overview"},
                {"doc_type": "policy", "section_type": "career"}]
        self.assertEqual(prefer_sections(rows, "Карьера"), tuple(rows))


if __name__ == "__main__":
    unittest.main()


class CatalogQueryTests(unittest.TestCase):
    def search(self, query):
        from types import SimpleNamespace
        from backend.app.rag.pipeline import search
        payloads = [{"source": f"data/md/x{i}.md", "doc_type": "handbook", "text": f"шум {i}", "section": "s"} for i in range(6)]
        payloads.append({"source": "data/md/programs/index.md#se", "doc_type": "catalog", "text": "SE", "section": "SE", "chunk_index": 1})
        payloads.append({"source": "data/md/programs/index.md", "doc_type": "catalog", "text": "16 программ", "section": "list", "chunk_index": 0})
        rows = [{"id": str(i), "payload": p} for i, p in enumerate(payloads)]

        def dense_search(vector, scope, k):  # catalog is the least similar chunk, as in the real corpus
            allowed = [r for r in rows if not scope.get("doc_types") or r["payload"]["doc_type"] in scope["doc_types"]]
            return [{**r, "score": 1 - i / 10} for i, r in enumerate(allowed)][:k]
        corpus = SimpleNamespace(rows=rows, bm25=None, dense_search=dense_search)
        results, _, _ = search(query, corpus=corpus, encode=lambda _: [1.0])
        return [r["source"] for r in results]

    def test_program_list_questions_put_catalog_first(self):
        for query in ("Какие программы есть в айту?", "Какие специальности есть в AITU?", "На кого можно учиться?"):
            self.assertEqual(self.search(query)[0], "data/md/programs/index.md", query)

    def test_other_questions_keep_retrieval_order(self):
        self.assertNotIn("data/md/programs/index.md", self.search("Сколько стоит общежитие?"))
