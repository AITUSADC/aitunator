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
