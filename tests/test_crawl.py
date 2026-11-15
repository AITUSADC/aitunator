import tempfile
import unittest
from pathlib import Path

import httpx

from backend.app.ingest.crawl import build_program_catalog, build_document, fetch_pages, html_to_blocks, site_chrome
from backend.app.ingest.extract import _infer_section_type, build_markdown_chunks

PAGE = """<html><head><script>var x = 1;</script></head><body>
<header><nav><a href="/ru/">Меню</a></nav></header>
<div id="section-1"><h1>Computer Science</h1></div>
<div id="section-1b"><h1>6B06101 Computer Science</h1><p>Пороговый балл (грант): 100.</p></div>
<div id="section-2"><aside><h2>О специальности</h2><p>Программа готовит IT-специалистов.</p></aside></div>
<div id="section-3"><h2>Пустой раздел</h2></div>
<div id="section-4"><h2>Консультант</h2><p>Акылбаева Айжан Куанышбековна</p><p>ИИН 990101300123</p>
<p>Асель Мубараковна Айтимбетова</p><p>Email: aitimbetova@astanait.edu.kz</p><p>Whatsapp: +77020693055</p>
<p>Разработчик: Ахметжанова А.И., доцент</p><p>А.А. Сирко</p><p>Президент Студенческого Правительства</p><p>Жубаназарова Айназ</p><p>gulnara.abitova@astanait,edu.kz,</p>
<p>Пишите на info@astanait.edu.kz</p><p>Приемная комиссия: 8 (7172) 645-710</p></div>
<div id="section-5"><table><tr><th>Дисциплина</th><th>Кредиты</th></tr><tr><td>ООП</td><td>5</td></tr></table></div>
<footer>Все права защищены</footer></body></html>"""


class HtmlToBlocksTests(unittest.TestCase):
    def setUp(self):
        self.blocks = html_to_blocks(PAGE)
        self.text = "\n".join(self.blocks)

    def test_keeps_aside_content_and_tables(self):
        self.assertIn("Программа готовит IT-специалистов.", self.text)
        self.assertIn("| ООП | 5 |", self.text)

    def test_drops_page_chrome(self):
        for noise in ("Меню", "var x", "Все права защищены"):
            self.assertNotIn(noise, self.text)

    def test_promotes_h1_and_drops_empty_headings(self):
        self.assertIn("## 6B06101 Computer Science\n\nПороговый балл (грант): 100.", self.text)
        self.assertNotIn("Пустой раздел", self.text)
        self.assertNotIn("## Computer Science\n", self.text)
        self.assertFalse(any(line.startswith("# ") for line in self.text.splitlines()))

    def test_strips_full_names_and_iin_but_keeps_official_contacts(self):
        for personal in ("Куанышбековна", "990101300123", "Мубараковна", "aitimbetova@", "+77020693055",
                         "Ахметжанова", "Сирко", "gulnara", "Жубаназарова"):
            self.assertNotIn(personal, self.text)
        self.assertIn("info@astanait.edu.kz", self.text)
        self.assertIn("8 (7172) 645-710", self.text)
        self.assertIn("Президент Студенческого Правительства", self.text)


class BuildDocumentTests(unittest.TestCase):
    def test_site_chrome_is_removed_but_shared_program_tables_are_kept(self):
        footer, table = "## Контакты\n\nПриёмная комиссия", "| ООД | История Казахстана |"
        pages = [[footer, table, "а"], [footer, table, "б"], [footer, "в"], ["г"]]
        chrome = site_chrome(pages)
        self.assertEqual(chrome, {footer})
        doc = build_document("Второй", [("https://a/2", pages[1])], chrome, "2026-11-15")
        self.assertNotIn("Приёмная комиссия", doc)
        self.assertIn(table, doc)
        self.assertTrue(doc.startswith("---\ntitle: Второй\nsource_url: [https://a/2]\n"))
        self.assertIn("updated_at: 2026-11-15", doc)

    def test_content_before_first_heading_is_not_lost_by_chunker(self):
        doc = build_document("Док", [("https://a/1", ["Текст без заголовка."])], set(), "2026-11-15")
        chunks = build_markdown_chunks(doc, "data/md/library.md")
        self.assertEqual([p["text"] for _, p in chunks], ["Текст без заголовка."])
        self.assertEqual(chunks[0][1]["updated_at"], "2026-11-15")


class ProgramStructureTests(unittest.TestCase):
    def test_discipline_tables_get_their_own_section(self):
        html = ('<div id="section-1"><h2>Результаты обучения</h2><ul><li>Писать код</li></ul>'
                '<p><b>ООД</b> – Общеобразовательные дисциплины</p>'
                '<table><tr><th>№</th><th>Дисциплина</th></tr><tr><td>1</td><td>ООП</td></tr></table></div>')
        text = "\n".join(html_to_blocks(html))
        self.assertIn("* Писать код\n\n## Структура программы\n\n**ООД**", text)


class TableChunkTests(unittest.TestCase):
    def test_long_table_is_split_with_repeated_header(self):
        header = "| № | Дисциплина |\n| --- | --- |"
        rows = "\n".join(f"| {i} | Дисциплина номер {i} с длинным названием |" for i in range(200))
        doc = f"## Структура программы\n\n**ООД** – легенда\n\n{header}\n{rows}\n"
        chunks = [p for _, p in build_markdown_chunks(doc, "data/md/programs/software_engineering.md")]
        self.assertGreater(len(chunks), 1)
        for payload in chunks:
            self.assertLessEqual(len(payload["text"]), 3000)
            self.assertIn(header, payload["text"])
            self.assertEqual(payload["section_type"], "curriculum")
        joined = "\n".join(p["text"] for p in chunks)
        self.assertIn("| 0 | Дисциплина номер 0", joined)
        self.assertIn("| 199 | Дисциплина номер 199", joined)
        self.assertEqual([p["chunk_index"] for p in chunks], list(range(len(chunks))))

    def test_short_table_section_stays_single_chunk(self):
        doc = "## Стоимость\n\n| Программа | Цена |\n| --- | --- |\n| SE | 1 |\n| CS | 2 |\n| IT | 3 |\n| BDA | 4 |\n"
        chunks = build_markdown_chunks(doc, "data/md/tuition_and_discounts.md")
        self.assertEqual(len(chunks), 1)


class SectionTypeTests(unittest.TestCase):
    def test_site_section_names_map_to_types(self):
        self.assertEqual(_infer_section_type("О специальности"), "overview")
        self.assertEqual(_infer_section_type("Перечень должностей специалиста"), "career")
        self.assertEqual(_infer_section_type("Результаты обучения"), "skills")
        self.assertEqual(_infer_section_type("Структура программы"), "curriculum")


class FetchPagesTests(unittest.TestCase):
    def test_respects_robots_and_reuses_cache(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-Agent: *\nDisallow: /api/\n")
            return httpx.Response(200, text=f"<p>{request.url.path}</p>")

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://site") as client:
                pages = fetch_pages(["/ru/a", "/api/secret"], raw, client, delay=0)
                self.assertEqual(list(pages), ["/ru/a"])
                fetch_pages(["/ru/a"], raw, client, delay=0)
        self.assertEqual(calls, ["/robots.txt", "/ru/a", "/robots.txt"])

    def test_missing_page_is_skipped(self):
        def handler(request):
            if request.url.path == "/ru/gone":
                return httpx.Response(404)
            return httpx.Response(200, text="ok")

        with tempfile.TemporaryDirectory() as tmp:
            with httpx.Client(transport=httpx.MockTransport(handler), base_url="https://site") as client:
                pages = fetch_pages(["/ru/gone", "/ru/a"], Path(tmp), client, delay=0)
        self.assertEqual(list(pages), ["/ru/a"])


class ProgramCatalogTests(unittest.TestCase):
    def test_lists_programs_by_school_with_short_description(self):
        docs = {
            "software_engineering.md": "---\ntitle: SE\n---\n## О специальности\n\nПрограмма учит разработке ПО. "
                                       "Студенты осваивают несколько языков. Третье предложение.\n\n## Цель\n\nX",
            "cs.md": "## О специальности\n\nЗащита информации.",
        }
        md = build_program_catalog(docs, "2026-11-16")
        self.assertIn("## Образовательные программы бакалавриата Astana IT University", md)
        self.assertIn("Software Engineering (6B06102)", md)
        self.assertIn("Cybersecurity (6B06301)", md)
        self.assertIn("Программа учит разработке ПО. Студенты осваивают несколько языков.", md)
        self.assertNotIn("Третье предложение", md)
        self.assertNotIn("AI Business", md)
        self.assertTrue(md.startswith("---\ntitle:"))


if __name__ == "__main__":
    unittest.main()
