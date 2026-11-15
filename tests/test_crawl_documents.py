import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import pymupdf

from backend.app.ingest import documents
from backend.app.ingest.documents import document_links, document_to_markdown, fetch_documents, strip_personal_tables, drop_signature_pages

PAGE = """<div id="section-1">
<a href="https://drive.google.com/file/d/AAA111/view?usp=sharing">Каталог элективных дисциплин Software Engineering</a>
<a href="https://drive.google.com/file/d/PLAN99/view">Рабочий учебный план (1 курс) Software Engineering</a>
<a href="https://docs.google.com/document/d/BBB222/edit?usp=sharing">Список обладателей гранта</a>
<a href="https://drive.google.com/file/d/LAT777/view">Cписок обладателей гранта (латинская C)</a>
<a href="https://drive.google.com/file/d/CCC333/view">Прейскурант стоимости кредитов на 2026 - 2027 учебный год</a>
<a href="https://drive.google.com/drive/u/0/folders/FFF">Внутренние документы</a>
<a href="https://drive.google.com/file/d/DDD444/view">здесь</a>
<a href="https://drive.google.com/file/d/AAA111/view">Учебный план (дубль)</a>
<a href="https://drive.google.com/file/d/EEE555/view">Программа развития Software Engineering</a>
</div>"""


def _pdf(*texts: str) -> bytes:
    doc = pymupdf.open()
    for text in texts:
        doc.new_page().insert_text((72, 72), text)
    return doc.tobytes()


class DocumentLinksTests(unittest.TestCase):
    def test_filters_personal_lists_folders_and_noise_and_routes_by_title(self):
        links = document_links(PAGE, "programs/software_engineering.md", "Software Engineering")
        self.assertEqual(links, [
            ("AAA111", "Каталог элективных дисциплин Software Engineering", "programs/software_engineering.md"),
            ("CCC333", "Прейскурант стоимости кредитов на 2026 - 2027 учебный год", "tuition_and_discounts.md"),
            ("DDD444", "Документ: Software Engineering", "programs/software_engineering.md"),
        ])


class PersonalTablesTests(unittest.TestCase):
    def test_tables_with_name_column_are_dropped_and_text_is_kept(self):
        md = ("Правила приема документов.\n\n"
              "|**№**<br>**ФИО**|**Должность**|\n|---|---|\n|1|Иванов|\n\n"
              "| Уровень | Стоимость |\n|---|---|\n| Бакалавриат | 20000 |\n\n"
              "| № | Фамилия, имя, отчество | Балл |\n| 1 | Алиева Томирис | 90 |")
        self.assertEqual(strip_personal_tables(md),
                         "Правила приема документов.\n\n\n| Уровень | Стоимость |\n|---|---|\n| Бакалавриат | 20000 |\n")

    def test_prose_mentioning_names_is_kept(self):
        text = "Документ, удостоверяющий личность, содержит фамилию, имя, отчество."
        self.assertEqual(strip_personal_tables(text), text)


class SignaturePagesTests(unittest.TestCase):
    def test_pages_with_signature_blocks_are_dropped(self):
        md = ("Цель программы: подготовка инженеров.\f"
              "Лист согласования\nОбучающийся/ Student:\nИванова Анна\n(Аты-жөн/ФИО/Name)\n(подпись/қолы/signature)\f"
              "Срок обучения 4 года.")
        self.assertEqual(drop_signature_pages(md), "Цель программы: подготовка инженеров.\n\nСрок обучения 4 года.")


    def test_single_page_documents_are_kept(self):
        docx = "Отбор проводится в четыре этапа.\n\nНачальнику военной кафедры (Ф.И.О (при наличии))"
        self.assertEqual(drop_signature_pages(docx), docx)


class FetchDocumentsTests(unittest.TestCase):
    def test_downloads_binary_files_once_and_skips_html_pages(self):
        calls = []

        def handler(request):
            file_id = request.url.params["id"]
            calls.append(file_id)
            if file_id == "PDF1":
                return httpx.Response(200, content=b"%PDF-1.4 data")
            return httpx.Response(200, text="<html>Sign in</html>")

        with tempfile.TemporaryDirectory() as tmp:
            with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                first = fetch_documents(["PDF1", "LOGIN"], Path(tmp), client, delay=0)
                second = fetch_documents(["PDF1"], Path(tmp), client, delay=0)
        self.assertEqual(first, {"PDF1": b"%PDF-1.4 data"})
        self.assertEqual(second, first)
        self.assertEqual(calls, ["PDF1", "LOGIN"])


class DocumentToMarkdownTests(unittest.TestCase):
    def test_pdf_with_text_layer_is_converted_without_ocr(self):
        with patch.object(documents, "_ocr_page", side_effect=AssertionError("OCR not expected")):
            md = document_to_markdown(_pdf("Summer semester lasts six weeks and allows up to fifteen credits."))
        self.assertIn("Summer semester lasts six weeks", md)

    def test_pdf_without_readable_text_falls_back_to_ocr(self):
        with patch.object(documents, "_ocr_page", return_value="Летний семестр длится шесть недель.") as ocr:
            md = document_to_markdown(_pdf("F E P ,! E q B 8, 8: I E8 E= d- ;^$*Es"))
        ocr.assert_called_once()
        self.assertIn("Летний семестр длится шесть недель.", md)

    def test_only_unreadable_pages_of_mixed_pdf_are_ocred(self):
        with patch.object(documents, "_ocr_page", return_value="Летний семестр длится шесть недель.") as ocr:
            md = document_to_markdown(_pdf("Summer semester lasts six weeks and allows up to fifteen credits.",
                                           "F E P ,! E q B 8, 8: I E8 E= d- ;^$*Es"))
        ocr.assert_called_once()
        self.assertIn("Summer semester lasts six weeks", md)
        self.assertTrue(md.endswith("\fЛетний семестр длится шесть недель."))

    def test_converted_markdown_is_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(documents, "document_to_markdown", return_value="Текст документа") as convert:
                first = documents.cached_markdown("ID1", b"%PDF", Path(tmp))
                second = documents.cached_markdown("ID1", b"%PDF", Path(tmp))
        convert.assert_called_once()
        self.assertEqual((first, second), ("Текст документа", "Текст документа"))

    def test_ocr_noise_lines_are_dropped(self):
        raw = "| $ an | Sqn |\nУЧЕБНЫЙ ПЛАН\n2 > | Г\nЛетний семестр: 1 июня – 15 июля 2026\n~ |< ¦\nФорма обучения: очная"
        self.assertEqual(documents._clean_ocr_text(raw),
                         "УЧЕБНЫЙ ПЛАН\nЛетний семестр: 1 июня – 15 июля 2026\nФорма обучения: очная")

    def test_unknown_format_returns_empty(self):
        self.assertEqual(document_to_markdown(b"\x00\x01binary"), "")


if __name__ == "__main__":
    unittest.main()
