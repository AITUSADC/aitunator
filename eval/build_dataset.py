"""Build a versioned retrieval-only set from local legacy evals and reviewed cases.

Evidence groups are AND across groups, OR within each group. Labels concern the
local KB snapshot, not whether the university's real-world information is current.
"""
import hashlib
import json
import re
from pathlib import Path

from backend.app.rag.programs import PROGRAM_MARKDOWN_REGISTRY

ROOT = Path(__file__).resolve().parents[1]


def build():
    cases, seen = [], set()

    def add(case):
        question = case["question"].strip()
        if question in seen:
            return
        seen.add(question)
        family = case["family"]
        case["split"] = "holdout" if int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 4 == 0 else "calibration"
        cases.append(case)

    for name in ("curated_golden", "curated_general", "curated_programs", "hard_negative_golden"):
        for original in json.loads((ROOT / f".local/legacy-evals/datasets/{name}.json").read_text(encoding="utf-8")):
            sources = original.get("expected_sources", [])
            answerable = bool(sources) and not original.get("expected_guardrail_reason")
            family = original.get("program_code") or (sources[0] if sources else "negative:" + original["id"])
            question = original["question"]
            numeric = any(word in question.lower() for word in ("сколько", "балл", "стоимост", "цен", "2026", "кредит"))
            add({"id": original["id"], "question": question, "family": family,
                 "category": "unanswerable" if not answerable else "numeric" if numeric else "factual",
                 "answerable": answerable, "evidence_groups": [sources] if answerable else [],
                 "program_code": original.get("program_code"), "language": original.get("language"),
                 "doc_types": original.get("doc_types", []), "origin": "legacy:" + name})

    profiles = [(source, meta) for source, meta in PROGRAM_MARKDOWN_REGISTRY.items()
                if meta.get("program_code")]
    for source, meta in profiles:
        code, name = meta["program_code"], meta["program_name"]
        path = "data/md/programs/" + Path(source).name
        variants = [
            ("exact", f"Как называется образовательная программа {code}?"),
            ("numeric", f"Какой пороговый балл указан у программы {code}?"),
            ("semantic", f"Какие умения я получу, если выберу {name}?"),
        ]
        for category, question in variants:
            add({"id": f"program_{code}_{category}", "question": question, "family": code,
                 "category": category, "answerable": True, "evidence_groups": [[path]],
                 "origin": "reviewed:program_profile"})

    pairs = [
        ("Software Engineering", "software_engineering", "Big Data Analysis", "big_data_analysis"),
        ("Computer Science", "it", "Cybersecurity", "cs"),
        ("IT Management", "it_management", "IT Entrepreneurship", "it_entrepreneurship"),
        ("Cybersecurity", "cs", "Smart Security Technologies", "smart_security_technologies"),
        ("Media Technologies", "media_technologies", "Digital Journalism", "digital_journalism"),
        ("AI Business", "ai_business", "Big Data Analysis", "big_data_analysis"),
        ("Electronic Engineering", "electronic_engineering", "Smart Technologies", "smart_technologies"),
        ("Software Engineering", "software_engineering", "Computer Science", "it"),
    ]
    for i, (a, source_a, b, source_b) in enumerate(pairs):
        add({"id": f"comparison_{i}", "question": f"Сравни {a} и {b}: чему учат на каждой программе?",
             "family": f"comparison:{source_a}:{source_b}", "category": "comparison", "answerable": True,
             "evidence_groups": [[f"data/md/programs/{source_a}.md"], [f"data/md/programs/{source_b}.md"]],
             "origin": "reviewed:two_programs"})

    extra = [
        ("numeric", "Сколько недель длится один триместр?", "academic_calendar"),
        ("numeric", "Сколько всего триместров в учебном году?", "academic_calendar"),
        ("numeric", "Какой максимум кредитов разрешён на летний семестр?", "academic_calendar"),
        ("numeric", "25–49 баллов за итоговый контроль: это какая оценка?", "first_year_guide"),
        ("numeric", "Что означает скидка Грант Ректора и какой у неё размер?", "tuition_and_discounts"),
        ("numeric", "Какой GPA нужен платнику для академической скидки 30%?", "tuition_and_discounts"),
        ("numeric", "Какая скидка указана для детей сотрудников AITU?", "tuition_and_discounts"),
        ("numeric", "Какие программы входят в группу B058?", "student_grants"),
        ("semantic", "Учиться оплачиваю сам. Есть способ получить освободившееся бюджетное место?", "student_grants"),
        ("semantic", "Мне тяжело эмоционально, к кому обратиться в университете за поддержкой?", "student_support"),
        ("semantic", "Хочу часть обучения провести в иностранном вузе. Какие есть условия?", "academic_mobility"),
        ("semantic", "Не могу открыть полный текст электронной книги. Как получить доступ?", "library"),
    ]
    for i, (category, question, source) in enumerate(extra):
        path = f"data/md/{source}.md"
        add({"id": f"additional_{i}", "question": question, "family": path, "category": category,
             "answerable": True, "evidence_groups": [[path]], "origin": "reviewed:general_document"})

    negatives = [
        "Какая будет стоимость Software Engineering в 2035 году?",
        "Сколько свободных мест в общежитии прямо сейчас?",
        "Покажи мой текущий GPA из личного кабинета.",
        "Кто из студентов получил 100 баллов на экзамене вчера?",
        "Какой пароль от моей учётной записи Moodle?",
        "Какой будет курс доллара завтра?",
        "Назови точное число заявлений на грант, поданных сегодня.",
        "Какая аудитория у моего следующего занятия?",
    ]
    for i, question in enumerate(negatives):
        add({"id": f"unanswerable_{i}", "question": question, "family": f"unanswerable:{i}",
             "category": "unanswerable", "answerable": False, "evidence_groups": [],
             "origin": "reviewed:unavailable_live_private_or_future_data"})
    for i, (name, source) in enumerate([
        ("Big Data Analysis", "big_data_analysis"), ("Software Engineering", "software_engineering"),
        ("Cybersecurity", "cs"), ("AI Business", "ai_business"),
    ]):
        meta = next(meta for key, meta in profiles if Path(key).stem == source)
        add({"id": f"followup_{i}", "question": f"А чему там учат? (после вопроса о {name})",
             "history": [{"role": "user", "content": f"Расскажи про {name}."}],
             "standalone_question": f"Какие дисциплины изучают на {name}?",
             "family": meta["program_code"], "category": "followup_resolved", "answerable": True,
             "evidence_groups": [[f"data/md/programs/{source}.md"]],
             "origin": "manual_rewrite:retriever_only"})
    # Add section-level evidence independently of retrieval results. Legacy
    # general labels remain source-level until their individual facts are reviewed.
    for case in cases:
        if not case["answerable"]:
            continue
        key = case["id"]
        section_prefixes = None
        if case["category"] == "comparison":
            section_prefixes = ("Основные дисциплины", "Направления дисциплин", "Чему научишься")
        elif case["category"] == "followup_resolved" or "subjects" in key:
            section_prefixes = ("Основные дисциплины", "Направления дисциплин")
        elif "electives" in key:
            section_prefixes = ("Элективные дисциплины",)
        elif "career" in key:
            section_prefixes = ("Карьера",)
        elif "skills" in key or key.startswith("program_") and case["category"] == "semantic":
            section_prefixes = ("Чему научишься",)
        elif "threshold" in key or key.startswith("program_"):
            section_prefixes = ("Основная информация",)
        if section_prefixes:
            groups = []
            for group in case["evidence_groups"]:
                alternatives = []
                for source in group:
                    if "/programs/" not in source or source.endswith("/index.md"):
                        continue
                    text = (ROOT / source).read_text(encoding="utf-8")
                    for heading in re.findall(r"^##\s+(.+)$", text, re.MULTILINE):
                        if heading.startswith(section_prefixes):
                            alternatives.append({"source": source, "section": heading.strip()})
                if alternatives:
                    groups.append(alternatives)
            if len(groups) == len(case["evidence_groups"]):
                case["section_evidence_groups"] = groups
    return cases


if __name__ == "__main__":
    cases = build()
    output = ROOT / "eval/golden.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {output}")
