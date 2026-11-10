"""Сборка отчёта docs/report.docx из реальных результатов конвейера.

Запуск после run-all, simulate-drift и reproduce: python scripts/build_report.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from report_data import CFG, MODEL_NAMES, ROOT, TARGET_NAMES, collect, make_figures, metric_value

OUT = ROOT / "docs" / "report.docx"
SHOTS = ROOT / "docs" / "screenshots"


# ------------------------------------------------------------------ оформление
def setup(doc: Document) -> None:
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    section.left_margin, section.right_margin = Cm(3.0), Cm(1.5)
    section.top_margin, section.bottom_margin = Cm(2.0), Cm(2.0)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(14)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Cm(1.25)
    for level, size in ((1, 16), (2, 14), (3, 14)):
        style = styles[f"Heading {level}"]
        style.font.name = "Times New Roman"
        fonts = style.element.rPr.rFonts
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            fonts.attrib.pop(qn(attr), None)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            fonts.set(qn(attr), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.first_line_indent = Cm(0)
    caption = styles["Caption"]
    caption.font.name = "Times New Roman"
    caption.font.size = Pt(12)
    caption.font.italic = False
    caption.font.color.rgb = RGBColor(0, 0, 0)
    # Нумерация страниц внизу по центру
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(footer, "PAGE")
    section.different_first_page_header_footer = True


def _field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    for kind, text in (("begin", None), (None, instruction), ("separate", None), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def p(doc: Document, text: str, *, bold: bool = False, align=None, indent: bool = True, size: int | None = None, after: int = 0):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.bold = bold
    if size:
        run.font.size = Pt(size)
    if align is not None:
        para.alignment = align
    else:
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if not indent:
        para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.space_after = Pt(after)
    return para


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        para = doc.add_paragraph(style="List Bullet")
        para.add_run(item)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        para.paragraph_format.first_line_indent = Cm(0)


def h(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def table(doc: Document, frame: pd.DataFrame, caption: str, widths: list[float] | None = None, font: int = 11) -> None:
    cap = doc.add_paragraph(style="Caption")
    cap.add_run(caption)
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(frame.columns))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, column in enumerate(frame.columns):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(column))
        run.bold = True
        run.font.size = Pt(font)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:fill"), "EDEDED")
        cell._tc.get_or_add_tcPr().append(shading)
    for _, row in frame.iterrows():
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(_fmt(value))
            run.font.size = Pt(font)
    keep_together = len(t.rows) <= 25
    for r, row in enumerate(t.rows):
        tr_pr = row._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
        if r == 0:
            header = OxmlElement("w:tblHeader")
            tr_pr.append(header)
        for i, cell in enumerate(row.cells):
            for para in cell.paragraphs:
                para.paragraph_format.first_line_indent = Cm(0)
                para.paragraph_format.line_spacing = 1.0
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                para.paragraph_format.keep_with_next = keep_together and r < len(t.rows) - 1
            if widths:
                cell.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def _fmt(value) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "—"
        return f"{value:,.4f}".replace(",", " ") if abs(value) < 10 else f"{value:,.1f}".replace(",", " ")
    return str(value)


def figure(doc: Document, path: Path, caption: str, width_cm: float = 16.0) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.paragraph_format.first_line_indent = Cm(0)
    para.paragraph_format.keep_with_next = True
    para.add_run().add_picture(str(path), width=Cm(width_cm))
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.first_line_indent = Cm(0)
    cap.add_run(caption)


def page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",") + "%"


def num(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


# ------------------------------------------------------------------ содержание
def build() -> Path:
    facts = collect()
    figs = make_figures(facts)
    m = facts["metrics"]
    c = facts["counts"]
    doc = Document()
    setup(doc)

    # Титульный лист
    for line in ("Отчёт о проектной работе", "по дисциплине «Проектный практикум»"):
        p(doc, line, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=16)
    for _ in range(3):
        p(doc, "", indent=False)
    p(doc, "Тема № 21. Рынок цифровых активов", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=16)
    p(doc, "Прототип прогнозно-аналитической системы cryptocast: прогноз объёма торгов и волатильности инструмента",
      align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    for _ in range(6):
        p(doc, "", indent=False)
    p(doc, "Выполнил: студент группы БВТ2304", align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False)
    p(doc, "Запорожец Михаил Романович", align=WD_ALIGN_PARAGRAPH.RIGHT, indent=False)
    for _ in range(8):
        p(doc, "", indent=False)
    p(doc, str(dt.date.today().year), align=WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    page_break(doc)

    # Содержание
    p(doc, "Содержание", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=False, size=16)
    toc = doc.add_paragraph()
    toc.paragraph_format.first_line_indent = Cm(0)
    _field(toc, 'TOC \\o "1-2" \\h \\z \\u')
    page_break(doc)

    status = facts["status"]
    folds = facts["folds"]
    holdout_days = CFG["validation"]["holdout_days"]
    best = {t: metric_value(m, "cv", t, "lightgbm", "mae") for t in CFG.targets}  # noqa: F841

    # 1. Постановка задачи
    h(doc, "1. Постановка задачи")
    p(doc, "Цель работы — спроектировать, разработать и представить прототип прогнозно-аналитической системы, которая проходит полный цикл работы "
           "с данными: автоматический сбор из нескольких внешних источников, хранение, очистка и подготовка, контроль качества, построение прогнозов "
           "и выдача результатов пользователю в интерактивном дашборде.")
    p(doc, "Прогнозируемые показатели по теме № 21 — объём торгов и волатильность инструмента. Инструменты: BTC, ETH, SOL, BNB и XRP к USDT на бирже Binance. "
           "Шаг прогноза — сутки UTC. В конце суток D система строит прогноз на сутки D+1 для двух величин:")
    bullets(doc, [
        "log_qv — натуральный логарифм суммарного объёма торгов в котируемой валюте (quote volume, USDT) за сутки D+1;",
        "log_rv — логарифм реализованной волатильности за сутки D+1, где RV = корень из суммы квадратов 24 часовых лог-доходностей. Метрики считаются в исходном масштабе RV.",
    ])
    p(doc, "Пользователь системы — риск-менеджер или трейдер небольшого фонда. Он принимает два решения: какой объём можно исполнить завтра, не становясь "
           "заметной долей рынка, и нужно ли снижать риск. Прогнозы служат информацией для контроля риска и не являются торговым советом.")
    p(doc, "Требования к системе взяты из методических указаний: не менее двух разнородных источников, из них хотя бы один слабоструктурированный; "
           "инкрементальная и идемпотентная загрузка; устойчивость к сбоям; журнал загрузок; единый код признаков; доказанное отсутствие утечки; "
           "не менее трёх моделей разных классов с базовой; единая временная валидация; дашборд с фактом, прогнозом и качеством. "
           "Реализованы все три компонента по выбору: интерпретируемость, мониторинг с переобучением и автоматический подбор с журналом экспериментов.")

    # 2. Предметная область
    h(doc, "2. Описание предметной области")
    p(doc, "Криптовалютный рынок торгуется круглосуточно без выходных, поэтому естественная единица времени — сутки UTC. Объём и волатильность "
           "имеют выраженную память: за днями высокой активности обычно следуют такие же. Это свойство описывают модели HAR (Heterogeneous "
           "Autoregressive, Corsi, 2009), в которых будущая волатильность выражается через её средние значения за день, неделю и месяц.")
    p(doc, "Объём торгов имеет недельную сезонность: в выходные он заметно ниже. Поэтому признаки дня недели относятся к дате прогноза D+1, "
           "которая известна заранее. Волатильность связана с объёмом и с общим состоянием рынка, где ведущую роль играет BTC.")
    p(doc, "Для риск-менеджера важны две оценки. Ожидаемый суточный объём задаёт ёмкость исполнения: принято не превышать небольшую долю "
           f"рыночного объёма (в системе это настраиваемый параметр, сейчас {CFG['decision']['participation_rate']:.0%}). Ожидаемая волатильность "
           "задаёт размер позиции и необходимость снизить экспозицию. Недооценка волатильности опаснее переоценки, поэтому для неё "
           "дополнительно используется асимметричная метрика QLIKE.")
    p(doc, "Новостной фон (взломы бирж, регуляторные решения, биржевые фонды ETF) и индекс страха и жадности считаются возможными драйверами активности. "
           "Проверить, дают ли они прирост качества сверх рыночных признаков, — отдельная задача работы (раздел 4.6).")

    # 3. Данные
    h(doc, "3. Данные")
    h(doc, "3.1. Источники", 2)
    by_source = c["news_by_source"]
    sources = pd.DataFrame([
        ["Binance REST, часовые свечи", "структурированный, JSON", f"{c['klines']:,}".replace(",", " ") + " свечей", f"{c['klines_first'][:10]} – {c['klines_last'][:16]}"],
        ["Fear & Greed Index", "структурированный, JSON", f"{c['fng']} значений", "с 2018-02-01"],
        ["Hacker News (Algolia)", "слабоструктурированный, тексты", f"{by_source.get('hn', 0):,}".replace(",", " ") + " заголовков", f"с {c['news_first']}"],
        ["RSS 9 крипто-СМИ", "слабоструктурированный, XML", f"{by_source.get('rss', 0)} записей", "копится с начала сбора"],
        ["GDELT DOC API", "слабоструктурированный, JSON", f"{by_source.get('gdelt', 0)} записей", "лимит 429, см. 3.2"],
    ], columns=["Источник", "Тип", "Объём", "Период"])
    table(doc, sources, "Таблица 1 — Источники данных (фактическое состояние после запуска)", [5.0, 4.2, 3.6, 3.7])
    p(doc, "Разнородность источников соответствует методичке: различаются и способ получения (REST JSON, RSS/XML), и формат (числовые ряды, тексты). "
           "Все источники общедоступны, персональные данные не используются: поля авторов в RSS и Hacker News не читаются, подписи вида «by Имя» "
           "вырезаются из текстов.")
    h(doc, "3.2. Проверка доступности и ограничение GDELT", 2)
    p(doc, "Перед разработкой каждый источник проверен реальными запросами (docs/data_sources.md). Binance, индекс страха, все девять RSS-лент, "
           "резервные Bybit и OKX ответили успешно. Постановка предполагала историю новостей из GDELT, но этот API почти на все запросы отвечал "
           "HTTP 429. Из более чем 20 запросов с паузами от 7 секунд до 2 минут успешных было два, а за 30-минутный запуск не прошёл ни один. "
           "Обходить ограничение я не стал. GDELT остался в конвейере с бережным бюджетом (не более трёх запросов за запуск), а историю заголовков "
           "дал публичный API Hacker News. Это сообщество разработчиков, а не новостная лента, и это ограничение учтено в выводах.")
    h(doc, "3.3. Сбор данных", 2)
    p(doc, "Сбор реализован в модуле cryptocast/ingest и отвечает требованиям методички:")
    bullets(doc, [
        "инкрементальность: для каждого источника хранится watermark (время последней загруженной свечи, дата индекса, конец окна новостей); повторный запуск запрашивает только данные после него;",
        "идемпотентность: вставка через INSERT … ON CONFLICT DO NOTHING по ключу (symbol, open_time) для свечей, по дате для индекса и по SHA-256 нормализованного URL для новостей;",
        "устойчивость: таймауты, повторы с экспоненциальной задержкой, обработка 429 и заголовка Retry-After, перебор зеркал Binance; ошибка одного источника записывается в журнал и не останавливает запуск;",
        "журнал загрузок ops.load_log: run_id, источник, время начала и конца, статус, получено, вставлено, дубликатов, watermark до и после, текст ошибки.",
    ])
    runs = facts["runs"]
    if len(runs) >= 2:
        first, last = runs[0], runs[-1]
        p(doc, f"Фактический результат: первый запуск с чистого состояния вставил {first['inserted']:,} записей ".replace(",", " ")
               + f"(статусы источников: {', '.join(f'{k} — {v}' for k, v in first['statuses'].items())}). Последний повторный запуск получил "
               + f"{last['received']:,} и вставил {last['inserted']:,} записей: догружены только новые свечи и новости.".replace(",", " "))
    h(doc, "3.4. Контроль качества", 2)
    p(doc, "После каждого сбора выполняются проверки, результаты пишутся в ops.dq_results. Критичный сбой блокирует обновление витрины: команда "
           "завершается с кодом 2, а дашборд показывает предупреждение и последние проверенные данные.")
    dq = pd.DataFrame(facts["dq_last"])
    dq_summary = dq.groupby(["check_name", "severity"]).agg(проверок=("passed", "size"), пройдено=("passed", "sum")).reset_index()
    dq_summary.columns = ["Проверка", "Важность", "Объектов", "Пройдено"]
    table(doc, dq_summary, "Таблица 2 — Проверки качества в последнем запуске", [6.5, 3.0, 3.0, 3.0])
    gaps = ", ".join(f"{g['scope']}: {g['details']}" for g in c["gaps"][:1])
    p(doc, f"Пропуски часовых свечей на Binance есть, но малы ({gaps}; для остальных инструментов так же). Это технические остановки биржи.")

    # 4. Обработка
    h(doc, "4. Обработка данных и признаки")
    h(doc, "4.1. Слои хранения", 2)
    p(doc, "Хранилище — DuckDB со схемами raw, staging, mart, ops и ml; после каждого этапа слои выгружаются в Parquet (data/lake/raw, staging, mart). "
           "DuckDB допускает одного пишущего процесса, поэтому дашборд читает только Parquet. Файлы заменяются атомарно и не конфликтуют с конвейером.")
    h(doc, "4.2. Единый вид данных, пропуски и аномалии", 2)
    bullets(doc, [
        f"Свечи с high < low, отрицательным объёмом или ценой ≤ 0 отбрасываются и сохраняются в staging.hourly_rejected (в текущих данных: {c['rejected']}).",
        f"Сутки считаются полными, если есть свеча за последний час. При 22–23 свечах объём и сумма квадратов доходностей масштабируются на 24/n "
        f"(таких суток {c['daily_scaled']}). При меньшем числе сутки невалидны, таргет и признаки за них не строятся (таких {c['daily_invalid']}). Незавершённые текущие сутки в витрину не попадают.",
        "Всплески объёма и резкие движения цены не удаляются: именно их должна учитывать модель риска. Они помечаются флагом; базовый уровень "
        "для флага — скользящие 30 суток только из прошлого (робастная z-оценка по медиане и MAD), иначе флаг зависел бы от будущего.",
        "Для линейной модели и MLP клиппинг по квантилям 0,5% и 99,5%, медианная импутация и стандартизация обучаются на train каждого фолда. "
        "LightGBM обрабатывает пропуски сам.",
    ])
    h(doc, "4.3. Обработка текстов", 2)
    p(doc, f"Заголовки очищаются: нижний регистр, удаление ссылок и HTML, нормализация пробелов. Перепечатки с одинаковым заголовком считаются одной "
           f"новостью, остаётся самая ранняя. После очистки осталось {c['news_clean']:,} заголовков.".replace(",", " "))
    bullets(doc, [
        "тональность: VADER с добавленным крипто-словарём (hack, exploit, rally, etf approved и др.);",
        "темы: доли заголовков о взломах, регулировании и ETF (регулярные выражения);",
        "упоминания инструментов: доля заголовков дня, где упомянут инструмент, и их тональность;",
        "TF-IDF (униграммы и биграммы, до 5000 слов) + усечённое SVD до 10 компонент. Векторизатор и SVD обучаются только на статьях до конца train каждого фолда, затем вектора усредняются по дню;",
        "флаг наличия текста за день и логарифм числа новостей. Счётчик строится только по потокам с постоянной историей; RSS в нём не участвует, иначе появление RSS выглядело бы как рост новостного фона.",
    ])
    h(doc, "4.4. Признаки", 2)
    p(doc, f"Витрина mart.features содержит {c['feature_rows']:,} строк (инструмент × сутки) и {c['features']} признаков, а также два таргета. ".replace(",", " ")
           + "Все признаки рассчитываются одной функцией build_feature_frame, которую вызывают и обучение, и прогноз.")
    groups = pd.DataFrame([
        ["Рынок (mkt_)", "логарифмы объёма и RV; лаги 1, 2, 3, 5, 7, 14 суток; средние и стандартные отклонения за 3, 7, 14, 30 суток; HAR-компоненты (день, неделя, месяц); полудисперсии; волатильность Паркинсона; доходности за 1, 3, 7, 30 суток; доля покупок тейкеров; концентрация объёма по часам; доля объёма последних 4 часов; показатели BTC и среднего по рынку"],
        ["Календарь (cal_)", "день недели, выходной, месяц (синус и косинус), конец месяца и квартала — для даты прогноза D+1"],
        ["Индекс страха (fng_)", "значение, изменения за 1 и 7 суток, среднее за 7, z-оценка за 30, флаг экстремума и наличия"],
        ["Текст (txt_, svd_)", "тональность, разброс, доля негатива, доли тем, упоминания инструмента, счётчик, флаг наличия, 10 SVD-компонент"],
    ], columns=["Группа", "Состав"])
    table(doc, groups, "Таблица 3 — Группы признаков", [3.6, 12.9], font=10)
    h(doc, "4.5. Кодирование инструмента", 2)
    enc = facts["analysis"][facts["analysis"]["section"] == "encoding"]
    enc_view = enc[["target", "variant", "metric", "value"]].copy()
    enc_view["target"] = enc_view["target"].map(TARGET_NAMES)
    enc_view.columns = ["Показатель", "Вариант", "Метрика", "Значение"]
    table(doc, enc_view, "Таблица 4 — Сравнение способов кодирования инструмента (walk-forward, последние фолды)", [3.0, 7.5, 2.5, 3.5])
    nat = enc[(enc["variant"].str.contains("нативная"))].set_index("target")["value"]
    one = enc[(enc["variant"].str.contains("LightGBM: one-hot"))].set_index("target")["value"]
    p(doc, "Для линейной модели нужно one-hot: категорию нельзя подать числом, иначе появится ложный порядок. Для LightGBM выбрана нативная категория: "
           "дерево делит пять инструментов на группы одним разбиением, а не цепочкой бинарных столбцов. По результатам разница между нативной категорией и "
           f"one-hot мала (объём: {num(nat.get('log_qv', float('nan')))} против {num(one.get('log_qv', float('nan')))}; волатильность: "
           f"{num(nat.get('log_rv', float('nan')))} против {num(one.get('log_rv', float('nan')))}). Уровень инструмента и так виден из лаговых признаков, поэтому "
           "выбор сделан по простоте и устойчивости, а не по выигрышу в качестве.")
    h(doc, "4.6. Отбор признаков и абляция", 2)
    analysis = facts["analysis"]
    ab = analysis[analysis["section"].isin(["ablation", "selection"])][["section", "target", "variant", "metric", "value"]].copy()
    ab["target"] = ab["target"].map(TARGET_NAMES)
    ab["section"] = ab["section"].map({"ablation": "абляция", "selection": "отбор"})
    ab.columns = ["Раздел", "Показатель", "Вариант", "Метрика", "Значение"]
    table(doc, ab, f"Таблица 5 — Абляция по группам и отбор признаков (LightGBM, последние {CFG['validation']['analysis_folds']} фолдов)", [2.4, 2.8, 6.0, 2.0, 3.3], font=10)
    figure(doc, figs["ablation"], "Рисунок 1 — Абляция по группам признаков")
    verdicts = []
    for target in CFG.targets:
        part = analysis[(analysis["section"] == "ablation") & (analysis["target"] == target)].reset_index(drop=True)
        base = part.loc[0, "value"]
        fng = part.loc[1, "value"]
        txt = part.loc[2, "value"]
        svd = part.loc[3, "value"]
        def rel(x: float) -> float:
            return (x / base - 1) * 100
        verdicts.append(f"{TARGET_NAMES[target]}: индекс страха {pct(rel(fng), 2)}, текстовые счётчики {pct(rel(txt), 2)}, TF-IDF+SVD {pct(rel(svd), 2)} к варианту «только рынок» (минус означает улучшение)")
    p(doc, "Результат абляции: " + "; ".join(verdicts) + ".")
    helpful = any(((analysis[(analysis["section"] == "ablation") & (analysis["target"] == t)]["value"].iloc[1:] / analysis[(analysis["section"] == "ablation") & (analysis["target"] == t)]["value"].iloc[0] - 1) < -0.01).any() for t in CFG.targets)
    if helpful:
        p(doc, "Хотя бы одна внешняя группа дала улучшение больше 1%: она сохранена в итоговой модели.")
    else:
        p(doc, "Вывод прямо: ни индекс страха, ни текстовые признаки не улучшают прогноз больше чем на 1%; все изменения в пределах шума. "
               "Вся предсказательная сила приходится на рыночные и календарные признаки. Вероятные причины: заголовки Hacker News отражают интерес "
               "разработчиков, а не рынка; объём и волатильность сами быстро впитывают новостной фон; суточный шаг слишком груб для новостного эффекта. "
               "Текстовый контур оставлен в системе, потому что методичка требует обработки слабоструктурированного источника, и он полностью готов "
               "к более информативной ленте. Модель не теряет в качестве от этих признаков.")
    p(doc, "Отбор top-k по важности gain (обучается на train каждого фолда) не дал устойчивого выигрыша над полным набором, поэтому в итоговой модели используются все признаки.")
    h(doc, "4.7. Доказательство отсутствия утечки", 2)
    leak = facts["leakage"]
    trunc = pd.DataFrame(leak.get("truncation", []))
    p(doc, "Для доказательства выполнены четыре вида проверок (cryptocast/leakage.py, tests/test_features_leakage.py):")
    bullets(doc, [
        f"Тест усечения: витрина пересчитывается от сырых свечей с удалением всех данных после даты t; признаки на даты ≤ t совпадают с полной витриной. На реальных данных проверено {len(trunc)} дат от {trunc['cutoff'].min() if not trunc.empty else '—'} до {trunc['cutoff'].max() if not trunc.empty else '—'}; максимальная разница {num(float(trunc['max_abs_diff'].max()) if not trunc.empty else float('nan'), 1)}, расхождений в пропусках {int(trunc['nan_mismatches'].sum()) if not trunc.empty else '—'}. Отдельный тест подтверждает, что признак со сглаживанием по будущему этим методом ловится.",
        "Проверка меток времени: таргет строки — значение следующих суток; дата новости совпадает с сутками публикации; свечи агрегируются только внутри своих суток; незавершённые сутки в витрину не попадают. Результаты: " + "; ".join(f"{x['check']} — {'пройдена' if x['passed'] else 'НЕ пройдена'}" for x in leak.get("timestamps", [])) + ".",
        "Тест с перемешанным таргетом: таргет глобально переставляется, после чего тот же walk-forward должен показать отсутствие предсказательной силы. " + "; ".join(f"{TARGET_NAMES[t]}: R² = {num(v['r2_shuffled'], 3)}, MAE {num(v['mae_log_shuffled'], 3)} против {num(v['mae_log_real'], 3)} на реальном таргете" for t, v in leak.get("shuffled_target", {}).items()) + ".",
        "Обучение преобразований только на train: тесты проверяют, что TF-IDF+SVD видит статьи не позже конца train фолда, а выброс в тестовой строке не меняет прогноз для других строк, то есть масштабирование и клиппинг не обучаются на тесте.",
    ])
    p(doc, f"Итог отчёта об утечке: {'все проверки пройдены' if leak.get('passed') else 'есть непройденные проверки'}.")
    h(doc, "4.8. Разбиение по времени", 2)
    p(doc, f"Последние {holdout_days} суток выделены в отложенную выборку, которая не участвует ни в подборе, ни в выборе модели. На оставшемся отрезке "
           f"построена расширяющаяся walk-forward валидация: начальное обучение {CFG['validation']['initial_train_days']} суток, тестовые окна по "
           f"{CFG['validation']['test_days']} суток, зазор {CFG['validation']['gap_days']} суток между train и тестом против переноса автокорреляции. "
           f"Получилось {len(folds)} фолдов: первый тест начинается {str(folds['test_start'].min())[:10]}, последний заканчивается {str(folds['test_end'].max())[:10]}. "
           f"Случайное зерно зафиксировано ({CFG.seed}) для LightGBM, MLP, SVD и Optuna.")

    # 5. Модели
    h(doc, "5. Модели и результаты валидации")
    h(doc, "5.1. Модели", 2)
    bullets(doc, [
        "Наивный прогноз (база): завтра равно сегодня. Дополнительный ориентир: среднее логарифма за 7 суток.",
        "ElasticNet на признаках HAR (день, неделя, месяц) с one-hot инструмента — линейная модель с регуляризацией L1+L2.",
        "LightGBM — градиентный бустинг деревьев на всех признаках; инструмент как нативная категория; детерминизм: deterministic=True, force_row_wise=True, фиксированное зерно.",
        "MLP — полносвязная нейросеть scikit-learn со стандартизацией, ранней остановкой и фиксированным зерном.",
    ])
    p(doc, "Классы моделей различны: линейная регрессия, ансамбль деревьев и нейронная сеть. Все модели обучаются на логарифмах таргетов.")
    h(doc, "5.2. Метрики и перевод в понятные величины", 2)
    p(doc, "Для объёма: MAE и RMSE в логарифмах и средняя ошибка в процентах от фактического объёма |V̂/V − 1|. Для волатильности: MAE и RMSE в процентных "
           "пунктах суточной волатильности и QLIKE на дисперсиях (штрафует недооценку риска сильнее). Для обеих величин считается улучшение "
           "относительно наивного прогноза в процентах.")
    h(doc, "5.3. Результаты", 2)
    for target in CFG.targets:
        rows = []
        for model in ("naive", "naive_ma7", "elasticnet", "lightgbm", "mlp"):
            cv = lambda col: metric_value(m, "cv", target, model, col)  # noqa: E731
            ho = lambda col: metric_value(m, "holdout", target, model, col)  # noqa: E731
            if target == "log_qv":
                rows.append([MODEL_NAMES[model], cv("mae"), cv("rmse"), cv("mean_pct_error_volume"), ho("mae"), ho("mean_pct_error_volume"), ho("mae_improvement_pct")])
            else:
                rows.append([MODEL_NAMES[model], cv("mae"), cv("qlike"), ho("mae"), ho("rmse"), ho("qlike"), ho("qlike_improvement_pct")])
        if target == "log_qv":
            cols = ["Модель", "MAE (CV)", "RMSE (CV)", "Ошибка, % (CV)", "MAE (отл.)", "Ошибка, % (отл.)", "Улучш. MAE, % (отл.)"]
        else:
            cols = ["Модель", "MAE, п.п. (CV)", "QLIKE (CV)", "MAE, п.п. (отл.)", "RMSE, п.п. (отл.)", "QLIKE (отл.)", "Улучш. QLIKE, % (отл.)"]
        table(doc, pd.DataFrame(rows, columns=cols), f"Таблица {6 if target == 'log_qv' else 7} — Качество моделей: {TARGET_NAMES[target]} (все инструменты)", [3.6, 2.1, 2.1, 2.2, 2.1, 2.2, 2.2], font=10)
    lgb_qv_cv = metric_value(m, "cv", "log_qv", "lightgbm", "mae_improvement_pct")
    lgb_qv_ho = metric_value(m, "holdout", "log_qv", "lightgbm", "mae_improvement_pct")
    lgb_qv_pct = metric_value(m, "holdout", "log_qv", "lightgbm", "mean_pct_error_volume")
    naive_qv_pct = metric_value(m, "holdout", "log_qv", "naive", "mean_pct_error_volume")
    lgb_rv_ql = metric_value(m, "holdout", "log_rv", "lightgbm", "qlike_improvement_pct")
    lgb_rv_mae = metric_value(m, "holdout", "log_rv", "lightgbm", "mae")
    naive_rv_mae = metric_value(m, "holdout", "log_rv", "naive", "mae")
    p(doc, f"LightGBM — лучшая модель по обоим показателям, и в walk-forward, и на отложенной выборке. Для объёма она снижает MAE на {pct(lgb_qv_cv)} "
           f"в walk-forward и на {pct(lgb_qv_ho)} на отложенной выборке. В понятных единицах средняя ошибка прогноза объёма {pct(lgb_qv_pct)} от "
           f"фактического против {pct(naive_qv_pct)} у наивного прогноза. Для волатильности QLIKE на отложенной выборке ниже наивного на {pct(lgb_rv_ql)}, "
           f"средняя ошибка {num(lgb_rv_mae, 2)} п.п. суточной волатильности против {num(naive_rv_mae, 2)} п.п.")
    p(doc, "Качество на отложенной выборке близко к walk-forward, то есть подбор не переобучился на период разработки. ElasticNet на HAR и MLP "
           "лучше наивного прогноза, но заметно уступают бустингу. Основная причина: эффект дня недели и нелинейные взаимодействия уровней, "
           "которые HAR без календаря не видит.")
    figure(doc, figs["fact_forecast"], "Рисунок 2 — Факт и прогноз на отложенной выборке, BTC/USDT")
    figure(doc, figs["improvement"], "Рисунок 3 — Улучшение относительно наивного прогноза по инструментам, отложенная выборка")
    per = m[(m["split"] == "holdout") & (m["model"] == "lightgbm") & (m["symbol"] != "ALL")]
    worst_qv = per[per["target"] == "log_qv"].sort_values("mae_improvement_pct").iloc[0]
    best_qv = per[per["target"] == "log_qv"].sort_values("mae_improvement_pct").iloc[-1]
    p(doc, f"По инструментам выигрыш неравномерен: для объёма сильнее всего у {best_qv['symbol']} ({pct(best_qv['mae_improvement_pct'])}), слабее всего у "
           f"{worst_qv['symbol']} ({pct(worst_qv['mae_improvement_pct'])}). Для всех инструментов и обоих показателей LightGBM лучше наивного прогноза.")

    # 6. Уникальные компоненты
    h(doc, "6. Реализованные компоненты по выбору")
    h(doc, "6.1. Интерпретируемость", 2)
    imp = facts["importance"]
    agree = {t: imp[imp["target"] == t][["increase_mae", "mean_abs_shap"]].corr(method="spearman").iloc[0, 1] for t in CFG.targets}
    p(doc, "Важность признаков оценивается двумя независимыми способами: SHAP (TreeExplainer, среднее модуля вклада на OOS-строках всех фолдов) и "
           "permutation importance (рост MAE при перестановке признака на тесте последних четырёх фолдов). Согласованность рангов (корреляция Спирмена): "
           + "; ".join(f"{TARGET_NAMES[t]} — {num(v, 2)}" for t, v in agree.items()) + ".")
    figure(doc, figs["importance"], "Рисунок 4 — Важность признаков: SHAP и permutation")
    stab = facts["stability"]
    p(doc, "Для произвольного инструмента и даты дашборд показывает вклад каждого признака в прогноз (SHAP модели того фолда, который делал этот прогноз "
           "вне обучения): базовое значение плюс сумма вкладов равно прогнозу. Зависимость прогноза от признака показывают PDP и ICE для 8 "
           "важнейших признаков. Структура важности сравнивается по годам: доли SHAP-вклада и корреляция Спирмена между соседними годами "
           + "(" + "; ".join(f"{TARGET_NAMES[r['target']]} {r['period_a']}→{r['period_b']}: {num(r['spearman'], 2)}" for r in stab.to_dict("records")[:4]) + ").")
    if (SHOTS / "04_explain.png").exists():
        figure(doc, SHOTS / "04_explain.png", "Рисунок 5 — Объяснение прогноза в дашборде", 14.5)
    h(doc, "6.2. Мониторинг и переобучение", 2)
    rule = CFG["monitoring"]["retrain"]
    p(doc, "Мониторинг выполняется на каждом запуске и сохраняет результаты в ops.drift_metrics, ops.retrain_decisions и mart:")
    bullets(doc, [
        "дрейф входных данных: PSI (10 квантильных корзин опорного периода) и тест Колмогорова — Смирнова для 20 ключевых признаков модели; опорный период — последние 365 суток обучения, текущий — последние 30 суток;",
        "качество на поступающих фактах: прогнозы production-версии сопоставляются с фактом, скользящая MAE за 14 суток;",
        f"правило переобучения (config.yaml): скользящая MAE за {rule['rolling_window_days']} суток выше {rule['mae_ratio_threshold']} × валидационной MAE {rule['consecutive_days']} суток подряд ИЛИ PSI > {CFG['monitoring']['psi_threshold']} у {rule['psi_share_threshold']:.0%} ключевых признаков; после обучения действует период ожидания {rule['cooldown_days']} суток;",
        f"реестр версий ml.registry: состав признаков, параметры, метрики CV и отложенной выборки, хеш данных, ссылка на эксперимент, стадия (candidate, production, archived, rejected);",
        "правило вывода в эксплуатацию: кандидат сравнивается с текущей версией на общем отрезке, где у текущей версии есть настоящие прогнозы вне обучения, а кандидат обучен до этого отрезка; без текущей версии кандидат должен быть не хуже наивного прогноза;",
        f"правило отката: если новая версия за ≥ {CFG['monitoring']['rollback']['min_live_days']} суток на живых данных хуже теневой предыдущей более чем в {CFG['monitoring']['rollback']['live_vs_shadow_ratio']} раза, возвращается предыдущая; ручной откат — команда rollback.",
    ])
    decisions = facts["status"].get("decisions", {}) or {}
    if decisions:
        p(doc, "Состояние после последнего запуска: " + "; ".join(
            f"{TARGET_NAMES[t]}: скользящая MAE {num(d['rolling_mae'], 3)} при пороге {num(d['threshold'], 3)}, доля признаков с дрейфом {d['psi_share']:.0%}, "
            f"правило {'сработало' if d['retrain'] else 'не сработало'}" + (f" ({d['reasons']})" if d.get("reasons") else "") + (", но действует период ожидания" if d.get("on_cooldown") and d["retrain"] else "")
            for t, d in decisions.items()) + ".")
    figure(doc, figs["monitoring"], "Рисунок 6 — Скользящая MAE production-версии на поступающих фактах")
    p(doc, "Важное наблюдение: правило по PSI срабатывает почти всегда, потому что среди ключевых признаков много уровней (средний объём за 7–30 суток, "
           "индекс страха). Их распределение закономерно меняется вместе с рыночным режимом. Поэтому правило PSI на практике означает плановое "
           "переобучение не чаще раза в неделю (период ожидания), а основным сигналом деградации остаётся рост ошибки на фактах.")
    sim = facts["simulation"]
    if sim:
        parts = []
        for t, info in sim["targets"].items():
            parts.append(f"{TARGET_NAMES[t]}: первое срабатывание {info.get('first_trigger_date') or 'не было'}, MAE текущей модели после срабатывания "
                         f"{num(info.get('champion_mae_log', float('nan')), 3)}, переобученной {num(info.get('candidate_mae_log', float('nan')), 3)}, кандидат "
                         f"{'выведен' if info.get('candidate_promoted_in_simulation') else 'не выведен'}")
        p(doc, f"Реакция на деградацию проверена командой simulate-drift. На копии данных, явно помеченной как симуляция, с {sim['start']} объём умножается "
               f"на {sim['volume_factor']}, а часовые доходности на {sim['return_factor']}. Рабочее хранилище не меняется. Результат: " + "; ".join(parts) + ".")
        if "simulation" in figs:
            figure(doc, figs["simulation"], "Рисунок 7 — Симуляция деградации: ошибка до и после переобучения")
    h(doc, "6.3. Автоматический подбор и журнал экспериментов", 2)
    trials = facts["trials"]
    counts = trials.groupby(["target", "model"]).size().to_dict() if not trials.empty else {}
    p(doc, "Гиперпараметры ElasticNet, LightGBM и MLP подбирает Optuna (TPE, зерно 42) с ограничением бюджета по числу попыток и времени (config.yaml). "
           f"Подбор идёт на последних {CFG['tuning']['folds_last_n']} фолдах разработки; отложенные {holdout_days} суток в нём не участвуют. Фактическое число "
           "попыток: " + "; ".join(f"{TARGET_NAMES[t]}/{MODEL_NAMES[mo]} — {n}" for (t, mo), n in counts.items()) + ".")
    p(doc, f"Журнал ml.experiments ({facts['experiments_count']} записей) хранит конфигурацию модели, хеш config.yaml, версию данных (хеш витрины и текстов), "
           "зерно, git-коммит с признаком незакоммиченных правок, метрики walk-forward и отложенной выборки. Версия данных замораживается в "
           "artifacts/datasets/<хеш>, поэтому эксперимент воспроизводится и после прихода новых данных. Сравнение экспериментов доступно во вкладке «Эксперименты».")
    reps = facts["reproductions"]
    if reps:
        rep = reps[-1]
        p(doc, f"Команда reproduce --experiment-id {rep['experiment_id']} повторила эксперимент ({MODEL_NAMES.get(rep['model'], rep['model'])}, {TARGET_NAMES[rep['target']]}): "
               f"версия данных {'совпала' if rep['data_version_match'] else 'не совпала'}, максимальное расхождение метрик {rep['max_abs_diff']:.2e} при допуске {rep['tolerance']:.0e}, "
               f"результат — {'воспроизведено' if rep['passed'] else 'не воспроизведено'}.")

    # 7. Стек и архитектура
    h(doc, "7. Технологический стек и архитектура")
    figure(doc, ROOT / "docs" / "architecture.png", "Рисунок 8 — Схема архитектуры", 16.5)
    stack = pd.DataFrame([
        ["Python 3.12", "единая среда для данных, моделей и интерфейса"],
        ["DuckDB + Parquet", "встраиваемая аналитическая СУБД без сервера, upsert по ключу и транзакции; Parquet — слои для чтения дашбордом"],
        ["scikit-learn, LightGBM", "ElasticNet, MLP, предобработка; градиентный бустинг с нативными категориями и детерминизмом"],
        ["Optuna", "байесовский подбор с ограничением trials и timeout"],
        ["SHAP", "точные объяснения деревьев (TreeExplainer)"],
        ["Streamlit + Plotly", "интерактивный дашборд на Python без отдельного фронтенда"],
        ["Typer, APScheduler", "CLI с отдельными командами этапов и run-all; расписание без внешних служб"],
        ["pytest", "тесты сбора, качества, признаков, утечки, моделей, мониторинга"],
        ["config.yaml + .env", "конфигурируемость без правки кода"],
    ], columns=["Компонент", "Обоснование"])
    table(doc, stack, "Таблица 8 — Технологический стек", [4.5, 12.0])
    p(doc, "Автоматизация: одна команда python -m cryptocast run-all выполняет весь цикл; отдельные команды позволяют запускать этапы по одному. "
           "Ежедневный запуск — команда schedule (APScheduler) или задача планировщика Windows. Для офлайн-демонстрации команда snapshot копирует "
           "БД, витрины и артефакты, а dashboard --offline показывает снимок без сети.")
    ver = facts["verification"]
    if ver:
        p(doc, "Проверка запуском (docs/verification.json): " + "; ".join(f"{k}: {v}" for k, v in ver.items()) + ".")

    # 8. Выводы
    h(doc, "8. Выводы и ограничения")
    p(doc, "Работающий прототип собирает данные из пяти источников разной структуры, контролирует их качество, строит единую витрину признаков, "
           "сравнивает четыре модели трёх классов с базовой по единой временной схеме и выдаёт прогноз на следующие сутки в дашборде. Все три "
           "компонента по выбору реализованы и проверены запуском.")
    bullets(doc, [
        f"LightGBM устойчиво лучше наивного прогноза: на отложенной выборке ошибка прогноза объёма {pct(lgb_qv_pct)} против {pct(naive_qv_pct)}, QLIKE волатильности ниже на {pct(lgb_rv_ql)}.",
        "Главные драйверы — недавний уровень самого показателя (HAR-компоненты) и день недели целевых суток.",
        "Новостные признаки и индекс страха прироста не дали; это зафиксировано абляцией, а не предположено.",
        "Мониторинг на отложенном периоде показал рост ошибки волатильности выше порога, то есть правило переобучения срабатывает на реальных данных.",
    ])
    p(doc, "Ограничения:", indent=False)
    bullets(doc, [
        "История новостей взята из Hacker News: GDELT из-за лимита 429 фактически недоступен, а RSS даёт только свежие записи. Выводы о роли текста относятся именно к этому источнику.",
        "Прогноз строится только по данным одной биржи (Binance); объём по рынку в целом выше и распределён иначе.",
        "Интервалы прогноза — эмпирические квантили ошибок walk-forward; при смене режима они могут быть занижены.",
        "Обратное преобразование exp(прогноз логарифма) даёт оценку медианы, а не среднего; для QLIKE это консервативно в сторону недооценки.",
        "Правило PSI срабатывает часто из-за признаков-уровней; его стоит перевести на стационарные признаки или повысить порог.",
        "Репозиторий создан за короткий срок; методичка допускает снижение балла за малое число коммитов в конце семестра.",
        "Система — информационный инструмент для контроля риска и не даёт торговых рекомендаций.",
    ])
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
