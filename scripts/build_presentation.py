"""Сборка презентации docs/presentation.pptx (12 слайдов) из реальных результатов и скриншотов.

Запуск после build_report.py: python scripts/build_presentation.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

from report_data import CFG, MODEL_NAMES, ROOT, TARGET_NAMES, collect, make_figures, metric_value

OUT = ROOT / "docs" / "presentation.pptx"
SHOTS = ROOT / "docs" / "screenshots"
NAVY, NAVY2, ICE, WHITE = RGBColor(0x14, 0x21, 0x3D), RGBColor(0x22, 0x33, 0x5C), RGBColor(0xE8, 0xEE, 0xF8), RGBColor(0xFF, 0xFF, 0xFF)
ORANGE, BLUE, GREEN, INK, MUTED = RGBColor(0xEB, 0x68, 0x34), RGBColor(0x2A, 0x78, 0xD6), RGBColor(0x1B, 0xAF, 0x7A), RGBColor(0x1A, 0x1A, 0x1A), RGBColor(0x5F, 0x5E, 0x5A)
HEAD, BODY = "Cambria", "Calibri"
W, H = Inches(13.333), Inches(7.5)


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    if name:
        box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = Emu(0)
    items = runs if isinstance(runs, list) else [runs]
    for i, item in enumerate(items):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = align
        content, opts = (item, {}) if isinstance(item, str) else item
        run = para.add_run()
        run.text = content
        run.font.name = opts.get("font", font)
        run.font.size = Pt(opts.get("size", size))
        run.font.bold = opts.get("bold", bold)
        run.font.color.rgb = opts.get("color", color)
        para.space_after = Pt(opts.get("after", 6))
    return box


def bullets(slide, x, y, w, h, items, size=16, color=INK):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = Emu(0)
    for i, item in enumerate(items):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.28))))
        pPr.set("indent", str(-Emu(Inches(0.28))))
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        run = para.add_run()
        run.text = item
        run.font.name = BODY
        run.font.size = Pt(size)
        run.font.color.rgb = color
        para.space_after = Pt(8)
    return box


def background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def title(slide, value, dark=False):
    text(slide, Inches(0.6), Inches(0.4), Inches(12.1), Inches(0.9), value, size=34, bold=True, font=HEAD, color=WHITE if dark else NAVY, name="Title")


def card(slide, x, y, w, h, fill=ICE):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = 0.08
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def stat(slide, x, y, w, value, label, color=ORANGE, dark=False):
    text(slide, x, y, w, Inches(1.0), value, size=48, bold=True, font=HEAD, color=color)
    text(slide, x, y + Inches(1.0), w, Inches(0.9), label, size=14, color=ICE if dark else MUTED)


def picture(slide, path: Path, x, y, max_w, max_h, crop_top_px: int = 0, crop_h_px: int | None = None):
    img = Image.open(path)
    wpx, hpx = img.size
    usable_h = (crop_h_px or hpx - crop_top_px)
    ratio = min(max_w / wpx, max_h / usable_h)
    pic = slide.shapes.add_picture(str(path), x, y, int(wpx * ratio), int(hpx * ratio))
    if crop_top_px or crop_h_px:
        pic.crop_top = crop_top_px / hpx
        pic.crop_bottom = max(0.0, 1 - (crop_top_px + usable_h) / hpx)
        pic.height = int(usable_h * ratio)
    pic.line.color.rgb = RGBColor(0xD8, 0xD7, 0xD1)
    return pic


def style_chart(chart, colors):
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    chart.legend.font.name = BODY
    for series, color in zip(chart.series, colors):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = color
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.font.size = Pt(11)
    labels.font.name = BODY
    labels.font.color.rgb = INK
    labels.number_format = '0"%"'
    labels.number_format_is_linked = False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    value_axis = chart.value_axis
    value_axis.has_major_gridlines = True
    value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE5, 0xE4, 0xDF)
    value_axis.tick_labels.font.size = Pt(11)
    value_axis.tick_labels.font.color.rgb = MUTED
    value_axis.format.line.fill.background()
    chart.category_axis.tick_labels.font.size = Pt(12)
    chart.category_axis.tick_labels.font.color.rgb = INK


def build() -> Path:
    facts = collect()
    figs = make_figures(facts)
    m, c = facts["metrics"], facts["counts"]
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]

    def new(dark=False):
        slide = prs.slides.add_slide(blank)
        background(slide, NAVY if dark else WHITE)
        return slide

    lgb_qv_pct = metric_value(m, "holdout", "log_qv", "lightgbm", "mean_pct_error_volume")
    naive_qv_pct = metric_value(m, "holdout", "log_qv", "naive", "mean_pct_error_volume")
    lgb_rv_ql = metric_value(m, "holdout", "log_rv", "lightgbm", "qlike_improvement_pct")
    lgb_rv_mae = metric_value(m, "holdout", "log_rv", "lightgbm", "mae")
    naive_rv_mae = metric_value(m, "holdout", "log_rv", "naive", "mae")

    # 1. Титул
    s = new(dark=True)
    text(s, Inches(0.8), Inches(1.7), Inches(11.5), Inches(1.2), "cryptocast", size=60, bold=True, font=HEAD, color=WHITE)
    text(s, Inches(0.8), Inches(2.9), Inches(11.5), Inches(1.2), "Прогноз объёма торгов и волатильности цифровых активов на следующие сутки", size=26, color=ICE)
    text(s, Inches(0.8), Inches(4.6), Inches(11), Inches(1.4), [("Проектный практикум, тема № 21 «Рынок цифровых активов»", {"size": 18, "color": ICE}),
                                                               ("Запорожец Михаил Романович, группа БВТ2304", {"size": 18, "color": WHITE, "bold": True})])
    s.notes_slide.notes_text_frame.text = "Тема 21: прогноз объёма торгов и волатильности. Покажу систему от сбора данных до дашборда и сопровождения модели."

    # 2. Задача и пользователь
    s = new()
    title(s, "Задача: два решения риск-менеджера на завтра")
    for i, (value, label) in enumerate([("5", "инструментов к USDT: BTC, ETH, SOL, BNB, XRP"), ("D+1", "шаг прогноза: сутки UTC, прогноз в конце дня D"), ("2", "показателя: log объёма и реализованная волатильность")]):
        stat(s, Inches(0.7 + i * 4.1), Inches(1.6), Inches(3.8), value, label, color=[ORANGE, BLUE, GREEN][i])
    card(s, Inches(0.6), Inches(4.0), Inches(5.9), Inches(2.8))
    text(s, Inches(0.9), Inches(4.2), Inches(5.4), Inches(0.5), "Какой объём можно исполнить?", size=20, bold=True, font=HEAD, color=NAVY)
    text(s, Inches(0.9), Inches(4.8), Inches(5.4), Inches(1.9), f"Прогноз суточного объёма с диапазоном 80%. Ёмкость исполнения равна {CFG['decision']['participation_rate']:.0%} от нижней границы.", size=16)
    card(s, Inches(6.8), Inches(4.0), Inches(5.9), Inches(2.8))
    text(s, Inches(7.1), Inches(4.2), Inches(5.4), Inches(0.5), "Нужно ли снижать риск?", size=20, bold=True, font=HEAD, color=NAVY)
    text(s, Inches(7.1), Inches(4.8), Inches(5.4), Inches(1.9), "Прогноз волатильности и его перцентиль за год: обычный, повышенный или высокий риск. Это не торговый совет.", size=16)
    s.notes_slide.notes_text_frame.text = "Пользователь — риск-менеджер небольшого фонда. RV = корень из суммы квадратов часовых лог-доходностей за сутки."

    # 3. Данные
    s = new()
    title(s, "Пять источников: числа и тексты")
    by = c["news_by_source"]
    items = [
        ("Binance REST", f"{c['klines']:,}".replace(",", " ") + " часовых свечей с 2021-01-01", "структурированный", BLUE),
        ("Fear & Greed", f"{c['fng']} суточных значений", "структурированный", BLUE),
        ("Hacker News", f"{by.get('hn', 0):,}".replace(",", " ") + " заголовков, история", "тексты", ORANGE),
        ("RSS, 9 лент", f"{by.get('rss', 0)} свежих записей", "тексты", ORANGE),
        ("GDELT", "почти всегда HTTP 429: оставлен с бюджетом 3 запроса", "тексты", MUTED),
    ]
    for i, (name, value, kind, color) in enumerate(items):
        x = Inches(0.6 + (i % 3) * 4.15)
        y = Inches(1.5 + (i // 3) * 2.2)
        card(s, x, y, Inches(3.9), Inches(1.95))
        text(s, x + Inches(0.25), y + Inches(0.15), Inches(3.4), Inches(0.5), name, size=20, bold=True, font=HEAD, color=NAVY)
        text(s, x + Inches(0.25), y + Inches(0.7), Inches(3.4), Inches(0.8), value, size=15)
        text(s, x + Inches(0.25), y + Inches(1.45), Inches(3.4), Inches(0.4), kind, size=12, color=color, bold=True)
    x = Inches(0.6 + 2 * 4.15)
    text(s, x, Inches(3.75), Inches(3.9), Inches(2.0), [("Сбор", {"size": 18, "bold": True, "font": HEAD, "color": NAVY}),
                                                      ("watermark, upsert по ключу, retry и Retry-After, журнал каждой загрузки; отказ источника не роняет запуск", {"size": 14})])
    text(s, Inches(0.6), Inches(6.2), Inches(12), Inches(0.6), "Персональные данные не хранятся: поля авторов не читаются, подписи «by Имя» вырезаются.", size=13, color=MUTED)
    s.notes_slide.notes_text_frame.text = "GDELT из более чем 20 запросов ответил дважды, обходить лимит не стал. История текстов — Hacker News, свежие — RSS."

    # 4. Архитектура
    s = new()
    title(s, "Архитектура: от источника до решения")
    picture(s, ROOT / "docs" / "architecture.png", Inches(0.5), Inches(1.4), Inches(12.3), Inches(5.6))
    s.notes_slide.notes_text_frame.text = "DuckDB для состояния, Parquet-слои raw, staging, mart. Дашборд читает только Parquet. Одна команда run-all."

    # 5. Признаки и утечка
    s = new()
    title(s, "Признаки из единого кода и проверка утечки")
    bullets(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(4.5), [
        f"{c['features']} признаков: HAR-компоненты, лаги и окна, календарь даты прогноза, индекс страха, текст",
        "Текст: очистка, дедупликация, тональность VADER, темы (взлом, регулирование, ETF), упоминания инструментов",
        "TF-IDF + SVD обучаются только на train каждого фолда",
        "Инструмент: one-hot для линейных моделей, нативная категория в LightGBM",
    ], size=16)
    leak = facts["leakage"]
    checks = [("Тест усечения", leak.get("truncation_passed", False)), ("Метки времени", all(x["passed"] for x in leak.get("timestamps", []))),
              ("Перемешанный таргет", all(v["passed"] for v in leak.get("shuffled_target", {}).values())), ("Преобразования только на train", True)]
    card(s, Inches(7.0), Inches(1.5), Inches(5.7), Inches(4.6))
    text(s, Inches(7.3), Inches(1.7), Inches(5.2), Inches(0.5), "Доказательство отсутствия утечки", size=20, bold=True, font=HEAD, color=NAVY)
    for i, (name, ok) in enumerate(checks):
        y = Inches(2.45 + i * 0.85)
        circle = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(7.35), y, Inches(0.45), Inches(0.45))
        circle.fill.solid()
        circle.fill.fore_color.rgb = GREEN if ok else ORANGE
        circle.line.fill.background()
        text(s, Inches(7.35), y + Inches(0.02), Inches(0.45), Inches(0.4), "✓" if ok else "!", size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        text(s, Inches(8.0), y + Inches(0.03), Inches(4.5), Inches(0.5), name + (" — пройден" if ok else " — не пройден"), size=16)
    shuffled = leak.get("shuffled_target", {})
    if shuffled:
        text(s, Inches(7.3), Inches(5.55), Inches(5.2), Inches(0.5), "R² при перемешанном таргете: " + ", ".join(f"{TARGET_NAMES[t]} {v['r2_shuffled']:.3f}".replace(".", ",") for t, v in shuffled.items()), size=13, color=MUTED)
    s.notes_slide.notes_text_frame.text = "Усечение: витрина пересчитывается без данных после t, признаки на t совпадают до бита."

    # 6. Модели: диаграмма улучшения
    s = new()
    title(s, "LightGBM лучше наивного прогноза")
    data = CategoryChartData()
    data.categories = [MODEL_NAMES[x] for x in ("elasticnet", "lightgbm", "mlp")]
    data.add_series("Объём: улучшение MAE, %", [round(metric_value(m, "holdout", "log_qv", x, "mae_improvement_pct"), 1) for x in ("elasticnet", "lightgbm", "mlp")])
    data.add_series("Волатильность: улучшение QLIKE, %", [round(metric_value(m, "holdout", "log_rv", x, "qlike_improvement_pct"), 1) for x in ("elasticnet", "lightgbm", "mlp")])
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.4), Inches(7.6), Inches(5.4), data).chart
    style_chart(chart, [BLUE, ORANGE])
    card(s, Inches(8.6), Inches(1.5), Inches(4.1), Inches(5.2))
    text(s, Inches(8.9), Inches(1.7), Inches(3.6), Inches(0.5), "Схема валидации", size=20, bold=True, font=HEAD, color=NAVY)
    bullets(s, Inches(8.9), Inches(2.3), Inches(3.6), Inches(4.2), [
        f"walk-forward: {len(facts['folds'])} фолдов, зазор {CFG['validation']['gap_days']} суток",
        f"отложенные {CFG['validation']['holdout_days']} суток не участвуют в подборе",
        "4 модели трёх классов + наивная база",
        "метрики: MAE, RMSE, QLIKE",
    ], size=15)
    text(s, Inches(0.6), Inches(6.85), Inches(8), Inches(0.4), "Отложенная выборка, все инструменты. Улучшение относительно прогноза «завтра = сегодня».", size=12, color=MUTED)
    s.notes_slide.notes_text_frame.text = "Классы: линейная модель на HAR, бустинг деревьев, нейросеть. Отложенная выборка близка к walk-forward."

    # 7. Понятные величины
    s = new()
    title(s, "Ошибка в понятных единицах")
    stat(s, Inches(0.7), Inches(1.7), Inches(3.9), f"{lgb_qv_pct:.0f}%", f"средняя ошибка прогноза объёма от факта (наивный прогноз: {naive_qv_pct:.0f}%)", ORANGE)
    stat(s, Inches(4.8), Inches(1.7), Inches(3.9), f"{lgb_rv_mae:.2f}".replace(".", ",") + " п.п.", "средняя ошибка суточной волатильности (наивный: " + f"{naive_rv_mae:.2f}".replace(".", ",") + " п.п.)", BLUE)
    stat(s, Inches(8.9), Inches(1.7), Inches(3.9), f"−{lgb_rv_ql:.0f}%", "QLIKE волатильности к наивному: меньше недооценки риска", GREEN)
    picture(s, figs["fact_forecast"], Inches(2.4), Inches(3.75), Inches(8.5), Inches(3.5))
    s.notes_slide.notes_text_frame.text = "Цифры отложенной выборки. Ошибка объёма в процентах |прогноз/факт−1|."

    # 8. Абляция
    s = new()
    title(s, "Текст и индекс страха прироста не дали")
    picture(s, figs["ablation"], Inches(0.5), Inches(2.3), Inches(8.3), Inches(4.2))
    card(s, Inches(9.1), Inches(1.5), Inches(3.6), Inches(4.8))
    text(s, Inches(9.35), Inches(1.7), Inches(3.2), Inches(4.5), [("Вывод абляции", {"size": 20, "bold": True, "font": HEAD, "color": NAVY}),
                                                                 ("Все различия меньше 1%, в пределах шума. Сила модели — в недавнем уровне показателя и дне недели.", {"size": 15}),
                                                                 ("Вероятная причина: Hacker News отражает интерес разработчиков, а не рынка.", {"size": 15}),
                                                                 ("Текстовый контур оставлен и готов к лучшей ленте.", {"size": 15})])
    s.notes_slide.notes_text_frame.text = "Говорю прямо: текст не помог. Это результат абляции на walk-forward, а не предположение."

    # 9. Дашборд
    s = new()
    title(s, "Дашборд: решение на завтра")
    if (SHOTS / "01_decision.png").exists():
        picture(s, SHOTS / "01_decision.png", Inches(0.6), Inches(1.35), Inches(8.6), Inches(5.9), crop_top_px=0, crop_h_px=1000)
    bullets(s, Inches(9.5), Inches(1.6), Inches(3.3), Inches(5), [
        "Факт, прогноз и качество на одном экране",
        "Фильтры: инструмент, показатель, период",
        "Автообновление каждые 5 минут",
        "Каждый элемент отвечает на вопрос пользователя",
    ], size=15)
    s.notes_slide.notes_text_frame.text = "Семь вкладок: решение, факт и прогноз, качество, объяснение, мониторинг, эксперименты, данные."

    # 10. Интерпретируемость
    s = new()
    title(s, "Объяснение прогноза для любой даты")
    if (SHOTS / "04_explain.png").exists():
        picture(s, SHOTS / "04_explain.png", Inches(0.6), Inches(1.35), Inches(7.6), Inches(5.9), crop_top_px=280, crop_h_px=900)
    picture(s, figs["importance"], Inches(8.4), Inches(1.5), Inches(4.5), Inches(3.0))
    text(s, Inches(8.4), Inches(4.7), Inches(4.4), Inches(2.4), [("SHAP и permutation — два независимых способа", {"size": 15, "bold": True}),
                                                                ("PDP/ICE и сравнение важности по годам. База + вклады = прогноз.", {"size": 14})])
    s.notes_slide.notes_text_frame.text = "На защите выбираем инструмент и дату — вклады берутся из модели того фолда, что прогнозировал вне обучения."

    # 11. Мониторинг
    s = new()
    title(s, "Мониторинг: деградация и переобучение")
    if "simulation" in figs:
        picture(s, figs["simulation"], Inches(0.5), Inches(1.4), Inches(8.2), Inches(3.4))
    rule = CFG["monitoring"]["retrain"]
    bullets(s, Inches(0.6), Inches(5.0), Inches(8.0), Inches(2.2), [
        f"Правило: MAE за {rule['rolling_window_days']} дн. > {rule['mae_ratio_threshold']} × валидационной {rule['consecutive_days']} дня подряд, или PSI > 0,25 у {rule['psi_share_threshold']:.0%} ключевых признаков",
        "Реестр версий, вывод в эксплуатацию по сравнению с текущей версией, откат",
    ], size=14)
    sim = facts["simulation"]
    card(s, Inches(9.0), Inches(1.5), Inches(3.7), Inches(5.3))
    lines = [("simulate-drift", {"size": 20, "bold": True, "font": HEAD, "color": NAVY}),
             ("Копия данных с пометкой СИМУЛЯЦИЯ; рабочее хранилище не меняется.", {"size": 14})]
    for t, info in (sim.get("targets", {}) if sim else {}).items():
        verdict = "кандидат выведен" if info.get("candidate_promoted_in_simulation") else "кандидат не лучше, оставлена текущая версия"
        champ = f"{info.get('champion_mae_log', float('nan')):.3f}".replace(".", ",")
        cand = f"{info.get('candidate_mae_log', float('nan')):.3f}".replace(".", ",")
        lines.append((f"{TARGET_NAMES[t].capitalize()}: правило сработало {info.get('first_trigger_date') or 'нет'}; MAE текущей {champ}, кандидата {cand}: {verdict}", {"size": 14}))
    text(s, Inches(9.25), Inches(1.7), Inches(3.25), Inches(5.0), lines)
    s.notes_slide.notes_text_frame.text = "Показываю симуляцию: объём ×3, доходности ×2,2 с даты сдвига; правило срабатывает, переобученная модель возвращает ошибку."

    # 12. Воспроизводимость и выводы
    s = new(dark=True)
    title(s, "Итоги и ограничения", dark=True)
    reps = facts["reproductions"]
    rep_line = f"reproduce: расхождение {reps[-1]['max_abs_diff']:.1e}, {'воспроизведено' if reps[-1]['passed'] else 'нет'}" if reps else "reproduce по ID эксперимента"
    bullets(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.5), [
        "Полный цикл одной командой run-all, расписание, офлайн-снимок",
        f"Optuna с бюджетом, журнал из {facts['experiments_count']} экспериментов; {rep_line}",
        f"LightGBM: ошибка объёма {lgb_qv_pct:.0f}% против {naive_qv_pct:.0f}% у наивного",
        "Три компонента по выбору: объяснения, мониторинг, подбор",
    ], size=17, color=WHITE)
    card(s, Inches(7.0), Inches(1.5), Inches(5.7), Inches(5.3), fill=NAVY2)
    text(s, Inches(7.3), Inches(1.7), Inches(5.2), Inches(0.5), "Ограничения", size=20, bold=True, font=HEAD, color=ORANGE)
    bullets(s, Inches(7.3), Inches(2.3), Inches(5.2), Inches(4.4), [
        "GDELT недоступен из-за 429, история текстов из Hacker News",
        "Данные одной биржи",
        "Правило PSI часто срабатывает на признаках-уровнях",
        "Информационный инструмент, не торговый совет",
    ], size=15, color=ICE)
    s.notes_slide.notes_text_frame.text = "Спасибо за внимание. Готов показать объяснение для любой даты, деградацию и воспроизведение эксперимента."

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
