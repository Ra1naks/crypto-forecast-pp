"""Русские названия признаков для дашборда и отчёта."""
from __future__ import annotations

FEATURE_NAMES = {
    "mkt_log_qv": "объём сегодня (лог)", "mkt_log_rv": "волатильность сегодня (лог)",
    "mkt_log_qv_ma7": "средний объём 7д (лог)", "mkt_log_qv_ma30": "средний объём 30д (лог)",
    "mkt_log_rv_ma7": "средняя волатильность 7д (лог)", "mkt_log_rv_ma30": "средняя волатильность 30д (лог)",
    "mkt_log_qv_vs_ma30": "объём относительно 30д", "mkt_log_rv_vs_ma30": "волатильность относительно 30д",
    "mkt_log_qv_vs_ma7": "объём относительно 7д", "mkt_log_rv_vs_ma7": "волатильность относительно 7д",
    "mkt_log_trades": "число сделок (лог)", "mkt_taker_buy_ratio": "доля покупок тейкеров",
    "mkt_abs_return_1d": "модуль доходности за день", "mkt_range_log": "дневной диапазон (лог)",
    "mkt_parkinson": "волатильность Паркинсона", "mkt_max_abs_ret": "макс. часовая доходность",
    "mkt_rs_pos_share": "доля положительной вариации", "mkt_volume_hhi": "концентрация объёма в часах",
    "mkt_last4h_share": "доля объёма за последние 4 ч", "mkt_btc_log_qv": "объём BTC (лог)",
    "mkt_btc_log_rv": "волатильность BTC (лог)", "mkt_btc_return_1d": "доходность BTC",
    "mkt_cross_rv_mean": "средняя волатильность рынка", "mkt_cross_qv_mean": "средний объём рынка",
    "mkt_rv_vs_cross": "волатильность относительно рынка", "cal_is_weekend": "прогноз на выходной день",
    "cal_dow": "день недели прогноза", "fng_value": "индекс страха и жадности", "fng_change_7d": "индекс: изменение 7д",
    "fng_ma7": "индекс: среднее 7д", "fng_z30": "индекс: отклонение от 30д", "txt_sent_mean": "тональность новостей",
    "txt_log_n_articles": "число новостей (лог)", "txt_topic_hack_share": "доля новостей о взломах",
    "txt_topic_regulation_share": "доля новостей о регулировании", "txt_topic_etf_share": "доля новостей об ETF",
    "txt_mention_share": "доля упоминаний инструмента", "txt_available": "есть текстовые данные",
    "symbol": "инструмент", "cal_dow_sin": "день недели прогноза (sin)", "cal_dow_cos": "день недели прогноза (cos)",
    "cal_month_sin": "месяц (sin)", "cal_month_cos": "месяц (cos)", "cal_is_month_end": "конец месяца",
    "mkt_return_1d": "доходность за день", "mkt_return_3d": "доходность за 3 дня", "mkt_return_7d": "доходность за 7 дней",
    "mkt_return_30d": "доходность за 30 дней", "mkt_log_trades_ma7": "число сделок, среднее 7д (лог)",
    "mkt_taker_ratio_ma7": "доля покупок тейкеров, среднее 7д", "mkt_n_anomalies": "часов с аномальным объёмом",
    "txt_neg_share": "доля негативных заголовков", "txt_sent_ma7": "тональность, среднее 7д",
    "txt_mention_sentiment": "тональность упоминаний инструмента", "txt_sent_std": "разброс тональности",
}


def human(feature: str) -> str:
    if feature in FEATURE_NAMES:
        return FEATURE_NAMES[feature]
    for suffix, label in (("_lag", "лаг"), ("_ma", "среднее"), ("_std", "ст. откл.")):
        if suffix in feature:
            base, _, n = feature.rpartition(suffix)
            if n.isdigit() and base in FEATURE_NAMES:
                return f"{FEATURE_NAMES[base]} ({label} {n}д)"
    return feature
