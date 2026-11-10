"""Единая walk-forward валидация и модели разных классов.

Все преобразования (импутация, клиппинг, масштабирование, one-hot, отбор признаков и
TF-IDF+SVD) обучаются строго на train текущего фолда внутри fit().
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

from cryptocast.config import Config
from cryptocast.features import HAR_COLUMNS, NAIVE_COLUMN, feature_columns
from cryptocast.ml.metrics import compute_metrics

SVD_PREFIX = "svd_"


# ---------------------------------- Разбиения -------------------------------------
@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def split_holdout(dates: pd.Series, holdout_days: int, gap_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Возвращает (последняя дата разработки, первая дата отложенной выборки)."""
    last = pd.Timestamp(dates.max())
    holdout_start = last - pd.Timedelta(days=holdout_days - 1)
    return holdout_start - pd.Timedelta(days=gap_days + 1), holdout_start


def walk_forward_folds(dev_dates: pd.Series, cfg: Config) -> list[Fold]:
    params = cfg["validation"]
    first, last = pd.Timestamp(dev_dates.min()), pd.Timestamp(dev_dates.max())
    train_end = first + pd.Timedelta(days=params["initial_train_days"])
    folds: list[Fold] = []
    while True:
        test_start = train_end + pd.Timedelta(days=params["gap_days"] + 1)
        test_end = test_start + pd.Timedelta(days=params["test_days"] - 1)
        if test_end > last:
            break
        folds.append(Fold(len(folds), train_end, test_start, test_end))
        train_end = train_end + pd.Timedelta(days=params["test_days"])
    return folds


# ---------------------------------- Текст -----------------------------------------
class TextSVD:
    """TF-IDF + SVD по заголовкам: fit строго на статьях до конца train."""

    def __init__(self, cfg: Config, seed: int):
        self.params = cfg["features"]["text"]
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.fitted_max_date: pd.Timestamp | None = None
        self.columns = [f"{SVD_PREFIX}{i}" for i in range(self.params["svd_components"])]

    def fit(self, news: pd.DataFrame) -> "TextSVD":
        if len(news) < 200:
            return self
        self.fitted_max_date = pd.Timestamp(news["date"].max())
        self.vectorizer = TfidfVectorizer(
            max_features=self.params["tfidf_max_features"], min_df=self.params["tfidf_min_df"],
            ngram_range=(1, self.params["ngram_max"]), stop_words="english", sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(news["title_clean"])
        components = min(self.params["svd_components"], min(matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed)
        self.svd.fit(matrix)
        return self

    def transform_daily(self, news: pd.DataFrame) -> pd.DataFrame:
        if news.empty or self.vectorizer is None or self.svd is None:
            return pd.DataFrame(columns=["date"] + self.columns)
        vectors = self.svd.transform(self.vectorizer.transform(news["title_clean"]))
        frame = pd.DataFrame(vectors, columns=self.columns[: vectors.shape[1]])
        frame["date"] = news["date"].values
        return frame.groupby("date", as_index=False).mean()


# ---------------------------------- Предобработка ---------------------------------
class TabularPreprocessor:
    """Импутация, клиппинг, стандартизация и one-hot инструмента для линейных моделей и MLP."""

    def __init__(self, clip_q: tuple[float, float]):
        self.clip_q = clip_q

    def fit(self, frame: pd.DataFrame, numeric: list[str], symbols: list[str]) -> "TabularPreprocessor":
        self.numeric = numeric
        self.symbols = symbols
        values = frame[numeric]
        self.low_ = values.quantile(self.clip_q[0])
        self.high_ = values.quantile(self.clip_q[1])
        clipped = values.clip(self.low_, self.high_, axis=1)
        self.medians_ = clipped.median().fillna(0.0)
        filled = clipped.fillna(self.medians_)
        self.mean_ = filled.mean()
        self.std_ = filled.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.numeric].clip(self.low_, self.high_, axis=1).fillna(self.medians_)
        scaled = ((values - self.mean_) / self.std_).to_numpy(dtype=float)
        onehot = np.column_stack([(frame["symbol"] == s).to_numpy(dtype=float) for s in self.symbols])
        return np.hstack([scaled, onehot])

    @property
    def output_names(self) -> list[str]:
        return list(self.numeric) + [f"symbol={s}" for s in self.symbols]


# ---------------------------------- Модели ----------------------------------------
class BaseModel:
    name = "base"

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class NaiveModel(BaseModel):
    """Сегодня = завтра: значение таргета на дату прогноза."""
    name = "naive"

    def __init__(self, target: str):
        self.column = NAIVE_COLUMN[target]

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "NaiveModel":
        self.fallback_ = float(np.nanmean(y))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].fillna(self.fallback_).to_numpy(dtype=float)


class NaiveMA7Model(NaiveModel):
    """Дополнительный ориентир: среднее логарифма за 7 суток."""
    name = "naive_ma7"

    def __init__(self, target: str):
        super().__init__(target)
        self.column = f"{NAIVE_COLUMN[target]}_ma7"


class ElasticNetHAR(BaseModel):
    """ElasticNet на признаках HAR (день, неделя, месяц) и one-hot инструмента."""
    name = "elasticnet"

    def __init__(self, target: str, params: dict, symbols: list[str], clip_q: tuple[float, float]):
        self.target, self.params, self.symbols, self.clip_q = target, dict(params), symbols, clip_q

    def fit(self, train: pd.DataFrame, y: np.ndarray) -> "ElasticNetHAR":
        columns = HAR_COLUMNS[self.target]
        self.prep = TabularPreprocessor(self.clip_q).fit(train, columns, self.symbols)
        self.model = ElasticNet(**self.params, random_state=0).fit(self.prep.transform(train), y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.prep.transform(frame))


class LightGBMModel(BaseModel):
    """Градиентный бустинг; инструмент передаётся как нативная категория."""
    name = "lightgbm"

    def __init__(self, params: dict, seed: int, symbols: list[str], encoding: str = "native"):
        self.params = dict(params)
        self.seed = seed
        self.symbols = symbols
        self.encoding = encoding

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        data = frame[self.features].copy()
        if self.encoding == "native":
            data["symbol"] = pd.Categorical(frame["symbol"], categories=self.symbols)
        elif self.encoding == "onehot":
            for symbol in self.symbols:
                data[f"symbol={symbol}"] = (frame["symbol"] == symbol).astype(float)
        return data

    def fit(self, train: pd.DataFrame, y: np.ndarray, features: list[str] | None = None) -> "LightGBMModel":
        self.features = list(features or [c for c in train.columns if c.startswith(("mkt_", "cal_", "fng_", "txt_", SVD_PREFIX))])
        self.model = LGBMRegressor(
            **self.params, random_state=self.seed, deterministic=True, force_row_wise=True, verbose=-1,
        )
        matrix = self._matrix(train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(matrix, y, categorical_feature=["symbol"] if self.encoding == "native" else "auto")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict(self._matrix(frame))


class MLPModel(BaseModel):
    """Полносвязная нейросеть (иной класс алгоритмов), воспроизводимая при фиксированном зерне."""
    name = "mlp"

    def __init__(self, params: dict, seed: int, symbols: list[str], clip_q: tuple[float, float]):
        self.params, self.seed, self.symbols, self.clip_q = dict(params), seed, symbols, clip_q

    def fit(self, train: pd.DataFrame, y: np.ndarray, features: list[str] | None = None) -> "MLPModel":
        columns = list(features or [c for c in train.columns if c.startswith(("mkt_", "cal_", "fng_", "txt_", SVD_PREFIX))])
        self.prep = TabularPreprocessor(self.clip_q).fit(train, columns, self.symbols)
        params = dict(self.params)
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        self.y_mean_, self.y_std_ = float(np.mean(y)), float(np.std(y) or 1.0)
        self.model = MLPRegressor(**params, random_state=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.prep.transform(train), (y - self.y_mean_) / self.y_std_)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.prep.transform(frame)) * self.y_std_ + self.y_mean_


# ---------------------------------- Обучаемый конвейер -------------------------------
@dataclass
class Spec:
    """Полное описание обучаемой модели: достаточно для повторения эксперимента."""
    target: str
    model: str
    params: dict = field(default_factory=dict)
    feature_set: list[str] = field(default_factory=lambda: ["market", "calendar", "fng", "text"])
    use_svd: bool = True
    top_k: int | None = None
    encoding: str = "native"
    seed: int = 42


class FittedPipeline:
    """Обучает модель на train и применяет её к любым будущим строкам тем же кодом."""

    def __init__(self, spec: Spec, cfg: Config):
        self.spec = spec
        self.cfg = cfg
        self.svd: TextSVD | None = None
        self.model: BaseModel | None = None
        self.features: list[str] = []

    def _svd_join(self, frame: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        if self.svd is None:
            return frame
        daily = self.svd.transform_daily(news)
        return frame.merge(daily, on="date", how="left") if not daily.empty else frame.assign(**{c: np.nan for c in self.svd.columns})

    def _build_model(self) -> BaseModel:
        s = self.spec
        clip_q = tuple(self.cfg["models"]["clip_quantiles"])
        symbols = self.cfg.symbols
        if s.model == "naive":
            return NaiveModel(s.target)
        if s.model == "naive_ma7":
            return NaiveMA7Model(s.target)
        if s.model == "elasticnet":
            return ElasticNetHAR(s.target, s.params, symbols, clip_q)
        if s.model == "lightgbm":
            return LightGBMModel(s.params, s.seed, symbols, s.encoding)
        if s.model == "mlp":
            return MLPModel(s.params, s.seed, symbols, clip_q)
        raise ValueError(s.model)

    def _select(self, train: pd.DataFrame, y: np.ndarray, columns: list[str]) -> list[str]:
        """Отбор: убрать почти константы, затем top_k по gain LightGBM (только train)."""
        usable = [c for c in columns if train[c].notna().mean() > 0.05 and train[c].nunique(dropna=True) > 1]
        if self.spec.top_k is None or self.spec.top_k >= len(usable):
            return usable
        selector = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=self.spec.seed,
                                 deterministic=True, force_row_wise=True, num_threads=4, verbose=-1, importance_type="gain")
        selector.fit(train[usable], y)
        order = np.argsort(selector.feature_importances_)[::-1][: self.spec.top_k]
        return [usable[i] for i in sorted(order)]

    def fit(self, train: pd.DataFrame, news: pd.DataFrame) -> "FittedPipeline":
        train = train.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.train_end_ = pd.Timestamp(train["date"].max())
        if self.spec.model in ("lightgbm", "mlp") and self.spec.use_svd and "text" in self.spec.feature_set:
            train_news = news[news["date"] <= self.train_end_] if not news.empty else news
            if not train_news.empty:
                self.svd = TextSVD(self.cfg, self.spec.seed).fit(train_news)
                if self.svd.vectorizer is None:
                    self.svd = None
        data = self._svd_join(train, news)
        y = data[f"target_{self.spec.target}"].to_numpy(dtype=float)
        columns = feature_columns(data, self.spec.feature_set)
        if self.svd is not None:
            columns += [c for c in data.columns if c.startswith(SVD_PREFIX)]
        self.model = self._build_model()
        if self.spec.model in ("lightgbm", "mlp"):
            self.features = self._select(data, y, columns)
            self.model.fit(data, y, features=self.features)  # type: ignore[call-arg]
        else:
            self.model.fit(data, y)
            self.features = list(HAR_COLUMNS[self.spec.target]) if self.spec.model == "elasticnet" else [NAIVE_COLUMN[self.spec.target]]
        return self

    def prepare(self, frame: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        return self._svd_join(frame, news)

    def predict(self, frame: pd.DataFrame, news: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.prepare(frame.reset_index(drop=True), news))  # type: ignore[union-attr]


def run_walk_forward(spec: Spec, frame: pd.DataFrame, news: pd.DataFrame, folds: list[Fold], cfg: Config) -> pd.DataFrame:
    """OOS-прогнозы по всем фолдам: символ, дата, факт, прогноз, номер фолда."""
    target = f"target_{spec.target}"
    data = frame[frame[target].notna()]
    results: list[pd.DataFrame] = []
    for fold in folds:
        train = data[data["date"] <= fold.train_end]
        test = data[(data["date"] >= fold.test_start) & (data["date"] <= fold.test_end)]
        pipeline = FittedPipeline(spec, cfg).fit(train, news)
        results.append(pd.DataFrame({
            "symbol": test["symbol"].values, "date": test["date"].values,
            "y_true": test[target].values, "y_pred": pipeline.predict(test, news), "fold": fold.index,
        }))
    return pd.concat(results, ignore_index=True)


def evaluate_spec(spec: Spec, frame: pd.DataFrame, news: pd.DataFrame, folds: list[Fold], cfg: Config) -> tuple[pd.DataFrame, dict]:
    oos = run_walk_forward(spec, frame, news, folds, cfg)
    return oos, compute_metrics(spec.target, oos["y_true"], oos["y_pred"])
