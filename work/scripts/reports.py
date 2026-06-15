"""
Модуль формирования аналитических отчётов.

Реализует все 7 типов отчётов: 3 текстовых и 4 графических.
Все функции работают в функциональном стиле без использования ООП.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _merge_data(passengers: pd.DataFrame, fares: pd.DataFrame) -> pd.DataFrame:
    """Объединить справочники Passengers и Fares в единый DataFrame.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.

    Returns:
        Объединённый DataFrame.
    """
    return pd.merge(passengers, fares, on='PassengerId', how='left')


def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Применить фильтры к DataFrame.

    Args:
        df: Исходный DataFrame.
        filters: Словарь {колонка: значение}.  Пустая строка или None
                 означает «без фильтра по данному полю».

    Returns:
        Отфильтрованный DataFrame.
    """
    result = df.copy()
    for col, val in filters.items():
        if val is not None and val != '' and col in result.columns:
            result = result[result[col].astype(str) == str(val)]
    return result


# ---------------------------------------------------------------------------
# Текстовые отчёты
# ---------------------------------------------------------------------------

def report_simple(passengers: pd.DataFrame, fares: pd.DataFrame,
                  columns: list, filters: dict) -> pd.DataFrame:
    """Простой текстовый отчёт — фильтрация и проекция данных.

    Формирует таблицу путём объединения справочников, применения фильтров
    и выбора указанных столбцов.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        columns: Список столбцов для отображения.
        filters: Словарь фильтров {колонка: значение}.

    Returns:
        Отфильтрованный и спроецированный DataFrame.
    """
    df = _merge_data(passengers, fares)
    df = _apply_filters(df, filters)
    available = [c for c in columns if c in df.columns]
    if not available:
        available = df.columns.tolist()
    return df[available].reset_index(drop=True)


def report_statistics(passengers: pd.DataFrame, fares: pd.DataFrame,
                      attributes: list) -> dict:
    """Статистический текстовый отчёт.

    Для качественных переменных возвращает таблицу частот.
    Для количественных — описательные статистики.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        attributes: Список атрибутов для анализа.

    Returns:
        Словарь {атрибут: DataFrame со статистикой}.
    """
    qualitative = {'Sex', 'Embarked', 'Survived', 'Pclass'}
    df = _merge_data(passengers, fares)
    results = {}

    for attr in attributes:
        if attr not in df.columns:
            continue
        if attr in qualitative:
            freq = df[attr].value_counts().reset_index()
            freq.columns = ['Значение', 'Частота']
            total = freq['Частота'].sum()
            freq['Процент, %'] = (freq['Частота'] / total * 100).round(2)
            results[attr] = freq
        else:
            series = pd.to_numeric(df[attr], errors='coerce').dropna()
            stats = pd.DataFrame({
                'Показатель': ['Минимум', 'Максимум', 'Среднее',
                               'Дисперсия', 'Ст. отклонение'],
                'Значение': [
                    round(series.min(), 4),
                    round(series.max(), 4),
                    round(series.mean(), 4),
                    round(series.var(), 4),
                    round(series.std(), 4),
                ]
            })
            results[attr] = stats

    return results


def report_pivot(passengers: pd.DataFrame, fares: pd.DataFrame,
                 index_col: str, columns_col: str,
                 values_col: str, aggfunc: str) -> pd.DataFrame:
    """Сводная таблица по двум качественным атрибутам.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        index_col: Атрибут для строк сводной таблицы.
        columns_col: Атрибут для столбцов сводной таблицы.
        values_col: Атрибут для значений.
        aggfunc: Функция агрегации ('count', 'mean', 'sum', 'min', 'max').

    Returns:
        Сводный DataFrame.
    """
    df = _merge_data(passengers, fares)
    func_map = {
        'Количество': 'count',
        'Среднее': 'mean',
        'Сумма': 'sum',
        'Минимум': 'min',
        'Максимум': 'max',
    }
    func = func_map.get(aggfunc, aggfunc)
    pivot = pd.pivot_table(
        df,
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc=func,
        fill_value=0,
    )
    pivot = pivot.round(2)
    return pivot


# ---------------------------------------------------------------------------
# Графические отчёты
# ---------------------------------------------------------------------------

def _get_fig_size(config) -> tuple:
    """Получить размер фигуры из конфигурации.

    Args:
        config: ConfigParser объект.

    Returns:
        Кортеж (ширина, высота).
    """
    w = float(config['REPORTS'].get('figure_width', '9'))
    h = float(config['REPORTS'].get('figure_height', '6'))
    return (w, h)


def chart_bar(passengers: pd.DataFrame, fares: pd.DataFrame,
              x_attr: str, hue_attr: str, config,
              save_path: str = None) -> plt.Figure:
    """Кластеризованная столбчатая диаграмма для двух качественных атрибутов.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        x_attr: Атрибут для оси X.
        hue_attr: Атрибут группировки.
        config: ConfigParser объект.
        save_path: Путь для сохранения (None — не сохранять).

    Returns:
        Объект Figure matplotlib.
    """
    df = _merge_data(passengers, fares)
    grouped = df.groupby([x_attr, hue_attr]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=_get_fig_size(config))
    x = np.arange(len(grouped.index))
    width = 0.8 / len(grouped.columns)
    colors = plt.cm.Set2.colors

    for i, col in enumerate(grouped.columns):
        offset = (i - len(grouped.columns) / 2 + 0.5) * width
        bars = ax.bar(x + offset, grouped[col], width * 0.9,
                      label=str(col), color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(str(int(h)),
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(v) for v in grouped.index], rotation=15, ha='right'
    )
    ax.set_xlabel(x_attr, fontsize=11)
    ax.set_ylabel('Количество', fontsize=11)
    ax.set_title(f'Распределение «{x_attr}» по «{hue_attr}»', fontsize=13)
    ax.legend(title=hue_attr, fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()

    if save_path:
        dpi = int(config['REPORTS'].get('figure_dpi', '100'))
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def chart_histogram(passengers: pd.DataFrame, fares: pd.DataFrame,
                    num_attr: str, cat_attr: str, config,
                    save_path: str = None) -> plt.Figure:
    """Категоризированная гистограмма (количественный × качественный).

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        num_attr: Количественный атрибут.
        cat_attr: Качественный атрибут (категория).
        config: ConfigParser объект.
        save_path: Путь для сохранения.

    Returns:
        Объект Figure matplotlib.
    """
    df = _merge_data(passengers, fares)
    categories = df[cat_attr].dropna().unique()
    colors = plt.cm.Set1.colors
    fig, ax = plt.subplots(figsize=_get_fig_size(config))

    for i, cat in enumerate(sorted(categories, key=str)):
        subset = pd.to_numeric(
            df.loc[df[cat_attr] == cat, num_attr], errors='coerce'
        ).dropna()
        ax.hist(subset, bins=20, alpha=0.6, label=str(cat),
                color=colors[i % len(colors)], edgecolor='white')

    ax.set_xlabel(num_attr, fontsize=11)
    ax.set_ylabel('Частота', fontsize=11)
    ax.set_title(
        f'Гистограмма «{num_attr}» по группам «{cat_attr}»',
        fontsize=13
    )
    ax.legend(title=cat_attr, fontsize=9)
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()

    if save_path:
        dpi = int(config['REPORTS'].get('figure_dpi', '100'))
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def chart_boxplot(passengers: pd.DataFrame, fares: pd.DataFrame,
                  num_attr: str, cat_attr: str, config,
                  save_path: str = None) -> plt.Figure:
    """Категоризированная диаграмма Бокса–Вискера.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        num_attr: Количественный атрибут.
        cat_attr: Качественный атрибут (группировка).
        config: ConfigParser объект.
        save_path: Путь для сохранения.

    Returns:
        Объект Figure matplotlib.
    """
    df = _merge_data(passengers, fares)
    categories = sorted(df[cat_attr].dropna().unique(), key=str)
    data_groups = []
    labels = []

    for cat in categories:
        subset = pd.to_numeric(
            df.loc[df[cat_attr] == cat, num_attr], errors='coerce'
        ).dropna()
        if len(subset) > 0:
            data_groups.append(subset.values)
            labels.append(str(cat))

    fig, ax = plt.subplots(figsize=_get_fig_size(config))
    bp = ax.boxplot(data_groups, labels=labels, patch_artist=True,
                    medianprops=dict(color='#C0392B', linewidth=2))
    colors = plt.cm.Pastel1.colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel(cat_attr, fontsize=11)
    ax.set_ylabel(num_attr, fontsize=11)
    ax.set_title(f'Диаграмма Бокса–Вискера: «{num_attr}» по «{cat_attr}»',
                 fontsize=13)
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()

    if save_path:
        dpi = int(config['REPORTS'].get('figure_dpi', '100'))
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


def chart_scatter(passengers: pd.DataFrame, fares: pd.DataFrame,
                  x_attr: str, y_attr: str, cat_attr: str, config,
                  save_path: str = None) -> plt.Figure:
    """Категоризированная диаграмма рассеивания.

    Args:
        passengers: Справочник пассажиров.
        fares: Справочник тарифов.
        x_attr: Количественный атрибут для оси X.
        y_attr: Количественный атрибут для оси Y.
        cat_attr: Качественный атрибут для цветовой категоризации.
        config: ConfigParser объект.
        save_path: Путь для сохранения.

    Returns:
        Объект Figure matplotlib.
    """
    df = _merge_data(passengers, fares)
    categories = sorted(df[cat_attr].dropna().unique(), key=str)
    colors = plt.cm.Set1.colors
    fig, ax = plt.subplots(figsize=_get_fig_size(config))

    for i, cat in enumerate(categories):
        subset = df[df[cat_attr] == cat].copy()
        xs = pd.to_numeric(subset[x_attr], errors='coerce')
        ys = pd.to_numeric(subset[y_attr], errors='coerce')
        mask = xs.notna() & ys.notna()
        ax.scatter(xs[mask], ys[mask], label=str(cat),
                   color=colors[i % len(colors)], alpha=0.6, s=40,
                   edgecolors='white', linewidths=0.5)

    ax.set_xlabel(x_attr, fontsize=11)
    ax.set_ylabel(y_attr, fontsize=11)
    ax.set_title(
        f'Диаграмма рассеивания: «{x_attr}» vs «{y_attr}» по «{cat_attr}»',
        fontsize=13)
    ax.legend(title=cat_attr, fontsize=9)
    ax.grid(alpha=0.4)
    fig.tight_layout()

    if save_path:
        dpi = int(config['REPORTS'].get('figure_dpi', '100'))
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Сохранение текстовых отчётов
# ---------------------------------------------------------------------------

def save_text_report(
    df: pd.DataFrame, filepath: str, fmt: str = 'csv'
) -> None:
    """Сохранить текстовый отчёт в файл.

    Args:
        df: DataFrame с результатами отчёта.
        filepath: Путь к выходному файлу.
        fmt: Формат файла — 'csv' или 'txt'.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if fmt == 'txt':
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(df.to_string(index=True))
    else:
        df.to_csv(filepath, index=True, encoding='utf-8-sig')
