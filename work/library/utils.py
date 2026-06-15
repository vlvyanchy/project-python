"""
Модуль вспомогательных функций приложения
«Анализ пассажиров Титаника».

Содержит универсальные утилиты для работы
с конфигурацией, данными и файловой системой.
"""

import configparser
import os
import pickle


def load_config(config_path: str) -> configparser.ConfigParser:
    """Загрузить конфигурационный файл приложения.

    Args:
        config_path: Путь к файлу config.ini.

    Returns:
        Объект ConfigParser с параметрами приложения.
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config


def get_base_dir() -> str:
    """Вернуть корневой каталог приложения (work/).

    Returns:
        Абсолютный путь к каталогу work/.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dirs(
    base_dir: str,
    config: configparser.ConfigParser
) -> None:
    """Создать рабочие каталоги, если они не существуют.

    Args:
        base_dir: Корневой каталог приложения.
        config: Объект ConfigParser с путями
            из секции [PATHS].
    """
    for key in ('data_dir', 'output_dir', 'graphics_dir', 'notes_dir'):
        path = os.path.join(base_dir, config['PATHS'][key])
        os.makedirs(path, exist_ok=True)


def load_pickle(filepath: str):
    """Загрузить объект из файла формата Pickle (.pkl).

    Args:
        filepath: Путь к .pkl файлу.

    Returns:
        Десериализованный объект
        (обычно pandas.DataFrame).

    Raises:
        FileNotFoundError: Если файл не найден.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Файл не найден: {filepath}')
    with open(filepath, 'rb') as fh:
        return pickle.load(fh)


def save_pickle(obj, filepath: str) -> None:
    """Сохранить объект в файл формата Pickle (.pkl).

    Args:
        obj: Объект для сериализации
            (обычно pandas.DataFrame).
        filepath: Путь к выходному .pkl файлу.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as fh:
        pickle.dump(obj, fh)


def format_float(value: float, decimals: int = 2) -> str:
    """Форматировать число с плавающей точкой для отображения.

    Args:
        value: Числовое значение.
        decimals: Количество знаков после запятой.

    Returns:
        Строка с форматированным числом.
    """
    try:
        return f'{float(value):.{decimals}f}'
    except (TypeError, ValueError):
        return str(value)


def sex_display(value: str) -> str:
    """Перевести значение пола на русский язык.

    Args:
        value: Значение 'male' или 'female'.

    Returns:
        Русскоязычное обозначение пола.
    """
    mapping = {'male': 'Мужской', 'female': 'Женский'}
    return mapping.get(str(value).lower(), str(value))


def survived_display(value) -> str:
    """Перевести признак выживания в читаемый вид.

    Args:
        value: 0 или 1.

    Returns:
        'Да' или 'Нет'.
    """
    return 'Да' if int(value) == 1 else 'Нет'


def embarked_display(value: str) -> str:
    """Расшифровать код порта посадки.

    Args:
        value: Код порта ('S', 'C', 'Q').

    Returns:
        Название порта.
    """
    mapping = {
        'S': 'Саутгемптон',
        'C': 'Шербур',
        'Q': 'Квинстаун',
    }
    return mapping.get(str(value).upper(), str(value))
