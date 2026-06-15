"""
Главный исполняемый модуль приложения «Анализ пассажиров Титаника».

Запуск:
    python scripts/main.py

Приложение предназначено для анализа статистических данных о пассажирах
лайнера «Титаник» с использованием Tkinter, Pandas и Matplotlib.
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk, filedialog

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Добавляем корень проекта в путь, чтобы импортировать library и scripts
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from library.utils import (load_config, ensure_dirs, load_pickle,
                            save_pickle, format_float,
                            sex_display, survived_display, embarked_display)
import scripts.reports as reports

# ---------------------------------------------------------------------------
# Глобальные переменные состояния
# ---------------------------------------------------------------------------
CONFIG = None
PASSENGERS = None  # pandas.DataFrame
FARES = None       # pandas.DataFrame


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def _load_or_rebuild(data_dir, pass_path, fare_path):
    """Загрузить pkl-справочники или пересоздать их из CSV при ошибке.

    Args:
        data_dir: Путь к папке data/.
        pass_path: Путь к passengers.pkl.
        fare_path: Путь к fares.pkl.

    Returns:
        Кортеж (passengers DataFrame, fares DataFrame).
    """
    import pickle
    import numpy as np
    import pandas as pd

    def _rebuild():
        """Пересоздать pkl-справочники из titanic_raw.csv.

        Returns:
            Кортеж (passengers DataFrame, fares DataFrame).
        """
        for name in ('titanic_raw.csv', 'titanic.csv'):
            csv_path = os.path.join(data_dir, name)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                p_df = pd.DataFrame({
                    'PassengerId': df['PassengerId'].to_numpy(dtype=np.int64),
                    'Survived':    df['Survived'].to_numpy(dtype=np.int64),
                    'Pclass':      df['Pclass'].to_numpy(dtype=np.int64),
                    'Sex': df['Sex'].fillna('male').to_numpy(
                        dtype=object),
                    'Age': df['Age'].fillna(
                        df['Age'].median()).to_numpy(dtype=np.float64),
                    'Embarked': df['Embarked'].fillna('S').to_numpy(
                        dtype=object),
                })
                f_df = pd.DataFrame({
                    'PassengerId': df['PassengerId'].to_numpy(dtype=np.int64),
                    'Fare': df['Fare'].fillna(
                        df['Fare'].median()).to_numpy(dtype=np.float64),
                })
                with open(pass_path, 'wb') as fh:
                    pickle.dump(p_df, fh, protocol=2)
                with open(fare_path, 'wb') as fh:
                    pickle.dump(f_df, fh, protocol=2)
                return p_df, f_df
        messagebox.showerror(
            'Ошибка',
            'Файл titanic_raw.csv не найден в папке data/'
        )
        sys.exit(1)

    try:
        with open(pass_path, 'rb') as fh:
            passengers = pickle.load(fh)
        with open(fare_path, 'rb') as fh:
            fares = pickle.load(fh)
        # Проверяем что типы корректны
        _ = passengers['Sex'].iloc[0]
        return passengers, fares
    except Exception:
        return _rebuild()


def init_app() -> None:
    """Инициализировать конфигурацию и загрузить данные справочников."""
    global CONFIG, PASSENGERS, FARES
    config_path = os.path.join(BASE_DIR, 'config.ini')
    CONFIG = load_config(config_path)
    ensure_dirs(BASE_DIR, CONFIG)

    data_dir = os.path.join(BASE_DIR, CONFIG['PATHS']['data_dir'])
    pass_path = os.path.join(data_dir, 'passengers.pkl')
    fare_path = os.path.join(data_dir, 'fares.pkl')

    PASSENGERS, FARES = _load_or_rebuild(data_dir, pass_path, fare_path)


# ---------------------------------------------------------------------------
# Вспомогательные GUI-функции
# ---------------------------------------------------------------------------

def apply_style(widget, cfg: dict) -> None:
    """Применить словарь конфигурации стиля к виджету Tkinter.

    Args:
        widget: Виджет Tkinter.
        cfg: Словарь с параметрами стиля (bg, fg и т.д.).
    """
    try:
        widget.config(**cfg)
    except tk.TclError:
        pass


def make_label(parent, text: str, **kwargs) -> tk.Label:
    """Создать стилизованный Label.

    Args:
        parent: Родительский виджет.
        text: Текст метки.
        **kwargs: Дополнительные параметры.

    Returns:
        Настроенный tk.Label.
    """
    cfg = CONFIG['INTERFACE']
    kwargs.setdefault('bg', cfg.get('bg_color', '#F0F4F8'))
    kwargs.setdefault('fg', cfg.get('fg_color', '#1A1A2E'))
    font_family = cfg.get('font_family', 'Arial')
    font_size = int(cfg.get('font_size', '11'))
    kwargs.setdefault('font', (font_family, font_size))
    lbl = tk.Label(parent, text=text, **kwargs)
    return lbl


def make_button(parent, text: str, command, **kwargs) -> tk.Button:
    """Создать стилизованную кнопку.

    Args:
        parent: Родительский виджет.
        text: Текст кнопки.
        command: Команда при нажатии.
        **kwargs: Дополнительные параметры.

    Returns:
        Настроенный tk.Button.
    """
    cfg = CONFIG['INTERFACE']
    btn = tk.Button(
        parent, text=text, command=command,
        bg=cfg.get('button_color', '#2C3E8C'),
        fg=cfg.get('button_fg', '#FFFFFF'),
        activebackground=cfg.get('accent_color', '#2C3E8C'),
        relief=tk.FLAT, padx=10, pady=5, cursor='hand2',
        font=(cfg.get('font_family', 'Arial'),
              int(cfg.get('font_size', '11'))),
        **kwargs
    )
    return btn


def make_combobox(parent, values: list, width: int = 18) -> ttk.Combobox:
    """Создать выпадающий список.

    Args:
        parent: Родительский виджет.
        values: Список допустимых значений.
        width: Ширина виджета в символах.

    Returns:
        Настроенный ttk.Combobox (только для чтения).
    """
    cb = ttk.Combobox(parent, values=values, width=width, state='readonly')
    if values:
        cb.current(0)
    return cb


def make_treeview(parent, columns: list, heights: int = 18) -> ttk.Treeview:
    """Создать таблицу Treeview с полосой прокрутки.

    Args:
        parent: Родительский виджет.
        columns: Список заголовков столбцов.
        heights: Количество видимых строк.

    Returns:
        Настроенный ttk.Treeview.
    """
    cfg = CONFIG['INTERFACE']
    style = ttk.Style()
    style.configure('Custom.Treeview',
                     background=cfg.get('table_bg', '#FFFFFF'),
                     foreground=cfg.get('table_fg', '#1A1A2E'),
                     rowheight=24,
                     fieldbackground=cfg.get('table_bg', '#FFFFFF'))
    style.configure('Custom.Treeview.Heading',
                     background=cfg.get('table_heading_bg', '#2C3E8C'),
                     foreground=cfg.get('table_heading_fg', '#FFFFFF'),
                     font=(cfg.get('font_family', 'Arial'),
                           int(cfg.get('font_size', '11')), 'bold'))

    frame = tk.Frame(parent,
                     bg=cfg.get('bg_color', '#F0F4F8'))
    frame.pack(fill=tk.BOTH, expand=True)

    tv = ttk.Treeview(frame, columns=columns, show='headings',
                      height=heights, style='Custom.Treeview')

    vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tv.yview)
    hsb = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tv.xview)
    tv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    for col in columns:
        tv.heading(col, text=col)
        tv.column(col, width=110, minwidth=60, anchor=tk.CENTER)

    tv.grid(row=0, column=0, sticky='nsew')
    vsb.grid(row=0, column=1, sticky='ns')
    hsb.grid(row=1, column=0, sticky='ew')
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)

    return tv


def fill_treeview(tv: ttk.Treeview, df) -> None:
    """Заполнить Treeview данными из DataFrame.

    Args:
        tv: Виджет Treeview.
        df: pandas.DataFrame с данными.
    """
    tv.delete(*tv.get_children())
    for _, row in df.iterrows():
        tv.insert('', tk.END, values=list(row))


# ---------------------------------------------------------------------------
# Окно справочника «Пассажиры»
# ---------------------------------------------------------------------------

def open_passengers_window(root: tk.Tk) -> None:
    """Открыть окно управления справочником «Пассажиры».

    Обеспечивает просмотр, добавление, редактирование и удаление записей,
    а также сохранение и загрузку данных в формате Pickle.

    Args:
        root: Главное окно приложения.
    """
    global PASSENGERS
    cfg = CONFIG['INTERFACE']
    win = tk.Toplevel(root)
    win.title('Справочник — Пассажиры')
    win.geometry('900x560')
    win.configure(bg=cfg.get('bg_color', '#F0F4F8'))

    cols = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Embarked']

    # --- Заголовок ---
    make_label(win, 'Справочник: Пассажиры',
               font=(cfg.get('font_family', 'Arial'),
                     int(cfg.get('title_font_size', '14')), 'bold')
               ).pack(pady=(10, 4))

    tv = make_treeview(win, cols, heights=16)

    def refresh():
        """Обновить отображение таблицы."""
        fill_treeview(tv, PASSENGERS[cols])

    refresh()

    # --- Панель управления ---
    ctrl = tk.Frame(win, bg=cfg.get('bg_color'))
    ctrl.pack(pady=8, fill=tk.X, padx=10)

    # Поля ввода
    fields = {}
    field_labels = {
        'PassengerId': 'ID', 'Survived': 'Выжил (0/1)',
        'Pclass': 'Класс (1-3)', 'Sex': 'Пол (male/female)',
        'Age': 'Возраст', 'Embarked': 'Порт (S/C/Q)'
    }
    for i, (col, lbl_text) in enumerate(field_labels.items()):
        tk.Label(ctrl, text=lbl_text,
                 bg=cfg.get('bg_color'), fg=cfg.get('fg_color'),
                 font=(cfg.get('font_family'), int(cfg.get('font_size'))
                       )).grid(row=0, column=i * 2, padx=3, sticky='w')
        var = tk.StringVar()
        entry = tk.Entry(ctrl, textvariable=var, width=10)
        entry.grid(row=1, column=i * 2, padx=3)
        fields[col] = var

    def on_select(event):
        """Заполнить поля ввода при выборе строки."""
        sel = tv.selection()
        if not sel:
            return
        vals = tv.item(sel[0], 'values')
        for i, col in enumerate(cols):
            fields[col].set(vals[i])

    tv.bind('<<TreeviewSelect>>', on_select)

    def add_record():
        """Добавить новую запись в справочник."""
        global PASSENGERS
        try:
            new_row = {
                'PassengerId': int(fields['PassengerId'].get()),
                'Survived': int(fields['Survived'].get()),
                'Pclass': int(fields['Pclass'].get()),
                'Sex': fields['Sex'].get().strip().lower(),
                'Age': float(fields['Age'].get()),
                'Embarked': fields['Embarked'].get().strip().upper(),
            }
        except ValueError as exc:
            messagebox.showwarning('Ошибка ввода', f'Проверьте данные: {exc}')
            return
        import pandas as pd
        PASSENGERS = pd.concat(
            [PASSENGERS, pd.DataFrame([new_row])], ignore_index=True)
        refresh()

    def edit_record():
        """Сохранить изменения в выбранной записи."""
        global PASSENGERS
        sel = tv.selection()
        if not sel:
            messagebox.showinfo('Выберите запись', 'Сначала выберите строку.')
            return
        try:
            pid = int(fields['PassengerId'].get())
        except ValueError:
            messagebox.showwarning('Ошибка', 'Некорректный PassengerId.')
            return
        idx = PASSENGERS.index[PASSENGERS['PassengerId'] == pid]
        if idx.empty:
            messagebox.showwarning(
                'Не найдено', f'PassengerId {pid} не найден.'
            )
            return
        try:
            PASSENGERS.loc[idx[0], 'Survived'] = int(fields['Survived'].get())
            PASSENGERS.loc[idx[0], 'Pclass'] = int(fields['Pclass'].get())
            PASSENGERS.loc[idx[0], 'Sex'] = fields['Sex'].get().strip().lower()
            PASSENGERS.loc[idx[0], 'Age'] = float(fields['Age'].get())
            PASSENGERS.loc[idx[0], 'Embarked'] = (
                fields['Embarked'].get().strip().upper())
        except ValueError as exc:
            messagebox.showwarning('Ошибка ввода', str(exc))
            return
        refresh()

    def delete_record():
        """Удалить выбранную запись."""
        global PASSENGERS
        sel = tv.selection()
        if not sel:
            messagebox.showinfo('Выберите запись', 'Сначала выберите строку.')
            return
        if not messagebox.askyesno('Удаление', 'Удалить выбранную запись?'):
            return
        try:
            pid = int(tv.item(sel[0], 'values')[0])
        except (IndexError, ValueError):
            return
        PASSENGERS = PASSENGERS[PASSENGERS['PassengerId'] != pid].reset_index(
            drop=True)
        refresh()

    def save_data():
        """Сохранить справочник в файл .pkl."""
        path = os.path.join(BASE_DIR, CONFIG['PATHS']['data_dir'],
                            'passengers.pkl')
        save_pickle(PASSENGERS, path)
        messagebox.showinfo('Сохранено', f'Данные сохранены:\n{path}')

    def load_data():
        """Загрузить справочник из файла .pkl."""
        global PASSENGERS
        path = filedialog.askopenfilename(
            title='Загрузить справочник',
            filetypes=[('Pickle файлы', '*.pkl')])
        if path:
            PASSENGERS = load_pickle(path)
            refresh()

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=5)
    for text, cmd in [('Добавить', add_record), ('Изменить', edit_record),
                       ('Удалить', delete_record), ('Сохранить', save_data),
                       ('Загрузить', load_data)]:
        make_button(btn_row, text, cmd).pack(side=tk.LEFT, padx=5)


# ---------------------------------------------------------------------------
# Окно справочника «Тарифы»
# ---------------------------------------------------------------------------

def open_fares_window(root: tk.Tk) -> None:
    """Открыть окно управления справочником «Тарифы».

    Обеспечивает просмотр, добавление, редактирование и удаление записей,
    а также сохранение и загрузку данных в формате Pickle.

    Args:
        root: Главное окно приложения.
    """
    global FARES
    cfg = CONFIG['INTERFACE']
    win = tk.Toplevel(root)
    win.title('Справочник — Тарифы')
    win.geometry('500x520')
    win.configure(bg=cfg.get('bg_color', '#F0F4F8'))

    cols = ['PassengerId', 'Fare']

    make_label(win, 'Справочник: Тарифы',
               font=(cfg.get('font_family', 'Arial'),
                     int(cfg.get('title_font_size', '14')), 'bold')
               ).pack(pady=(10, 4))

    tv = make_treeview(win, cols, heights=16)

    def refresh():
        """Обновить таблицу тарифов."""
        fill_treeview(tv, FARES[cols])

    refresh()

    ctrl = tk.Frame(win, bg=cfg.get('bg_color'))
    ctrl.pack(pady=8, fill=tk.X, padx=20)

    fields = {}
    for col in cols:
        tk.Label(ctrl, text=col, bg=cfg.get('bg_color'),
                 fg=cfg.get('fg_color'),
                 font=(cfg.get('font_family'), int(cfg.get('font_size')))
                 ).grid(row=0, column=cols.index(col), padx=6)
        var = tk.StringVar()
        tk.Entry(ctrl, textvariable=var, width=14).grid(
            row=1, column=cols.index(col), padx=6)
        fields[col] = var

    def on_select(event):
        """Заполнить поля при выборе строки."""
        sel = tv.selection()
        if sel:
            vals = tv.item(sel[0], 'values')
            for i, col in enumerate(cols):
                fields[col].set(vals[i])

    tv.bind('<<TreeviewSelect>>', on_select)

    def add_record():
        """Добавить тариф."""
        global FARES
        try:
            row = {'PassengerId': int(fields['PassengerId'].get()),
                   'Fare': float(fields['Fare'].get())}
        except ValueError as exc:
            messagebox.showwarning('Ошибка', str(exc))
            return
        import pandas as pd
        FARES = pd.concat([FARES, pd.DataFrame([row])], ignore_index=True)
        refresh()

    def edit_record():
        """Изменить выбранный тариф."""
        global FARES
        sel = tv.selection()
        if not sel:
            return
        try:
            pid = int(fields['PassengerId'].get())
            fare = float(fields['Fare'].get())
        except ValueError as exc:
            messagebox.showwarning('Ошибка', str(exc))
            return
        idx = FARES.index[FARES['PassengerId'] == pid]
        if not idx.empty:
            FARES.loc[idx[0], 'Fare'] = fare
        refresh()

    def delete_record():
        """Удалить выбранный тариф."""
        global FARES
        sel = tv.selection()
        if not sel:
            return
        if not messagebox.askyesno('Удаление', 'Удалить запись?'):
            return
        pid = int(tv.item(sel[0], 'values')[0])
        FARES = FARES[FARES['PassengerId'] != pid].reset_index(drop=True)
        refresh()

    def save_data():
        """Сохранить тарифы в .pkl."""
        path = os.path.join(BASE_DIR, CONFIG['PATHS']['data_dir'], 'fares.pkl')
        save_pickle(FARES, path)
        messagebox.showinfo('Сохранено', f'Сохранено:\n{path}')

    def load_data():
        """Загрузить тарифы из .pkl."""
        global FARES
        path = filedialog.askopenfilename(
            title='Загрузить тарифы',
            filetypes=[('Pickle файлы', '*.pkl')])
        if path:
            FARES = load_pickle(path)
            refresh()

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=5)
    for text, cmd in [('Добавить', add_record), ('Изменить', edit_record),
                       ('Удалить', delete_record), ('Сохранить', save_data),
                       ('Загрузить', load_data)]:
        make_button(btn_row, text, cmd).pack(side=tk.LEFT, padx=5)


# ---------------------------------------------------------------------------
# Окно простого текстового отчёта
# ---------------------------------------------------------------------------

def open_simple_report(root: tk.Tk) -> None:
    """Открыть окно формирования простого текстового отчёта.

    Позволяет выбрать столбцы и применить фильтры по полу,
    классу каюты и признаку выживания.

    Args:
        root: Главное окно приложения.
    """
    cfg = CONFIG['INTERFACE']
    win = tk.Toplevel(root)
    win.title('Отчёт: Простой текстовый')
    win.geometry('960x620')
    win.configure(bg=cfg.get('bg_color'))

    make_label(win, 'Простой текстовый отчёт — фильтрация и проекция',
               font=(cfg.get('font_family'),
                     int(cfg.get('title_font_size')), 'bold')
               ).pack(pady=(10, 4))

    all_cols = ['PassengerId', 'Survived', 'Pclass', 'Sex',
                'Age', 'Embarked', 'Fare']

    # --- Выбор столбцов ---
    col_frame = tk.LabelFrame(win, text='Выбор столбцов',
                               bg=cfg.get('bg_color'), fg=cfg.get('fg_color'))
    col_frame.pack(fill=tk.X, padx=15, pady=5)

    col_vars = {}
    for i, col in enumerate(all_cols):
        var = tk.BooleanVar(value=True)
        tk.Checkbutton(col_frame, text=col, variable=var,
                       bg=cfg.get('bg_color'), fg=cfg.get('fg_color'),
                       selectcolor=cfg.get('bg_color')
                       ).grid(row=0, column=i, padx=8, pady=4)
        col_vars[col] = var

    # --- Фильтры ---
    flt_frame = tk.LabelFrame(win, text='Фильтры',
                               bg=cfg.get('bg_color'), fg=cfg.get('fg_color'))
    flt_frame.pack(fill=tk.X, padx=15, pady=5)

    tk.Label(flt_frame, text='Пол:', bg=cfg.get('bg_color'),
             fg=cfg.get('fg_color')).grid(row=0, column=0, padx=8)
    sex_cb = make_combobox(flt_frame, ['', 'male', 'female'], width=12)
    sex_cb.grid(row=0, column=1, padx=4)

    tk.Label(flt_frame, text='Класс:', bg=cfg.get('bg_color'),
             fg=cfg.get('fg_color')).grid(row=0, column=2, padx=8)
    pclass_cb = make_combobox(flt_frame, ['', '1', '2', '3'], width=6)
    pclass_cb.grid(row=0, column=3, padx=4)

    tk.Label(flt_frame, text='Выжил:', bg=cfg.get('bg_color'),
             fg=cfg.get('fg_color')).grid(row=0, column=4, padx=8)
    surv_cb = make_combobox(flt_frame, ['', '0', '1'], width=6)
    surv_cb.grid(row=0, column=5, padx=4)

    # --- Таблица результата ---
    result_frame = tk.Frame(win, bg=cfg.get('bg_color'))
    result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

    tv_holder = [None]  # mutable container for the treeview

    def build_report():
        """Построить отчёт по текущим параметрам."""
        selected_cols = [c for c, v in col_vars.items() if v.get()]
        if not selected_cols:
            messagebox.showwarning(
                'Нет столбцов', 'Выберите хотя бы один столбец.'
            )
            return
        filters = {
            'Sex': sex_cb.get(),
            'Pclass': pclass_cb.get(),
            'Survived': surv_cb.get(),
        }
        df = reports.report_simple(PASSENGERS, FARES, selected_cols, filters)

        for w in result_frame.winfo_children():
            w.destroy()

        tv = make_treeview(result_frame, list(df.columns), heights=16)
        tv_holder[0] = df
        fill_treeview(tv, df)

        make_label(win, f'Найдено записей: {len(df)}').pack()

    def save_report():
        """Сохранить сформированный отчёт."""
        if tv_holder[0] is None:
            messagebox.showinfo('Нет данных', 'Сначала постройте отчёт.')
            return
        fmt = CONFIG['REPORTS'].get('default_text_format', 'csv')
        out_dir = os.path.join(BASE_DIR, CONFIG['PATHS']['output_dir'])
        path = filedialog.asksaveasfilename(
            initialdir=out_dir, defaultextension=f'.{fmt}',
            filetypes=[('CSV', '*.csv'), ('Текст', '*.txt')],
            title='Сохранить отчёт')
        if path:
            ext = os.path.splitext(path)[1].lstrip('.')
            reports.save_text_report(tv_holder[0], path, fmt=ext or fmt)
            messagebox.showinfo('Сохранено', f'Отчёт сохранён:\n{path}')

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=6)
    make_button(
        btn_row, 'Построить отчёт', build_report
    ).pack(side=tk.LEFT, padx=8)
    make_button(btn_row, 'Сохранить', save_report).pack(side=tk.LEFT, padx=8)


# ---------------------------------------------------------------------------
# Окно статистического отчёта
# ---------------------------------------------------------------------------

def open_statistics_report(root: tk.Tk) -> None:
    """Открыть окно статистического текстового отчёта.

    Для качественных переменных формирует таблицу частот,
    для количественных — описательные статистики.

    Args:
        root: Главное окно приложения.
    """
    cfg = CONFIG['INTERFACE']
    win = tk.Toplevel(root)
    win.title('Отчёт: Статистический')
    win.geometry('860x580')
    win.configure(bg=cfg.get('bg_color'))

    make_label(win, 'Статистический текстовый отчёт',
               font=(cfg.get('font_family'),
                     int(cfg.get('title_font_size')), 'bold')
               ).pack(pady=(10, 4))

    all_attrs = ['Sex', 'Embarked', 'Survived', 'Pclass', 'Age', 'Fare']

    sel_frame = tk.LabelFrame(win, text='Выберите атрибуты',
                               bg=cfg.get('bg_color'), fg=cfg.get('fg_color'))
    sel_frame.pack(fill=tk.X, padx=15, pady=5)

    attr_vars = {}
    for i, a in enumerate(all_attrs):
        var = tk.BooleanVar(value=True)
        tk.Checkbutton(sel_frame, text=a, variable=var,
                       bg=cfg.get('bg_color'), fg=cfg.get('fg_color'),
                       selectcolor=cfg.get('bg_color')
                       ).grid(row=0, column=i, padx=10, pady=4)
        attr_vars[a] = var

    result_area = tk.Frame(win, bg=cfg.get('bg_color'))
    result_area.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

    report_data = [None]

    def build_report():
        """Построить статистический отчёт."""
        selected = [a for a, v in attr_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning('Нет атрибутов', 'Выберите хотя бы один.')
            return
        stat = reports.report_statistics(PASSENGERS, FARES, selected)
        report_data[0] = stat

        for w in result_area.winfo_children():
            w.destroy()

        notebook = ttk.Notebook(result_area)
        notebook.pack(fill=tk.BOTH, expand=True)

        for attr, df in stat.items():
            tab = tk.Frame(notebook, bg=cfg.get('bg_color'))
            notebook.add(tab, text=attr)
            tv = make_treeview(tab, list(df.columns), heights=12)
            fill_treeview(tv, df)

    def save_report():
        """Сохранить все таблицы статистики в файл."""
        if report_data[0] is None:
            messagebox.showinfo('Нет данных', 'Сначала постройте отчёт.')
            return
        out_dir = os.path.join(BASE_DIR, CONFIG['PATHS']['output_dir'])
        path = filedialog.asksaveasfilename(
            initialdir=out_dir, defaultextension='.txt',
            filetypes=[('Текст', '*.txt'), ('CSV', '*.csv')],
            title='Сохранить статистику')
        if not path:
            return
        ext = os.path.splitext(path)[1].lstrip('.')
        lines = []
        for attr, df in report_data[0].items():
            lines.append(f'\n=== {attr} ===\n')
            lines.append(df.to_string(index=False))
            lines.append('\n')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines))
        messagebox.showinfo('Сохранено', f'Сохранено:\n{path}')

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=6)
    make_button(btn_row, 'Построить отчёт', build_report).pack(
        side=tk.LEFT, padx=8)
    make_button(btn_row, 'Сохранить', save_report).pack(side=tk.LEFT, padx=8)


# ---------------------------------------------------------------------------
# Окно сводной таблицы
# ---------------------------------------------------------------------------

def open_pivot_report(root: tk.Tk) -> None:
    """Открыть окно формирования сводной таблицы.

    Использует pandas.pivot_table() для агрегации данных
    по двум качественным атрибутам.

    Args:
        root: Главное окно приложения.
    """
    cfg = CONFIG['INTERFACE']
    win = tk.Toplevel(root)
    win.title('Отчёт: Сводная таблица')
    win.geometry('860x580')
    win.configure(bg=cfg.get('bg_color'))

    make_label(win, 'Сводная таблица',
               font=(cfg.get('font_family'),
                     int(cfg.get('title_font_size')), 'bold')
               ).pack(pady=(10, 4))

    all_cols = ['Survived', 'Pclass', 'Sex', 'Embarked', 'Age', 'Fare']
    aggfuncs = ['Количество', 'Среднее', 'Сумма', 'Минимум', 'Максимум']

    param_frame = tk.LabelFrame(
        win, text='Параметры',
        bg=cfg.get('bg_color'), fg=cfg.get('fg_color')
    )
    param_frame.pack(fill=tk.X, padx=15, pady=5)

    labels_and_cbs = [
        ('Строки (Index):', all_cols, 'Sex'),
        ('Столбцы:', all_cols, 'Survived'),
        ('Значения:', all_cols, 'Fare'),
        ('Функция:', aggfuncs, 'Среднее'),
    ]
    combos = []
    for i, (lbl, vals, default) in enumerate(labels_and_cbs):
        tk.Label(
            param_frame, text=lbl,
            bg=cfg.get('bg_color'),
            fg=cfg.get('fg_color')
        ).grid(row=0, column=i * 2, padx=6, pady=4)
        cb = make_combobox(param_frame, vals, width=14)
        cb.set(default)
        cb.grid(row=0, column=i * 2 + 1, padx=4)
        combos.append(cb)

    result_frame = tk.Frame(win, bg=cfg.get('bg_color'))
    result_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

    pivot_data = [None]

    def build_report():
        """Построить сводную таблицу."""
        index_col, col_col, val_col, agg = [cb.get() for cb in combos]
        try:
            pivot = reports.report_pivot(
                PASSENGERS, FARES, index_col, col_col, val_col, agg)
        except Exception as exc:
            messagebox.showerror('Ошибка построения', str(exc))
            return
        pivot_data[0] = pivot

        for w in result_frame.winfo_children():
            w.destroy()

        df_reset = pivot.reset_index()
        tv = make_treeview(result_frame, list(df_reset.columns), heights=14)
        fill_treeview(tv, df_reset)

    def save_report():
        """Сохранить сводную таблицу."""
        if pivot_data[0] is None:
            messagebox.showinfo('Нет данных', 'Сначала постройте отчёт.')
            return
        out_dir = os.path.join(BASE_DIR, CONFIG['PATHS']['output_dir'])
        path = filedialog.asksaveasfilename(
            initialdir=out_dir, defaultextension='.csv',
            filetypes=[('CSV', '*.csv'), ('Текст', '*.txt')])
        if path:
            ext = os.path.splitext(path)[1].lstrip('.')
            reports.save_text_report(
                pivot_data[0].reset_index(), path, fmt=ext
            )
            messagebox.showinfo('Сохранено', f'Сохранено:\n{path}')

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=6)
    make_button(btn_row, 'Построить таблицу', build_report).pack(
        side=tk.LEFT, padx=8)
    make_button(btn_row, 'Сохранить', save_report).pack(side=tk.LEFT, padx=8)


# ---------------------------------------------------------------------------
# Универсальное окно графического отчёта
# ---------------------------------------------------------------------------

def open_chart_window(root: tk.Tk, chart_type: str) -> None:
    """Открыть окно формирования графического отчёта.

    Поддерживает четыре типа диаграмм: столбчатая, гистограмма,
    Бокса–Вискера и рассеивания.

    Args:
        root: Главное окно приложения.
        chart_type: Тип диаграммы: 'bar', 'hist', 'box', 'scatter'.
    """
    cfg = CONFIG['INTERFACE']
    titles = {
        'bar': 'Столбчатая диаграмма',
        'hist': 'Гистограмма',
        'box': 'Диаграмма Бокса–Вискера',
        'scatter': 'Диаграмма рассеивания',
    }
    win = tk.Toplevel(root)
    win.title(f'Отчёт: {titles.get(chart_type, "График")}')
    win.geometry('980x680')
    win.configure(bg=cfg.get('bg_color'))

    make_label(win, titles.get(chart_type, 'График'),
               font=(cfg.get('font_family'),
                     int(cfg.get('title_font_size')), 'bold')
               ).pack(pady=(10, 4))

    qual_attrs = ['Survived', 'Pclass', 'Sex', 'Embarked']
    quant_attrs = ['Age', 'Fare']
    all_attrs = quant_attrs + qual_attrs

    param_frame = tk.LabelFrame(
        win, text='Параметры диаграммы',
        bg=cfg.get('bg_color'), fg=cfg.get('fg_color')
    )
    param_frame.pack(fill=tk.X, padx=15, pady=5)

    combos = {}
    if chart_type == 'bar':
        specs = [('Ось X (качественный):', qual_attrs, 'Pclass'),
                 ('Группировка:', qual_attrs, 'Survived')]
    elif chart_type in ('hist', 'box'):
        specs = [('Количественный:', quant_attrs, 'Age'),
                 ('Категория:', qual_attrs, 'Survived')]
    else:  # scatter
        specs = [('Ось X (колич.):', quant_attrs, 'Age'),
                 ('Ось Y (колич.):', quant_attrs, 'Fare'),
                 ('Цвет (кач.):', qual_attrs, 'Survived')]

    for i, (lbl, vals, default) in enumerate(specs):
        tk.Label(
            param_frame, text=lbl,
            bg=cfg.get('bg_color'),
            fg=cfg.get('fg_color')
        ).grid(row=0, column=i * 2, padx=6, pady=4)
        cb = make_combobox(param_frame, vals, width=14)
        cb.set(default)
        cb.grid(row=0, column=i * 2 + 1, padx=4)
        combos[i] = cb

    canvas_frame = tk.Frame(win, bg=cfg.get('bg_color'))
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

    fig_holder = [None]
    canvas_holder = [None]

    def build_chart():
        """Построить выбранную диаграмму."""
        vals = [cb.get() for cb in combos.values()]
        try:
            if chart_type == 'bar':
                fig = reports.chart_bar(PASSENGERS, FARES,
                                        vals[0], vals[1], CONFIG)
            elif chart_type == 'hist':
                fig = reports.chart_histogram(PASSENGERS, FARES,
                                              vals[0], vals[1], CONFIG)
            elif chart_type == 'box':
                fig = reports.chart_boxplot(PASSENGERS, FARES,
                                            vals[0], vals[1], CONFIG)
            else:
                fig = reports.chart_scatter(PASSENGERS, FARES,
                                            vals[0], vals[1], vals[2], CONFIG)
        except Exception as exc:
            messagebox.showerror('Ошибка построения', str(exc))
            return

        fig_holder[0] = fig

        for w in canvas_frame.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_holder[0] = canvas

    def save_chart():
        """Сохранить диаграмму в файл."""
        if fig_holder[0] is None:
            messagebox.showinfo('Нет графика', 'Сначала постройте диаграмму.')
            return
        fmt = CONFIG['REPORTS'].get('default_image_format', 'png')
        gfx_dir = os.path.join(BASE_DIR, CONFIG['PATHS']['graphics_dir'])
        path = filedialog.asksaveasfilename(
            initialdir=gfx_dir,
            defaultextension=f'.{fmt}',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf')],
            title='Сохранить диаграмму')
        if path:
            dpi = int(CONFIG['REPORTS'].get('figure_dpi', '100'))
            fig_holder[0].savefig(path, dpi=dpi, bbox_inches='tight')
            messagebox.showinfo('Сохранено', f'График сохранён:\n{path}')

    btn_row = tk.Frame(win, bg=cfg.get('bg_color'))
    btn_row.pack(pady=6)
    make_button(btn_row, 'Построить', build_chart).pack(side=tk.LEFT, padx=8)
    make_button(btn_row, 'Сохранить', save_chart).pack(side=tk.LEFT, padx=8)


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

def build_main_window() -> tk.Tk:
    """Создать и настроить главное окно приложения.

    Returns:
        Настроенный объект tk.Tk.
    """
    cfg = CONFIG['INTERFACE']
    root = tk.Tk()
    root.title('Анализ пассажиров Титаника | МИЭМ НИУ ВШЭ')
    w = int(cfg.get('window_width', '1100'))
    h = int(cfg.get('window_height', '720'))
    root.geometry(f'{w}x{h}')
    root.configure(bg=cfg.get('bg_color', '#F0F4F8'))
    root.resizable(True, True)

    # ---- Заголовок ----
    header = tk.Frame(root, bg=cfg.get('accent_color', '#2C3E8C'), height=60)
    header.pack(fill=tk.X)
    tk.Label(
        header, text='⚓  Анализ пассажиров Титаника',
        bg=cfg.get('accent_color'), fg='#FFFFFF',
        font=(cfg.get('font_family', 'Arial'),
              int(cfg.get('title_font_size', '14')), 'bold')
    ).pack(side=tk.LEFT, padx=20, pady=12)

    tk.Label(
        header,
        text='МИЭМ НИУ ВШЭ  |  Группа БИВ255  |  Анчы В.В.',
        bg=cfg.get('accent_color'), fg='#BDC3E8',
        font=(cfg.get('font_family'), 9)
    ).pack(side=tk.RIGHT, padx=20)

    # ---- Боковая панель навигации ----
    sidebar = tk.Frame(root, bg=cfg.get('accent_color'), width=200)
    sidebar.pack(side=tk.LEFT, fill=tk.Y)
    sidebar.pack_propagate(False)

    nav_sections = [
        ('📂 СПРАВОЧНИКИ', None),
        ('  Пассажиры', lambda: open_passengers_window(root)),
        ('  Тарифы', lambda: open_fares_window(root)),
        ('', None),
        ('📄 ТЕКСТОВЫЕ ОТЧЁТЫ', None),
        ('  Простой отчёт', lambda: open_simple_report(root)),
        ('  Статистика', lambda: open_statistics_report(root)),
        ('  Сводная таблица', lambda: open_pivot_report(root)),
        ('', None),
        ('📊 ГРАФИЧЕСКИЕ ОТЧЁТЫ', None),
        ('  Столбчатая', lambda: open_chart_window(root, 'bar')),
        ('  Гистограмма', lambda: open_chart_window(root, 'hist')),
        ('  Бокса–Вискера', lambda: open_chart_window(root, 'box')),
        ('  Рассеивания', lambda: open_chart_window(root, 'scatter')),
    ]

    for text, cmd in nav_sections:
        if not text:
            tk.Frame(sidebar, bg='#3D5A9F', height=1).pack(
                fill=tk.X, padx=10, pady=2)
            continue
        if cmd is None:
            tk.Label(sidebar, text=text,
                     bg=cfg.get('accent_color'), fg='#8CA0D7',
                     font=(cfg.get('font_family'), 9, 'bold'),
                     anchor='w'
                     ).pack(fill=tk.X, padx=12, pady=(8, 2))
        else:
            btn = tk.Button(
                sidebar, text=text, command=cmd,
                bg=cfg.get('accent_color'), fg='#FFFFFF',
                activebackground='#3D5A9F',
                relief=tk.FLAT, anchor='w',
                font=(cfg.get('font_family'), int(cfg.get('font_size'))),
                cursor='hand2',
                pady=4
            )
            btn.pack(fill=tk.X, padx=8, pady=1)
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg='#3D5A9F'))
            btn.bind('<Leave>', lambda e, b=btn: b.config(
                bg=cfg.get('accent_color')))

    # ---- Центральная область — дашборд ----
    main_area = tk.Frame(root, bg=cfg.get('bg_color'))
    main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    _build_dashboard(main_area)

    return root


def _build_dashboard(parent: tk.Frame) -> None:
    """Построить информационный дашборд на главном экране.

    Отображает ключевые статистики датасета в виде карточек.

    Args:
        parent: Родительский фрейм.
    """
    import pandas as pd
    cfg = CONFIG['INTERFACE']

    tk.Label(
        parent, text='Сводная информация о наборе данных',
        bg=cfg.get('bg_color'), fg=cfg.get('fg_color'),
        font=(cfg.get('font_family'), int(cfg.get('title_font_size')), 'bold')
    ).pack(pady=(20, 10))

    df = pd.merge(PASSENGERS, FARES, on='PassengerId', how='left')
    total = len(df)
    survived = int(df['Survived'].sum())
    avg_age = df['Age'].mean()
    avg_fare = df['Fare'].mean()
    pct_female = (df['Sex'] == 'female').sum() / total * 100

    cards = [
        ('Всего пассажиров', str(total), '#2C3E8C'),
        ('Выжило', f'{survived} ({survived / total * 100:.1f}%)', '#27AE60'),
        ('Средний возраст', f'{avg_age:.1f} лет', '#E67E22'),
        ('Средний тариф', f'£{avg_fare:.2f}', '#8E44AD'),
        ('Доля женщин', f'{pct_female:.1f}%', '#C0392B'),
    ]

    card_row = tk.Frame(parent, bg=cfg.get('bg_color'))
    card_row.pack(fill=tk.X, padx=20, pady=10)

    for title, value, color in cards:
        card = tk.Frame(card_row, bg=color, relief=tk.FLAT,
                        padx=16, pady=12)
        card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=6)
        tk.Label(card, text=value, bg=color, fg='#FFFFFF',
                 font=(cfg.get('font_family'), 18, 'bold')
                 ).pack()
        tk.Label(card, text=title, bg=color, fg='#D5E8FF',
                 font=(cfg.get('font_family'), 9)
                 ).pack()

    # Краткое описание
    desc = (
        'Набор данных: Titanic Dataset (Kaggle)  •  891 запись'
        '  •  2 справочника в 3НФ\n'
        'Справочник «Пассажиры»: PassengerId, Survived,'
        ' Pclass, Sex, Age, Embarked\n'
        'Справочник «Тарифы»: PassengerId, Fare\n\n'
        'Используйте меню слева для работы'
        ' со справочниками и формирования отчётов.'
    )
    tk.Label(
        parent, text=desc,
        bg=cfg.get('bg_color'), fg=cfg.get('fg_color'),
        font=(cfg.get('font_family'), 10),
        justify=tk.LEFT, wraplength=700
    ).pack(padx=30, pady=15, anchor='w')


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    """Точка входа. Инициализирует приложение и запускает главный цикл."""
    init_app()
    root = build_main_window()
    root.mainloop()


if __name__ == '__main__':
    main()
