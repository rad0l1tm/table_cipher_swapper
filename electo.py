"""
Интерактивный помощник для подбора следующих столбцов.
Оценка сочетания сделана через логарифм вероятностей n-грамм (2..4).
"""

import math
import os
import sys

# ---------------- ЗАГРУЗКА N-ГРАММ ----------------

def load_ngram_counts(path, n=2):
    counts = {}
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                gram = parts[0]
                try:
                    cnt = int(parts[1])
                except:
                    continue
                counts[gram] = counts.get(gram, 0) + cnt
                total += cnt
    except FileNotFoundError:
        print(f"Файл {path} не найден. Убедитесь, что он есть в рабочей папке.")
        sys.exit(1)
    return counts, total

# ---------------- ЧТЕНИЕ ТАБЛИЦЫ ----------------

def read_table(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.rstrip('\n').replace('\t', ' ')
            if s == '':
                continue
            rows.append(list(s))

    if not rows:
        raise ValueError("Empty table")

    max_len = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_len:
            r.extend([' '] * (max_len - len(r)))

    keep_idxs = []
    for j in range(max_len):
        if any(rows[i][j] != ' ' for i in range(len(rows))):
            keep_idxs.append(j)

    trimmed_rows = []
    for r in rows:
        trimmed_rows.append([r[j] for j in keep_idxs])

    num_cols = len(trimmed_rows[0])
    cols = []
    for c in range(num_cols):
        col = [trimmed_rows[r][c] for r in range(len(trimmed_rows))]
        cols.append(col)

    return cols

# ----------------- Модель вероятностей -----------------

class NgramModel:
    def __init__(self, big_counts, big_total, tri_counts, tri_total, tet_counts, tet_total,
                 add_k=1.0):
        self.big = big_counts
        self.big_total = max(1, big_total)
        self.V2 = max(1, len(self.big))
        self.tri = tri_counts
        self.tri_total = max(1, tri_total)
        self.V3 = max(1, len(self.tri))
        self.tet = tet_counts
        self.tet_total = max(1, tet_total)
        self.V4 = max(1, len(self.tet))
        self.k = float(add_k)

    def logprob_big(self, bi):
        # P(bi) = (count(bi)+k) / (total + k*V)
        c = self.big.get(bi, 0)
        num = c + self.k
        den = self.big_total + self.k * self.V2
        return math.log(num) - math.log(den)

    def logprob_tri(self, tri):
        c = self.tri.get(tri, 0)
        num = c + self.k
        den = self.tri_total + self.k * self.V3
        return math.log(num) - math.log(den)

    def logprob_tet(self, tet):
        c = self.tet.get(tet, 0)
        num = c + self.k
        den = self.tet_total + self.k * self.V4
        return math.log(num) - math.log(den)

# ----------------- Оценка сочетания столбцов -----------------

def compute_score(prev_cols, cand_idx, columns, ngram_model, weights):
    """
    prev_cols: список индексов уже выбранных столбцов (в порядке)
    cand_idx: индекс кандидата для оценки (целое)
    columns: список столбцов (каждый столбец - список символов по строкам)
    ngram_model: экземпляр NgramModel
    weights: dict с ключами w2,w3,w4 (веса для лог-проб)
    возвращает суммарный лог-скор (чем больше — тем лучше)
    """
    if not prev_cols:
        return 0.0

    R = len(columns[0])
    last_idx = prev_cols[-1]
    last_col = columns[last_idx]
    cand_col = columns[cand_idx]

    w2 = weights.get('w2', 1.0)
    w3 = weights.get('w3', 1.0)
    w4 = weights.get('w4', 1.0)

    total_log = 0.0
    for r in range(R):
        bi = last_col[r] + cand_col[r]
        total_log += w2 * ngram_model.logprob_big(bi)

        if len(prev_cols) >= 2:
            prev2_col = columns[prev_cols[-2]]
            tri = prev2_col[r] + last_col[r] + cand_col[r]
            total_log += w3 * ngram_model.logprob_tri(tri)

        if len(prev_cols) >= 3:
            prev3_col = columns[prev_cols[-3]]
            tet = prev3_col[r] + columns[prev_cols[-2]][r] + last_col[r] + cand_col[r]
            total_log += w4 * ngram_model.logprob_tet(tet)

    total_log /= max(1, R)
    return total_log

# ----------------- Вспомогательные печати -----------------

def print_table(columns):
    N = len(columns)
    print("\nТаблица (столбцы пронумерованы):")
    print(" ".join(f"{i:3d}" for i in range(N)))
    print("_".join('___' for i in range(N)))
    rows = len(columns[0])
    for r in range(rows):
        for c in range(N):
            print(' ', end = ' ')
            print(columns[c][r], end = ' ')
        print()
    print()

def print_column(col):
    for ch in col:
        print(ch)
    print()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_sequence(seq, columns, l = 0):
    if not l:
        if not seq:
            print("Текущая последовательность пуста.")
            return
        rows = len(columns[0])
        print("\nСобранная пользователем последовательность (строки):")
        for r in range(rows):
            print("".join(columns[c][r] for c in seq))
        print("Индексы:", seq)
        print()
    else:
        N = len(columns)
        print("Cols left:")
        print(" ".join(f"{i:3d}" if i not in seq else '\b' for i in range(N) ))
        print("_".join('___' if i not in seq else '\b' for i in range(N) ))
        rows = len(columns[0])
        for r in range(rows):
            if 0 in seq:
                print(end = ' ')
            for c in range(N):

                if c not in seq:
                    print(' ', end = ' ')
                    print(columns[c][r], end = ' ')
            print()
        print()

# ----------------- Главная интерактивная функция -----------------

def interactive_loop(columns, ngram_model):
    N = len(columns)
    user_seq = []
    show_n = 5

    # начальные веса (можно корректировать внутри программы командой 'set w2 w3 w4')
    weights = {'w2': 1.0, 'w3': 1.5, 'w4': 2.0}
    add_k = ngram_model.k

    print_table(columns)

    help_text = """
Команды:
 - <num>         : выбрать столбец с номером num и добавить в вашу последовательность
 - show n        : показывать n лучших кандидатов (по убыванию score)
 - show all      : вывести текущую собранную таблицу (в виде строк)
 - del n         : удалить n последних выбранных столбцов из вашей последовательности
 - set w w w     : установить веса w2 w3 w4 (три числа), например: set 1.0 2.0 3.0
 - set addk k    : установить параметр сглаживания add_k (положительное число)
 - help          : показать эту подсказку
 - exit          : выход
 - last          : оставшиеся столбцы
"""
    print(help_text)

    while True:
        cmd = input(": ").strip()

        if not cmd:
            continue
        if cmd == "exit":
            print("Выход.")
            return
        if cmd == "help":
            clear_screen()
            print(help_text)
            continue

        if cmd == "show all":

            print_sequence(user_seq, columns)
            continue

        if cmd.startswith("show "):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    s_n = int(parts[1])
                    print(f"Топ-{s_n} кандидатов на продолжение (score логарифмический):")
                    for sc, c in scored[:s_n]:
                        print(f" idx={c:3d}   score={sc:.6f}")
                    print()
                except:
                    print("Неверный аргумент для show.")
            else:
                print("Команда show принимает один аргумент: show N")
            continue

        if cmd.startswith("del "):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    k = int(parts[1])
                    if k <= 0:
                        print("k должно быть положительным.")
                        continue
                    if k > len(user_seq):
                        print("Указано больше, чем текущая длина последовательности. Очищаю всё.")
                        user_seq = []
                    else:
                        user_seq = user_seq[:-k]
                        print(f"Удалено {k} последних столбцов.")
                except:
                    print("Ошибка в аргументе del.")
            else:
                print("Команда del принимает один аргумент: del N")
            continue

        if cmd.startswith("set "):
            parts = cmd.split()
            if len(parts) == 4:
                try:
                    w2 = float(parts[1])
                    w3 = float(parts[2])
                    w4 = float(parts[3])
                    weights['w2'] = w2
                    weights['w3'] = w3
                    weights['w4'] = w4
                    print(f"Установлены веса: w2={w2}, w3={w3}, w4={w4}")
                except:
                    print("Ошибка: требуются три числа для set w2 w3 w4")
            elif len(parts) == 3 and parts[1] == "addk":
                try:
                    newk = float(parts[2])
                    if newk <= 0:
                        print("add_k должно быть > 0")
                    else:
                        ngram_model.k = newk
                        print(f"Установлен add_k = {newk}")
                except:
                    print("Ошибка при установке add_k.")
            else:
                print("Команда set: либо 'set w2 w3 w4' либо 'set addk k'")
            continue

        # если число — добавляем столбец
        try:
            idx = int(cmd)
        except:
            print("Нераспознанная команда. Введите help для подсказки.")
            continue

        if not (0 <= idx < N):
            print("Индекс столбца вне диапазона.")
            continue

        user_seq.append(idx)
        clear_screen()
        print_sequence(user_seq, columns, 1)
        print(f"\nДобавлен столбец {idx}. Его содержимое:")
        print_column(columns[idx])

        remaining = [i for i in range(N) if i not in user_seq]
        if not remaining:
            print("Больше нет свободных столбцов.")
            continue

        scored = []
        for cand in remaining:
            sc = compute_score(user_seq, cand, columns, ngram_model, weights)
            scored.append((sc, cand))
        scored.sort(reverse=True, key=lambda x: x[0])

        print(f"Топ-{show_n} кандидатов на продолжение (score логарифмический):")
        for sc, c in scored[:show_n]:
            print(f" idx={c:3d}   score={sc:.6f}")
        print()

# ----------------- Запуск -----------------

def main():
    # файлы n-грамм (ожидаются в текущей папке)
    big_path = "bis.txt"
    tri_path = "tris.txt"
    tet_path = "tets.txt"
    table_path = "table.txt"

    print("Загрузка n-грамм...")
    big_counts, big_total = load_ngram_counts(big_path, n=2)
    tri_counts, tri_total = load_ngram_counts(tri_path, n=3)
    tet_counts, tet_total = load_ngram_counts(tet_path, n=4)

    # параметр сглаживания (add-k)
    add_k = 1.0

    ngram_model = NgramModel(big_counts, big_total, tri_counts, tri_total, tet_counts, tet_total, add_k=add_k)

    print("Чтение таблицы...")
    columns = read_table(table_path)

    interactive_loop(columns, ngram_model)


if __name__ == "__main__":
    main()
