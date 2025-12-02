import argparse
import math
import random
import sys
from collections import defaultdict
import statistics

def calculate_weight_coefficients(bi_probs, tri_probs, tetra_probs):

    # логи вероятности для всех n-грамм
    bi_logs = list(bi_probs.values())
    tri_logs = list(tri_probs.values())
    tetra_logs = list(tetra_probs.values())

    bi_mean = statistics.mean(bi_logs) if bi_logs else -8
    tri_mean = statistics.mean(tri_logs) if tri_logs else -10
    tetra_mean = statistics.mean(tetra_logs) if tetra_logs else -12

    bi_median = statistics.median(bi_logs) if bi_logs else -8
    tri_median = statistics.median(tri_logs) if tri_logs else -10
    tetra_median = statistics.median(tetra_logs) if tetra_logs else -12

    print(f"Средние log вероятности: биграммы={bi_mean:.2f}, триграммы={tri_mean:.2f}, тетраграммы={tetra_mean:.2f}")
    print(f"Медианы log вероятностей: биграммы={bi_median:.2f}, триграммы={tri_median:.2f}, тетраграммы={tetra_median:.2f}")

    base_tetra = 15.0
    base_tri = 8.0
    base_bi = 4.0

    if abs(tri_mean) > 1e-6 and abs(bi_mean) > 1e-6:
        tri_to_bi_ratio = abs(tri_mean / bi_mean)
        tetra_to_tri_ratio = abs(tetra_mean / tri_mean) if abs(tri_mean) > 1e-6 else 1.5

        w4 = base_tetra * tetra_to_tri_ratio
        w3 = base_tri * tri_to_bi_ratio
        w2 = base_bi

        total = w4 + w3 + w2
        w4 = w4 / total * 25
        w3 = w3 / total * 25
        w2 = w2 / total * 25

        print(f"Автоматически рассчитанные веса: w4={w4:.2f}, w3={w3:.2f}, w2={w2:.2f}")

        return w4, w3, w2

    return 12.0, 6.0, 3.0

def load_grams(path, n):
    counts = defaultdict(int)
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue
            gram = parts[0]
            try:
                cnt = int(parts[1])
            except:
                try:
                    cnt = int(parts[-1])
                    gram = ''.join(parts[:-1])
                except:
                    continue
            if len(gram) != n: 
                continue
            counts[gram] += cnt
            total += cnt
    return counts, total

def build_probs(counts, total, alphabet_size_estimate=100, addk=0.1):
    probs = {}
    V = max(len(counts), 0)
    denom = total + addk * V
    for g, c in counts.items():
        probs[g] = math.log((c + addk) / denom)
    missing_log = math.log(addk / denom)
    return probs, missing_log

def read_table(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            print(line)
            line = line.replace('\t', ' ')
            row = list(line)
            if not row:
                continue

            rows.append(row)

    if not rows:
        raise ValueError("Пустая таблица")

    max_len = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_len:
            r.extend([' '] * (max_len - len(r)))
    l = [i for i in range(0, len(rows[0]))]
    print(l)
    for i in rows:
        for j in range(0, len(i)):
            if i[j] != ' ' and j in l:
                l.remove(j)
    for i in range(len(rows)):
        for j in l[::-1]:
            rows[i] = rows[i][:j] + rows[i][j + 1:]
        print(rows[i])

    return rows, len(rows), max_len

def assemble_text(rows, perm):
    r = len(rows)
    text_chars = []
    for i in range(r):
        row = rows[i]
        for j in perm:
            text_chars.append(row[j])
    return ''.join(text_chars)


def score_text(text, tri_probs, tri_missing_log,
               bi_probs, bi_missing_log,
               tetra_probs, tetra_missing_log,
               w4=13.49, w3=8.0, w2=3.48):
    n = len(text)

    score = 0.0
    bad_run = 0

    # тетра-граммы (4 буквы)
    tetra_count = 0
    for i in range(n - 3):
        g4 = text[i:i+4]
        s = tetra_probs.get(g4, tetra_missing_log)
        if s > -2:
            bad_run += 1
        else:
            bad_run = 0
        if bad_run > 2:
            binus = 1.0 + (bad_run - 2) * 0.1
            score += w4 * s * binus
        else:
            score += w4 * s
        tetra_count += 1
    bad_run = 0

    # триграммы
    tri_count = 0
    for i in range(n - 2):
        g3 = text[i:i+3]
        s = tri_probs.get(g3, tri_missing_log)
        if s > -5:
            bad_run += 1
        else:
            bad_run = 0
        if bad_run > 2:
            binus = 1.0 + (bad_run - 2) * 0.1
            score += w3 * s * binus
        else:
            score += w3 * s
        tri_count += 1
    bad_run = 0

    # биграммы
    bi_count = 0
    for i in range(n - 1):
        g2 = text[i:i+2]
        s = bi_probs.get(g2, bi_missing_log)
        if s > -6:
            bad_run += 1
        else:
            bad_run = 0
        if bad_run > 4:
            binus = 1.0 + (bad_run - 4) * 0.1
            score += w2 * s * binus
        else:
            score += w2 * s
        bi_count += 1

    # НОРМАЛИЗАЦИЯ по количеству n-грамм
    total_ngrams = tetra_count + tri_count + bi_count
    if total_ngrams > 0:
        score = score / total_ngrams * n  # или просто score / total_ngrams

    return score


def op_swap(perm):
    a = random.randrange(len(perm))
    b = random.randrange(len(perm))
    if a == b:
        b = (a + 1) % len(perm)
    perm[a], perm[b] = perm[b], perm[a]
    return ('swap', a, b)

def op_triple_cycle(perm):
    n = len(perm)
    i = random.randrange(n)
    j = random.randrange(n)
    k = random.randrange(n)
    if i == j or j == k or i == k:
        return None
    tmp = perm[i]
    perm[i] = perm[j]
    perm[j] = perm[k]
    perm[k] = tmp
    return ('3cycle', i, j, k)

def op_block_shift(perm):
    n = len(perm)
    a = random.randrange(n)
    b = random.randrange(n)
    if a > b:
        a, b = b, a
    if a == b:
        return None
    k = random.randint(1, max(1, b - a))
    seg = perm[a:b+1]
    seg = seg[k:] + seg[:k]
    perm[a:b+1] = seg
    return ('bshift', a, b, k)

def op_large_shake(perm):
    n = len(perm)
    k = max(2, n // 4)
    for _ in range(k):
        a = random.randrange(n)
        b = random.randrange(n)
        perm[a], perm[b] = perm[b], perm[a]
    return ('shake', k)

def undo_op(perm, op):
    pass

def anneal(rows, tri_probs, tri_missing_log, bi_probs, bi_missing_log, tetra_probs, tetra_missing_log,
           max_iters=200000, T0=5.0, Tmin=0.001, restart_seed=None,
           moves_weights=(0.5, 0.2, 0.2, 0.1)):
    r = len(rows)
    c = len(rows[0])
    if restart_seed is not None:
        random.seed(restart_seed)
    perm = list(range(c))
    random.shuffle(perm)
    best_perm = perm.copy()
    best_text = assemble_text(rows, perm)
    best_score = score_text(
        best_text,
        tri_probs, tri_missing_log,
        bi_probs, bi_missing_log,
        tetra_probs, tetra_missing_log
    )
    current_score = best_score
    current_perm = perm.copy()

    moves = [op_swap, op_triple_cycle, op_block_shift, op_large_shake]
    weights = moves_weights
    s = sum(weights)
    weights = [w/s for w in weights]

    for i in range(max_iters):
        t = T0 * ((Tmin / T0) ** (i / max_iters))
        mv = random.choices(moves, weights)[0]
        prev_perm = current_perm.copy()
        opinfo = None
        for attempt in range(3):
            opinfo = mv(current_perm)
            if opinfo is not None:
                break
        if opinfo is None:
            opinfo = op_swap(current_perm)

        new_text = assemble_text(rows, current_perm)
        new_score = score_text(
            new_text,
            tri_probs, tri_missing_log,
            bi_probs, bi_missing_log,
            tetra_probs, tetra_missing_log
        )

        if new_score > current_score:
            accept = True
        else:
            delta = new_score - current_score
            try:
                ap = math.exp(delta / max(t, 1e-12))
            except OverflowError:
                ap = Tmin
            accept = (random.random() < ap)

        if accept:
            current_score = new_score
            # update best
            if new_score > best_score:
                best_score = new_score
                best_perm = current_perm.copy()
                best_text = new_text
        else:
            current_perm = prev_perm

        if i % 3000 == 0 and i > 0:
            tmp = current_perm.copy()
            op_large_shake(tmp)
            tmp_text = assemble_text(rows, tmp)
            tmp_score = score_text(
                tmp_text,
                tri_probs, tri_missing_log,
                bi_probs, bi_missing_log,
                tetra_probs, tetra_missing_log
            )
            if tmp_score > current_score or random.random() < 0.001:
                current_perm = tmp
                current_score = tmp_score

        if i % 2000 == 0:
            sys.stdout.write(f"\riter {i}/{max_iters} cur_score {current_score:.3f} best {best_score:.3f}")
            sys.stdout.flush()

    print()
    return best_perm, best_text, best_score

def full_search(tri_path, bi_path, table_path, restarts=4, iters=200000, seed=None):

    tri_counts, tri_total = load_grams(tri_path, 3)
    bi_counts, bi_total = load_grams(bi_path, 2)
    tetra_counts, tetra_total = load_grams(args.tetragrams, 4)
    tetra_probs, tetra_missing_log = build_probs(tetra_counts, tetra_total, alphabet_size_estimate=200, addk=3.1)

    tri_probs, tri_missing_log = build_probs(tri_counts, tri_total, alphabet_size_estimate=100, addk=10.01)
    bi_probs, bi_missing_log = build_probs(bi_counts, bi_total, alphabet_size_estimate=100, addk=1.01)
    rows, r, c = read_table(table_path)
    print(f"Table: {r} rows x {c} cols. Restarts: {restarts}, iters/restart: {iters}")

    global_best_score = -1e300
    global_best_text = None
    global_best_perm = None

    for run in range(restarts):
        rs = None if seed is None else seed + run
        print(f"\n=== Restart {run+1}/{restarts} seed={rs} ===")
        perm, text, score = anneal(rows, tri_probs, tri_missing_log, bi_probs, bi_missing_log, tetra_probs, tetra_missing_log,
                                   max_iters=iters, T0=10.0, Tmin=0.0001,
                                   restart_seed=rs,
                                   moves_weights=(0.45, 0.2, 0.25, 0.1))
        print(f"Run {run+1} best score {score:.3f}")
        print("Decoded (first 300 chars):")
        print(text[:300])
        if score > global_best_score:
            global_best_score = score
            global_best_text = text
            global_best_perm = perm.copy()

    print("\n=== Global best ===")
    print(f"Score {global_best_score:.3f}")
    print("Permutation (column indices):", global_best_perm)
    print("Decoded text:")
    print(global_best_text)
    return global_best_text, global_best_perm, global_best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Column transposition decryption with simulated annealing")
    parser.add_argument("--trigrams", "-t", required=True, help="file with trigram counts")
    parser.add_argument("--bigrams", "-b", required=True, help="file with bigram counts")
    parser.add_argument("--tetragrams", "-q", required=True, help="file with 4-gram counts")
    parser.add_argument("--table", "-g", required=True, help="table file (rows of symbols separated by spaces or contiguous)")
    parser.add_argument("--restarts", "-r", type=int, default=4)
    parser.add_argument("--iters", "-i", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    full_search(args.trigrams, args.bigrams, args.table, restarts=args.restarts, iters=args.iters, seed=args.seed)

