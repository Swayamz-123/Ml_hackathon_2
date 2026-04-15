"""Analyze the 155 disagreement cases to understand patterns."""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import re
from collections import Counter
from data_loader.data_loader import load_data

benchmark = pd.read_csv('submission_95score.csv')
ours      = pd.read_csv('submission_v2_kaggle.csv')

benchmark['id'] = benchmark['id'].astype(str).str.lstrip('0').replace('', '0').astype(int)
ours['id']      = ours['id'].astype(int)

merged = benchmark.merge(ours, on='id', suffixes=('_bench','_ours'))
bench_label = merged['real_text_id_bench'].values
our_label   = merged['real_text_id_ours'].values

diff_ids = merged['id'].values[bench_label != our_label]
same_ids = merged['id'].values[bench_label == our_label]

df_test = load_data('data/test', None)
df_test = df_test.set_index('id')

def word_count(t): return len(str(t).split())
def char_count(t): return len(str(t))
def has_informal(t):
    informal = {'wow','amazing','incredible','awesome','cool','great','fantastic',
                'wonderful','brilliant','super','nice','hope','hopefully',
                'dont','wont','cant','im','ive','youre','theyre','basically',
                'literally','actually','totally','definitely','obviously'}
    words = set(re.findall(r"[a-z']+", str(t).lower()))
    return len(words & informal)
def num_count(t): return len(re.findall(r'\d+', str(t)))
def empty_or_short(t): return word_count(t) < 10

print("=== DISAGREEMENT ANALYSIS ===\n")
print(f"Disagreements: {len(diff_ids)}, Agreements: {len(same_ids)}\n")

# Analyze disagreement cases
wc_a_diff, wc_b_diff = [], []
wc_a_same, wc_b_same = [], []
inf_diff, inf_same = [], []
empty_diff, empty_same = 0, 0
len_ratio_diff, len_ratio_same = [], []

for id_ in diff_ids:
    if id_ not in df_test.index: continue
    row = df_test.loc[id_]
    a, b = str(row['summary_A']), str(row['summary_B'])
    wca, wcb = word_count(a), word_count(b)
    wc_a_diff.append(wca); wc_b_diff.append(wcb)
    inf_diff.append(has_informal(a) + has_informal(b))
    if empty_or_short(a) or empty_or_short(b): empty_diff += 1
    if wcb > 0: len_ratio_diff.append(wca / wcb)

for id_ in same_ids[:500]:  # sample
    if id_ not in df_test.index: continue
    row = df_test.loc[id_]
    a, b = str(row['summary_A']), str(row['summary_B'])
    wca, wcb = word_count(a), word_count(b)
    wc_a_same.append(wca); wc_b_same.append(wcb)
    inf_same.append(has_informal(a) + has_informal(b))
    if empty_or_short(a) or empty_or_short(b): empty_same += 1
    if wcb > 0: len_ratio_same.append(wca / wcb)

print("--- Word counts ---")
print(f"Disagreements: A={np.mean(wc_a_diff):.0f}±{np.std(wc_a_diff):.0f}  B={np.mean(wc_b_diff):.0f}±{np.std(wc_b_diff):.0f}")
print(f"Agreements:    A={np.mean(wc_a_same):.0f}±{np.std(wc_a_same):.0f}  B={np.mean(wc_b_same):.0f}±{np.std(wc_b_same):.0f}")

print(f"\n--- Length ratio (A/B) ---")
print(f"Disagreements: mean={np.mean(len_ratio_diff):.2f}  std={np.std(len_ratio_diff):.2f}")
print(f"Agreements:    mean={np.mean(len_ratio_same):.2f}  std={np.std(len_ratio_same):.2f}")

print(f"\n--- Informal word count ---")
print(f"Disagreements: mean={np.mean(inf_diff):.2f}")
print(f"Agreements:    mean={np.mean(inf_same):.2f}")

print(f"\n--- Empty/very short texts ---")
print(f"Disagreements: {empty_diff}/{len(diff_ids)} ({100*empty_diff/len(diff_ids):.1f}%)")
print(f"Agreements:    {empty_same}/{len(same_ids[:500])} ({100*empty_same/500:.1f}%)")

# Show 10 disagreement examples
print("\n--- 10 sample disagreement cases ---")
for id_ in diff_ids[:10]:
    if id_ not in df_test.index: continue
    row = df_test.loc[id_]
    a, b = str(row['summary_A'])[:120], str(row['summary_B'])[:120]
    bench = bench_label[merged['id'].values == id_][0]
    ours_ = our_label[merged['id'].values == id_][0]
    print(f"\nID={id_}  benchmark says real={bench}, we say real={ours_}")
    print(f"  A: {a}")
    print(f"  B: {b}")
