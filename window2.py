import pandas as pd
import numpy as np


def clean_float(x):
    """
    转换器函数：先去除前后空白，再转换为float。
    如果为空或转换失败则返回 np.nan
    """
    x = str(x).strip()
    if x == '':
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan


def get_sliding_windows(seq, window_size=6):
    """
    返回DNA序列 seq 所有长度为 window_size 的滑动窗口列表，
    每个元素为 (起始索引, 子序列)。
    """
    seq = seq.upper()
    return [(i, seq[i : i + window_size]) for i in range(len(seq) - window_size + 1)]


THRESHOLD = 0.5


df9mer = pd.read_csv(
    '9mer_400bp.csv',
    sep=',',
    header=None,
    usecols=[0, 1],
    names=['seq', 'signal'],
    dtype={'seq': str},
    converters={'signal': clean_float},
    skipinitialspace=True,
)

df6mer = pd.read_csv(
    '6mer_450bp.csv',
    sep=',',
    header=None,
    usecols=[0, 1],
    names=['seq', 'signal'],
    dtype={'seq': str},
    converters={'signal': clean_float},
    skipinitialspace=True,
)


sixmer_dict = {}
for idx, row in df6mer.iterrows():
    if pd.isna(row['seq']) or pd.isna(row['signal']):
        continue
    seq6 = str(row['seq']).strip().upper()
    sig6 = row['signal']
    sixmer_dict.setdefault(seq6, []).append((sig6, idx, 0))


results = []
for idx, row in df9mer.iterrows():
    if pd.isna(row['seq']):
        continue
    seq9 = str(row['seq']).strip().upper()
    sig9 = row['signal']
    for pos, sub6 in get_sliding_windows(seq9, 6):
        if sub6 in sixmer_dict and not pd.isna(sig9):
            best_diff = float('inf')
            best = None
            for sig6, six_idx, six_pos in sixmer_dict[sub6]:
                diff = abs(sig9 - sig6)
                if diff < best_diff:
                    best_diff, best = diff, (sig6, six_idx, six_pos)
            if best is not None:
                quality = '匹配较好' if best_diff < THRESHOLD else '匹配较差'
                results.append(
                    {
                        '9mer_index': idx,
                        'window_start': pos,
                        'window_seq': sub6,
                        'signal_9mer': sig9,
                        'matched_signal_6mer': best[0],
                        '6mer_index': best[1],
                        '6mer_window_start': best[2],
                        'signal_diff': best_diff,
                        'match_quality': quality,
                    }
                )
        else:
            results.append(
                {
                    '9mer_index': idx,
                    'window_start': pos,
                    'window_seq': sub6,
                    'signal_9mer': sig9,
                    'matched_signal_6mer': np.nan,
                    '6mer_index': None,
                    '6mer_window_start': None,
                    'signal_diff': np.nan,
                    'match_quality': '无匹配',
                }
            )


df_results = pd.DataFrame(results)
df_results.to_csv('window_matching_results.csv', index=False)
print('所有窗口匹配已输出到 window_matching_results.csv')


df_valid = df_results.dropna(subset=['signal_diff']).copy()

summary = (
    df_valid.groupby(['window_start', '6mer_window_start'], as_index=False)
    .agg(mean_diff=('signal_diff', 'mean'), count_pairs=('signal_diff', 'size'))
)

top3 = summary.sort_values('mean_diff').head(4)

top3['9mer_range'] = top3['window_start'].add(1).astype(str) + '-' + top3['window_start'].add(6).astype(str)
top3['6mer_range'] = top3['6mer_window_start'].add(1).astype(str) + '-' + top3['6mer_window_start'].add(6).astype(str)

print('\n全局最优的 3 组窗口位置对：')
print(top3[['9mer_range', '6mer_range', 'mean_diff', 'count_pairs']])

top3.to_csv('global_top3_window_pairs.csv', index=False)
print('全局最优窗口位置对已保存到 global_top3_window_pairs.csv')
