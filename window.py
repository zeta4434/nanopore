import pandas as pd
import numpy as np

# 转换器函数：先去除前后空白，再转换为float。
# 如果为空或转换失败则返回 np.nan
def clean_float(x):
    x = str(x).strip()
    if x == '':
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan

# 获取滑动窗口函数
def get_sliding_windows(seq, window_size=3):
    return [(i, seq[i:i+window_size]) for i in range(len(seq) - window_size + 1)]

# 匹配阈值
THRESHOLD = 0.5

# 读取9mer数据
df9mer = pd.read_csv(
    '9mer_400bp.csv', sep=",", header=None, usecols=[0,1],
    names=['seq','signal'], dtype={'seq': str},
    converters={'signal': clean_float}, skipinitialspace=True
)

# 读取6mer数据
df6mer = pd.read_csv(
    '6mer_450bp.csv', sep=",", header=None, usecols=[0,1],
    names=['seq','signal'], dtype={'seq': str},
    converters={'signal': clean_float}, skipinitialspace=True
)

# 构建6mer字典
sixmer_dict = {}
for idx, row in df6mer.iterrows():
    if pd.isna(row['seq']):
        continue
    seq6 = str(row['seq']).strip().upper()
    sig6 = row['signal']
    for pos, subseq in get_sliding_windows(seq6, 3):
        sixmer_dict.setdefault(subseq, []).append((sig6, idx, pos))

# 匹配9mer
results = []
for idx, row in df9mer.iterrows():
    if pd.isna(row['seq']):
        continue
    seq9 = str(row['seq']).strip().upper()
    sig9 = row['signal']
    for pos, sub9 in get_sliding_windows(seq9, 3):
        if sub9 in sixmer_dict:
            # 找到最接近的6mer信号
            best_diff = float('inf')
            best = None
            for sig6, six_idx, six_pos in sixmer_dict[sub9]:
                diff = abs(sig9 - sig6)
                if diff < best_diff:
                    best_diff, best = diff, (sig6, six_idx, six_pos)
            quality = '匹配较好' if best_diff < THRESHOLD else '匹配较差'
            results.append({
                '9mer_index': idx,
                'window_start': pos,
                'window_seq': sub9,
                'signal_9mer': sig9,
                'matched_signal_6mer': best[0],
                '6mer_index': best[1],
                '6mer_window_start': best[2],
                'signal_diff': best_diff,
                'match_quality': quality
            })
        else:
            results.append({
                '9mer_index': idx,
                'window_start': pos,
                'window_seq': sub9,
                'signal_9mer': sig9,
                'matched_signal_6mer': np.nan,
                '6mer_index': None,
                '6mer_window_start': None,
                'signal_diff': np.nan,
                'match_quality': '无匹配'
            })

# 保存所有窗口匹配结果
df_results = pd.DataFrame(results)
df_results.to_csv('window_matching_results.csv', index=False)
print("所有窗口匹配已输出到 window_matching_results.csv")

# ====== 基于线性回归的回归系数统计 ======
# 只保留有匹配的条目
df_valid = df_results.dropna(subset=['signal_diff']).copy()

# 按窗口起始位置对分组，计算回归斜率和配对数量
records = []
for (w9, w6), group in df_valid.groupby(['window_start', '6mer_window_start']):
    x = group['matched_signal_6mer'].values
    y = group['signal_9mer'].values
    if len(x) > 1:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = np.nan
    records.append({
        'window_start': w9,
        '6mer_window_start': w6,
        'slope': slope,
        'count_pairs': len(x)
    })

# 构建回归系数DataFrame，按与1的差值排序，取前12
df_slopes = pd.DataFrame(records)
df_slopes['slope_diff'] = (df_slopes['slope'] - 1).abs()
top12 = df_slopes.nsmallest(12, 'slope_diff').copy()

# 为可读性添加 1-based 范围字符串
top12['9mer_range'] = top12['window_start'].add(1).astype(str) + '-' + top12['window_start'].add(3).astype(str)
top12['6mer_range'] = top12['6mer_window_start'].add(1).astype(str) + '-' + top12['6mer_window_start'].add(3).astype(str)

print("\n回归系数最接近1的 12 组窗口位置对：")
print(top12[['9mer_range', '6mer_range', 'slope', 'count_pairs']])

# 保存结果
top12.to_csv('window_pair_slopes_top12.csv', index=False)
print("窗口位置对回归斜率已保存到 window_pair_slopes_top12.csv")
