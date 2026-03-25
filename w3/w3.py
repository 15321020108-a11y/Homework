import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_current_figure(output_path):
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存: {output_path}")


# =========================================================
# 1. 利用闭包 + 惰性加载 构造情绪分析器
# =========================================================
def build_emotion_analyzer(lexicon_path):
    """
    返回两个函数：
    1) mixed_emotion(text): 混合情绪分析
    2) unique_emotion(text): 唯一情绪分析

    使用闭包保存 lexicon，且只在第一次真正调用时加载一次（惰性加载）。
    """

    lexicon = None  # 闭包变量：第一次为 None，之后加载完成会一直保存在这里
    emotion_categories = ['anger', 'disgust', 'fear', 'sadness', 'joy']

    def load_lexicon_once():
        """
        惰性加载：第一次调用时加载词典，后续直接复用。
        """
        nonlocal lexicon

        if lexicon is not None:
            return lexicon

        if not os.path.exists(lexicon_path):
            raise FileNotFoundError(f"未找到情绪词典路径: {lexicon_path}")

        lexicon = {emo: set() for emo in emotion_categories}

        if os.path.isdir(lexicon_path):
            for emo in emotion_categories:
                txt_path = os.path.join(lexicon_path, f"{emo}.txt")
                if not os.path.exists(txt_path):
                    raise ValueError(f"词典目录中未找到 {emo}.txt: {txt_path}")

                with open(txt_path, 'rb') as f:
                    words = []
                    for line in f:
                        try:
                            word = line.decode('utf-8').strip()
                        except UnicodeDecodeError:
                            word = line.decode('gbk', errors='ignore').strip()
                        if word:
                            words.append(word)
                    lexicon[emo] = set(words)
        else:
            with zipfile.ZipFile(lexicon_path, 'r') as z:
                file_list = z.namelist()

                for emo in emotion_categories:
                    matched_file = None

                    # 尽量兼容不同文件命名
                    for name in file_list:
                        lower_name = name.lower()
                        if emo in lower_name and not name.endswith('/'):
                            matched_file = name
                            break

                    if matched_file is None:
                        raise ValueError(f"压缩包中未找到 {emo} 对应词典文件")

                    with z.open(matched_file) as f:
                        words = []
                        for line in f:
                            try:
                                word = line.decode('utf-8').strip()
                            except UnicodeDecodeError:
                                word = line.decode('gbk', errors='ignore').strip()
                            if word:
                                words.append(word)
                        lexicon[emo] = set(words)

        print("情绪词典已加载，仅加载一次。")
        return lexicon

    def count_emotions(text):
        """
        统计一条评论中五类情绪词出现次数。
        返回:
            counts: dict
            total: 总情绪词数
            tokens: 分词列表
        """
        lex = load_lexicon_once()

        if pd.isna(text):
            text = ""

        tokens = str(text).strip().split()
        counts = {emo: 0 for emo in emotion_categories}

        for token in tokens:
            for emo in emotion_categories:
                if token in lex[emo]:
                    counts[emo] += 1

        total = sum(counts.values())
        return counts, total, tokens

    def mixed_emotion(text):
        """
        混合情绪分析：
        返回五类情绪比例，以及正负极性值。
        极性值定义为:
            valence = joy_ratio - (anger_ratio + disgust_ratio + fear_ratio + sadness_ratio)
        也可写成:
            valence = (joy_count - neg_count) / total
        """
        counts, total, tokens = count_emotions(text)

        if total == 0:
            return {
                'anger': 0.0,
                'disgust': 0.0,
                'fear': 0.0,
                'sadness': 0.0,
                'joy': 0.0,
                'total_emotion_words': 0,
                'coverage': 0,
                'valence': 0.0,
                'label': 'neutral'
            }

        ratios = {emo: counts[emo] / total for emo in emotion_categories}
        neg_ratio = ratios['anger'] + ratios['disgust'] + ratios['fear'] + ratios['sadness']
        valence = ratios['joy'] - neg_ratio

        if valence > 0:
            label = 'positive'
        elif valence < 0:
            label = 'negative'
        else:
            label = 'neutral'

        return {
            **ratios,
            'total_emotion_words': total,
            'coverage': 1,
            'valence': valence,
            'label': label
        }

    def unique_emotion(text):
        """
        唯一情绪分析：
        返回出现次数最多的情绪类别。
        特殊情况：
        1. 无情绪词 -> neutral
        2. 并列最多 -> mixed_tie
        """
        counts, total, tokens = count_emotions(text)

        if total == 0:
            return {
                'dominant_emotion': 'neutral',
                'counts': counts,
                'total_emotion_words': 0,
                'coverage': 0,
                'polarity': 'neutral'
            }

        max_count = max(counts.values())
        top_emotions = [emo for emo, c in counts.items() if c == max_count and c > 0]

        if len(top_emotions) > 1:
            dominant = 'mixed_tie'
            polarity = 'neutral'
        else:
            dominant = top_emotions[0]
            polarity = 'positive' if dominant == 'joy' else 'negative'

        return {
            'dominant_emotion': dominant,
            'counts': counts,
            'total_emotion_words': total,
            'coverage': 1,
            'polarity': polarity
        }

    return mixed_emotion, unique_emotion


# =========================================================
# 2. 对整张表做情绪分析
# =========================================================
def apply_emotion_analysis(df, text_col, mixed_func, unique_func):
    """
    对 DataFrame 批量做情绪分析，并拼接结果。
    """
    mixed_results = df[text_col].apply(mixed_func).apply(pd.Series)
    unique_results = df[text_col].apply(unique_func).apply(pd.Series)

    # unique_func 里的 counts 是字典，拆开
    unique_counts = unique_results['counts'].apply(pd.Series).add_prefix('unique_count_')
    unique_results = unique_results.drop(columns=['counts'])
    unique_results = unique_results.rename(columns={
        'total_emotion_words': 'unique_total_emotion_words',
        'coverage': 'unique_coverage'
    })

    result = pd.concat([df, mixed_results, unique_results, unique_counts], axis=1)
    return result


# =========================================================
# 3. 时间模式分析函数
# =========================================================
def plot_time_pattern(
    df,
    shop_id=None,
    sentiment='positive',
    mode='hour',
    analysis_type='mixed',
    output_dir='outputs'
):
    """
    参数:
        shop_id: 指定店铺ID，如 518986
        sentiment:
            - 若 analysis_type='mixed'，可选:
              'joy', 'anger', 'disgust', 'fear', 'sadness',
              'positive', 'negative', 'valence'
            - 若 analysis_type='unique'，可选:
              'positive', 'negative', 'neutral'
        mode: 'hour' / 'weekday' / 'month'
        analysis_type: 'mixed' / 'unique'
    """

    temp = df.copy()

    if shop_id is not None:
        temp = temp[temp['shopID'] == shop_id]

    if temp.empty:
        print("筛选后没有数据。")
        return

    if mode not in ['hour', 'weekday', 'month']:
        raise ValueError("mode 只能是 'hour' / 'weekday' / 'month'")

    x_col = mode

    if analysis_type == 'mixed':
        if sentiment == 'positive':
            temp['sent_value'] = temp['joy']
        elif sentiment == 'negative':
            temp['sent_value'] = temp[['anger', 'disgust', 'fear', 'sadness']].sum(axis=1)
        elif sentiment == 'valence':
            temp['sent_value'] = temp['valence']
        else:
            temp['sent_value'] = temp[sentiment]

        agg = temp.groupby(x_col)['sent_value'].mean().reset_index()
        y_label = f"{sentiment} 平均值"

    elif analysis_type == 'unique':
        if sentiment not in ['positive', 'negative', 'neutral']:
            raise ValueError("analysis_type='unique' 时 sentiment 只能是 positive/negative/neutral")

        temp['sent_value'] = (temp['polarity'] == sentiment).astype(int)
        agg = temp.groupby(x_col)['sent_value'].mean().reset_index()
        y_label = f"{sentiment} 比例"

    else:
        raise ValueError("analysis_type 只能是 'mixed' 或 'unique'")

    output_dir = ensure_output_dir(output_dir)
    shop_tag = f"shop_{shop_id}" if shop_id is not None else "all_shops"
    file_stub = f"time_pattern_{shop_tag}_{analysis_type}_{sentiment}_{mode}"
    csv_path = os.path.join(output_dir, f"{file_stub}.csv")
    fig_path = os.path.join(output_dir, f"{file_stub}.png")

    agg.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存: {csv_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(agg[x_col], agg['sent_value'], marker='o')
    plt.title(f"店铺 {shop_id} 的 {sentiment} {mode} 模式 ({analysis_type})")
    plt.xlabel(mode)
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    save_current_figure(fig_path)

    return agg


# =========================================================
# 4. 评分与情绪对比分析
# =========================================================
def compare_emotion_with_rating(df, output_dir='outputs'):
    """
    分析情绪值与评分的关系，并进行可视化。
    """
    temp = df.copy()
    output_dir = ensure_output_dir(output_dir)

    # 只保留有评分的数据
    temp = temp.dropna(subset=['stars'])

    # 评分分组统计
    star_group = temp.groupby('stars').agg(
        avg_valence=('valence', 'mean'),
        avg_joy=('joy', 'mean'),
        avg_negative=('anger', 'mean'),
        coverage_rate=('coverage', 'mean'),
        sample_size=('stars', 'size')
    ).reset_index()

    print("按评分聚合后的统计结果：")
    print(star_group)
    star_group_path = os.path.join(output_dir, "rating_emotion_summary.csv")
    star_group.to_csv(star_group_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存: {star_group_path}")

    # 图1：评分 vs 平均valence
    plt.figure(figsize=(8, 5))
    plt.plot(star_group['stars'], star_group['avg_valence'], marker='o')
    plt.title("评分与平均情绪极性值（valence）关系")
    plt.xlabel("评分")
    plt.ylabel("平均 valence")
    plt.grid(alpha=0.3)
    save_current_figure(os.path.join(output_dir, "rating_vs_avg_valence.png"))

    # 图2：评分 vs 词典覆盖率
    plt.figure(figsize=(8, 5))
    plt.bar(star_group['stars'], star_group['coverage_rate'])
    plt.title("评分与情绪词典覆盖率关系")
    plt.xlabel("评分")
    plt.ylabel("覆盖率")
    plt.grid(axis='y', alpha=0.3)
    save_current_figure(os.path.join(output_dir, "rating_vs_coverage.png"))

    # 图3：散点图（评分与valence）
    plt.figure(figsize=(8, 5))
    plt.scatter(temp['stars'], temp['valence'], alpha=0.2)
    plt.title("评分与评论 valence 散点图")
    plt.xlabel("评分")
    plt.ylabel("valence")
    plt.grid(alpha=0.3)
    save_current_figure(os.path.join(output_dir, "rating_vs_valence_scatter.png"))

    return star_group


# =========================================================
# 5. 额外分析：积极情绪差评
# =========================================================
def positive_low_rating_analysis(df, output_dir='outputs'):
    """
    分析“带积极情绪的差评”
    差评：stars <= 2
    积极：valence > 0
    """
    temp = df.copy()
    output_dir = ensure_output_dir(output_dir)
    temp = temp.dropna(subset=['stars'])

    low_rating = temp[temp['stars'] <= 2]
    pos_low_rating = low_rating[low_rating['valence'] > 0]

    print("差评总数：", len(low_rating))
    print("积极差评数：", len(pos_low_rating))

    if len(low_rating) > 0:
        print("积极差评占差评比例：", len(pos_low_rating) / len(low_rating))

    pos_low_rating_path = os.path.join(output_dir, "positive_low_rating_comments.csv")
    pos_low_rating.to_csv(pos_low_rating_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存: {pos_low_rating_path}")

    return pos_low_rating


# =========================================================
# 6. 主程序示例
# =========================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_dir(os.path.join(base_dir, "outputs"))

    # 1) 读取数据
    df = pd.read_csv(os.path.join(base_dir, "data3.csv"))

    # 2) 构造分析器（默认读取当前目录下的 emotion_lexicon 文件夹）
    mixed_func, unique_func = build_emotion_analyzer(os.path.join(base_dir, "emotion_lexicon"))

    # 3) 批量分析
    result_df = apply_emotion_analysis(df, "cus_comment", mixed_func, unique_func)

    # 4) 看看结果
    print(result_df[['cus_comment', 'joy', 'anger', 'sadness', 'valence', 'label', 'dominant_emotion', 'polarity']].head())
    result_path = os.path.join(output_dir, "emotion_analysis_result.csv")
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"统计结果已保存: {result_path}")

    # 5) 时间模式分析示例
    plot_time_pattern(result_df, shop_id=518986, sentiment='positive', mode='hour', analysis_type='mixed', output_dir=output_dir)
    plot_time_pattern(result_df, shop_id=520004, sentiment='negative', mode='weekday', analysis_type='mixed', output_dir=output_dir)

    # 6) 评分对比
    compare_emotion_with_rating(result_df, output_dir=output_dir)

    # 7) 积极差评
    positive_low_rating_analysis(result_df, output_dir=output_dir)
