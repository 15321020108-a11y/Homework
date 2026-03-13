from __future__ import annotations

from collections import Counter
from math import sqrt
from pathlib import Path

try:
    import jieba
    import jieba.posseg as pseg
except ImportError as exc:
    raise SystemExit("缺少 jieba，请先安装：pip install jieba") from exc

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None


DATA_PATH = Path("p2_data.txt")
TOP_K = 10
MIN_WORD_LENGTH = 2
MIN_FREQ_FOR_WORDCLOUD = 2

STOPWORDS = {
    "的", "了", "和", "是", "在", "就", "都", "而", "及", "与", "着", "或", "一个",
    "没有", "我们", "你们", "他们", "她们", "是否", "自己", "不会", "不是", "可以",
    "这个", "那个", "一种", "一些", "什么", "怎么", "怎么说", "以及", "因为", "所以",
    "如果", "但是", "并且", "然后", "需要", "进行", "通过", "对于", "其中", "非常",
    "可能", "已经", "还是", "这样", "那样", "一个", "一种", "一下", "时候", "问题",
}


def read_documents(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"找不到数据文件：{path.resolve()}")
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def print_preview(documents: list[str], lines: int = 10) -> None:
    print("1. 前10行文本预览：")
    for index, line in enumerate(documents[:lines], start=1):
        print(f"{index}: {line}")
    print()


def tokenize_documents(documents: list[str]) -> list[list[str]]:
    tokenized = []
    for document in documents:
        words = [word.strip() for word in jieba.lcut(document) if word.strip()]
        tokenized.append(words)
    return tokenized


def is_valid_word(word: str, stopwords: set[str] | None = None) -> bool:
    if len(word) < MIN_WORD_LENGTH:
        return False
    if word.isdigit():
        return False
    if stopwords and word in stopwords:
        return False
    return True


def count_words(tokenized_documents: list[list[str]], stopwords: set[str] | None = None) -> Counter[str]:
    counter: Counter[str] = Counter()
    for words in tokenized_documents:
        filtered_words = [word for word in words if is_valid_word(word, stopwords)]
        counter.update(filtered_words)
    return counter


def print_top_words(counter: Counter[str], title: str, top_k: int = TOP_K) -> None:
    print(title)
    for word, freq in counter.most_common(top_k):
        print(f"{word}: {freq}")
    print()


def analyze_pos(documents: list[str], stopwords: set[str]) -> tuple[Counter[str], Counter[str]]:
    pos_counter: Counter[str] = Counter()
    adjective_counter: Counter[str] = Counter()
    for document in documents:
        for pair in pseg.cut(document):
            word = pair.word.strip()
            flag = pair.flag
            if not is_valid_word(word, stopwords):
                continue
            pos_counter[flag] += 1
            if flag.startswith("a"):
                adjective_counter[word] += 1
    return pos_counter, adjective_counter


def count_bigrams(tokenized_documents: list[list[str]], stopwords: set[str]) -> Counter[tuple[str, str]]:
    bigram_counter: Counter[tuple[str, str]] = Counter()
    for words in tokenized_documents:
        filtered_words = [word for word in words if is_valid_word(word, stopwords)]
        for i in range(len(filtered_words) - 1):
            bigram = (filtered_words[i], filtered_words[i + 1])
            bigram_counter[bigram] += 1
    return bigram_counter


def build_feature_words(counter: Counter[str], top_k: int = 50) -> list[str]:
    return [word for word, _ in counter.most_common(top_k)]


def vectorize_document(words: list[str], feature_words: list[str]) -> list[int]:
    word_counter = Counter(words)
    return [word_counter[word] for word in feature_words]


def cosine_similarity(vec1: list[int], vec2: list[int]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sqrt(sum(a * a for a in vec1))
    norm2 = sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def save_wordcloud(counter: Counter[str], output_name: str) -> None:
    if WordCloud is None:
        print(f"跳过词云 {output_name}：缺少 wordcloud，请先安装：pip install wordcloud")
        print()
        return

    filtered_counter = {word: freq for word, freq in counter.items() if freq >= MIN_FREQ_FOR_WORDCLOUD}
    if not filtered_counter:
        print(f"跳过词云 {output_name}：没有满足最低词频要求的词")
        print()
        return

    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    font_path = next((path for path in font_candidates if Path(path).exists()), None)

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        font_path=font_path,
    ).generate_from_frequencies(filtered_counter)
    wordcloud.to_file(output_name)
    print(f"词云已保存：{Path(output_name).resolve()}")
    print()


def main() -> None:
    documents = read_documents(DATA_PATH)
    print_preview(documents)

    tokenized_documents = tokenize_documents(documents)

    word_counter = count_words(tokenized_documents)
    print_top_words(word_counter, "2-3. 未过滤停用词时，词频最高的前10个词：")

    filtered_counter = count_words(tokenized_documents, STOPWORDS)
    print_top_words(filtered_counter, "4. 过滤停用词后，词频最高的前10个词：")

    save_wordcloud(filtered_counter, "wordcloud_top_words.png")

    pos_counter, adjective_counter = analyze_pos(documents, STOPWORDS)
    print_top_words(pos_counter, "6. 词性频率前10名：")
    print_top_words(adjective_counter, "6. 形容词前10名：")
    save_wordcloud(adjective_counter, "wordcloud_adjectives.png")

    bigram_counter = count_bigrams(tokenized_documents, STOPWORDS)
    print("7. 高频 bigram 前10名：")
    for bigram, freq in bigram_counter.most_common(TOP_K):
        print(f"{bigram}: {freq}")
    print()

    bigram_for_cloud = Counter({" / ".join(bigram): freq for bigram, freq in bigram_counter.items()})
    save_wordcloud(bigram_for_cloud, "wordcloud_bigrams.png")

    feature_words = build_feature_words(filtered_counter, top_k=20)
    print("8. 选取的特征词：")
    print(feature_words)
    print()

    filtered_documents = [
        [word for word in words if is_valid_word(word, STOPWORDS)]
        for words in tokenized_documents
    ]
    document_vectors = [vectorize_document(words, feature_words) for words in filtered_documents]

    print("8. 前3条文本的向量表示：")
    for index, vector in enumerate(document_vectors[:3], start=1):
        print(f"doc{index}: {vector}")
    print()

    if len(document_vectors) >= 2:
        similarity = cosine_similarity(document_vectors[0], document_vectors[1])
        print(f"8. 第1条和第2条文本的余弦相似度：{similarity:.4f}")


if __name__ == "__main__":
    main()
