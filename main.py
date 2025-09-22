import sys
import os
import unicodedata
import difflib
import re
import math
from collections import Counter
import jieba


def try_read_file(path):
    """
    尝试使用多种编码读取文件，返回文本内容
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']

    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    except Exception:
        return ''


def preprocess(text, mode='normal'):
    """
    简化的预处理函数，提高性能
    """
    if not text:
        return ''

    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text.strip())

    if mode == 'aggressive':
        # 激进模式：只保留中文字符和数字
        return re.sub(r'[^\u4e00-\u9fa50-9]', '', text)

    elif mode == 'semantic':
        # 语义模式：使用jieba分词
        words = jieba.cut(text)
        # 只保留长度>=2的词
        return ' '.join(word for word in words if len(word) >= 2)

    else:
        # 常规模式
        return text


def lcs_length(a, b):
    """
    优化的LCS计算，使用滚动数组减少内存使用
    """
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a

    m, n = len(short), len(long)
    if m == 0 or n == 0:
        return 0

    # 使用滚动数组
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if long[i - 1] == short[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev

    return prev[m]


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    magnitude1 = math.sqrt(sum(v * v for v in vec1.values()))
    magnitude2 = math.sqrt(sum(v * v for v in vec2.values()))

    if magnitude1 * magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def compute_semantic_similarity(orig_text, copy_text):
    """
    简化的语义相似度计算
    """
    if not orig_text or not copy_text:
        return 0.0

    # 预处理文本
    orig_processed = preprocess(orig_text, 'semantic')
    copy_processed = preprocess(copy_text, 'semantic')

    if not orig_processed or not copy_processed:
        return 0.0

    # 分词并计算词频
    orig_words = orig_processed.split()
    copy_words = copy_processed.split()

    if not orig_words:
        return 0.0

    # 计算词频向量
    orig_counter = Counter(orig_words)
    copy_counter = Counter(copy_words)

    return cosine_similarity(orig_counter, copy_counter)


def compute_keyword_similarity(orig_text, copy_text):
    """
    基于关键词的相似度计算
    """
    orig_processed = preprocess(orig_text, 'semantic')
    copy_processed = preprocess(copy_text, 'semantic')

    orig_words = set(orig_processed.split())
    copy_words = set(copy_processed.split())

    if not orig_words:
        return 0.0

    common_words = orig_words & copy_words
    return len(common_words) / len(orig_words)


def compute_structural_similarity(orig_text, copy_text):
    """
    简化的结构相似度计算
    """
    # 只检查前20个句子以提高性能
    orig_sentences = [s.strip() for s in re.split(r'[。！？.!?]', orig_text) if s.strip()][:20]
    copy_sentences = [s.strip() for s in re.split(r'[。！？.!?]', copy_text) if s.strip()][:20]

    if not orig_sentences:
        return 0.0

    matched_count = 0
    for orig_sent in orig_sentences:
        for copy_sent in copy_sentences:
            if len(orig_sent) > 10 and orig_sent[:20] in copy_sent:
                matched_count += 1
                break

    return matched_count / len(orig_sentences)


def compute_text_similarity(orig_text, copy_text):
    """
    基础文本相似度计算
    """
    if not orig_text:
        return 0.0

    orig_processed = preprocess(orig_text, 'normal')
    copy_processed = preprocess(copy_text, 'normal')

    if not orig_processed:
        return 0.0

    # 使用快速匹配算法
    matcher = difflib.SequenceMatcher(None, orig_processed, copy_processed)
    return matcher.ratio()


def compute_ngram_similarity(orig_text, copy_text, n=2):
    """
    简化的N-gram相似度计算
    """
    orig_processed = preprocess(orig_text, 'aggressive')
    copy_processed = preprocess(copy_text, 'aggressive')

    if len(orig_processed) < n or len(copy_processed) < n:
        return 0.0

    # 只采样部分ngram以提高性能
    def sample_ngrams(text, n, sample_size=100):
        step = max(1, len(text) // sample_size)
        return set(text[i:i + n] for i in range(0, len(text) - n + 1, step))

    orig_ngrams = sample_ngrams(orig_processed, n)
    copy_ngrams = sample_ngrams(copy_processed, n)

    if not orig_ngrams:
        return 0.0

    common_ngrams = orig_ngrams & copy_ngrams
    return len(common_ngrams) / len(orig_ngrams)


def compute_similarity_ratio_fast(orig_text, copy_text):
    """
    快速文本相似度计算算法
    """
    if not orig_text:
        return 0.0

    # 1. 基于LCS的相似度（主要指标）
    orig_aggressive = preprocess(orig_text, 'aggressive')
    copy_aggressive = preprocess(copy_text, 'aggressive')

    if len(orig_aggressive) == 0:
        return 0.0

    lcs_similarity = lcs_length(orig_aggressive, copy_aggressive) / len(orig_aggressive)

    # 2. 基于字符级别的相似度
    char_similarity = compute_text_similarity(orig_text, copy_text)

    # 3. 基于语义的相似度（简化版）
    semantic_similarity = compute_semantic_similarity(orig_text, copy_text)

    # 4. 基于N-gram的相似度
    ngram_similarity = compute_ngram_similarity(orig_text, copy_text, 2)

    # 使用固定权重，减少计算复杂度
    weights = [0.4, 0.3, 0.2, 0.1]

    # 计算最终相似度
    similarities = [
        lcs_similarity,
        char_similarity,
        semantic_similarity,
        ngram_similarity
    ]

    final_similarity = sum(s * w for s, w in zip(similarities, weights))

    # 确保结果在合理范围内
    final_similarity = max(0.0, min(1.0, final_similarity))

    return final_similarity * 100


def compute_repetition_rate(orig_text, copy_text):
    """
    计算文本重复率
    """
    return compute_similarity_ratio_fast(orig_text, copy_text)


def write_answer(path, rate):
    """
    将重复率结果写入答案文件
    """
    dirp = os.path.dirname(path)
    if dirp and not os.path.exists(dirp):
        os.makedirs(dirp)

    with open(path, 'w', encoding='utf-8') as f:
        f.write("{:.2f}".format(rate))


def main():
    """
    主函数
    """
    if len(sys.argv) != 4:
        print("Usage: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        print("示例: python main.py orig.txt copy.txt answer.txt")
        sys.exit(1)

    orig_path = sys.argv[1]
    copy_path = sys.argv[2]
    ans_path = sys.argv[3]

    if not os.path.exists(orig_path):
        print(f"错误：原始文件 {orig_path} 不存在")
        sys.exit(1)

    if not os.path.exists(copy_path):
        print(f"错误：待检测文件 {copy_path} 不存在")
        sys.exit(1)

    print("正在读取文件...")
    orig_text = try_read_file(orig_path)
    copy_text = try_read_file(copy_path)

    if not orig_text:
        print(f"错误：无法读取原始文件 {orig_path} 或文件为空")
        sys.exit(1)

    if not copy_text:
        print(f"错误：无法读取待检测文件 {copy_path} 或文件为空")
        sys.exit(1)

    print(f"原始文件字符数: {len(orig_text)}")
    print(f"待检测文件字符数: {len(copy_text)}")

    print("正在计算重复率...")
    rate = compute_repetition_rate(orig_text, copy_text)

    try:
        write_answer(ans_path, rate)
        print("=" * 50)
        print(f"重复率计算结果: {rate:.2f}%")
        print(f"结果已保存到: {ans_path}")
        print("=" * 50)
    except Exception as e:
        print(f"错误：写入答案文件失败: {e}")
        sys.exit(2)


if __name__ == "__main__":
    # 初始化jieba分词，禁用不必要的功能以提高性能
    try:
        jieba.initialize()
        # 禁用HMM以加快分词速度
        jieba.enable_parallel(4)  # 启用并行分词
    except:
        pass

    main()