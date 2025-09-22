"""
main.py - 文本查重算法实现

功能：
1. 读取文件（支持多种编码）
2. 文本预处理（normal/aggressive/semantic）
3. 多种相似度算法：
   - LCS 最长公共子序列
   - 余弦相似度
   - 基于关键词的相似度
   - n-gram 相似度
4. 计算综合重复率并输出到文件
"""

import sys
import os
import difflib
import re
import math
from collections import Counter
import jieba  # pylint: disable=import-error


class TextProcessor:
    """文本处理器，用于统一的文本预处理"""

    def __init__(self):
        # 缓存预处理结果，避免重复计算
        self._cache = {}

    def preprocess(self, text, mode='normal'):
        """文本预处理，支持不同处理模式"""
        if not text:
            return ''

        # 使用文本内容和模式作为缓存键
        cache_key = (hash(text), mode)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 统一空白字符处理
        text = re.sub(r'\s+', ' ', text.strip())

        if mode == 'aggressive':
            # 激进模式：只保留中文字符和数字
            result = re.sub(r'[^\u4e00-\u9fa50-9]', '', text)
        elif mode == 'semantic':
            # 语义模式：使用分词处理
            if len(text) > 1000:
                # 长文本处理：采样关键部分以提高效率
                sample_text = text[:500] + text[-500:] if len(text) > 1500 else text
                words = jieba.cut(sample_text)
            else:
                words = jieba.cut(text)
            # 过滤短词，保留有意义的词汇
            result = ' '.join(word for word in words if len(word) >= 2)
        else:
            # 普通模式：仅进行基础清洗
            result = text

        # 维护缓存大小，防止内存过度使用
        if len(self._cache) > 100:
            self._cache.clear()

        self._cache[cache_key] = result
        return result


# 创建全局文本处理器实例
text_processor = TextProcessor()


def try_read_file(path):
    """
    尝试使用多种编码读取文件，返回文本内容。
    支持utf-8, gbk, gb2312, utf-8-sig等常见编码格式。
    """
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']

    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as file_handler:
                return file_handler.read()
        except (UnicodeDecodeError, OSError):
            continue

    # 如果标准编码都失败，尝试二进制读取并解码
    try:
        with open(path, 'rb') as file_handler:
            return file_handler.read().decode('utf-8', errors='replace')
    except (UnicodeDecodeError, OSError):
        return ''


def lcs_length(text_a, text_b):
    """计算两个文本的最长公共子序列长度"""
    # 确保较短的文本作为比较基准
    if len(text_a) < len(text_b):
        short, long_text = text_a, text_b
    else:
        short, long_text = text_b, text_a

    short_len, long_len = len(short), len(long_text)

    # 处理空文本情况
    if short_len == 0 or long_len == 0:
        return 0

    # 对于超长文本，采用采样策略平衡精度和性能
    if short_len > 2000:
        # 采样文本的关键部分：开头、中间、结尾
        sample_short = short[:500] + short[short_len//2-250:short_len//2+250] + short[-500:]
        sample_long = long_text[:500] + long_text[long_len//2-250:long_len//2+250] + long_text[-500:]
        # 递归计算采样文本的LCS，并按比例缩放
        return lcs_length(sample_short, sample_long) * (short_len / len(sample_short))

    # 使用动态规划计算LCS
    prev = [0] * (short_len + 1)  # 前一行状态
    curr = [0] * (short_len + 1)  # 当前行状态

    # 遍历长文本的每个字符
    for i in range(1, long_len + 1):
        long_char = long_text[i - 1]
        # 遍历短文本的每个字符
        for j in range(1, short_len + 1):
            if long_char == short[j - 1]:
                # 字符匹配，LCS长度加1
                curr[j] = prev[j - 1] + 1
            else:
                # 字符不匹配，取上方或左方的最大值
                curr[j] = max(curr[j - 1], prev[j])

        # 更新状态行
        prev, curr = curr, prev

        # 重置当前行状态
        if i < long_len:
            curr = [0] * (short_len + 1)

    return prev[short_len]


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度。
    用于衡量两个文本在词汇分布上的相似性。
    """
    # 找出两个向量的共同词汇
    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0.0

    # 计算点积
    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    # 计算向量模长
    magnitude1 = math.sqrt(sum(v * v for v in vec1.values()))
    magnitude2 = math.sqrt(sum(v * v for v in vec2.values()))

    # 避免除零错误
    if magnitude1 * magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def compute_semantic_similarity(orig_text, copy_text):
    """
    计算语义相似度，基于分词后的词汇分布。
    """
    if not orig_text or not copy_text:
        return 0.0

    # 使用语义模式预处理文本
    orig_processed = text_processor.preprocess(orig_text, 'semantic')
    copy_processed = text_processor.preprocess(copy_text, 'semantic')

    if not orig_processed or not copy_processed:
        return 0.0

    # 对长文本进行采样以提升计算效率
    if len(orig_processed) > 1000:
        orig_words = orig_processed.split()
        copy_words = copy_processed.split()
        # 取文本开头和结尾部分的关键词汇
        orig_counter = Counter(orig_words[:500] + orig_words[-500:])
        copy_counter = Counter(copy_words[:500] + copy_words[-500:])
    else:
        # 短文本使用完整计算
        orig_counter = Counter(orig_processed.split())
        copy_counter = Counter(copy_processed.split())

    return cosine_similarity(orig_counter, copy_counter)


def compute_text_similarity(orig_text, copy_text):
    """
    计算基础文本相似度，基于字符级别的序列匹配。
    """
    if not orig_text:
        return 0.0

    # 普通模式预处理，保留标点等字符信息
    orig_processed = text_processor.preprocess(orig_text, 'normal')
    copy_processed = text_processor.preprocess(copy_text, 'normal')

    if not orig_processed:
        return 0.0

    # 对超长文本使用采样策略
    if len(orig_processed) > 5000:
        sample_orig = orig_processed[:2000] + orig_processed[-2000:]
        sample_copy = copy_processed[:2000] + copy_processed[-2000:]
        matcher = difflib.SequenceMatcher(None, sample_orig, sample_copy)
        return matcher.ratio()

    # 使用Python内置的序列匹配器
    matcher = difflib.SequenceMatcher(None, orig_processed, copy_processed)
    return matcher.ratio()


def compute_ngram_similarity(orig_text, copy_text, n_value=2):
    """
    基于N-gram的相似度计算，捕捉局部文本模式。
    n_value表示gram的大小，默认为2（bigram）。
    """
    # 激进模式预处理，去除所有非中文字符和数字
    orig_processed = text_processor.preprocess(orig_text, 'aggressive')
    copy_processed = text_processor.preprocess(copy_text, 'aggressive')

    # 检查文本长度是否支持N-gram分析
    if len(orig_processed) < n_value or len(copy_processed) < n_value:
        return 0.0

    def sample_ngrams(text, n, sample_size=50):
        """从文本中采样生成N-gram集合"""
        # 短文本生成所有可能的N-gram
        if len(text) <= sample_size * n:
            return {text[i:i + n] for i in range(0, len(text) - n + 1)}

        # 长文本按步长采样生成N-gram
        step = max(1, len(text) // sample_size)
        return {text[i:i + n] for i in range(0, len(text) - n + 1, step)}

    # 生成原文和抄袭文的N-gram集合
    orig_ngrams = sample_ngrams(orig_processed, n_value)
    copy_ngrams = sample_ngrams(copy_processed, n_value)

    if not orig_ngrams:
        return 0.0

    # 计算Jaccard相似度：交集大小除以并集大小
    intersection_size = len(orig_ngrams & copy_ngrams)
    return intersection_size / len(orig_ngrams)


def compute_similarity_ratio(orig_text, copy_text):
    """
    综合计算文本相似度，结合多种算法结果。
    返回0-100之间的相似度百分比。
    """
    if not orig_text:
        return 0.0

    # 获取激进预处理后的文本，用于LCS计算
    orig_aggressive = text_processor.preprocess(orig_text, 'aggressive')

    if not orig_aggressive:
        return 0.0

    copy_aggressive = text_processor.preprocess(copy_text, 'aggressive')

    # 计算四种不同的相似度指标
    lcs_similarity = lcs_length(orig_aggressive, copy_aggressive) / len(orig_aggressive)
    char_similarity = compute_text_similarity(orig_text, copy_text)
    semantic_similarity = compute_semantic_similarity(orig_text, copy_text)
    ngram_similarity = compute_ngram_similarity(orig_text, copy_text, 2)

    # 设置各算法的权重系数
    weights = [0.4, 0.3, 0.2, 0.1]
    similarities = [
        lcs_similarity,      # LCS相似度，权重40%
        char_similarity,     # 字符相似度，权重30%
        semantic_similarity, # 语义相似度，权重20%
        ngram_similarity,    # N-gram相似度，权重10%
    ]

    # 加权平均计算最终相似度
    final_similarity = sum(s * w for s, w in zip(similarities, weights))

    # 确保结果在0-1范围内，然后转换为百分比
    final_similarity = max(0.0, min(1.0, final_similarity))
    return final_similarity * 100


def compute_repetition_rate(orig_text, copy_text):
    """
    计算文本重复率，主入口函数。
    """
    return compute_similarity_ratio(orig_text, copy_text)


def write_answer(path, rate):
    """
    将重复率结果写入答案文件。
    自动创建不存在的目录路径。
    """
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(path, 'w', encoding='utf-8') as file_handler:
        # 格式化输出，保留两位小数
        file_handler.write(f"{rate:.2f}")


def main():
    """
    主函数：命令行运行入口。
    使用方式：python main.py [原文文件] [抄袭版论文的文件] [答案文件]
    """
    if len(sys.argv) != 4:
        print("Usage: python main.py [原文文件] [抄袭版论文的文件] [答案文件]")
        sys.exit(1)

    orig_path, copy_path, ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 检查输入文件是否存在
    if not os.path.exists(orig_path):
        print(f"错误：原始文件 {orig_path} 不存在")
        sys.exit(1)

    if not os.path.exists(copy_path):
        print(f"错误：待检测文件 {copy_path} 不存在")
        sys.exit(1)

    # 读取文件内容
    orig_text = try_read_file(orig_path)
    copy_text = try_read_file(copy_path)

    # 检查文件是否读取成功
    if not orig_text:
        print(f"错误：无法读取原始文件 {orig_path} 或文件为空")
        sys.exit(1)

    if not copy_text:
        print(f"错误：无法读取待检测文件 {copy_path} 或文件为空")
        sys.exit(1)

    print("正在计算重复率...")
    # 计算重复率
    rate = compute_repetition_rate(orig_text, copy_text)

    try:
        # 写入结果文件
        write_answer(ans_path, rate)
        print(f"重复率计算结果: {rate:.2f}% (已保存到 {ans_path})")
    except OSError as error:
        print(f"错误：写入答案文件失败: {error}")
        sys.exit(2)


if __name__ == "__main__":
    # 初始化jieba分词器
    try:
        jieba.initialize()
        # 启用并行分词，提升分词速度
        jieba.enable_parallel(4)
    except (OSError, RuntimeError):
        # 如果并行分词失败，回退到单线程模式
        pass

    main()