import sys
import os
import unicodedata
import difflib
import re
from collections import Counter
import jieba  # 用于中文分词，提高语义理解


def try_read_file(path):
    """
    尝试使用多种编码读取文件，返回文本内容

    参数:
        path: 文件路径

    返回:
        str: 文件内容，如果读取失败返回空字符串
    """
    # 尝试的编码列表，按常见程度排序
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']

    # 按顺序尝试不同编码
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    # 如果所有编码都失败，尝试二进制读取并使用UTF-8解码（替换不可解码字符）
    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    except Exception:
        return ''  # 如果所有方法都失败，返回空字符串


def preprocess(text, mode='normal'):
    """
    改进的预处理函数，支持多种处理模式

    参数:
        text: 待处理的文本
        mode: 处理模式
            'normal' - 常规模式：规范化空白，保留标点
            'aggressive' - 激进模式：去除所有空白和标点
            'semantic' - 语义模式：分词并保留关键词

    返回:
        str: 处理后的文本
    """
    if not text:
        return ''

    if mode == 'aggressive':
        # 激进模式：去除所有空白和标点
        out_chars = []
        for ch in text:
            if ch.isspace():
                continue
            cat = unicodedata.category(ch)
            if cat.startswith('P'):
                continue
            out_chars.append(ch)
        return ''.join(out_chars)

    elif mode == 'semantic':
        # 语义模式：分词并过滤停用词
        # 先进行常规预处理
        text = re.sub(r'\s+', ' ', text).strip()
        # 使用jieba分词
        words = jieba.cut(text)
        # 过滤停用词和短词（长度小于2的词）
        filtered_words = [word for word in words if len(word) >= 2]
        return ' '.join(filtered_words)

    else:
        # 常规模式：只规范化空白，保留标点
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


def lcs_length(a, b):
    """
    计算两个字符串的最长公共子序列（LCS）长度
    """
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a

    m = len(short)
    n = len(long)

    if m == 0 or n == 0:
        return 0

    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        li = long[i - 1]
        for j in range(1, m + 1):
            if li == short[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev

    return prev[m]


def compute_semantic_similarity(orig_text, copy_text):
    """
    计算语义相似度，使用词频和关键信息匹配
    """
    # 语义预处理
    orig_semantic = preprocess(orig_text, 'semantic')
    copy_semantic = preprocess(copy_text, 'semantic')

    if not orig_semantic:
        return 0.0

    # 计算词频相似度
    orig_words = orig_semantic.split()
    copy_words = copy_semantic.split()

    orig_counter = Counter(orig_words)
    copy_counter = Counter(copy_words)

    # 计算余弦相似度
    common_words = set(orig_words) & set(copy_words)
    dot_product = sum(orig_counter[word] * copy_counter[word] for word in common_words)

    magnitude_orig = sum(count ** 2 for count in orig_counter.values()) ** 0.5
    magnitude_copy = sum(count ** 2 for count in copy_counter.values()) ** 0.5

    if magnitude_orig * magnitude_copy == 0:
        return 0.0

    return dot_product / (magnitude_orig * magnitude_copy)


def compute_structural_similarity(orig_text, copy_text):
    """
    计算结构相似度，关注段落和句子结构
    """
    # 分割段落
    orig_paragraphs = [p.strip() for p in orig_text.split('\n') if p.strip()]
    copy_paragraphs = [p.strip() for p in copy_text.split('\n') if p.strip()]

    if not orig_paragraphs:
        return 0.0

    # 计算段落级别的相似度
    paragraph_similarity = 0.0
    matched_paragraphs = 0

    for orig_para in orig_paragraphs:
        best_similarity = 0.0
        for copy_para in copy_paragraphs:
            # 使用模糊匹配计算段落相似度
            para_sim = compute_text_similarity(orig_para, copy_para, use_semantic=True)
            best_similarity = max(best_similarity, para_sim)

        if best_similarity > 0.6:  # 阈值可调整
            matched_paragraphs += 1

    paragraph_similarity = matched_paragraphs / len(orig_paragraphs)

    return paragraph_similarity


def compute_text_similarity(orig_text, copy_text, use_semantic=False):
    """
    计算两个文本的相似度（基础函数）
    """
    if not orig_text:
        return 0.0

    if use_semantic:
        # 语义模式下的相似度计算
        orig_processed = preprocess(orig_text, 'semantic')
        copy_processed = preprocess(copy_text, 'semantic')
    else:
        # 字符模式下的相似度计算
        orig_processed = preprocess(orig_text, 'normal')
        copy_processed = preprocess(copy_text, 'normal')

    matcher = difflib.SequenceMatcher(None, orig_processed, copy_processed)
    return matcher.ratio()


def compute_similarity_ratio_v3(orig_text, copy_text):
    """
    改进的文本相似度计算算法v3，针对字符错乱、内容增删、语序调整优化
    """
    if not orig_text:
        return 0.0

    # 1. 基于LCS的相似度（激进预处理）
    orig_aggressive = preprocess(orig_text, 'aggressive')
    copy_aggressive = preprocess(copy_text, 'aggressive')

    if len(orig_aggressive) == 0:
        return 0.0

    lcs_similarity = lcs_length(orig_aggressive, copy_aggressive) / len(orig_aggressive)

    # 2. 基于字符级别的相似度（常规预处理）
    char_similarity = compute_text_similarity(orig_text, copy_text, use_semantic=False)

    # 3. 基于语义的相似度（新增）
    semantic_similarity = compute_semantic_similarity(orig_text, copy_text)

    # 4. 基于结构相似度（新增）
    structural_similarity = compute_structural_similarity(orig_text, copy_text)

    # 5. 基于词级别的相似度
    orig_conservative = preprocess(orig_text, 'normal')
    copy_conservative = preprocess(copy_text, 'normal')
    orig_words = orig_conservative.split()
    copy_words = copy_conservative.split()

    if orig_words:
        orig_set = set(orig_words)
        copy_set = set(copy_words)
        common_words = orig_set & copy_set
        word_similarity = len(common_words) / len(orig_set)
    else:
        word_similarity = 0.0

    # 动态权重调整：根据文本特征选择最佳权重组合
    text_length = len(orig_aggressive)

    # 对于长文本，更注重语义和结构相似度
    if text_length > 1000:
        weights = [0.25, 0.20, 0.25, 0.15, 0.15]  # LCS, 字符, 语义, 结构, 词汇
    # 对于短文本，更注重字符和词汇相似度
    else:
        weights = [0.30, 0.25, 0.20, 0.10, 0.15]

    final_similarity = (lcs_similarity * weights[0] +
                        char_similarity * weights[1] +
                        semantic_similarity * weights[2] +
                        structural_similarity * weights[3] +
                        word_similarity * weights[4])

    return final_similarity * 100


def compute_repetition_rate(orig_text, copy_text):
    """
    计算文本重复率（使用改进的算法v3）
    """
    return compute_similarity_ratio_v3(orig_text, copy_text)


def write_answer(path, rate):
    """
    将重复率结果写入答案文件
    """
    dirp = os.path.dirname(path)
    if dirp and not os.path.exists(dirp):
        os.makedirs(dirp)

    with open(path, 'w', encoding='utf-8') as f:
        f.write("{:.2f}".format(rate))


def batch_test():
    """
    批量测试函数
    """
    orig_file = "orig.txt"
    test_files = [
        "orig_0.8_add.txt",
        "orig_0.8_del.txt",
        "orig_0.8_dis_1.txt",
        "orig_0.8_dis_10.txt",
        "orig_0.8_dis_15.txt"
    ]

    orig_text = try_read_file(orig_file)
    if not orig_text:
        print(f"错误：无法读取原始文件 {orig_file}")
        return

    result_dir = "results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    results = {}

    print("开始批量测试...")
    print("-" * 60)

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"警告：文件 {test_file} 不存在，跳过")
            continue

        test_text = try_read_file(test_file)
        if not test_text:
            print(f"警告：无法读取文件 {test_file}，跳过")
            continue

        rate = compute_repetition_rate(orig_text, test_text)
        results[test_file] = rate

        answer_file = os.path.join(result_dir, f"answer_{test_file}")
        write_answer(answer_file, rate)
        print(f"{test_file}: {rate:.2f}%")

    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("重复率检测报告（改进算法v3）\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始文件: {orig_file}\n\n")
        f.write("测试结果：\n")
        f.write("-" * 40 + "\n")

        for file_name, rate in results.items():
            f.write(f"{file_name}: {rate:.2f}%\n")

        f.write("\n统计信息：\n")
        f.write("-" * 40 + "\n")
        if results:
            avg_rate = sum(results.values()) / len(results)
            f.write(f"平均重复率: {avg_rate:.2f}%\n")
            f.write(f"最高重复率: {max(results.values()):.2f}%\n")
            f.write(f"最低重复率: {min(results.values()):.2f}%\n")

    print("-" * 60)
    print(f"\n测试完成！结果已保存到 {result_dir} 目录")
    print(f"详细报告见: {report_file}")


def main():
    """
    主函数
    """
    if len(sys.argv) == 1:
        batch_test()
    elif len(sys.argv) == 4:
        orig_path = sys.argv[1]
        copy_path = sys.argv[2]
        ans_path = sys.argv[3]

        orig_text = try_read_file(orig_path)
        copy_text = try_read_file(copy_path)

        if not orig_text:
            print(f"错误：无法读取原始文件 {orig_path}")
            sys.exit(1)
        if not copy_text:
            print(f"错误：无法读取待检测文件 {copy_path}")
            sys.exit(1)

        rate = compute_repetition_rate(orig_text, copy_text)

        try:
            write_answer(ans_path, rate)
            print(f"重复率: {rate:.2f}%")
        except Exception as e:
            print("Error writing answer file:", e)
            sys.exit(2)
    else:
        print("Usage:")
        print("  单个文件测试: python main.py [orig_file] [copy_file] [answer_file]")
        print("  批量测试: python main.py")
        sys.exit(1)


if __name__ == "__main__":
    # 初始化jieba分词
    try:
        jieba.initialize()
    except:
        pass  # 如果初始化失败，继续运行

    main()