import sys
import os
import unicodedata
import difflib
import re


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


def preprocess(text, aggressive=False):
    """
    预处理文本，根据模式选择不同的处理方式

    参数:
        text: 待处理的文本
        aggressive: 处理模式标志
            True - 激进模式：去除所有空白和标点
            False - 保守模式：只规范化空白，保留标点

    返回:
        str: 处理后的文本
    """
    if not text:
        return ''

    if aggressive:
        # 激进模式：去除所有空白和标点（主要用于LCS算法）
        out_chars = []
        for ch in text:
            # 跳过所有空白字符（空格、制表符、换行等）
            if ch.isspace():
                continue
            # 获取字符的Unicode分类
            cat = unicodedata.category(ch)
            # 跳过所有标点符号（分类以'P'开头的字符）
            if cat.startswith('P'):
                continue
            out_chars.append(ch)
        return ''.join(out_chars)
    else:
        # 保守模式：只规范化空白，保留标点（用于精确比较）
        # 将多个连续空白字符替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除字符串首尾的空白字符
        return text.strip()


def lcs_length(a, b):
    """
    计算两个字符串的最长公共子序列（LCS）长度

    参数:
        a: 第一个字符串
        b: 第二个字符串

    返回:
        int: 最长公共子序列的长度

    算法说明:
        使用动态规划的空间优化版本，将空间复杂度从O(m*n)降低到O(min(m,n))
    """
    # 选择较短的字符串作为内循环，减少内存使用
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a

    m = len(short)  # 较短字符串的长度
    n = len(long)  # 较长字符串的长度

    # 处理空字符串的情况
    if m == 0 or n == 0:
        return 0

    # 初始化动态规划表的两行（空间优化）
    prev = [0] * (m + 1)  # 上一行的状态
    curr = [0] * (m + 1)  # 当前行的状态

    # 动态规划过程
    for i in range(1, n + 1):  # 遍历长字符串的每个字符
        li = long[i - 1]  # 当前处理的字符
        for j in range(1, m + 1):  # 遍历短字符串的每个字符
            if li == short[j - 1]:
                # 字符匹配，长度加1
                curr[j] = prev[j - 1] + 1
            else:
                # 字符不匹配，取左边或上边的最大值
                curr[j] = max(curr[j - 1], prev[j])
        # 交换行，准备处理下一个字符
        prev, curr = curr, prev

    # 返回最长公共子序列的长度
    return prev[m]


def compute_similarity_ratio(orig_text, copy_text):
    """
    计算两个文本的相似度比率，使用多种方法综合评估

    参数:
        orig_text: 原始文本
        copy_text: 待比较的文本

    返回:
        float: 相似度百分比（0-100）

    算法说明:
        使用三种不同的相似度计算方法，并加权综合：
        1. LCS相似度（激进预处理） - 权重30%
        2. 字符级别相似度（保守预处理 + difflib） - 权重50%
        3. 词级别相似度 - 权重20%
    """
    if not orig_text:
        return 0.0

    # 方法1：基于LCS的相似度（使用激进预处理）
    orig_aggressive = preprocess(orig_text, aggressive=True)
    copy_aggressive = preprocess(copy_text, aggressive=True)

    # 处理原始文本为空的情况
    if len(orig_aggressive) == 0:
        return 0.0

    # 计算LCS相似度（最长公共子序列长度 / 原始文本长度）
    lcs_similarity = lcs_length(orig_aggressive, copy_aggressive) / len(orig_aggressive)

    # 方法2：基于字符级别的相似度（使用保守预处理）
    orig_conservative = preprocess(orig_text, aggressive=False)
    copy_conservative = preprocess(copy_text, aggressive=False)

    # 使用difflib的SequenceMatcher计算字符级别相似度
    matcher = difflib.SequenceMatcher(None, orig_conservative, copy_conservative)
    difflib_similarity = matcher.ratio()

    # 方法3：基于词级别的相似度
    orig_words = orig_conservative.split()  # 将原始文本分割为单词列表
    copy_words = copy_conservative.split()  # 将比较文本分割为单词列表

    if orig_words:
        # 计算共同词汇的比例
        common_words = set(orig_words) & set(copy_words)  # 求词汇交集
        word_similarity = len(common_words) / len(orig_words)
    else:
        word_similarity = 0.0

    # 综合三种方法的权重（加权平均）
    # difflib相似度权重最高（50%），因为它最能反映实际修改程度
    # LCS相似度权重30%，词级别相似度权重20%
    final_similarity = (difflib_similarity * 0.5 +
                        lcs_similarity * 0.3 +
                        word_similarity * 0.2)

    # 转换为百分比形式
    return final_similarity * 100


def compute_repetition_rate(orig_text, copy_text):
    """
    计算文本重复率（对外接口）

    参数:
        orig_text: 原始文本
        copy_text: 待检测文本

    返回:
        float: 重复率百分比
    """
    return compute_similarity_ratio(orig_text, copy_text)


def write_answer(path, rate):
    """
    将重复率结果写入答案文件

    参数:
        path: 答案文件路径
        rate: 重复率数值
    """
    # 确保输出目录存在
    dirp = os.path.dirname(path)
    if dirp and not os.path.exists(dirp):
        os.makedirs(dirp)

    # 写入结果，保留两位小数
    with open(path, 'w', encoding='utf-8') as f:
        f.write("{:.2f}".format(rate))


def batch_test():
    """
    批量测试函数：对指定文件列表进行重复率检测

    功能:
        1. 读取原始文件
        2. 对每个测试文件计算重复率
        3. 生成答案文件和汇总报告
    """
    # 原始文件名
    orig_file = "orig.txt"
    # 测试文件列表
    test_files = [
        "orig_0.8_add.txt",  # 添加干扰字符的版本
        "orig_0.8_del.txt",  # 删除部分内容的版本
        "orig_0.8_dis_1.txt",  # 不同程度干扰的版本
        "orig_0.8_dis_10.txt",
        "orig_0.8_dis_15.txt"
    ]

    # 读取原始文件内容
    orig_text = try_read_file(orig_file)
    if not orig_text:
        print(f"错误：无法读取原始文件 {orig_file}")
        return

    # 创建结果目录
    result_dir = "results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 存储测试结果
    results = {}

    # 对每个测试文件进行处理
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"警告：文件 {test_file} 不存在，跳过")
            continue

        # 读取测试文件内容并计算重复率
        test_text = try_read_file(test_file)
        rate = compute_repetition_rate(orig_text, test_text)
        results[test_file] = rate

        # 生成答案文件
        answer_file = os.path.join(result_dir, f"answer_{test_file}")
        write_answer(answer_file, rate)
        print(f"{test_file}: {rate:.2f}%")

    # 生成汇总报告
    report_file = os.path.join(result_dir, "report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("重复率检测报告（改进算法）\n")
        f.write("=" * 50 + "\n")
        f.write(f"原始文件: {orig_file}\n\n")

        # 写入每个文件的测试结果
        for file_name, rate in results.items():
            f.write(f"{file_name}: {rate:.2f}%\n")

    print(f"\n测试完成！结果已保存到 {result_dir} 目录")


def main():
    """
    主函数：程序入口点

    支持两种运行模式：
    1. 批量测试模式（无参数）：测试预设的文件列表
    2. 单文件测试模式（3个参数）：测试指定的文件对
    """
    if len(sys.argv) == 1:
        # 无参数：执行批量测试
        batch_test()
    elif len(sys.argv) == 4:
        # 3个参数：执行单文件测试
        orig_path = sys.argv[1]  # 原始文件路径
        copy_path = sys.argv[2]  # 待检测文件路径
        ans_path = sys.argv[3]  # 答案文件路径

        # 读取文件内容
        orig_text = try_read_file(orig_path)
        copy_text = try_read_file(copy_path)

        # 计算重复率
        rate = compute_repetition_rate(orig_text, copy_text)

        try:
            # 写入结果文件
            write_answer(ans_path, rate)
            print(f"重复率: {rate:.2f}%")
        except Exception as e:
            # 处理写入错误
            print("Error writing answer file:", e)
            sys.exit(2)
    else:
        # 参数数量错误，显示用法说明
        print("Usage:")
        print("  单个文件测试: python main.py [orig_file] [copy_file] [answer_file]")
        print("  批量测试: python main.py")
        sys.exit(1)


if __name__ == "__main__":
    # 程序入口
    main()