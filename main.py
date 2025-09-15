# main.py
import sys
import os
import unicodedata

def try_read_file(path):
    """尝试使用若干编码读取文件，返回文本（不抛出异常，出错时返回空串）。"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # 最后尝试二进制读取并解码替换不可解码字符
    try:
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    except Exception:
        return ''

def preprocess(text):
    """
    简单预处理：
    - 去掉所有空白字符（空格、制表符、换行等）
    - 去掉 Unicode 分类为标点的字符（减少标点差异带来的影响）
    该函数返回处理后的字符串（字符序列级别比较）
    """
    if not text:
        return ''
    out_chars = []
    for ch in text:
        # 跳过空白
        if ch.isspace():
            continue
        # 跳过标点（Unicode 分类以 'P' 开头的被认为是标点）
        cat = unicodedata.category(ch)
        if cat.startswith('P'):
            continue
        out_chars.append(ch)
    return ''.join(out_chars)

def lcs_length(a, b):
    """
    计算字符串 a, b 的最长公共子序列长度（LCS length）。
    使用空间优化的动态规划，只保存两行（O(min(n,m)) 内存）。
    """
    # 将较短的作为列以减少内存
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

    # iterate over long's characters
    for i in range(1, n + 1):
        li = long[i - 1]
        # iterate over short's characters
        for j in range(1, m + 1):
            if li == short[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                # max of left or top in full DP table
                if curr[j - 1] >= prev[j]:
                    curr[j] = curr[j - 1]
                else:
                    curr[j] = prev[j]
        # swap rows
        prev, curr = curr, prev
    # prev 是最后填写完成的一行（因为我们在结尾做了swap）
    return prev[m]

def compute_repetition_rate(orig_text, copy_text):
    a = preprocess(orig_text)
    b = preprocess(copy_text)
    len_a = len(a)
    # 处理 edge cases
    if len_a == 0:
        # 若原文也为空，则视为完全重复；否则重复率为0
        if len(b) == 0:
            return 100.0
        else:
            return 0.0
    lcs_len = lcs_length(a, b)
    rate = (lcs_len / len_a) * 100.0
    return rate

def write_answer(path, rate):
    # 保证目录存在？
    dirp = os.path.dirname(path)
    if dirp and not os.path.exists(dirp):
        # 不创建目录以避免越界写入；但如果需要可以开启
        pass
    # 写入浮点数，保留两位小数
    with open(path, 'w', encoding='utf-8') as f:
        f.write("{:.2f}".format(rate))

def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py [orig_file] [copy_file] [answer_file]")
        sys.exit(1)
    orig_path = sys.argv[1]
    copy_path = sys.argv[2]
    ans_path = sys.argv[3]

    orig_text = try_read_file(orig_path)
    copy_text = try_read_file(copy_path)

    rate = compute_repetition_rate(orig_text, copy_text)

    try:
        write_answer(ans_path, rate)
    except Exception as e:
        # 若写文件失败，打印错误并退出非0
        print("Error writing answer file:", e)
        sys.exit(2)

if __name__ == "__main__":
    main()
