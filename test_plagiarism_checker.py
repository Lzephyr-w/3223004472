"""
单元测试 - 文本查重算法

测试策略：
1. 白盒测试：基于代码逻辑设计测试用例
2. 边界值测试：测试各种边界情况
3. 异常测试：测试错误处理能力
4. 性能测试：测试长文本处理能力
"""

import unittest
import os
import tempfile
import sys
from unittest.mock import patch, mock_open

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    TextProcessor, try_read_file, lcs_length, cosine_similarity,
    compute_semantic_similarity, compute_text_similarity,
    compute_ngram_similarity, compute_similarity_ratio,
    compute_repetition_rate, write_answer
)


class TestTextProcessor(unittest.TestCase):
    """测试文本处理器类"""

    def setUp(self):
        self.processor = TextProcessor()

    def test_preprocess_normal_mode(self):
        """测试普通模式预处理"""
        text = "   Hello    World!  \n\nTest   "
        result = self.processor.preprocess(text, 'normal')
        self.assertEqual(result, "Hello World! Test")

    def test_preprocess_aggressive_mode(self):
        """测试激进模式预处理"""
        text = "Hello 世界123! Test"
        result = self.processor.preprocess(text, 'aggressive')
        self.assertEqual(result, "世界123")

    def test_preprocess_semantic_mode(self):
        """测试语义模式预处理"""
        text = "这是一个测试句子"
        result = self.processor.preprocess(text, 'semantic')
        # 检查是否进行了分词
        self.assertIn(" ", result)

    def test_preprocess_empty_text(self):
        """测试空文本预处理"""
        result = self.processor.preprocess("", 'normal')
        self.assertEqual(result, "")

    def test_preprocess_cache(self):
        """测试预处理缓存功能"""
        text = "测试缓存功能"
        result1 = self.processor.preprocess(text, 'normal')
        result2 = self.processor.preprocess(text, 'normal')
        self.assertEqual(result1, result2)
        # 检查是否使用了缓存
        self.assertEqual(id(result1), id(result2))


class TestFileOperations(unittest.TestCase):
    """测试文件操作功能"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_try_read_file_success(self):
        """测试成功读取文件"""
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("测试内容")

        content = try_read_file(test_file)
        self.assertEqual(content, "测试内容")

    def test_try_read_file_nonexistent(self):
        """测试读取不存在的文件"""
        content = try_read_file("nonexistent_file.txt")
        self.assertEqual(content, "")

    def test_write_answer_success(self):
        """测试成功写入答案文件"""
        answer_file = os.path.join(self.test_dir, "results", "answer.txt")
        write_answer(answer_file, 85.67)

        self.assertTrue(os.path.exists(answer_file))
        with open(answer_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertEqual(content, "85.67")


class TestLCSCalculation(unittest.TestCase):
    """测试LCS算法"""

    def test_lcs_identical_texts(self):
        """测试完全相同文本的LCS"""
        text1 = "abcdefg"
        text2 = "abcdefg"
        result = lcs_length(text1, text2)
        self.assertEqual(result, 7)

    def test_lcs_partial_match(self):
        """测试部分匹配文本的LCS"""
        text1 = "abcdefg"
        text2 = "acdeg"
        result = lcs_length(text1, text2)
        self.assertEqual(result, 5)

    def test_lcs_no_match(self):
        """测试无匹配文本的LCS"""
        text1 = "abc"
        text2 = "def"
        result = lcs_length(text1, text2)
        self.assertEqual(result, 0)

    def test_lcs_empty_text(self):
        """测试空文本的LCS"""
        result = lcs_length("", "test")
        self.assertEqual(result, 0)
        result = lcs_length("test", "")
        self.assertEqual(result, 0)
        result = lcs_length("", "")
        self.assertEqual(result, 0)

    def test_lcs_long_text(self):
        """测试长文本的LCS（触发采样策略）"""
        text1 = "a" * 3000 + "b" * 3000
        text2 = "a" * 3000 + "c" * 3000
        result = lcs_length(text1, text2)
        # 应该返回合理的结果，不抛出异常
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 6000)


class TestCosineSimilarity(unittest.TestCase):
    """测试余弦相似度计算"""

    def test_cosine_identical_vectors(self):
        """测试相同向量的余弦相似度"""
        vec1 = {'a': 1, 'b': 2, 'c': 3}
        vec2 = {'a': 1, 'b': 2, 'c': 3}
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_cosine_orthogonal_vectors(self):
        """测试正交向量的余弦相似度"""
        vec1 = {'a': 1, 'b': 0}
        vec2 = {'c': 1, 'd': 1}
        result = cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_cosine_empty_vectors(self):
        """测试空向量的余弦相似度"""
        result = cosine_similarity({}, {})
        self.assertEqual(result, 0.0)


class TestSimilarityAlgorithms(unittest.TestCase):
    """测试各种相似度算法"""

    def test_semantic_similarity_identical(self):
        """测试相同文本的语义相似度"""
        text = "今天天气很好，我们一起去公园玩"
        result = compute_semantic_similarity(text, text)
        self.assertAlmostEqual(result, 1.0, places=1)

    def test_text_similarity_identical(self):
        """测试相同文本的字符相似度"""
        text = "相同的文本内容"
        result = compute_text_similarity(text, text)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_ngram_similarity_identical(self):
        """测试相同文本的N-gram相似度"""
        text = "测试文本"
        result = compute_ngram_similarity(text, text)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_similarity_ratio_boundary_cases(self):
        """测试边界情况的相似度计算"""
        # 空原文
        result = compute_similarity_ratio("", "测试文本")
        self.assertEqual(result, 0.0)

        # 完全相同文本
        text = "这是一个测试"
        result = compute_similarity_ratio(text, text)
        self.assertGreaterEqual(result, 90.0)  # 应该接近100%


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_complete_workflow(self):
        """测试完整工作流程"""
        orig_text = "这是一个原创的文本内容，用于测试查重系统的准确性。"
        copy_text = "这是一个抄袭的文本内容，用于测试查重系统的检测能力。"

        result = compute_repetition_rate(orig_text, copy_text)

        # 结果应该在合理范围内
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)
        self.assertIsInstance(result, float)

    def test_performance_large_text(self):
        """测试大文本性能"""
        # 生成大文本
        base_text = "这是一个基准文本。" * 1000  # 约10KB文本

        # 修改部分内容
        copy_text = base_text.replace("基准", "测试")[:len(base_text) // 2] + base_text[len(base_text) // 2:]

        # 测试是否能正常处理（不超时）
        import time
        start_time = time.time()
        result = compute_repetition_rate(base_text, copy_text)
        end_time = time.time()

        # 检查结果和性能
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)
        # 性能检查：处理10KB文本应该在合理时间内完成
        self.assertLess(end_time - start_time, 10.0)  # 10秒内完成


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""

    def test_invalid_file_encoding(self):
        """测试无效编码文件处理"""
        # 创建二进制文件（非文本文件）
        test_file = os.path.join(tempfile.gettempdir(), "binary_file.bin")
        with open(test_file, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00')

        content = try_read_file(test_file)
        # 应该能够处理而不崩溃
        self.assertIsInstance(content, str)

    @patch('builtins.open', side_effect=PermissionError("权限不足"))
    def test_file_permission_error(self, mock_file):
        """测试文件权限错误处理"""
        content = try_read_file("test.txt")
        self.assertEqual(content, "")


# 性能测试类
class TestPerformance(unittest.TestCase):
    """性能测试"""

    def test_large_text_processing(self):
        """测试大文本处理性能"""
        large_text = "测试文本" * 10000  # 生成大文本

        # 测试预处理性能
        processor = TextProcessor()
        start_time = os.times().elapsed
        processed = processor.preprocess(large_text, 'semantic')
        end_time = os.times().elapsed

        processing_time = end_time - start_time
        self.assertLess(processing_time, 5.0)  # 5秒内完成
        self.assertIsInstance(processed, str)


def create_test_files():
    """创建测试用的临时文件"""
    test_dir = tempfile.mkdtemp()

    # 创建原文文件
    orig_file = os.path.join(test_dir, "orig.txt")
    with open(orig_file, 'w', encoding='utf-8') as f:
        f.write("这是原创文本内容。")

    # 创建抄袭文件
    copy_file = os.path.join(test_dir, "copy.txt")
    with open(copy_file, 'w', encoding='utf-8') as f:
        f.write("这是抄袭文本内容。")

    # 创建答案文件路径
    answer_file = os.path.join(test_dir, "answer.txt")

    return orig_file, copy_file, answer_file, test_dir


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestTextProcessor,
        TestFileOperations,
        TestLCSCalculation,
        TestCosineSimilarity,
        TestSimilarityAlgorithms,
        TestIntegration,
        TestErrorHandling,
        TestPerformance
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    # 设置测试环境
    import jieba

    try:
        jieba.initialize()
    except:
        pass

    # 运行测试
    success = run_tests()

    # 退出码
    sys.exit(0 if success else 1)