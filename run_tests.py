#!/usr/bin/env python3
"""
自动化测试运行脚本
支持多种运行模式：
1. 完整测试套件
2. 快速测试（跳过性能测试）
3. 单个模块测试
"""

import unittest
import sys
import os
import argparse
import time
from test_plagiarism_checker import run_tests


def main():
    parser = argparse.ArgumentParser(description='文本查重系统单元测试')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式（跳过性能测试）')
    parser.add_argument('--module', type=str,
                        help='指定测试模块')
    parser.add_argument('--coverage', action='store_true',
                        help='生成测试覆盖率报告')

    args = parser.parse_args()

    # 设置环境变量
    if args.fast:
        os.environ['TEST_MODE'] = 'FAST'

    print("=" * 60)
    print("文本查重系统 - 单元测试")
    print("=" * 60)

    start_time = time.time()

    try:
        if args.coverage:
            # 覆盖率测试
            import coverage
            cov = coverage.Coverage()
            cov.start()

        # 运行测试
        success = run_tests()

        if args.coverage:
            cov.stop()
            cov.save()
            cov.report()
            cov.html_report(directory='htmlcov')

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\n测试完成，耗时: {elapsed:.2f}秒")

        if success:
            print("✅ 所有测试通过！")
            return 0
        else:
            print("❌ 测试失败！")
            return 1

    except Exception as e:
        print(f"测试执行错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())