"""
测试配置文件
"""

# 测试参数配置
TEST_CONFIG = {
    'performance_threshold': 10.0,  # 性能测试阈值（秒）
    'large_text_size': 10000,       # 大文本测试大小
    'test_timeout': 30,             # 单个测试超时时间
}

# 测试数据
TEST_DATA = {
    'identical_texts': [
        ("完全相同文本", "完全相同文本"),
        ("长文本" * 100, "长文本" * 100)
    ],
    'similar_texts': [
        ("今天天气很好", "今天天气不错"),
        ("我喜欢编程", "我热爱写代码")
    ],
    'different_texts': [
        ("苹果手机", "华为电脑"),
        ("北京上海", "广州深圳")
    ]
}