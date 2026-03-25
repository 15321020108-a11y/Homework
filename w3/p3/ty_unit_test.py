import unittest
from typing import List, Dict, Union, Tuple
from ty import process_data

class TestProcessData(unittest.TestCase):

    def test_average_score(self):
        # 测试正常情况
        name, avg_score = process_data("Alice", 25, [88.5, 92.0, 79.5], {"city": "New York", "years_of_experience": 3})
        self.assertEqual(name, "Alice")
        self.assertAlmostEqual(avg_score, 86.67, places=2)

    def test_empty_scores(self):
        # 测试空分数列表
        name, avg_score = process_data("Bob", 30, [], {"city": "Los Angeles", "years_of_experience": 5})
        self.assertEqual(name, "Bob")
        self.assertEqual(avg_score, 0.0)

    def test_single_score(self):
        # 测试单个分数
        name, avg_score = process_data("Charlie", 22, [100.0], {"city": "Chicago", "years_of_experience": 1})
        self.assertEqual(name, "Charlie")
        self.assertEqual(avg_score, 100.0)

if __name__ == '__main__':
    unittest.main()