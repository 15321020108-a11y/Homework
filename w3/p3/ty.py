from typing import List, Dict, Tuple, Union

def process_data(name: str, # 姓名
                 age: int, # 年龄
                 scores: List[float], # 考试分数列表 
                 metadata: Dict[str, Union[str, int]] # 额外的用户信息
                ) -> Tuple[str, float]: # 返回值类型
    """
    处理用户数据并返回摘要信息。

    参数：
    name (str): 用户姓名
    age (int): 用户年龄
    scores (List[float]): 用户的考试分数列表
    metadata (Dict[str, Union[str, int]]): 额外的用户信息，例如"city"（字符串）或"years_of_experience"（整数）

    返回：
    Tuple[str, float]: 包含用户姓名和平均分的元组
    """
    average_score = sum(scores) / len(scores) if scores else 0.0
    return name, average_score

# 示例调用
user_name, avg_score = process_data("Alice", 25, [88.5, 92.0, 79.5], {"city": "New York", "years_of_experience": 3})
print(user_name, avg_score)