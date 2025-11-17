"""
示例：使用 JudgementPipeline 进行评判
这个示例展示了如何使用 JudgementPipeline 类进行数据评判，并演示了参数覆盖功能
"""

import sys
import os

sys.path.append(os.getcwd())
import asyncio
from judgement import JudgementPipeline


def load_judgement_data():
    """
    加载示例评判数据
    """
    data_pool = [
        {
            "query": "人工智能的主要应用领域有哪些？",
            "answer": "AI技术在医疗诊断、金融风控、自动驾驶、智能教育、工业制造等方面都有重要应用",
            "GT": "人工智能广泛应用于医疗、金融、交通、教育、制造业等领域",
            "meta_info": {
                "model_name": "gpt-4o"
            },
            "system_prompt_kwargs": {
                "role": "专业AI评估师"
            },
            "user_prompt_kwargs": {
                "question": "人工智能的主要应用领域有哪些？",
                "reference_answer": "人工智能广泛应用于医疗、金融、交通、教育、制造业等领域",
                "model_response": "AI技术在医疗诊断、金融风控、自动驾驶、智能教育、工业制造等方面都有重要应用"
            }
        },
        {
            "query": "机器学习和深度学习有什么区别？",
            "answer": "机器学习是人工智能的一个分支，而深度学习是机器学习的一种特殊形式，它主要使用多层神经网络结构来模拟人脑的学习过程",
            "GT": "机器学习是AI的子集，深度学习是机器学习的子集。深度学习使用神经网络",
            "meta_info": {
                "model_name": "gpt-4o"
            },
            "system_prompt_kwargs": {
                "role": "专业AI评估师"
            },
            "user_prompt_kwargs": {
                "question": "机器学习和深度学习有什么区别？",
                "reference_answer": "机器学习是AI的子集，深度学习是机器学习的子集。深度学习使用神经网络",
                "model_response": "机器学习是人工智能的一个分支，而深度学习是机器学习的一种特殊形式，它主要使用多层神经网络结构来模拟人脑的学习过程"
            }
        }
    ]
    return data_pool


def count_tools_usage(input_data):
    """
    示例自定义规则函数：计算工具使用次数
    """
    # 这里只是一个示例，实际应用中可以根据 input_data 的内容计算工具使用次数
    answer = input_data.get("answer", "")
    # 假设在 answer 中查找工具使用次数（这里只是示例逻辑）
    return answer.count("工具")  # 示例：计算"工具"一词出现的次数


async def main():
    pipeline = JudgementPipeline(
        job_name="my_job",
        experiment_name="my_experiment",
        temperature=0.5,  # 覆盖模型温度
        max_tokens=1024,  # 覆盖最大token数
        system_prompt_path="prompt/system_prompts/default.prompt_evaluation.txt",  # 覆盖系统提示词路径
        user_prompt_path="prompt/user_prompts/default.prompt_evaluation.txt",  # 覆盖用户提示词路径
        output_data={
            "output_dir": "output",
            "experiment_name": "judgement_demo",
            "need_time_stamp": True,
        },
        need_time_stamp=False
    )
    data_pool = load_judgement_data()
    
    # 定义规则函数
    rule_functions = {
        "count-tools": count_tools_usage
    }
    
    results = await pipeline.run(
        data_pool,
        concurrency_limit=3,
        model_judgement_function=pipeline.make_judgement_extractor(),  # 使用内置的评判提取器
        rule_functions=rule_functions  # 传入自定义规则函数
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())