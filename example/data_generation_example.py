import sys
import os

sys.path.append(os.getcwd())
import asyncio
from data_generation import DataGenerationPipeline


def load_sample_data():
    """
    加载示例输入数据
    """
    data_pool = [
        {
            "user_prompt_kwargs": {"topic": "人工智能", "word_count": "100字"},
            "system_prompt_kwargs": {"tone": "专业"},
        },
        {
            "user_prompt_kwargs": {"topic": "机器学习", "word_count": "150字"},
            "system_prompt_kwargs": {"tone": "通俗易懂"},
        },
        {
            "user_prompt_kwargs": {"topic": "数据科学", "word_count": "200字"},
            "system_prompt_kwargs": {"tone": "学术"},
        },
        {
            "user_prompt_kwargs": {"topic": "深度学习", "word_count": "180字"},
            "system_prompt_kwargs": {"tone": "简洁"},
        },
        {
            "user_prompt_kwargs": {"topic": "自然语言处理", "word_count": "160字"},
            "system_prompt_kwargs": {"tone": "技术性"},
        },
    ]
    return data_pool


async def main():
    pipeline2 = DataGenerationPipeline(
        temperature=0.9,  # 覆盖模型温度
        max_tokens=2048,  # 覆盖最大token数
        system_prompt_path="prompt/system_prompts/default.prompt.txt",  # 覆盖系统提示词路径
        user_prompt_path="prompt/user_prompts/default.prompt.txt",  # 覆盖用户提示词路径
        output_data={
            "output_dir": "output",
            "experiment_name": "demo_test",
            "need_time_stamp": True,
        },
        need_time_stamp=False
    )
    data_pool = load_sample_data()
    results = await pipeline2.run(
        data_pool,
        concurrency_limit=5,
        extract_function=pipeline2.make_default_extractor(pattern_name="draft"),
        # 支持自定义的复杂 extractions
        # 类内部也提供了基本提取的 函数工厂模式
    )
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
