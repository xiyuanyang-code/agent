# Data Generations Pipeline

## Introduction

数据合成的基本模板

- 用户自定义 fuzz 输入数据
- 调用 LLM 实现合成数据的批量生成
    - 支持高并发处理
    - 实时保存结果到 JSONL 文件

## Configuration

在 `config.yaml` 中配置模型参数、提示词路径和输出设置：

```yaml
model:
  provider: openai
  api_key: your_api_key
  base_url: https://api.openai.com/v1
  model: "gpt-4o-2024-11-20"
  rate_limit: 20
  max_tokens: 5012
  temperature: 0.7

prompts:
  system_prompt_path: "prompts/system_prompts/default.txt"
  user_prompt_path: "prompts/user_prompts/default.txt"

output_data:
  output_dir: "output"
  experiment_name: "initial_test"
  need_time_stamp: true
```

注意，上述配置只是默认配置，后续也支持在代码中显示提供配置来覆盖默认配置。

## Usage

### Simple Workflow

> [!IMPORTANT]
> 最基本的合成数据的工作流，后续还会添加更多的工作流格式的数据。

- 在配置中准备好 user_prompt 和 system_prompt 的模版
- 自定义生成 data_pool 的函数，数据格式如下：
    - 要求返回一个 List，List 中的每一个元素是包含 `user_prompt_kwargs` 和 `system_prompt_kwargs` 的字典
    - 在后续的数据生成中，pipeline 会自动的将输入的这些模板填充到 user prompt 和 system prompt 中，形成 ready 的 对话版本的提示词
- 模型调用的版本已经封装完成（带安全限制的高并发模型）
- **需要提供一个 parse_response()** 函数，将模型输出的内容通过正则匹配等格式实现转换，返回结构化的数据。
    - 默认行为则会直接记录模型的原始 response
- 最终数据会记录在对应的 jsonl 文件中，可以选择文件是否需要带上时间戳

```python
def load_sample_data():
    """
    加载示例输入数据
    """
    # 增加更多示例数据以便更好地展示进度条
    data_pool = [
        {
            "user_prompt_kwargs": {
                "topic": "人工智能",
                "word_count": "100字"
            },
            "system_prompt_kwargs": {
                "tone": "专业"
            }
        },
        {
            "user_prompt_kwargs": {
                "topic": "机器学习",
                "word_count": "150字"
            },
            "system_prompt_kwargs": {
                "tone": "通俗易懂"
            }
        },
        {
            "user_prompt_kwargs": {
                "topic": "数据科学",
                "word_count": "200字"
            },
            "system_prompt_kwargs": {
                "tone": "学术"
            }
        },
        {
            "user_prompt_kwargs": {
                "topic": "深度学习",
                "word_count": "180字"
            },
            "system_prompt_kwargs": {
                "tone": "简洁"
            }
        },
        {
            "user_prompt_kwargs": {
                "topic": "自然语言处理",
                "word_count": "160字"
            },
            "system_prompt_kwargs": {
                "tone": "技术性"
            }
        }
    ]
    return data_pool
```

```python
"""
示例：使用 Pipeline 生成数据
这个示例展示了如何使用 Pipeline 类生成数据，并演示了参数覆盖功能
"""

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
```
