# Judgement Module

## Introduction

In this module, we hope we can judge agent's generated output with **model-based judgement** and **rule-based judgement**


## 数据读取

这一部分需要用户手动实现（为了适配各种数据模式），但是需要保证数据是这样的基本格式并且一定要有 answer 这一项。

```json
{
    "query": "",
    "answer": "",
    "GT": "",
    "system_prompt_kwargs": {},
    "user_prompt_kwargs": {}
}
```

## 评判

在这里需要提供一个 Judgement 的窗口，并且自动提取结构化的信息和统计量。

### Model-Based Judgement

仍然需要用户手动构建提示词，这一部分的参考方式可以见 `./data_generation/pipeline.py` 并且**用户需要手动提供一个提取函数**。

仿照`./data_generation/pipeline.py` 这里也要提供一个函数工厂，来解析这样格式的输出：

```json
{
  "accuracy": <1-10>,
  "relevance": <1-10>,
  "clarity": <1-10>,
  "completeness": <1-10>,
  "overall": <1-10>,
  "comment": "<50-char bilingual (EN/CN) comment highlighting key strengths and flaws>"
}
```

### Rule-Based Judgement

Rule-Based Judgement 更多作为一种统计信息出现，默认需要统计的信息是：

- Answer Token 的数目

这一部分需要支持用户手动传入 Rule-Based 的解析函数（例如解析调用工具的数量等等）

最终用户需要传入的参数是：

```
{
    "count-tools": count_tool,
    // count_tool 是传入的用户自定义函数
    ...
}
```

而对于每一个评测单元（一条数据），模型需要输出一个 json 数据：

```json
{
    "query-token-count": 100,
    "answer-token-count": 100,
    "GT-token-count": 100,
    // 上面的三个是默认统计的 Rule-based 的参数统计量，
    "count-tools": 12
    // 这些是用户传入的自定义函数的返回结果
}
```

因此，最终对于一个评测单元，需要生成如下的一个 json 格式的文件：

```json
{
    "model-based-judgement": {...},
    "rule-based-judgement": {...},
}
```

## 数据的存储

对于每一次实验，都需要指定:

- job_name
- experiment_name

存储的输出在 `./job_name/experiment_name/` 中，存储的信息是一个带时间戳的 jsonl 文件，这个文件每一行就是我的测评结果。

注意我希望和 `./data_generation/pipeline.py` 一样开并发 并且实时写入 json

## Usage

### Simple Workflow

> [!IMPORTANT]
> 最基本的评判工作流，支持对数据进行批量自动化评估。

- 在配置中准备好 user_prompt 和 system_prompt 的模版
- 自定义生成 data_pool 的函数，数据格式如下：
    - 要求返回一个 List，List 中的每一个元素是包含 `query`, `answer`, `GT` 的字典
    - 在后续的评判中，pipeline 会自动的将输入的这些模板填充到 user prompt 和 system prompt 中，形成 ready 的 对话版本的提示词
- 模型调用的版本已经封装完成（带安全限制的高并发模型）
- **需要提供一个 model_judgement_function()** 函数，将模型输出的评判内容进行格式化处理，返回结构化的评分结果。
- **需要提供一个 rule_functions** 字典，包含用户自定义的规则函数
- 最终评判结果会记录在对应的 jsonl 文件中，可以选择文件是否需要带上时间戳

