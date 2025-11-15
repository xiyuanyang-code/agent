# Agent

> [!IMPORTANT]
> The Agent Factory, to build and to use.

## Introduction

提升代码复用的效率，故封装成结构化的代码。

## Structures

```text
.
├── config
│   └── data_generation.example.yaml
├── data
├── data_generation
│   ├── __init__.py
│   ├── pipeline.py
│   └── README.md
├── example
│   └── data_generation_example.py
├── logs
├── main.py
├── prompt
│   ├── system_prompts
│   │   └── default.prompt.txt
│   └── user_prompts
│       └── default.prompt.txt
├── README.md
└── utils
    ├── llm_client.py
    └── logger_config.py
```

- `data_generation`: 数据合成模块

> More will be added in the future.

## Modules

> [!IMPORTANT]
> 这一部分会经常更新！欢迎 PR

- `data_generation`: 合成数据
    - 基本的 AI 合成数据（模型蒸馏）
    - 更加 advanced 的轨迹合成数据
        - MCP Tool Calling
        - Multi-turn data generation
        - Multi-agent data generation
- `agent`: 基本的 Agent
    - **Less is More**: We just need the simplest agent!
    - Multi-turn chat
    - Inner Memory and Reasoning
    - Outer Tool Calling and MCP
    - Interactions with multi-agents
- `environment`: Agent Interactions with the environment
    - Coding Environment
    - Searching Environment
- `evaluation`:
    - simple scripts for agent evaluations

## Todo List

- build and simple agent interactions
    - 关键在于明确简洁的接口