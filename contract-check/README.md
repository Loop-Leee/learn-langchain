# contract-check（LangChain + DeepSeek 合同合规审查接口）

## 你将得到什么
- 一个 HTTP 接口：上传「法规要求」与「合同内容」，调用 DeepSeek 大模型判断是否合规
- 返回结构化 JSON：`review_result` / `review_process` / `review_suggestion`

## 1. 环境变量
复制示例文件并填写（本目录使用 `example.env` 命名）：

```bash
cp example.env .env
```

需要配置：
- `MODEL_TYPE`: 模型类型（可选：`deepseek`，可选：`qwen3`）
- `DEEPSEEK_API_KEY`: DeepSeek 的 API Key
- `DEEPSEEK_BASE_URL`: DeepSeek OpenAI 兼容接口地址（默认已给）
- `DEEPSEEK_MODEL`: 模型名（默认 `deepseek-chat`）

### 切换到 Qwen3（通义千问）

如需使用 Qwen3，在 `example.env` 中配置：
- 'MODEL_TYPE': qwen3
- `QWEN3_API_KEY`: Qwen3 的 API Key
- `QWEN3_BASE_URL`: Qwen3 OpenAI 兼容接口地址（默认已给）
- `QWEN3_MODEL`: 模型名（默认 `qwen-plus`，可选：`qwen-max`, `qwen-turbo` 等）

在代码中使用（策略模式）：
```python
from app.llm import create_chat_model
from app.config import ModelType, LLMConfig

# 方式1：使用字符串（推荐）
llm = create_chat_model(model_type="qwen3")

# 方式2：使用枚举（类型安全）
llm = create_chat_model(model_type=ModelType.QWEN3)

# 方式3：使用策略模式的工厂方法
config = LLMConfig.from_model_type("qwen3")
llm = create_chat_model(config=config)

# 方式4：直接使用具体策略方法
config = LLMConfig.from_qwen3_env()
llm = create_chat_model(config=config)

# 切换到 DeepSeek（同样支持策略模式）
llm = create_chat_model(model_type="deepseek")
# 或
config = LLMConfig.from_deepseek_env()
llm = create_chat_model(config=config)
```

## 2. 安装依赖（Conda 推荐）

使用仓库提供的 `environment.yml` 创建环境（默认名 `contract310`）：

```bash
conda env create -f environment.yml
conda activate contract310
# 可选：升级 pip
python -m pip install --upgrade pip
```

如不使用 Conda，可自行创建虚拟环境并参考 `environment.yml` 中的依赖列表。

## 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 调试模式（可以断点调试）
- 安装调试依赖：`pip install debugpy`（或 `conda install debugpy`）
- 启动附加调试端口（默认 5678）：

```bash
python -m debugpy --listen 5678 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 在 IDE 中附加到 `localhost:5678`，即可在代码中下断点单步调试。

## 4. 调用接口

### 4.1 使用 curl（直接提交字符串）

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/contract-check" \
  -F "regulation=这里填写法规要求文本" \
  -F "contract=这里填写合同内容文本" | jq .
```

返回示例（字段与题目一致）：

```json
{
  "审查结果": "不合格",
  "审查过程": "……",
  "审查建议": "……"
}
```

### 4.2 Swagger
启动后访问：`http://127.0.0.1:8000/docs`