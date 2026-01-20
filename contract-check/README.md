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
- `DEEPSEEK_API_KEY`: DeepSeek 的 API Key
- `DEEPSEEK_BASE_URL`: DeepSeek OpenAI 兼容接口地址（默认已给）
- `DEEPSEEK_MODEL`: 模型名（默认 `deepseek-chat`）

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

## 4. 调用接口

### 4.1 使用 curl（上传两个文本文件）

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/contract-check" \
  -F "regulation=@regulation.txt" \
  -F "contract=@contract.txt" | jq .
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