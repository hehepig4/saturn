# Evaluation Module

结构化评测结果导出、聚合和可视化工具。

## 目录结构

```
source/evaluation/
├── __init__.py              # 模块初始化
├── export_utils.py          # 核心数据模型和导出工具
├── aggregate_results.py     # 结果聚合工具
├── visualize_results.py     # 可视化生成工具
└── README.md               # 本文件
```

## 核心组件

### 1. export_utils.py

提供统一的数据模型和导出功能。

**主要类:**
- `EvaluationRun`: 评测运行元数据
- `QueryResult`: 单个查询的检索结果
- `AggregatedMetrics`: 聚合指标
- `EvaluationResults`: 完整评测结果容器

**用法示例:**

```python
from evaluation.export_utils import EvaluationResults, EvaluationRun, QueryResult

# 创建运行元数据
run = EvaluationRun(
    run_id="fetaqa_hyde_hybrid_20260204",
    timestamp="2026-02-04T12:00:00",
    dataset="fetaqa",
    method="hyde",
    hyde_mode="combined",
    retriever_type="hybrid"
)

# 创建结果容器
results = EvaluationResults(run)

# 添加查询结果
results.add_query_result(QueryResult(
    query_id=0,
    query_text="Which team won?",
    gt_tables=["table_123"],
    rank=3,
    top_k_table_ids=["t1", "t2", "t3"],
    top_k_scores=[0.9, 0.85, 0.82]
))

# 计算指标
metrics = results.compute_metrics()
print(f"Hit@10: {metrics.hit_at_10:.1%}")
print(f"MRR: {metrics.mrr:.4f}")

# 导出
results.save_json("output.json")
results.save_parquet("output.parquet")
```

### 2. aggregate_results.py

聚合多个评测运行的结果。

**用法:**

```bash
# 聚合目录下所有 parquet 文件
python -m evaluation.aggregate_results \
    --input-dir data/lake/lancedb/eval_results/exports/parquet \
    --output data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet \
    --summary data/lake/lancedb/eval_results/exports/aggregated/summary.csv

# 聚合特定文件
python -m evaluation.aggregate_results \
    --input-files run1.parquet run2.parquet run3.parquet \
    --output aggregated.parquet
```

### 3. visualize_results.py

从聚合数据生成可视化图表。

**可用图表:**
- `hyde_comparison`: HyDE 模式对比柱状图
- `retriever_comparison`: 检索器类型对比热力图
- `recall_curve`: Hit@K 曲线
- `dataset_comparison`: 数据集对比图
- `performance_matrix`: 方法性能矩阵

**用法:**

```bash
# 生成所有默认图表
python -m evaluation.visualize_results \
    --data data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet \
    --output-dir data/lake/lancedb/eval_results/visualizations

# 生成特定图表
python -m evaluation.visualize_results \
    --data aggregated.parquet \
    --output-dir viz/ \
    --plots hyde_comparison recall_curve

# 过滤特定数据集
python -m evaluation.visualize_results \
    --data aggregated.parquet \
    --output-dir viz/ \
    --filter-dataset fetaqa public_bi \
    --filter-method hyde
```

## 完整工作流

### 步骤 1: 运行评测并导出

修改评测脚本以支持导出（参见集成说明）：

```bash
python scripts/eval/analyze_hyde_retrieval.py \
    -d fetaqa \
    --retriever hybrid \
    --compare-combined \
    --export data/lake/lancedb/eval_results/exports/parquet/hyde/fetaqa_hybrid
```

这会生成：
- `fetaqa_hybrid.json` - JSON 格式结果
- `fetaqa_hybrid.parquet` - Parquet 格式结果（推荐）

### 步骤 2: 聚合多个运行

```bash
python -m evaluation.aggregate_results \
    --input-dir data/lake/lancedb/eval_results/exports/parquet \
    --output data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet \
    --summary data/lake/lancedb/eval_results/exports/aggregated/summary.csv
```

### 步骤 3: 生成可视化

```bash
python -m evaluation.visualize_results \
    --data data/lake/lancedb/eval_results/exports/aggregated/all_results.parquet \
    --output-dir data/lake/lancedb/eval_results/visualizations
```

### 步骤 4: 分析结果

```python
import pandas as pd

# 加载聚合数据
df = pd.read_parquet('exports/aggregated/all_results.parquet')

# 按数据集和方法分组
summary = df.groupby(['dataset', 'method', 'hyde_mode']).agg({
    'hit_at_1': 'mean',
    'hit_at_10': 'mean',
    'mrr': 'mean',
    'query_id': 'count'
}).rename(columns={'query_id': 'num_queries'})

print(summary)

# 自定义分析
hyde_results = df[df['method'] == 'hyde']
print(hyde_results.groupby('hyde_mode')['hit_at_10'].describe())
```

## 数据格式

### Parquet Schema

每条记录包含：

**元数据字段:**
- `run_id`: 运行唯一标识
- `timestamp`: 时间戳
- `dataset`: 数据集名称
- `method`: 评测方法
- `hyde_mode`: HyDE 模式（可选）
- `retriever_type`: 检索器类型（可选）
- `llm`: LLM 类型

**查询级别字段:**
- `query_id`: 查询ID
- `query_text`: 查询文本
- `gt_tables`: Ground truth 表ID列表
- `rank`: GT 最佳排名
- `top_k_table_ids`: Top-K 检索结果
- `top_k_scores`: 对应分数

**指标字段:**
- `hit_at_1`, `hit_at_5`, `hit_at_10`, `hit_at_50`, `hit_at_100`
- `mrr`

### JSON Schema

```json
{
  "metadata": {
    "run_id": "fetaqa_hyde_hybrid_20260204",
    "dataset": "fetaqa",
    "method": "hyde",
    "hyde_mode": "combined",
    "retriever_type": "hybrid"
  },
  "metrics": {
    "hit_at_1": 0.452,
    "hit_at_10": 0.753,
    "mrr": 0.5621,
    "num_queries": 100
  },
  "query_results": [
    {
      "query_id": 0,
      "query_text": "Which team won?",
      "gt_tables": ["table_123"],
      "rank": 3,
      "top_k_table_ids": ["t1", "t2", "t3"],
      "top_k_scores": [0.9, 0.85, 0.82]
    }
  ]
}
```

## 集成到现有评测脚本

### analyze_hyde_retrieval.py 示例

在 `compare_hyde_modes()` 函数中添加：

```python
def compare_hyde_modes(..., export_prefix: Optional[str] = None):
    # ... 现有代码 ...
    
    # 导出结果
    if export_prefix:
        from evaluation.export_utils import (
            EvaluationResults, EvaluationRun, QueryResult, create_run_id
        )
        
        run = EvaluationRun(
            run_id=create_run_id(dataset, "hyde", 
                                retriever_type=retriever_type, 
                                hyde_mode=modes[-1]),
            timestamp=datetime.now().isoformat(),
            dataset=dataset,
            method="hyde",
            hyde_mode=modes[-1],
            retriever_type=retriever_type,
            llm=llm_suffix,
            num_queries=num_queries,
            top_k=top_k
        )
        
        eval_results = EvaluationResults(run)
        
        for i, item in enumerate(analysis_data):
            # ... 检索逻辑 ...
            
            eval_results.add_query_result(QueryResult(
                query_id=i,
                query_text=item['query'],
                gt_tables=gt_tables,
                rank=rank,
                top_k_table_ids=[tid for tid, _ in results[:10]],
                top_k_scores=[score for _, score in results[:10]],
                hyde_table_desc=analysis.get('hypothetical_table_description'),
                hyde_column_desc=analysis.get('hypothetical_column_descriptions')
            ))
        
        # 保存
        eval_results.save_json(f"{export_prefix}.json")
        eval_results.save_parquet(f"{export_prefix}.parquet")
        logger.info(f"Exported results to {export_prefix}.[json/parquet]")
```

添加 CLI 参数：

```python
parser.add_argument("--export", type=str, default=None,
                    help="Export results to structured format (path prefix)")
```

## 依赖

```bash
# 必需
pip install pandas pyarrow

# 可视化
pip install matplotlib seaborn

# 日志
pip install loguru
```

## 目录组织建议

```
data/lake/lancedb/eval_results/
├── runs/                   # 原始运行日志
│   └── eval_20260204_120000/
│       ├── fetaqa_semantic.log
│       └── summary.md
├── exports/                # 结构化导出
│   ├── json/
│   │   └── hyde/
│   │       └── fetaqa_hybrid_combined_20260204.json
│   ├── parquet/           # 推荐使用
│   │   ├── hyde/
│   │   │   ├── fetaqa_hybrid_combined_20260204.parquet
│   │   │   └── fetaqa_bm25_raw_20260204.parquet
│   │   └── structural/
│   │       └── fetaqa_structural_20260204.parquet
│   └── aggregated/        # 聚合汇总
│       ├── all_results.parquet
│       └── summary.csv
└── visualizations/        # 可视化结果
    ├── hyde_comparison.png
    ├── retriever_comparison.png
    └── recall_curve.png
```

## 常见问题

### Q: Parquet vs JSON，用哪个？
A: **推荐 Parquet**。优势：
- 列式存储，查询快
- 压缩率高，占用空间小
- pandas 原生支持，读取快

JSON 适合：调试、人类可读性、小数据集

### Q: 如何添加自定义指标？
A: 扩展 `AggregatedMetrics` 类：

```python
@dataclass
class AggregatedMetrics:
    # ... 现有字段 ...
    custom_metric: Optional[float] = None
```

### Q: 如何添加新的可视化？
A: 在 `visualize_results.py` 中添加函数并注册：

```python
def plot_my_custom_viz(df: pd.DataFrame, output_path: Path):
    # 实现可视化逻辑
    ...

PLOT_FUNCTIONS['my_custom_viz'] = plot_my_custom_viz
```

### Q: 数据太大怎么办？
A: 使用分片和增量聚合：

```bash
# 按数据集分别聚合
for dataset in fetaqa public_bi; do
    python -m evaluation.aggregate_results \
        --input-dir exports/parquet \
        --output aggregated/${dataset}.parquet \
        --no-recursive
done

# 最后合并
python -m evaluation.aggregate_results \
    --input-files aggregated/*.parquet \
    --output aggregated/all.parquet
```

## 许可证

MIT
