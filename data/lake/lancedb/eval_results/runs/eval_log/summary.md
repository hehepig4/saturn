# Retrieval Evaluation Results

Run timestamp: 2026-03-08 15:53:47

## Configuration
- LLM: local
- Num Queries: -1
- Methods: semantic
- Top-K: 100
- Compare Mode: false

## Datasets Evaluated
- adventure_works
- chembl
- fetaqa
- fetaqapn
- public_bi
- bird
- chicago

## Results

| Dataset | Method | R@1 | R@3 | R@5 | R@10 | R@50 | R@100 | MRR |
|---------|--------|------|------|------|------|------|------|------|
| adventure_works_semantic | bm25 | 85.6% | 95.2% | 97.0% | 98.9% | 100.0% | 100.0% | 0.8453 |
| adventure_works_semantic | hybrid | 81.5% | 92.6% | 95.6% | 98.4% | 100.0% | 100.0% | 0.8813 |
| adventure_works_semantic | vector | 75.5% | 85.2% | 90.2% | 94.5% | 99.4% | 100.0% | 0.8290 |
| bird_semantic | bm25 | 74.9% | 88.9% | 92.3% | 95.8% | 99.4% | 99.6% | 0.7077 |
| bird_semantic | hybrid | 66.5% | 83.1% | 89.1% | 94.0% | 99.1% | 99.6% | 0.7829 |
| bird_semantic | vector | 56.9% | 73.5% | 78.6% | 86.5% | 96.6% | 98.2% | 0.6935 |
| chembl_semantic | bm25 | 89.6% | 96.8% | 97.9% | 98.6% | 99.9% | 100.0% | 0.8550 |
| chembl_semantic | hybrid | 87.6% | 94.1% | 95.6% | 97.2% | 99.8% | 100.0% | 0.8858 |
| chembl_semantic | vector | 82.8% | 90.6% | 92.9% | 94.8% | 99.5% | 100.0% | 0.8522 |
| chicago_semantic | bm25 | 60.7% | 76.9% | 82.7% | 88.7% | 97.5% | 98.8% | 0.6296 |
| chicago_semantic | hybrid | 57.1% | 72.3% | 77.3% | 85.3% | 97.6% | 98.6% | 0.6733 |
| chicago_semantic | vector | 48.0% | 63.6% | 69.2% | 75.7% | 93.0% | 96.1% | 0.5728 |
| fetaqa_semantic | bm25 | 75.5% | 86.5% | 88.5% | 90.6% | 94.6% | 96.0% | 0.7119 |
| fetaqa_semantic | hybrid | 77.1% | 88.0% | 89.9% | 91.9% | 95.7% | 96.8% | 0.8261 |
| fetaqa_semantic | vector | 76.9% | 86.6% | 88.3% | 90.1% | 94.2% | 95.5% | 0.8311 |
| fetaqapn_semantic | bm25 | 56.9% | 68.0% | 72.4% | 77.9% | 88.0% | 90.9% | 0.6303 |
| fetaqapn_semantic | hybrid | 47.7% | 59.6% | 65.7% | 72.5% | 86.5% | 90.3% | 0.6308 |
| fetaqapn_semantic | vector | 38.7% | 49.9% | 54.8% | 61.2% | 74.7% | 81.0% | 0.5044 |
| public_bi_semantic | bm25 | 71.8% | 82.5% | 86.5% | 91.8% | 98.2% | 99.6% | 0.7930 |
| public_bi_semantic | hybrid | 67.1% | 77.9% | 80.9% | 86.9% | 97.9% | 99.8% | 0.8135 |
| public_bi_semantic | vector | 54.0% | 68.9% | 74.1% | 81.1% | 93.1% | 99.0% | 0.6323 |
