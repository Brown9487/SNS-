# SNS 选股

基于 AkShare 的 A 股选股脚本，默认股票池为中证 A500、沪深 300、中证红利和中证 1000 成分股。

## 环境

```bash
python3 -m pip install -r requirements.txt
```

## 运行

```bash
python3 sns_selector.py --max-workers 2 --batch-size 20 --sleep-min 0.3 --sleep-max 1.0
```

## 输出

- `sns_structure_top20.csv`
- `sns_narrative_top15.csv`
- `sns_result_YYYYMMDD_data_YYYYMMDD.xlsx`
