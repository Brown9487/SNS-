# SNS 选股

基于 AkShare 的 A 股选股脚本，默认股票池为中证 A500 和中证 1000 成分股。

当前评分框架分为两类：

- `Structure`：120 日收益、相对强弱、回撤控制，并叠加动态 PE 估值因子
- `Narrative`：20/60 日动量、成交额放大、逼近前高、EMA 拐点

其中动态 PE 采用正值清洗后再做 `log(PE)` 标准化，主要用于抑制估值过高标的对 `Structure` 排名的挤占。

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
