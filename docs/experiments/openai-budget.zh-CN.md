# OpenAI 闭源对照与预算

用户确认总预算为 **80 美元**。本轮实验共用 **60 美元**运行上限，保留至少 20 美元；试跑阶段合计最多 **1 美元**。这不是已花费金额。2026-11-03 准备接入时 GPU 节点尚未配置 API key，未发送付费推理请求。

先接入 GPT-6.1 Terra，关闭 reasoning。它支持图像输入和 Responses 函数调用；保持现有 96 / 384 输出 token、8192 输入 token、8 轮与工具预算 12。价格与功能见[官方模型页](https://developers.openai.com/api/docs/models/gpt-6.1-terra)。更强模型待首轮实际费用与接口验证后再安排，不切换到用户的 ChatGPT 登录凭据。

## 评测范围

- 沿用已冻结的 209 个论文案件族，不重新筛选成功样本。无工具、全证据参考、有工具三种条件均保留。
- 默认先跑 Sufficient，以便直接加入主表。完整运行是 627 个 episode；增加其余三个材料可获得性版本时是 2508 个 episode，须在同一共享预算内执行。
- 试跑按物品状态／身份各取两个固定 ID 的案件族，使用同样三个条件，共 12 个 episode。试跑只验证接口与费用，不能作为主表结果。
- 本地图像在发送前限制到 512² 像素；不发送标签、标注或未释放资产。保留原始区域与时点引用。
- 复用当前环境、动作参数与冻结 grounding 权重。该入口用 CPU 执行本地工具以免干扰四张卡上的实验；API 网络延迟与 CPU 工具耗时需另列，不与本地 GPU 时间作等硬件速度排名。
- 按源标签统计的结果仍为未审核开发结果，正式 grounded accuracy 留空。预算耗尽时保留部分进度，不用缩小分母后的分数进入主表。
- 该入口尚未做真实 API 集成验证；离线测试不代表闭源模型性能或账户已获访问权限。

## 花费控制

所有模型、阶段与重启必须使用同一个外部 SQLite 账本。每次生成前用官方输入计数接口核对完整历史、图片与工具 schema，再预留输入及最大输出的费用。只使用 Standard 服务与本地函数，不启用付费托管搜索、容器或文件存储。官方参考：[输入计数](https://developers.openai.com/api/docs/guides/token-counting)、[reasoning 与输出计费](https://developers.openai.com/api/docs/guides/reasoning)。

账本使用微美元整数、事务预留与持久阶段限额。输入按每百万 2.50 美元保守计入（高于 Terra 的标准输入价并覆盖当前列出的缓存写入价），输出按每百万 5 美元计入；不依赖缓存折扣。记录原始 API usage，账本金额是保守核算上界，不能当成账单实际费用。

超时、缺少 usage、异常 token 用量或费用无法核对时，保留预留并阻止新付费调用，不自动重试。已完成的相同请求可复用持久响应，保留原调用时间。费用异常应先对照 API 控制台及请求 ID 核对，不删账本重跑。该限制只覆盖使用此账本的实验，账户其他程序的花费需另行扣除。

## 启动

密钥仅通过服务器进程环境的 `OPENAI_API_KEY` 传入，不写进配置、Git、命令行参数或日志。用户需先在 API 控制台核对可用余额，确认额度关联正确组织。

在已分配的 macula GPU 节点，从仓库根目录执行：

```bash
RUNTIME=/home/cxu-serve/p62/zzh136/experiments/resolveai-poc
.venv/bin/python -m scripts.evaluate_openai \
  "$RUNTIME/evaluation/cvpr-20261101/plan.json" \
  docs/results/cvpr_20261102_candidate_scores.json \
  "$RUNTIME/evaluation/openai/terra-pilot" \
  --ledger "$RUNTIME/evaluation/openai/budget.sqlite" \
  --grounding-root "$RUNTIME/grounding"
```

默认是离线预览，不连接 API，不创建实验输出。只有显式加入 `--execute` 且配置了密钥才会请求 API。试跑确认后，用 `--stage paper` 与独立输出目录评测整个固定题组；继续使用同一账本。需要所有四种版本时加入 `--all-variants`，不改变 Sufficient 主表的题目 ID。
