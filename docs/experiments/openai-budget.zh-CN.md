# OpenAI 闭源对照与预算

用户确认总预算为 **80 美元**。本轮实验共用 **60 美元**运行上限，保留至少 20 美元；最初 Terra 试跑上限为 **1 美元**，后续各模型试跑上限见下方模型清单。这不是已花费金额。2026-11-03 已将用户授权的 API key 配置到 GPU 节点的私有文件（0600，位于 Git 仓库外）；官方模型目录认证通过，GPT-6.1 Terra 可访问。认证后已完成 12/12 次实际接口试跑，15 次模型调用，保守核算 0.085282 美元。4 次 Agent 运行均通过 finish 完成且无格式错误，其中实际出现 inspect → crop → compare → finish 的多轮过程。这只是接口验证，不作为 benchmark 分数。

GPT-6.1 Terra 关闭 reasoning，已完成固定主表的 627 次对照。额外材料版本运行到总账可用额度不足时停止，保留已完成的 1058/2508 次尝试；不重跑或借用预留给其他模型的额度。每个案件独立会话，顺序不改变模型观察或标签。它支持图像输入和 Responses 函数调用；保持现有 96 / 384 输出 token、8192 输入 token、8 轮与工具预算 12。价格与功能见[官方模型页](https://developers.openai.com/api/docs/models/gpt-6.1-terra)。后续模型按下方队列运行，不切换到用户的 ChatGPT 登录凭据。

## 评测范围

- 沿用已冻结的 209 个论文案件族，不重新筛选成功样本。无工具、全证据参考、有工具三种条件均保留。
- 默认先跑 Sufficient，以便直接加入主表。完整运行是 627 个 episode；增加其余三个材料可获得性版本时是 2508 个 episode，须在同一共享预算内执行。
- 试跑按物品状态／身份各取两个固定 ID 的案件族，使用同样三个条件，共 12 个 episode。试跑只验证接口与费用，不能作为主表结果。
- 本地图像在发送前限制到 512² 像素；不发送标签、标注或未释放资产。保留原始区域与时点引用。
- 复用当前环境、动作参数与冻结 grounding 权重。该入口用 CPU 执行本地工具以免干扰四张卡上的实验；API 网络延迟与 CPU 工具耗时需另列，不与本地 GPU 时间作等硬件速度排名。
- 按源标签统计的结果仍为未审核开发结果，正式 grounded accuracy 留空。预算耗尽时保留部分进度，不用缩小分母后的分数进入主表。
- 12 次真实 API 试跑已完成；原始汇总见[试跑记录](../results/openai_pilot.json)。正式模型成绩等待完整固定题组，不从试跑推算答对率。

## 花费控制

所有模型受同一个外部总账本的 60 美元上限约束。为并行运行，总账先足额预留每个模型的消费额度，模型仅能在获拨款的子账本内消费；预留不等于实际支出。模型结束后先封闭子账本，再回收未用额度，防止已返还的资金再次被消费。总账预留还包含在途请求的费用上界。每次生成前用官方输入计数接口核对完整历史、图片与工具 schema，再预留输入及最大输出的费用。只使用 Standard 服务与本地函数，不启用付费托管搜索、容器或文件存储。官方参考：[输入计数](https://developers.openai.com/api/docs/guides/token-counting)、[reasoning 与输出计费](https://developers.openai.com/api/docs/guides/reasoning)。

账本使用微美元整数、事务预留与持久阶段限额。输入按每百万 2.50 美元保守计入（高于 Terra 的标准输入价并覆盖当前列出的缓存写入价），输出按每百万 5 美元计入；不依赖缓存折扣。记录原始 API usage，账本金额是保守核算上界，不能当成账单实际费用。

超时、缺少 usage、异常 token 用量或费用无法核对时，保留受影响模型的费用上界并阻止它的新付费调用，不自动重试。其他模型只可消费已获总账担保的额度。已完成的相同请求可复用持久响应，保留原调用时间。费用异常应先对照 API 控制台及请求 ID 核对，不删账本重跑。该限制覆盖总账及其获拨款的全部子账本；账户其他程序的花费需另行扣除。

## 启动

密钥通过服务器进程环境的 `OPENAI_API_KEY` 传入，或使用 `--key-file` 从仓库外的本人私有文件加载；不把密钥值写进公开配置、Git、命令行参数或日志。入口拒绝组／其他用户可读的密钥文件及符号链接。用户需先在 API 控制台核对可用余额，确认额度关联正确组织。

在已分配的 macula GPU 节点，从仓库根目录执行：

```bash
RUNTIME=/home/cxu-serve/p62/zzh136/experiments/resolveai-poc
.venv/bin/python -m scripts.evaluate_openai \
  "$RUNTIME/evaluation/cvpr-20261101/plan.json" \
  docs/results/cvpr_20261102_candidate_scores.json \
  "$RUNTIME/evaluation/openai/terra-pilot" \
  --ledger "$RUNTIME/evaluation/openai/budget.sqlite" \
  --grounding-root "$RUNTIME/grounding" \
  --key-file "$RUNTIME/credentials/openai.env"
```

默认是离线预览，不连接 API，不创建实验输出。只有显式加入 `--execute` 且配置了密钥才会请求 API。macula 主机存在 DNS 别名，执行位置按 Slurm 的 `SLURMD_NODENAME` 核实。试跑确认后，用 `--stage paper` 与独立输出目录评测整个固定题组；继续使用同一账本。需要所有四种版本时加入 `--all-variants`，不改变 Sufficient 主表的题目 ID。


## 多模型与 Decisions 对照

优先完成 Table 1 的同一 209 族 Sufficient 版本，各普通模型共 627 次尝试。已在运行的 Terra 保留其完整版本计划；新增模型先做主表，不先铺开额外版本。

- GPT-6.1 Terra：固定主表已完成，额外版本因预算预留停止。
- GPT-6.1 Sol：low reasoning，每次总输出最多 2048 tokens；仍核对 96/384 非推理输出限额。隐藏推理同样计费。保留原生调用 ID 和加密推理上下文。
- GPT-5.6 Sol：none reasoning，代表另一代大模型。
- GPT-4.1、GPT-4.1 mini：固定 2025-04-14 快照。虽已列为 deprecated，当前组织仍可访问，作为历史系列对照；不将别名替换成新模型。
- GPT-5.4 mini：固定 2026-03-17 快照，none reasoning。
- GPT-6 Luna：普通 Responses 接口，可生成工具动作。
- GPT-6 Luna / Decisions：只做直接三分类，固定 209 题与全证据参考各一遍，共 418 次；不伪造工具参数，Table 1 的搜证列标记不适用。

Decisions 当前仅支持 GPT-6 Luna，不是所有模型共用的另一种端点。其输出不收费，但输入、缓存写入和重发均须计入预算；本实验不自动重试。官方说明：[Decisions](https://developers.openai.com/api/docs/guides/decisions)、[模型与价格](https://developers.openai.com/api/docs/pricing)。

配置见 [模型清单](openai-models.json)，每个新模型先跑最多 0.8 美元的接口试验（同时受其总拨款约束）。较小模型先启动，最多三个新增 API worker；不新增本地 GPU 申请，不释放原四卡。模型的单独拨款与父账本的直接花费合计始终受 60 美元保护，余下 20 美元不纳入本轮运行。

scripts/run_openai_suite.py 执行队列，scripts/score_api_results.py 严格检查固定题组完整性。全部固定题目和所支持设置完成后，才更新公开成绩和 paper/tables/benchmark_results.tex。Decisions 的两种直接设置单独核对，缺少 native agent 设置不能记成 0 分。API 的网络时延、CPU 工具、原生采样与推理 token 会在论文中说明，不能与本地 GPU 速度作等硬件排名。

## 已完成结果与客户端恢复（2026-11-03 17:58 EST）

固定 209 族：Terra 无工具 65.1%、有工具 70.8%；Luna Decisions 无工具 55.5%、全证据参考 56.5%，418 次已全部完成，保守费用核算 0.141703 美元。成绩衡量未审核的源标签一致率，不能称为证据充分性准确率。完整结果已进入主表、GitHub 与 Overleaf。

GPT-4.1 mini 在 197/627 次完成后被客户端 token 预计数与响应 usage 的差异中断。修复在每次校验前持久保存响应，并按完整 8192 输入限额预留费用；模型可见内容、输出限额和题库均不变。修复通过 30 项离线检查。原 197 条答案与首段运行清单按哈希保留；只补做剩余题目，已付费且可恢复的响应从缓存复用。旧客户端未保存的一次响应最多需要重新调用一次；恢复拨款额外冻结 1.11 美元，覆盖该调用按已公布最大上下文和输出长度计算的保守上界，不能视为实际账单。

恢复入口 scripts/resume_openai_mini.py 在主队列结束后尝试申请 6.11 美元的同一总账拨款（含上述 1.11 美元保留），没有余款则停止，不越过 60 美元总上限。恢复运行拒绝更换模型、题目、工具、轮数或工具预算，并逐条验证旧答案没有改变。独立 recovery_status.json 避免把主队列中已中断的状态误报为完成。
