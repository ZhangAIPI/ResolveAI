# 论文：任务说明与实验组织

2026-11-06 更新。

本次修改先让读者理解比较对象，再看结果。参考 MMMU 和 VSI-Bench 的实验组织方式，保留设置、主结果、训练结果及机制分析四个小节。

- **状态验证**：声明指定目标物品的某种缺陷，判断是否存在。正常参考可能来自另一实物，不当作目标的先前状态。
- **身份验证**：声明两组照片是否为同一实物。目前有标签控制为同序列正例与跨类别反例，同类别跨序列候选不强行赋真值。
- **访问设置**：Sufficient、Obtainable、Missing、Unavailable 用于上述两个任务，描述照片是否提供及能否请求。
- **回答方式**：No tools 直接回答；With tools 可以检查、裁剪、比较或请求；全部照片可见结果另列参考。
- **训练方式**：Base、SFT、从 Base 开始的 RL、SFT 后的 RL，定义放正文设置中。
- **指标**：准确率按来源标签；Finished 包含有效弃答；New image 可以是原帧高清版本，不等同新视角或充分证据；时间与输出 token 累计整个交互。

七个主图表采用短 caption，题量、模型配置、预算、分母和 token 含义移到正文。主表用 Condition / Identity、No tools / With tools、Output tokens 等完整名称。

主文把重复的逐模型计数收拢成几个结论：缺图造成的差距；SFT 的完成率与效率收益；这些收益能否转成索证能力；重复图片、失败反馈与实际像素内容分别怎样影响结果。条件弃答公式及完整细目移到补充材料。

新 Figure 3 展示完整 204 条像素内容对照：原图回答36 Supported、32 Refuted；灰图和置乱图各68 Refuted。它是68段已选请求历史上的单次判断检查，不混入主表的原生交互成绩。

参考写法：

- [MMMU（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.pdf)
- [VSI-Bench（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Thinking_in_Space_How_Multimodal_Large_Language_Models_See_Remember_CVPR_2025_paper.pdf)

