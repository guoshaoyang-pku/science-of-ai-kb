# KB 定向 Science prototype

从已有 KB 的矛盾出发，三个方向分别研究残差与特征学习、有效学习时间与容量、激活通路与泛化。此版本包含12项研究、7条短 KB；每条关联可读报告、提前预测及原始测量。

| 方向 | 已完成研究 | 具体发现 |
|---|---|---|
| A：残差与特征学习 | A01–A04，1740次训练 | head增长和LN affine更新均非偏移收益必要条件；固定LN几何下hidden weights学习可保留收益。 |
| B：有效时间与容量 | B01–B03，360次optimizer训练，另72条解析probe | 目标相关谱决定固定特征拟合时间；晚期特征学习改变方向，train增益仍可能损害test。 |
| C：激活与泛化 | C01–C05，624个recipe、1136条拟合轨迹 | 小尺度SiLU的偶通路受抑制；初始固定谱能提前预测封存条件的噪声风险曲线。 |

计数对应保存的recipe/cell；C04/C05每recipe包含多个head，不能当作独立目标。研究新增solver调用为0。固定谱、Taylor奇偶分解与bias/variance使用已知理论；当前贡献是受控区分、反例和条件内的前瞻验证，不是统一训练理论。

## 查看研究

- [A：残差与特征学习](directions/A_residual_features/handoff.md)：[A01固定head](directions/A_residual_features/report.md)、[A02新函数OOD](studies/A02_ood_residual/report.md)、[A03 LN与width](studies/A03_ln_width/report.md)、[A04参数路径](studies/A04_ln_affine/report.md)。
- [B：有效时间与容量](directions/B_effective_time_capacity/handoff.md)：[B01谱校准](studies/B01_effective_time/report.md)、[B02真实与切线](studies/B02_nonlinear_clock/report.md)、[B03交换核](studies/B03_kernel_direction/report.md)。
- [C：激活与泛化](directions/C_activation_generalization/handoff.md)：[C01通路](studies/C01_activation_scale/report.md)、[C02干预OOD](studies/C02_parity_ood/report.md)、[C03 hidden边界](studies/C03_hidden_boundary/report.md)、[C04噪声U型](studies/C04_noise_generalization/report.md)、[C05前瞻风险](studies/C05_risk_ood/report.md)。
- [具体KB](kb.json) 与 [总交接](HANDOFF.md)。

## 复算与恢复

在仓库根目录运行setup.sh后：

~~~sh
python tools/verify_science_program.py --download
~~~

首次下载三个v0.2.0测量附件；校验归档及逐文件SHA256，在临时副本重算全部15份analysis/summary，并核对B/C的源码、data、contracts及谱。后续可省略--download，全程不训练或调用模型。只检查完整证据hash可加--hash-only。没有结果附件时脚本会明确提示缺文件。

sources/保存e16/e17成功测量和KB的development导入；sources/manifest.json核对公开版本，original_sha256保留来源身份。恢复研究请使用原study/run.py或run_study.py，先匹配执行源码/data/recipe。成功cell直接复用；不要在已用OOD上改预测后重新称为首次验证。

公开版位于 science-of-ai-kb 的 programs/directed_science；原始数值证据由release附件提供。使用公共metadata变化映射区分原始身份与公开hash；不把缺少原始Git历史的clone称为原预注册历史。旧正式主线和publisher独立保留。
