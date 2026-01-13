AlphaGPT 仓库速读

这是一套“因子挖掘 + 实盘执行”的加密量化系统，核心思路是：用模型自动生成可解释的因子公式，通过回测打分筛选，再把高分公式用于实时扫描与交易执行。整体偏向 Solana meme 生态的数据与交易链路。

代码组织（按功能划分）
- data_pipeline/：数据管线。拉取 Birdeye/DexScreener 的代币与 OHLCV，写入 Postgres/Timescale。
- model_core/：策略挖掘。把原始行情转成特征（factors），定义算子语言（ops），用 Transformer 生成“公式 token 序列”，再用回测评分训练。
- strategy_manager/：实盘策略执行。周期性加载数据、生成信号、风控、下单与持仓管理。
- execution/：交易执行层。封装 Solana RPC + Jupiter 聚合器的报价/下单/签名。
- dashboard/：Streamlit 看板，展示持仓、市场快照与日志。
- paper/、lord/、times.py：研究材料或独立实验脚本（不直接参与主流程）。

主流程（从数据到实盘）
1) data_pipeline 抓取链上/行情数据并入库
2) model_core 训练生成最优公式（best_meme_strategy.json）
3) strategy_manager 读取公式，对 Top N 代币打分
4) risk 过滤流动性等风险，execution 完成交易
5) portfolio 持仓落地到本地 JSON，dashboard 展示

核心思想
- 不是直接预测价格，而是“生成公式→解释执行→回测评分→优化生成器”。
- 公式 = token 序列；token 由“特征 + 算子”组成，StackVM 执行成因子信号。
- 交易层只消费最终信号分数，负责风控与执行。

当前因子与算子一览
- 主流程因子（FeatureEngineer，6 个）
  - ret：对数收益
  - liq_score：流动性/FDV 健康度
  - pressure：买卖力量不平衡
  - fomo：成交量加速度
  - dev：价格偏离均值
  - log_vol：对数成交量
- 扩展因子（AdvancedFactorEngineer，12 个）
  - ret：对数收益
  - liq_score：流动性/FDV 健康度
  - pressure：买卖力量不平衡
  - fomo：成交量加速度
  - dev：价格偏离均值
  - log_vol：对数成交量
  - vol_cluster：波动率聚集
  - momentum_rev：动量反转
  - rel_strength：相对强弱（RSI 类）
  - hl_range：高低价振幅
  - close_pos：收盘在区间位置
  - vol_trend：成交量趋势
- 算子（OPS_CONFIG，12 个）
  - ADD：加法
  - SUB：减法
  - MUL：乘法
  - DIV：除法
  - NEG：取负
  - ABS：绝对值
  - SIGN：符号
  - GATE：门控选择（condition>0 选 x，否则 y）
  - JUMP：极端跳变检测（zscore>3）
  - DECAY：衰减叠加（t + 0.8*lag1 + 0.6*lag2）
  - DELAY1：滞后 1
  - MAX3：当前/滞后1/滞后2 最大值

现状与依赖（实话版）
- 需要外部服务：Postgres、Birdeye API、Solana RPC、Jupiter。
- 缺少依赖清单与 .env 模板；实盘需要私钥配置。
- best_meme_strategy.json 需要先训练生成，仓库默认不带。

Takeaway（可对外复述）
这不是一套“预测模型”，而是一个“自动写因子的系统”：它用 Transformer 生成公式，用回测奖励训练公式生成器，再把高分公式接入链上执行与风控。把“策略研究”和“交易执行”清晰分层，是它最值得借鉴的工程设计。
