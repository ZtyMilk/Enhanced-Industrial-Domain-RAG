"""
decision_model.py
真正的基于 PyTorch 的工业注塑因果决策 Transformer 模型 (Causal Decision Transformer)。
核心机制：
1. 以目标回报 (Target Reward R=1.0) 为指引，将物理状态 (State s) 与文献因果先验 (Prior p) 进行多头自注意力融合；
2. 连续动作头输出参数调节量 (ΔKnob)，结合 Action Mask 硬性屏蔽非相关参数；
3. 意图分类头识别高层工程策略；
4. 评价头 (Critic) 估计缺陷削减期望，转化为决策置信度 (Confidence)。
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (
    ProcessState,
    CausalPrior,
    TuningAction,
    PROCESS_PARAM_NAMES,
    PARAM_METADATA,
)
from .physical_rules import (
    GLOBAL_PARAM_RANGES,
    clamp_action_to_envelope,
)


# 高层工程调机意图类别枚举
INTENT_CLASSES = [
    "DECREASE_SHEAR_HEATING",   # 降低剪切生热与降解（发白对策）
    "BOOST_PACKING_FEED",       # 增强补缩保压（缩痕对策）
    "SUPPRESS_FLASH_PRESSURE",  # 泄压防飞边（披锋对策）
    "INCREASE_FILL_DRIVE",      # 强化流动充填（欠注对策）
    "RELEASE_RESIDUAL_STRESS",  # 热松弛释压（翘曲应力对策）
    "BALANCED_PARETO_TUNING"    # 综合平衡微调
]


class CausalDecisionTransformer(nn.Module):
    """
    工业工艺因果决策 Transformer 神经网络
    """
    def __init__(
        self,
        state_dim: int = 10,        # 9个参数 + 1个严重度
        prior_dim: int = 18,        # 9个参数 × (1个极性 + 1个掩码)
        reward_dim: int = 1,        # 目标期望回报 (R_target)
        hidden_dim: int = 128,      # 隐层维度
        num_heads: int = 4,         # 多头注意力头数
        num_layers: int = 2,        # Transformer 层数
        num_params: int = 9,        # 可控参数量
        num_intents: int = len(INTENT_CLASSES)
    ):
        super().__init__()
        self.num_params = num_params
        self.hidden_dim = hidden_dim

        # 1. 各通道特征映射层 (Embedding Projectors)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.prior_proj = nn.Linear(prior_dim, hidden_dim)
        self.reward_proj = nn.Linear(reward_dim, hidden_dim)

        # 模态类型编码 (Modal Embeddings: Reward=0, State=1, Prior=2)
        self.modal_embed = nn.Embedding(3, hidden_dim)

        # 2. Transformer 核心交互层 (Cross-Modal Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 多任务输出头 (Multi-Task Heads)
        # 3.1 连续动作输出头 (Continuous Action Head) -> Tanh 输出归一化增量 [-1, 1]
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_params),
            nn.Tanh()
        )

        # 3.2 高层意图分类头 (Intent Classification Head)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_intents)
        )

        # 3.3 价值/置信度评价头 (Critic / Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        state_tensor: torch.Tensor,     # (batch, state_dim)
        prior_tensor: torch.Tensor,     # (batch, prior_dim)
        reward_tensor: torch.Tensor,    # (batch, reward_dim)
        action_mask: torch.Tensor       # (batch, num_params) 1=允许调节, 0=锁定
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = state_tensor.size(0)

        # 特征投影
        r_emb = self.reward_proj(reward_tensor) + self.modal_embed(torch.zeros(batch_size, dtype=torch.long, device=state_tensor.device))
        s_emb = self.state_proj(state_tensor) + self.modal_embed(torch.ones(batch_size, dtype=torch.long, device=state_tensor.device))
        p_emb = self.prior_proj(prior_tensor) + self.modal_embed(torch.full((batch_size,), 2, dtype=torch.long, device=state_tensor.device))

        # 序列拼接: [Reward, State, Prior] 长度为 3
        seq = torch.stack([r_emb, s_emb, p_emb], dim=1)  # (batch, 3, hidden_dim)

        # Transformer 自注意力全序列融合
        hidden = self.transformer(seq)  # (batch, 3, hidden_dim)

        # 汇聚上下文表征 (取融合后的 State 节点表征)
        context_rep = hidden[:, 1, :]   # (batch, hidden_dim)

        # 连续动作预测与掩码硬约束
        raw_deltas = self.action_head(context_rep)       # [-1.0, +1.0]
        masked_deltas = raw_deltas * action_mask         # 严格封死无关参数

        # 意图 logits 与置信度
        intent_logits = self.intent_head(context_rep)
        confidence = self.value_head(context_rep)

        return masked_deltas, intent_logits, confidence


class ProcessDecisionEngine:
    """
    面向工业调机的全闭环决策引擎包装器
    负责张量组装、先验注入、物理尺度还原与安全围栏硬约束裁剪
    """
    def __init__(self, model: Optional[CausalDecisionTransformer] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model or CausalDecisionTransformer().to(self.device)
        self.model.eval()

        # 各参数最大单步调参幅度基准 (步长缩放因子)
        self.max_step_scales = {
            "mold_temp": 15.0,    # 单步最大调 ±15℃
            "melt_temp": 10.0,    # 单步最大调 ±10℃
            "inj_speed": 20.0,    # 单步最大调 ±20 mm/s
            "inj_press": 25.0,    # 单步最大调 ±25 MPa
            "pack_press": 20.0,   # 单步最大调 ±20 MPa
            "pack_time": 3.0,     # 单步最大调 ±3 s
            "cool_time": 5.0,     # 单步最大调 ±5 s
            "screw_rpm": 25.0,    # 单步最大调 ±25 rpm
            "back_press": 4.0,    # 单步最大调 ±4 MPa
        }

    def infer_action(
        self,
        state: ProcessState,
        prior: CausalPrior,
        target_reward: float = 1.0  # 默认许愿满分（缺陷完全消除）
    ) -> TuningAction:
        """
        根据当前状态、物理因果先验与目标期望回报，执行前向决策推断
        """
        # 1. 数据向量化
        s_vec = state.to_vector(GLOBAL_PARAM_RANGES)
        p_vec = prior.to_prior_vector()
        r_vec = np.array([target_reward], dtype=np.float32)

        # 动作掩码向量 (9维)
        mask_vec = np.array([
            1.0 if prior.action_mask.get(k, False) else 0.0
            for k in PROCESS_PARAM_NAMES
        ], dtype=np.float32)

        # 转为 PyTorch 张量
        s_tensor = torch.tensor(s_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        p_tensor = torch.tensor(p_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        r_tensor = torch.tensor(r_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_deltas, intent_logits, val = self.model(
                s_tensor, p_tensor, r_tensor, mask_tensor
            )

        deltas_np = raw_deltas.squeeze(0).cpu().numpy()
        intent_idx = int(torch.argmax(intent_logits, dim=-1).item())
        confidence_val = float(val.squeeze().item())

        # 2. 物理尺度还原：将 [-1, 1] 映射到各参数物理量纲步长 (℃, rpm, MPa)
        proposed_deltas: Dict[str, float] = {}
        for i, param in enumerate(PROCESS_PARAM_NAMES):
            # 引入物理因果敏感度方向引导先验 (Physics Alignment)
            sign_prior = prior.sensitivity_signs.get(param, 0)
            scale = self.max_step_scales.get(param, 5.0)

            # 模型原始预测输出
            model_delta = float(deltas_np[i])

            # 如果先验明确指示该参数恶化缺陷（sign=+1），调机动作强力偏向于负向降低 (-1)
            # 反之如果抑制缺陷（sign=-1），调机偏向于正向提升 (+1)
            if sign_prior != 0:
                heuristic_guide = -float(sign_prior) * 0.7  # 负向反馈引导
                combined_delta = 0.5 * model_delta + 0.5 * heuristic_guide
            else:
                combined_delta = model_delta

            scaled_delta = combined_delta * scale
            proposed_deltas[param] = round(scaled_delta, 2)

        # 3. 物理安全电子围栏严格裁剪 (Hard Clamping)
        current_knobs = state.to_knob_dict()
        safe_deltas, target_values, warnings = clamp_action_to_envelope(
            current_knobs=current_knobs,
            proposed_deltas=proposed_deltas,
            envelope=prior.safe_envelope
        )

        # 4. 构建调优动作输出
        intent_str = INTENT_CLASSES[intent_idx]
        estimated_reduction = min(0.95, confidence_val * 0.9 + 0.1)

        return TuningAction(
            primary_intent=intent_str,
            deltas=safe_deltas,
            target_values=target_values,
            confidence=round(confidence_val, 3),
            estimated_defect_reduction=round(estimated_reduction, 3),
            risk_warnings=warnings
        )
