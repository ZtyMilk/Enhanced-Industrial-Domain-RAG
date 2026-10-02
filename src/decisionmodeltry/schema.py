"""
schema.py
工业注塑工艺决策模型核心数据结构与契约定义。
定义注塑机台状态空间 (State)、调优动作空间 (Action)、因果物理先验 (Causal Prior) 及决策报告 (Decision Report)。
"""

from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import numpy as np


# 核心工艺可控参数白名单与其标准物理量纲
PROCESS_PARAM_NAMES = [
    "mold_temp",    # 模具温度 (℃)
    "melt_temp",    # 熔体温度 / 料筒温度 (℃)
    "inj_speed",    # 射胶速度 (mm/s)
    "inj_press",    # 射胶压力 (MPa)
    "pack_press",   # 保压压力 (MPa)
    "pack_time",    # 保压时间 (s)
    "cool_time",    # 冷却时间 (s)
    "screw_rpm",    # 螺杆转速 (rpm)
    "back_press",   # 背压 (MPa)
]

# 参数的标准量纲与中文映射
PARAM_METADATA = {
    "mold_temp": {"name_zh": "模具温度", "unit": "℃", "step_default": 5.0},
    "melt_temp": {"name_zh": "熔体温度", "unit": "℃", "step_default": 5.0},
    "inj_speed": {"name_zh": "射胶速度", "unit": "mm/s", "step_default": 5.0},
    "inj_press": {"name_zh": "射胶压力", "unit": "MPa", "step_default": 5.0},
    "pack_press": {"name_zh": "保压压力", "unit": "MPa", "step_default": 5.0},
    "pack_time": {"name_zh": "保压时间", "unit": "s", "step_default": 0.5},
    "cool_time": {"name_zh": "冷却时间", "unit": "s", "step_default": 1.0},
    "screw_rpm": {"name_zh": "螺杆转速", "unit": "rpm", "step_default": 5.0},
    "back_press": {"name_zh": "背压", "unit": "MPa", "step_default": 1.0},
}


class ProcessState(BaseModel):
    """
    注塑工艺机台与工况当前状态空间 (State, s_t)
    """
    material: str = Field(..., description="规范化材料名称，如 Polycarbonate (PC)")
    defect_name: str = Field(..., description="规范化缺陷名称，如 Gate Blushing")
    severity: float = Field(default=0.8, ge=0.0, le=1.0, description="缺陷严重程度 [0.0: 无缺陷, 1.0: 严重报废]")
    
    # 当前机台旋钮数值 (Knobs)
    mold_temp: float = Field(default=80.0, description="模具温度 (℃)")
    melt_temp: float = Field(default=280.0, description="熔体/料温 (℃)")
    inj_speed: float = Field(default=70.0, description="射胶速度 (mm/s)")
    inj_press: float = Field(default=90.0, description="射胶压力 (MPa)")
    pack_press: float = Field(default=60.0, description="保压压力 (MPa)")
    pack_time: float = Field(default=5.0, description="保压时间 (s)")
    cool_time: float = Field(default=15.0, description="冷却时间 (s)")
    screw_rpm: float = Field(default=75.0, description="螺杆转速 (rpm)")
    back_press: float = Field(default=8.0, description="背压 (MPa)")

    def to_knob_dict(self) -> Dict[str, float]:
        """提取纯参数数值字典"""
        return {k: getattr(self, k) for k in PROCESS_PARAM_NAMES}

    def to_vector(self, normalize_ranges: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """
        将连续状态转换为神经网络归一化输入向量 [0, 1] 空间，拼接缺陷严重度
        """
        vec = []
        for k in PROCESS_PARAM_NAMES:
            val = getattr(self, k)
            low, high = normalize_ranges.get(k, (0.0, 100.0))
            norm_val = (val - low) / (high - low + 1e-6)
            vec.append(np.clip(norm_val, 0.0, 1.0))
        vec.append(self.severity)
        return np.array(vec, dtype=np.float32)


class CausalPrior(BaseModel):
    """
    从文献图谱中检索提炼出的因果物理先验约束 (Causal Prior Context)
    """
    material: str
    target_defect: str
    matched_statements: List[str] = Field(default_factory=list, description="文献权威因果原句")
    safe_envelope: Dict[str, Tuple[float, float]] = Field(default_factory=dict, description="物理安全边界 {param: (min, max)}")
    sensitivity_signs: Dict[str, int] = Field(default_factory=dict, description="影响极性 {param: +1(恶化) / -1(抑制)}")
    action_mask: Dict[str, bool] = Field(default_factory=dict, description="动作可调节掩码 {param: True(允许调) / False(保持)}")
    dominant_conflict: Optional[str] = Field(default=None, description="潜在互斥矛盾描述")

    def to_prior_vector(self) -> np.ndarray:
        """
        将先验极性与动作掩码压平为固定维度的先验特征向量输入模型
        """
        vec = []
        for k in PROCESS_PARAM_NAMES:
            sign = float(self.sensitivity_signs.get(k, 0))  # -1, 0, 1
            mask = 1.0 if self.action_mask.get(k, False) else 0.0
            vec.extend([sign, mask])
        return np.array(vec, dtype=np.float32)


class TuningAction(BaseModel):
    """
    决策模型输出的调参动作决策空间 (Action, a_t)
    """
    primary_intent: str = Field(..., description="主要调控方向，如 DECREASE_SHEAR_RATE")
    deltas: Dict[str, float] = Field(default_factory=dict, description="各参数建议调整增量 (ΔKnob)")
    target_values: Dict[str, float] = Field(default_factory=dict, description="调优后的绝对目标值 (New Knob)")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="决策置信度 [0, 1]")
    estimated_defect_reduction: float = Field(default=0.8, description="预计缺陷削减率 (0.0~1.0)")
    risk_warnings: List[str] = Field(default_factory=list, description="边界越界或副作用风险预警")


class DecisionReport(BaseModel):
    """
    面向工业工程师或注塑机 MES/PLC 的终极工艺优化执行工单
    """
    timestamp: str
    material: str
    defect: str
    initial_severity: float
    action: TuningAction
    causal_rationale: str = Field(..., description="物理机理因果依据 (浓缩自文献先验)")
    convergence_criteria: str = Field(..., description="试模收敛判据与下一轮反馈判据")
    raw_evidence_sources: List[str] = Field(default_factory=list, description="权威文献溯源出处")
