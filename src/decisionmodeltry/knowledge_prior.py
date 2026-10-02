"""
knowledge_prior.py
知识库先验桥接器 (Causal Knowledge Bridge)。
负责将工业文献三元组与实体归一化结果，翻译为决策模型可计算的物理先验向量 (Prior Vector)
与动作空间掩码 (Action Mask)。
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .schema import CausalPrior, PROCESS_PARAM_NAMES
from .physical_rules import (
    get_material_envelope,
    DEFECT_CAUSAL_SENSITIVITIES,
    PARETO_CONFLICT_RULES,
)


# 极速轻量别名启发式兜底字典（在无需启动大型模型时秒级对齐，或与 BLINK 实体对齐模块协同）
DEFAULT_MATERIAL_MAPPING = {
    "pc": "Polycarbonate (PC)",
    "pc料": "Polycarbonate (PC)",
    "聚碳酸酯": "Polycarbonate (PC)",
    "聚碳酸酯(pc)": "Polycarbonate (PC)",
    "pa66": "Polyamide 66 (PA66-GF30)",
    "pa66-gf30": "Polyamide 66 (PA66-GF30)",
    "尼龙66": "Polyamide 66 (PA66-GF30)",
    "pp": "Polypropylene (PP)",
    "聚丙烯": "Polypropylene (PP)",
    "abs": "Acrylonitrile Butadiene Styrene (ABS)",
}

DEFAULT_DEFECT_MAPPING = {
    "浇口发白": "Gate Blushing",
    "发白": "Gate Blushing",
    "进料口发白": "Gate Blushing",
    "白印": "Gate Blushing",
    "飞边": "Flash",
    "披锋": "Flash",
    "毛边": "Flash",
    "溢料": "Flash",
    "欠注": "Short Shot",
    "缺胶": "Short Shot",
    "注不满": "Short Shot",
    "缩痕": "Sink Marks",
    "缩水": "Sink Marks",
    "缩瘪": "Sink Marks",
    "残余应力": "Residual Stress",
    "内应力": "Residual Stress",
    "翘曲": "Residual Stress",
    "变形": "Residual Stress",
}


class KnowledgePriorBridge:
    """
    工业知识因果先验桥接器
    负责为决策模型构建严密的物理先验约束环境
    """
    def __init__(self, entity_aligner=None):
        self.entity_aligner = entity_aligner

    def canonicalize_entity(self, raw_text: str, is_material: bool = False) -> str:
        """实体归一化，优先调用现有 BLINK 模块，未加载时走工业专家字典"""
        if not raw_text:
            return "Polycarbonate (PC)" if is_material else "Gate Blushing"
            
        clean_text = raw_text.strip().lower()

        # 优先使用外挂传入的 EntityAlign 实例
        if self.entity_aligner is not None:
            try:
                aligned = self.entity_aligner.aligner(raw_text)
                if aligned:
                    return aligned
            except Exception:
                pass

        # 启发式专家字典兜底
        mapping = DEFAULT_MATERIAL_MAPPING if is_material else DEFAULT_DEFECT_MAPPING
        for k, v in mapping.items():
            if k in clean_text:
                return v

        return raw_text.strip()

    def build_causal_prior(
        self,
        raw_material: str,
        raw_defect: str,
        literature_statements: Optional[List[str]] = None
    ) -> CausalPrior:
        """
        根据材料和目标缺陷，合成标准的 CausalPrior 对象
        """
        canonical_material = self.canonicalize_entity(raw_material, is_material=True)
        canonical_defect = self.canonicalize_entity(raw_defect, is_material=False)

        # 1. 获取材料安全加工窗口
        safe_envelope = get_material_envelope(canonical_material)

        # 2. 提取物理因果敏感度先验
        sensitivities = DEFECT_CAUSAL_SENSITIVITIES.get(canonical_defect, {})
        sensitivity_signs: Dict[str, int] = {}
        action_mask: Dict[str, bool] = {}

        for param in PROCESS_PARAM_NAMES:
            sens_val = sensitivities.get(param, 0.0)
            if sens_val > 0.05:
                sensitivity_signs[param] = +1  # 正相关：调大会加剧缺陷
                action_mask[param] = True      # 允许调控
            elif sens_val < -0.05:
                sensitivity_signs[param] = -1  # 负相关：调大能抑制缺陷
                action_mask[param] = True      # 允许调控
            else:
                sensitivity_signs[param] = 0   # 不相关
                action_mask[param] = False     # 动作掩码冻结，杜绝乱调

        # 3. 探查潜在互斥冲突 (Pareto Conflict)
        dominant_conflict: Optional[str] = None
        for rule in PARETO_CONFLICT_RULES:
            if rule["defect_a"] == canonical_defect:
                dominant_conflict = f"{rule['conflict_reason']} (关键冲突参数: {rule['key_param']})"
                break

        # 4. 文献因果陈述证据
        statements = literature_statements or [
            f"对于 {canonical_material} 材料，{canonical_defect} 与机台剪切、压力与热历程密切相关。",
            f"针对 {canonical_defect}，需在工艺窗口严格限制敏感参数，并在 Pareto 边界内协调动作。"
        ]

        return CausalPrior(
            material=canonical_material,
            target_defect=canonical_defect,
            matched_statements=statements,
            safe_envelope=safe_envelope,
            sensitivity_signs=sensitivity_signs,
            action_mask=action_mask,
            dominant_conflict=dominant_conflict
        )
