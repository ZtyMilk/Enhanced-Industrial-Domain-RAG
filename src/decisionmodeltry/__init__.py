"""
decisionmodeltry
知识增强的工业注塑成型工艺深度决策模型套件。
包含状态空间、因果先验、Causal Decision Transformer、Pareto 物理矛盾仲裁与闭环流水线。
"""

from .schema import (
    ProcessState,
    CausalPrior,
    TuningAction,
    DecisionReport,
    PROCESS_PARAM_NAMES,
    PARAM_METADATA,
)
from .physical_rules import (
    GLOBAL_PARAM_RANGES,
    MATERIAL_PROCESSING_WINDOWS,
    DEFECT_CAUSAL_SENSITIVITIES,
    PARETO_CONFLICT_RULES,
    get_material_envelope,
    clamp_action_to_envelope,
)
from .knowledge_prior import KnowledgePriorBridge
from .decision_model import (
    CausalDecisionTransformer,
    ProcessDecisionEngine,
    INTENT_CLASSES,
)
from .pareto_arbitrator import ParetoArbitrator
from .simulator import InjectionMoldingSimulator
from .pipeline import IndustrialDecisionPipeline

__all__ = [
    "ProcessState",
    "CausalPrior",
    "TuningAction",
    "DecisionReport",
    "PROCESS_PARAM_NAMES",
    "PARAM_METADATA",
    "GLOBAL_PARAM_RANGES",
    "MATERIAL_PROCESSING_WINDOWS",
    "DEFECT_CAUSAL_SENSITIVITIES",
    "PARETO_CONFLICT_RULES",
    "get_material_envelope",
    "clamp_action_to_envelope",
    "KnowledgePriorBridge",
    "CausalDecisionTransformer",
    "ProcessDecisionEngine",
    "INTENT_CLASSES",
    "ParetoArbitrator",
    "InjectionMoldingSimulator",
    "IndustrialDecisionPipeline",
]
