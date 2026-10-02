"""
physical_rules.py
注塑材料物性常数库、物理加工安全可行域 (Safety Processing Envelope)
以及领域物理因果敏感度先验矩阵。
为决策模型提供不可逾越的物理硬约束（Hard Constraints）。
"""

from typing import Dict, Tuple, List, Optional
from .schema import PROCESS_PARAM_NAMES


# 全局参数归一化参考基准 (Min, Max)，用于神经网络特征缩放
GLOBAL_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "mold_temp": (20.0, 150.0),    # 模温：20℃ ~ 150℃
    "melt_temp": (160.0, 360.0),   # 料温：160℃ ~ 360℃
    "inj_speed": (10.0, 150.0),    # 射速：10 ~ 150 mm/s
    "inj_press": (30.0, 180.0),    # 射压：30 ~ 180 MPa
    "pack_press": (20.0, 150.0),   # 保压：20 ~ 150 MPa
    "pack_time": (1.0, 30.0),      # 保压时间：1 ~ 30 s
    "cool_time": (5.0, 60.0),      # 冷却时间：5 ~ 60 s
    "screw_rpm": (20.0, 120.0),    # 转速：20 ~ 120 rpm
    "back_press": (2.0, 25.0),     # 背压：2 ~ 25 MPa
}


# 常见工业注塑材料物理极限与安全工艺窗口 (Safety Envelope)
MATERIAL_PROCESSING_WINDOWS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Polycarbonate (PC)": {
        "mold_temp": (80.0, 120.0),    # PC 为高粘度无定型料，需高模温以释放残余应力
        "melt_temp": (280.0, 320.0),   # 熔体温度区间，>320℃极易热分解
        "inj_speed": (30.0, 80.0),
        "inj_press": (80.0, 140.0),
        "pack_press": (50.0, 95.0),
        "pack_time": (3.0, 10.0),
        "cool_time": (12.0, 30.0),
        "screw_rpm": (30.0, 60.0),     # PC 对剪切敏感，螺杆转速严禁 >65rpm，否则剪切发白/变色
        "back_press": (5.0, 15.0),
    },
    "Polyamide 66 (PA66-GF30)": {
        "mold_temp": (80.0, 110.0),
        "melt_temp": (275.0, 305.0),
        "inj_speed": (40.0, 110.0),
        "inj_press": (90.0, 150.0),
        "pack_press": (60.0, 110.0),
        "pack_time": (3.0, 8.0),
        "cool_time": (10.0, 25.0),
        "screw_rpm": (40.0, 75.0),
        "back_press": (4.0, 12.0),
    },
    "Polypropylene (PP)": {
        "mold_temp": (20.0, 65.0),
        "melt_temp": (190.0, 250.0),
        "inj_speed": (40.0, 120.0),
        "inj_press": (50.0, 110.0),
        "pack_press": (35.0, 80.0),
        "pack_time": (2.0, 8.0),
        "cool_time": (8.0, 20.0),
        "screw_rpm": (50.0, 90.0),
        "back_press": (3.0, 10.0),
    },
    "Acrylonitrile Butadiene Styrene (ABS)": {
        "mold_temp": (50.0, 85.0),
        "melt_temp": (200.0, 250.0),
        "inj_speed": (35.0, 90.0),
        "inj_press": (60.0, 120.0),
        "pack_press": (40.0, 85.0),
        "pack_time": (2.5, 7.0),
        "cool_time": (10.0, 25.0),
        "screw_rpm": (40.0, 70.0),
        "back_press": (4.0, 10.0),
    }
}


# 物理缺陷机理因果先验字典：各参数对特定缺陷的敏感度极性
# +1.0: 该参数增大将剧烈恶化/诱发该缺陷
# -1.0: 该参数增大将强力抑制/改善该缺陷
#  0.0: 无明显一阶物理耦合
DEFECT_CAUSAL_SENSITIVITIES: Dict[str, Dict[str, float]] = {
    "Gate Blushing": {  # 浇口发白 (剪切过热与冷料诱发)
        "screw_rpm": +1.0,     # 螺杆转速高 -> 剪切热剧烈 -> 降解发白
        "inj_speed": +0.8,     # 浇口高剪切速率 -> 熔体破裂与折射率畸变
        "melt_temp": +0.3,     # 熔温过高加剧降解
        "mold_temp": -0.5,     # 提高模温有利于应力松弛与消除白印
        "pack_press": -0.2,
    },
    "Flash": {  # 飞边 / 披锋 (型腔压力超限与胀模)
        "pack_press": +1.0,    # 保压过大胀模
        "inj_press": +0.9,     # 射压过高
        "melt_temp": +0.6,     # 料温过高，熔体流动性过好容易钻入分型面
        "mold_temp": +0.4,
        "inj_speed": +0.5,
    },
    "Short Shot": {  # 欠注 / 缺胶 (充填流动阻力大或保压不足)
        "inj_press": -1.0,     # 提高射压直接抑制欠注
        "pack_press": -0.8,
        "inj_speed": -0.8,     # 提高射速减少充填热损耗
        "melt_temp": -0.7,     # 提高料温增加流动性
        "mold_temp": -0.6,
    },
    "Sink Marks": {  # 缩痕 / 缩瘪 (补缩不足与局部热节过慢冷却)
        "pack_press": -1.0,    # 增大保压是消除缩孔的第一手段
        "pack_time": -0.9,     # 延长保压封口时间
        "mold_temp": +0.4,     # 模温过高会导致冷却收缩率增大
        "melt_temp": +0.3,
        "cool_time": -0.5,
    },
    "Residual Stress": {  # 残余应力 / 翘曲变形 (冷却取向冻结与不均收缩)
        "mold_temp": -1.0,     # 高模温让分子链有充裕时间松弛展开，强效消除内应力
        "pack_press": +0.6,    # 过保压强制挤入，冻结大量残余应力
        "inj_speed": +0.4,
        "cool_time": -0.6,     # 均匀充分冷却可降应力
    }
}


# 典型互斥冲突规则库 (Pareto Conflicts)
# 两个目标在同方向调节下发生对抗
PARETO_CONFLICT_RULES = [
    {
        "defect_a": "Short Shot",
        "defect_b": "Flash",
        "key_param": "inj_press",
        "conflict_reason": "提升射压可彻底消除欠注(Short Shot)，但极易击穿分型面导致飞边溢料(Flash)"
    },
    {
        "defect_a": "Sink Marks",
        "defect_b": "Flash",
        "key_param": "pack_press",
        "conflict_reason": "增大保压与保压时间可补缩消除缩痕(Sink Marks)，但过大保压直接诱发严重披锋(Flash)"
    },
    {
        "defect_a": "Residual Stress",
        "defect_b": "Cooling Cycle (Efficiency)",
        "key_param": "mold_temp",
        "conflict_reason": "提高模温可大幅释放残余应力(Residual Stress)，但会显著延长冷却固化时间，拉低整机成型节拍"
    },
    {
        "defect_a": "Gate Blushing",
        "defect_b": "Short Shot",
        "key_param": "inj_speed",
        "conflict_reason": "降低射胶速度可降低浇口剪切消除发白，但过慢射速会导致远端充型不足甚至欠注"
    }
]


def get_material_envelope(material_name: str) -> Dict[str, Tuple[float, float]]:
    """获取指定材料的安全工艺窗口，未精确收录时回退到 PC 默认值"""
    for mat_key, envelope in MATERIAL_PROCESSING_WINDOWS.items():
        if mat_key.lower() in material_name.lower() or material_name.lower() in mat_key.lower():
            return envelope
    # 默认兜底使用 PC
    return MATERIAL_PROCESSING_WINDOWS["Polycarbonate (PC)"]


def clamp_action_to_envelope(
    current_knobs: Dict[str, float],
    proposed_deltas: Dict[str, float],
    envelope: Dict[str, Tuple[float, float]]
) -> Tuple[Dict[str, float], Dict[str, float], List[str]]:
    """
    【物理电子围栏硬限制 (Safety Clamping)】
    根据当前机台参数、提议调参量，严格裁剪至材料安全窗口内，绝不允许越界。
    
    返回：
        (safe_deltas, target_knobs, warnings)
    """
    safe_deltas: Dict[str, float] = {}
    target_knobs: Dict[str, float] = {}
    warnings: List[str] = []

    for param, cur_val in current_knobs.items():
        delta = proposed_deltas.get(param, 0.0)
        target_val = cur_val + delta

        if param in envelope:
            min_val, max_val = envelope[param]
            if target_val < min_val:
                warnings.append(
                    f"参数 [{param}] 目标值 {target_val:.1f} 低于安全下限 {min_val:.1f}，强制截断至安全下限。"
                )
                target_val = min_val
                delta = target_val - cur_val
            elif target_val > max_val:
                warnings.append(
                    f"参数 [{param}] 目标值 {target_val:.1f} 高于安全上限 {max_val:.1f}，强制截断至安全上限。"
                )
                target_val = max_val
                delta = target_val - cur_val

        safe_deltas[param] = round(delta, 2)
        target_knobs[param] = round(target_val, 2)

    return safe_deltas, target_knobs, warnings
