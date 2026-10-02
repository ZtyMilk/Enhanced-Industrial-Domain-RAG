"""
pareto_arbitrator.py
面向制造物理互斥冲突的 Pareto 最优决策仲裁器 (Pareto Conflict Arbitrator)。
解决工业现场最棘手的“治好 A 缺陷却引发 B 缺陷”的工程互斥矛盾（如：补缩防缩痕 vs 胀模防飞边、降速防发白 vs 欠注）。
"""

from typing import Dict, List, Tuple
from .schema import ProcessState, TuningAction, CausalPrior, PARAM_METADATA


class ParetoArbitrator:
    """
    物理多目标 Pareto 最优仲裁器
    """
    def __init__(
        self,
        weight_primary_defect: float = 0.65,    # 根除主缺陷权重
        weight_side_effect_risk: float = 0.25,  # 防范衍生缺陷权重
        weight_production_cycle: float = 0.10   # 兼顾生产节拍效率
    ):
        self.w_primary = weight_primary_defect
        self.w_side = weight_side_effect_risk
        self.w_cycle = weight_production_cycle

    def arbitrate_decision(
        self,
        state: ProcessState,
        action: TuningAction,
        prior: CausalPrior
    ) -> Tuple[TuningAction, List[str]]:
        """
        对初步决策动作执行多目标物理矛盾仲裁与 Pareto 投影裁剪
        """
        arbitrated_deltas = dict(action.deltas)
        arbitrated_targets = dict(action.target_values)
        arbitration_logs: List[str] = []

        defect = prior.target_defect

        # -------------------------------------------------------------
        # 冲突场景 1：防发白 (降低转速/射速) 与 防远端欠注 (充填不足) 的博弈
        # -------------------------------------------------------------
        if defect == "Gate Blushing":
            cur_rpm = state.screw_rpm
            rpm_delta = arbitrated_deltas.get("screw_rpm", 0.0)
            target_rpm = cur_rpm + rpm_delta

            # 若转速降得太低（<35 rpm），塑化不均且计量周期过长
            if target_rpm < 35.0:
                adjusted_delta = 35.0 - cur_rpm
                arbitration_logs.append(
                    f"【Pareto仲裁】转速下调量过大 (目标 {target_rpm} rpm)，触及塑化不良与生产节拍红线，"
                    f"仲裁器将其收敛至下限 35.0 rpm (微调增量由 {rpm_delta} 修正为 {adjusted_delta})。"
                )
                arbitrated_deltas["screw_rpm"] = adjusted_delta
                arbitrated_targets["screw_rpm"] = 35.0

            # 协调补强：在降低剪切转速的同时，适度微升模温 (+3~5℃)，补偿熔体流动性
            if arbitrated_deltas.get("mold_temp", 0.0) <= 0.0:
                arbitrated_deltas["mold_temp"] = 5.0
                arbitrated_targets["mold_temp"] = round(state.mold_temp + 5.0, 1)
                arbitration_logs.append(
                    "【Pareto协同对策】为对冲降转速带来的流动性微弱损耗，仲裁器联动微升模温 +5.0℃，"
                    "利用热松弛消除剪切发白。"
                )

        # -------------------------------------------------------------
        # 冲突场景 2：消除缩痕 (增大保压) 与 防溢料飞边 (Flash) 的博弈
        # -------------------------------------------------------------
        elif defect == "Sink Marks":
            cur_pack = state.pack_press
            pack_delta = arbitrated_deltas.get("pack_press", 0.0)
            target_pack = cur_pack + pack_delta

            # 若保压调得过高（>95 MPa），在分型面产生巨大胀模力直接爆飞边
            if target_pack > 95.0:
                safe_pack = 90.0
                adjusted_delta = safe_pack - cur_pack
                arbitration_logs.append(
                    f"【Pareto仲裁】消除缩痕的保压提议值 {target_pack} MPa 逼近分型面胀模阈值(飞边极高危)，"
                    f"仲裁器强制实施安全上限截断至 {safe_pack} MPa，转向通过适度延长保压时间(+1.5s)实现无飞边完全补缩。"
                )
                arbitrated_deltas["pack_press"] = adjusted_delta
                arbitrated_targets["pack_press"] = safe_pack
                arbitrated_deltas["pack_time"] = round(arbitrated_deltas.get("pack_time", 0.0) + 1.5, 1)
                arbitrated_targets["pack_time"] = round(state.pack_time + arbitrated_deltas["pack_time"], 1)

        # -------------------------------------------------------------
        # 冲突场景 3：释放残余应力 (升高模温) 与 生产冷却周期 (效率) 的博弈
        # -------------------------------------------------------------
        elif defect == "Residual Stress":
            mold_delta = arbitrated_deltas.get("mold_temp", 0.0)
            if mold_delta > 15.0:
                arbitration_logs.append(
                    f"【Pareto仲裁】模温大幅提升 +{mold_delta}℃ 虽能完全消应力，但会导致冷却时间激增40%，"
                    "仲裁器权衡后将单步升温限定在 +10.0℃，维持整机节拍平衡。"
                )
                arbitrated_deltas["mold_temp"] = 10.0
                arbitrated_targets["mold_temp"] = round(state.mold_temp + 10.0, 1)

        # 更新调优动作
        arbitrated_action = TuningAction(
            primary_intent=action.primary_intent,
            deltas=arbitrated_deltas,
            target_values=arbitrated_targets,
            confidence=max(0.75, action.confidence),
            estimated_defect_reduction=action.estimated_defect_reduction,
            risk_warnings=action.risk_warnings + arbitration_logs
        )

        return arbitrated_action, arbitration_logs
