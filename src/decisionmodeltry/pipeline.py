"""
pipeline.py
端到端工业工艺决策流水线 (Industrial Decision Pipeline)。
将实体对齐、物理先验注入、Causal Decision Transformer 前向推理、Pareto 最优仲裁与仿真验证
全部无缝串联，输出结构化调机工单 (Decision Report)。
"""

import datetime
from typing import Dict, Any, Optional, List

from .schema import ProcessState, CausalPrior, TuningAction, DecisionReport, PARAM_METADATA
from .knowledge_prior import KnowledgePriorBridge
from .decision_model import ProcessDecisionEngine
from .pareto_arbitrator import ParetoArbitrator
from .simulator import InjectionMoldingSimulator


class IndustrialDecisionPipeline:
    """
    工业注塑工艺决策模型总控流水线
    """
    def __init__(self, device: str = "cpu", entity_aligner=None):
        self.bridge = KnowledgePriorBridge(entity_aligner=entity_aligner)
        self.engine = ProcessDecisionEngine(device=device)
        self.arbitrator = ParetoArbitrator()

    def optimize_process(
        self,
        material: str,
        defect: str,
        current_knobs: Dict[str, float],
        severity: float = 0.85,
        target_reward: float = 1.0,
        literature_evidence: Optional[List[str]] = None
    ) -> DecisionReport:
        """
        单步工业工艺决策优化核心入口
        """
        # 1. 知识库先验桥接与实体归一化
        prior = self.bridge.build_causal_prior(
            raw_material=material,
            raw_defect=defect,
            literature_statements=literature_evidence
        )

        # 2. 构造当前机台物理状态
        state = ProcessState(
            material=prior.material,
            defect_name=prior.target_defect,
            severity=severity,
            **current_knobs
        )

        # 3. 决策 Transformer 模型前向求解初始动作
        raw_action = self.engine.infer_action(
            state=state,
            prior=prior,
            target_reward=target_reward
        )

        # 4. Pareto 物理矛盾仲裁与安全投影
        final_action, arbitration_notes = self.arbitrator.arbitrate_decision(
            state=state,
            action=raw_action,
            prior=prior
        )

        # 5. 生成因果依据与试模收敛判据
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        causal_rationale = (
            f"基于文献因果机理：针对 {prior.material} 材料的 {prior.target_defect} 缺陷，"
            f"主要病因为机台特定参数偏离安全加工窗口。决策模型执行 [{final_action.primary_intent}] 策略，"
            f"对关键敏感参数实施定向调节并锁定非相关参数。多目标仲裁器平衡了潜在副作用，保证动作安全收敛。"
        )
        if prior.dominant_conflict:
            causal_rationale += f" [重点协调矛盾: {prior.dominant_conflict}]"

        convergence_criteria = (
            f"机台设定参数调整完毕后，执行 3 模连续空注稳定熔体流场；随后取第 4~5 模制品进行外观质检。"
            f"收敛达标判定：{prior.target_defect} 缺陷严重度降低至 0.15 以下，且分型面无飞边溢料，"
            f"各向异性收缩率符合设计公差要求。"
        )

        return DecisionReport(
            timestamp=timestamp_str,
            material=prior.material,
            defect=prior.target_defect,
            initial_severity=severity,
            action=final_action,
            causal_rationale=causal_rationale,
            convergence_criteria=convergence_criteria,
            raw_evidence_sources=prior.matched_statements
        )

    def run_closed_loop_simulation(
        self,
        material: str,
        defect: str,
        initial_knobs: Dict[str, float],
        initial_severity: float = 0.85,
        max_steps: int = 3
    ) -> Dict[str, Any]:
        """
        运行多步闭环试模仿真，验证决策模型引导注塑机台从缺陷到满分合格的收敛过程
        """
        # 知识桥接
        prior = self.bridge.build_causal_prior(material, defect)
        init_state = ProcessState(
            material=prior.material,
            defect_name=prior.target_defect,
            severity=initial_severity,
            **initial_knobs
        )

        sim = InjectionMoldingSimulator(initial_state=init_state)
        trajectory_records = []

        print(f"\n=======================================================")
        print(f"[RUN] 开始执行注塑工艺自主调机闭环仿真验证")
        print(f"材料牌号: {prior.material} | 目标缺陷: {prior.target_defect} | 初始严重度: {initial_severity}")
        print(f"=======================================================")

        for step_idx in range(1, max_steps + 1):
            cur_state = sim.state.model_copy(deep=True)

            # 决策模型给出动作
            report = self.optimize_process(
                material=prior.material,
                defect=prior.target_defect,
                current_knobs=cur_state.to_knob_dict(),
                severity=cur_state.severity
            )
            action = report.action

            # 仿真机台执行动作
            next_state, reward, done, step_info = sim.step(action)

            step_record = {
                "step": step_idx,
                "before_severity": step_info["old_severity"],
                "after_severity": step_info["new_severity"],
                "action_intent": action.primary_intent,
                "confidence": action.confidence,
                "adjusted_deltas": {k: v for k, v in action.deltas.items() if abs(v) > 0.01},
                "reward": reward,
                "converged": done
            }
            trajectory_records.append(step_record)

            print(f"\n[第 {step_idx} 轮试模调机]")
            print(f"  - 决策意图: {action.primary_intent} (置信度: {action.confidence:.3f})")
            print(f"  - 关键调优量: {step_record['adjusted_deltas']}")
            print(f"  - 缺陷严重度演化: {step_info['old_severity']:.2f} ──► {step_info['new_severity']:.2f} (改善幅度: {step_info['improvement']:.2f})")
            print(f"  - 试模即时回报 (Reward): {reward:+.3f}")

            if done:
                print(f"  [SUCCESS] 缺陷已完全消除并收敛达标 (Severity <= 0.15)，闭环调机成功！")
                break

        return {
            "converged": trajectory_records[-1]["converged"],
            "total_steps": len(trajectory_records),
            "final_severity": sim.state.severity,
            "final_knobs": sim.state.to_knob_dict(),
            "trajectory": trajectory_records
        }
