"""
simulator.py
高分子注塑成型物理环境与机台状态转移仿真器 (Molding Environment Simulator)。
基于注塑物理传递函数模拟机台调优动作 (Action) 执行后的真实物理响应：
缺陷严重度变化、派生副作用风险（飞边/欠注）及闭环收敛判据。
"""

from typing import Dict, Tuple, Any
import numpy as np
from .schema import ProcessState, TuningAction, PROCESS_PARAM_NAMES


class InjectionMoldingSimulator:
    """
    轻量级注塑工艺物理机台环境模拟器
    提供标准的 Gym 风格状态转移闭环: state, reward, done, info = step(action)
    """
    def __init__(self, initial_state: ProcessState):
        self.state = initial_state.model_copy(deep=True)
        self.history = [self.state.model_copy(deep=True)]
        self.step_count = 0

    def step(self, action: TuningAction) -> Tuple[ProcessState, float, bool, Dict[str, Any]]:
        """
        执行一步调优动作，模拟注塑机物理响应并更新机台状态
        """
        self.step_count += 1
        knobs = self.state.to_knob_dict()
        deltas = action.deltas

        # 1. 更新机台旋钮参数
        for p in PROCESS_PARAM_NAMES:
            if p in deltas:
                knobs[p] += deltas[p]
                setattr(self.state, p, round(knobs[p], 2))

        # 2. 模拟物理机理对缺陷严重度的影响传递函数
        old_severity = self.state.severity
        defect = self.state.defect_name
        delta_sev = 0.0

        if defect == "Gate Blushing":
            # 螺杆转速降低强效削减发白，射速降低中度削减，模温提升微度辅助消除
            d_rpm = deltas.get("screw_rpm", 0.0)
            d_spd = deltas.get("inj_speed", 0.0)
            d_mold = deltas.get("mold_temp", 0.0)

            delta_sev = (d_rpm / 35.0) * 0.55 + (d_spd / 40.0) * 0.30 - (d_mold / 20.0) * 0.15

        elif defect == "Sink Marks":
            # 保压提升与保压时间延长强力抑制缩痕
            d_pack = deltas.get("pack_press", 0.0)
            d_time = deltas.get("pack_time", 0.0)
            delta_sev = -(d_pack / 30.0) * 0.60 - (d_time / 4.0) * 0.40

        elif defect == "Flash":
            # 降低保压与射压有效消除飞边
            d_pack = deltas.get("pack_press", 0.0)
            d_inj = deltas.get("inj_press", 0.0)
            delta_sev = (d_pack / 25.0) * 0.60 + (d_inj / 30.0) * 0.40

        elif defect == "Short Shot":
            # 提升射压、射速与料温消除欠注
            d_inj_p = deltas.get("inj_press", 0.0)
            d_spd = deltas.get("inj_speed", 0.0)
            delta_sev = -(d_inj_p / 30.0) * 0.50 - (d_spd / 30.0) * 0.40

        elif defect == "Residual Stress":
            # 提高模温与充分冷却强效释放应力
            d_mold = deltas.get("mold_temp", 0.0)
            d_cool = deltas.get("cool_time", 0.0)
            delta_sev = -(d_mold / 25.0) * 0.70 - (d_cool / 10.0) * 0.30

        # 新的缺陷严重度更新与物理下限截断
        new_severity = float(np.clip(old_severity + delta_sev, 0.0, 1.0))
        self.state.severity = round(new_severity, 3)

        # 3. 伴生副作用监测 (Side-Effect Risks)
        flash_risk = 0.0
        if self.state.pack_press > 92.0 or self.state.inj_press > 145.0:
            flash_risk = min(1.0, (self.state.pack_press - 92.0) * 0.08 + (self.state.inj_press - 145.0) * 0.05)

        short_shot_risk = 0.0
        if self.state.inj_press < 55.0 or self.state.inj_speed < 25.0:
            short_shot_risk = 0.6

        # 4. 计算综合工艺回报 (Reward)
        # 回报公式：质量改善收益 - 副作用惩罚
        improvement = old_severity - new_severity
        penalty = flash_risk * 0.4 + short_shot_risk * 0.4
        reward = float(round(improvement * 10.0 - penalty * 5.0, 3))

        # 5. 判定是否收敛完成试模 (Done Criteria)
        # 严重度降低至 0.15 以下且无致命副作用判定为合格收敛
        done = (self.state.severity <= 0.15) and (flash_risk < 0.1)

        info = {
            "step": self.step_count,
            "old_severity": old_severity,
            "new_severity": self.state.severity,
            "improvement": round(improvement, 3),
            "flash_risk": round(flash_risk, 3),
            "short_shot_risk": round(short_shot_risk, 3),
            "converged": done
        }

        self.history.append(self.state.model_copy(deep=True))
        return self.state, reward, done, info
