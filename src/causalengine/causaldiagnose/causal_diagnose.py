"""
Physics-Informed Causal Diagnosis Engine
Evaluates CAE multi-physics states against thermodynamic & rheological first principles.
Generates hard optimization bounds to eliminate LLM hallucinations.
"""

from typing import Any


def causal_diagnose(report_data: dict[str, Any], warpage_target_mm: float = 0.30):
    """
    接收 report_extract 的输出字典，执行因果拓扑诊断。

    返回:
        is_converged (bool): 翘曲是否达标
        active_causal_paths (list): 当前主导的物理致因链
        hard_constraints (dict): 强制锁死大模型的参数可行域
        llm_prompt_context (str): 结构化可直接拼入 Prompt 的因果约束文本
    """
    process_settings = report_data["process_settings"]
    sim = report_data["simulation_results"]
    summary = report_data["summary_results"]

    # 1. 提取核心物理状态 (Mediators & Objectives)
    current_warpage = sim["warpage"]["TotalDisplacement"]["max"]
    res_stress = (
        sim["warpage"].get("ThermallyInducedResidualVonMisesStress", {}).get("max")
        or 0.0
    )
    mold_temp_diff = (
        sim["cooling"].get("MoldTemperatureDifference", {}).get("max") or 0.0
    )

    current_pack_time = process_settings["packing_time_s"]
    gate_freeze_time = summary.get("gate_freeze_time_s") or sim["packing"].get(
        "GateFreezeTime", {}
    ).get("max")

    # 2. 目标收敛判定
    is_converged = current_warpage <= warpage_target_mm

    # 3. 因果逻辑推导与硬约束生成
    active_paths = []
    constraints = {}

    # --- 因果链 1: 浇口冻结门控 (Gate Freeze Mediation) ---
    if gate_freeze_time and current_pack_time < gate_freeze_time:
        active_paths.append(
            f"【因果阻断】保压时间({current_pack_time}s) < 浇口冻结时间({gate_freeze_time}s)。"
            "浇口未封闭即撤压，引发熔体倒流，导致型腔严重失压与体积收缩不均！"
        )
        # 锁死硬下界：保压时间必须覆盖冻结时间 + 5% 余量
        constraints["packing_time_s_min"] = round(gate_freeze_time * 1.05, 1)
    else:
        active_paths.append("【保压有效】保压时间已覆盖浇口冻结，补缩通道正常。")
        if gate_freeze_time:
            # 锁死上界：防止无意义的过量保压浪费周期与能耗
            constraints["packing_time_s_max"] = round(gate_freeze_time * 1.2, 1)

    # --- 因果链 2: 热应力 vs 收缩率主导判定 (Thermal vs Volumetric Dominance) ---
    if mold_temp_diff > 20.0 and res_stress > 10.0:
        active_paths.append(
            f"【热应力主导】动定模温差达 {mold_temp_diff}℃，诱发了高达 {res_stress} MPa 的热残余应力，"
            "为翘曲的第一致因！优先消除温差而非盲目加大压力。"
        )
        constraints["mold_temperature_action"] = "INCREASE"
        constraints["target_mold_temp_range_C"] = [70.0, 75.0]
    else:
        active_paths.append(
            "【比容收缩主导】模温场基本平衡，翘曲主要受体积收缩各向异性控制。"
        )
        constraints["mold_temperature_action"] = "MAINTAIN"

    # 4. 生成可直接喂给大模型（Claude / Qwen）的因果控制 Prompt
    llm_prompt_context = f"""
=== CAUSAL DIAGNOSIS & ACTION BOUNDS ===
- 状态判定: {"【已收敛 / PASS】" if is_converged else "【未达标 / NEED OPTIMIZATION】"}
- 当前实际翘曲量: {current_warpage} mm (设计目标: <= {warpage_target_mm} mm)
- 物理瓶颈因果链:
  * {" ".join(active_paths)}

[PHYSICAL HARD CONSTRAINTS] (大模型严禁违背的可行域边界):
1. 保压时间 (Packing Time): 必须设定为 >= {constraints.get("packing_time_s_min", current_pack_time)} 秒
2. 模具温度 (Mold Temperature): 策略为 {constraints.get("mold_temperature_action")} (建议设定区间: {constraints.get("target_mold_temp_range_C", [process_settings["mold_temperature_C"]])} ℃)
========================================
"""

    return {
        "is_converged": is_converged,
        "current_warpage_mm": current_warpage,
        "active_causal_paths": active_paths,
        "hard_constraints": constraints,
        "llm_prompt_context": llm_prompt_context,
    }
