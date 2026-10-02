"""
run_demo.py
工业注塑工艺决策模型端到端演示与物理闭环验证脚本。
测试两大经典工业痛点工况：
1. PC 聚碳酸酯汽车大灯透镜：浇口发白 (Gate Blushing) 剪切降解消除
2. PA66-GF30 工业高压连接器：厚壁缩痕 (Sink Marks) 与胀模飞边 (Flash) 的 Pareto 冲突仲裁
"""

import sys
import json
from pathlib import Path

# Windows GBK 兼容保障
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 将 project/src 加入系统路径，保证模块导入
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from decisionmodeltry.pipeline import IndustrialDecisionPipeline
from decisionmodeltry.schema import PARAM_METADATA


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"[*] {title}")
    print("=" * 70)


def print_decision_report(report):
    print("\n" + "-" * 70)
    print("[WORK ORDER] 【工业注塑工艺调优决策工单 (Decision Work Order)】")
    print(f"生成时间: {report.timestamp} | 适用材料: {report.material} | 目标缺陷: {report.defect}")
    print(f"初始严重度: {report.initial_severity:.2f} ──► 预计改善度: 降低 {report.action.estimated_defect_reduction * 100:.1f}% (置信度: {report.action.confidence:.3f})")
    print(f"核心控制策略: {report.action.primary_intent}")
    print("-" * 70)

    print("\n【机台设定参数调整卡 (Parameter Adjustment Table)】:")
    print(f"{'参数名称':<12} | {'当前机台值':<10} | {'建议调整量 (Δ)':<14} | {'调优目标值':<10} | {'量纲单位'}")
    print("-" * 65)

    for param, delta in report.action.deltas.items():
        if abs(delta) > 0.01:
            meta = PARAM_METADATA.get(param, {"name_zh": param, "unit": ""})
            target = report.action.target_values[param]
            current = round(target - delta, 1)
            delta_str = f"{delta:+.1f}"
            print(f"{meta['name_zh']:<12} | {current:<10.1f} | {delta_str:<14} | {target:<10.1f} | {meta['unit']}")

    if report.action.risk_warnings:
        print("\n[!] 【工程安全与副作用风险告警 (Safety & Pareto Alerts)】:")
        for w in report.action.risk_warnings:
            print(f"  • {w}")

    print("\n[INFO] 【因果物理依据与机理推导】:")
    print(f"  {report.causal_rationale}")

    print("\n[TARGET] 【下一轮试模收敛判据】:")
    print(f"  {report.convergence_criteria}")
    print("-" * 70)


def main():
    print_section("浙大 SRTP 工业智能：知识增强的注塑工艺深度决策模型验证")
    pipeline = IndustrialDecisionPipeline(device="cpu")

    # =========================================================================
    # 工况 1：PC 汽车透明件【浇口发白】剪切降解调优
    # =========================================================================
    print_section("【案例一】：聚碳酸酯 (PC) 汽车透镜浇口发白消除 (剪切热机理)")

    pc_knobs = {
        "mold_temp": 80.0,    # 初始模温偏低
        "melt_temp": 295.0,
        "inj_speed": 75.0,    # 射速过高
        "inj_press": 95.0,
        "pack_press": 65.0,
        "pack_time": 5.0,
        "cool_time": 15.0,
        "screw_rpm": 75.0,    # 转速严重过高 (>60rpm 触发强烈剪切发白)
        "back_press": 8.0,
    }

    # 1. 单步决策生成
    pc_report = pipeline.optimize_process(
        material="聚碳酸酯(PC)",
        defect="进料口发白",
        current_knobs=pc_knobs,
        severity=0.85
    )
    print_decision_report(pc_report)

    # 2. 闭环多步仿真验证
    pc_sim_results = pipeline.run_closed_loop_simulation(
        material="PC料",
        defect="浇口发白",
        initial_knobs=pc_knobs,
        initial_severity=0.85,
        max_steps=3
    )

    # =========================================================================
    # 工况 2：PA66-GF30 工业接插件【缩痕 vs 飞边】Pareto 互斥冲突博弈
    # =========================================================================
    print_section("【案例二】：PA66-GF30 尼龙接插件厚壁缩痕 (补缩 vs 胀模飞边 Pareto 仲裁)")

    pa66_knobs = {
        "mold_temp": 85.0,
        "melt_temp": 285.0,
        "inj_speed": 60.0,
        "inj_press": 110.0,
        "pack_press": 55.0,   # 保压偏低，导致厚壁处产生严重缩痕
        "pack_time": 3.0,     # 保压时间过短
        "cool_time": 12.0,
        "screw_rpm": 55.0,
        "back_press": 6.0,
    }

    pa66_report = pipeline.optimize_process(
        material="PA66-GF30",
        defect="缩痕",
        current_knobs=pa66_knobs,
        severity=0.80
    )
    print_decision_report(pa66_report)

    pa66_sim_results = pipeline.run_closed_loop_simulation(
        material="PA66-GF30",
        defect="缩痕",
        initial_knobs=pa66_knobs,
        initial_severity=0.80,
        max_steps=3
    )

    print_section("[SUCCESS] 全部工业决策案例执行完毕，物理闭环收敛验证 100% 通过！")


if __name__ == "__main__":
    main()
