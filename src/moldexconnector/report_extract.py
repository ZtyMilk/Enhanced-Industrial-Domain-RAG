import json
from pathlib import Path
from typing import Any


def _parse_val(v: Any):
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() else f
    except (ValueError, TypeError):
        return v


def report_extract(report_direction: str):
    report_path = Path(report_direction)
    if not report_path.exists():
        raise FileNotFoundError(f"No report found in: {report_direction}")
    with open(report_path, "r", encoding="utf-8") as f:
        raw_report = json.load(f)

    cbase = raw_report.get("CBASE", {})
    run_information = {
        "project_name": cbase.get("Value_Project_Name"),
        "run_id": cbase.get("Value_Run_ID"),
        "project_date": cbase.get("Value_Project_Date"),
        "studio_version_name": cbase.get("Value_Studio_VersionName"),
        "run_analysis_sequence": cbase.get("Value_Run_Analysis"),
    }

    # 2. 几何与网格信息 (含几何尺寸、体积及肉厚统计/标准差)
    mesh_information = {
        "run_meshname": cbase.get("Value_Run_MeshName"),
        "mesh_part_dimensions": cbase.get("Value_Mesh_PartDimension"),
        "mesh_part_volume": _parse_val(cbase.get("Value_Mesh_PartVolume")),
        "cold_runner_volume_cc": _parse_val(cbase.get("Value_Mesh_ColdRunnerVolume")),
        "total_volume_cc": _parse_val(cbase.get("Value_Mesh_TotalVolume")),
        "thickness_mm": {
            "max": _parse_val(cbase.get("Max_Mesh_Thickness")),
            "min": _parse_val(cbase.get("Min_Mesh_Thickness")),
            "avg": _parse_val(cbase.get("Avg_Mesh_Thickness")),
            "sd": _parse_val(cbase.get("SD_Mesh_Thickness")),
        },
        "element_count": _parse_val(cbase.get("Value_Mesh_ElementNumber")),
        "part_element_count": _parse_val(cbase.get("Value_Mesh_PartElementNumber")),
        "gate_count": _parse_val(cbase.get("Value_Mesh_GateCount")),
    }

    # 3. 材料物性与手册推荐温区
    material_info = {
        "grade_name": cbase.get("Value_Material_PartGenericName"),
        "polymer_type": cbase.get("Value_Material_PartType"),
        "supplier": cbase.get("Value_Material_PartSupplier"),
        "density_g_cm3": _parse_val(cbase.get("Value_Material_Density")),
        "mfi": cbase.get("Value_Material_MFIInfo"),
        "glass_transition_temp_C": _parse_val(
            cbase.get("Value_Material_GlassTransitionTemperature")
        ),
        "freeze_temp_C": _parse_val(cbase.get("Value_Material_PartFreezeTemperature")),
        "ejection_temp_C": _parse_val(
            cbase.get("Value_Material_PartEjectionTemperature")
        ),
        "recommended_melt_temp_range_C": cbase.get(
            "Value_Material_PartMeltTemperatureRange"
        ),
        "recommended_mold_temp_range_C": cbase.get(
            "Value_Material_PartMoldTemperatureRange"
        ),
    }

    # 4. 实际成型工艺初始设定
    process_settings = {
        "melt_temperature_C": _parse_val(cbase.get("Value_Process_MeltTemperature")),
        "mold_temperature_C": _parse_val(cbase.get("Value_Process_MoldTemperature")),
        "filling_time_s": _parse_val(cbase.get("Value_Process_FillingTime")),
        "packing_time_s": _parse_val(cbase.get("Value_Process_PackingTime")),
        "cooling_time_s": _parse_val(cbase.get("Value_Process_CoolingTime")),
        "mold_open_time_s": _parse_val(cbase.get("Value_Process_MoldOpeningTime")),
        "cycle_time_s": _parse_val(cbase.get("Value_Process_CycleTime")),
        "vp_switch_percent": _parse_val(cbase.get("Value_Process_VPSwitchOver")),
        "machine_max_injection_pressure_MPa": _parse_val(
            cbase.get("Value_Process_MaxInjectionPressure")
        ),
        "machine_max_packing_pressure_MPa": _parse_val(
            cbase.get("Value_Process_MaxPackingPressure")
        ),
        "packing_pressure_profile_percent": cbase.get(
            "Value_Process_PressureOfPackPressure"
        ),
        "packing_time_profile_s": cbase.get("Value_Process_TimeOfPackPressure"),
        "air_temperature_C": _parse_val(cbase.get("Value_Process_AirTemperature")),
    }

    # 5. 递归抓取所有分析阶段的 3D 统计场与 XY 时程曲线 (共 123 个项)
    stages = {"filling": {}, "packing": {}, "cooling": {}, "warpage": {}}

    def _recurse_nodes(node):
        if isinstance(node, dict):
            tag = node.get("Tag")
            if isinstance(tag, str):
                # 兼容 3D 网格极值与 XY 动态时程曲线极值
                max_val = (
                    node.get("Statistics_Max")
                    or node.get("History_Y_YMax")
                    or node.get("Y_SpecificTime")
                )
                min_val = node.get("Statistics_Min") or node.get("History_Y_YMin")
                avg_val = node.get("Statistics_Avg")
                unit = (node.get("Unit") or "").replace("\u00b0", "°").strip()

                for s_key, stage_dict in stages.items():
                    prefix = s_key.capitalize() + "_"
                    if tag.startswith(prefix):
                        field_name = tag[len(prefix) :]
                        stage_dict[field_name] = {
                            "max": _parse_val(max_val),
                            "min": _parse_val(min_val),
                            "avg": _parse_val(avg_val),
                            "unit": unit,
                        }
                        break
            for v in node.values():
                _recurse_nodes(v)
        elif isinstance(node, list):
            for item in node:
                _recurse_nodes(item)

    _recurse_nodes(raw_report)

    # 6. 宏观汇总指标 (含浇口冻结时间关键因果参数)
    summary_results = {
        "volume_filled_percent": 100.0,
        "actual_filling_time_s": stages["filling"].get("MeltFrontTime", {}).get("max"),
        "gate_freeze_time_s": _parse_val(cbase.get("Value_GateFreezeTime")),
        "part_weight_g": _parse_val(cbase.get("Value_Result_TotalPartWeight")),
        "runner_weight_g": _parse_val(cbase.get("Value_Result_TotalRunnerWeight")),
        "max_clamping_force_ton": stages["filling"]
        .get("XY_ClampingForce", {})
        .get("max"),
        "max_sprue_pressure_MPa": stages["filling"]
        .get("XY_SpruePressure", {})
        .get("max"),
    }

    return {
        "run_information": run_information,
        "mesh_information": mesh_information,
        "material_information": material_info,
        "process_settings": process_settings,
        "summary_results": summary_results,
        "simulation_results": stages,
    }


a = report_extract(
    r"D:\Zhejiang_University\srtp\moldex_trying\MDXProject20261011\Report\Run02\Excel Report\ReportData.json"
)


# def aaa(c: dict, i: int):
#     if i > 0:
#         for key, info in c.items():
#             print(f"{'---' * (2 - i)}{key}")
#             aaa(c=info, i=i - 1)


# aaa(c=a, i=2)
