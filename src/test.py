from causalengine import causal_diagnose
from moldexconnector import report_extract

# 1. 诊断 Run 01 ( baseline )
run01_data = report_extract(
    r"D:\Zhejiang_University\srtp\moldex_trying\MDXProject20261011\Report\Run01\Excel Report\ReportData.json"
)
diag01 = causal_diagnose(run01_data)
print("=== RUN 01 因果诊断输出 ===")
print(diag01["llm_prompt_context"])

# 2. 诊断 Run 02 ( 优化后 )
# run02_data = report_extract(
#     r"D:\Zhejiang_University\srtp\moldex_trying\MDXProject20261011\Report\Run02\Excel Report\ReportData.json"
# )
# diag02 = causal_diagnose(run02_data)
# print("=== RUN 02 因果诊断输出 ===")
# print(diag02["llm_prompt_context"])
