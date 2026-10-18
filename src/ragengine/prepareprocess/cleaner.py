import re


def is_noise_chunk(text: str) -> bool:
    """判定切片是否为中英文参考文献、专利法条或元数据标题"""
    clean_text = text.strip()

    # 1. 过滤短文本中的元数据、纯标题、关键词
    if len(clean_text) < 100:
        if "keywords:" in clean_text.lower():
            return True
        if re.match(r"^\d+(\.\d+)*\s+[A-Za-z\u4e00-\u9fa5]", clean_text):
            return True
        if clean_text.startswith("Defect band formation in high pressure die casting"):
            return True

    # 2. 过滤英文学术期刊参考文献 (如 Materials Characterization, 2009, 60: 1432-1441)
    if re.search(r"\[\d+\]\s*[A-Z][a-z]+", clean_text):
        return True
    if re.search(r"\d{4},\s*\d+(\(\d+\))?:\s*\d+[-–]\s*\d+", clean_text):
        return True

    # 3. 过滤中文期刊文献题录与学位论文
    if re.search(r"《.+?》\s*\.\s*\d{4}", clean_text):
        return True
    if "学位论文全文数据库" in clean_text or (
        "特种铸造及有色合金" in clean_text and "期" in clean_text
    ):
        return True

    # 4. 过滤专利法条与公开号列表
    if len(re.findall(r"CN\s*\d+", clean_text)) >= 2:
        return True
    if re.search(r"^\s*(\[\d+\]\s*)?(\d+[\.、]\s*)?根据权利要求", clean_text):
        return True
    return "所述期望缺陷类型包括无缺陷" in clean_text
