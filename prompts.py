"""Korean prompt templates."""

MAJOR_TEMPLATE = """당신은 응급의학과 전문의입니다. 다음 환자 설명을 바탕으로 대분류를 판단해주세요.

### 환자 설명:
{patient_text}

### 가능한 대분류 목록:
{major_categories}

위 목록 중에서 환자의 증상에 가장 적합한 대분류를 하나만 선택하여 답변해주세요.

### 답변:"""

MINOR_TEMPLATE = """당신은 응급의학과 전문의입니다. 다음 환자 설명을 바탕으로 중분류를 판단해주세요.

### 환자 설명:
{patient_text}

### 대분류:
{major_category}

### 가능한 중분류 목록:
{minor_categories}

위 목록 중에서 환자의 증상에 가장 적합한 중분류를 하나만 선택하여 답변해주세요.

### 답변:"""

KTAS_TEMPLATE = """당신은 응급의학과 전문의입니다. 다음 환자 설명을 바탕으로 KTAS level을 판단해주세요.

### 환자 설명:
{patient_text}

### 대분류:
{major_category}

### 중분류:
{minor_category}

### 가능한 KTAS level:
{ktas_levels}

KTAS 레벨 기준:
- 레벨 1 (소생): 생명이나 사지(팔다리)를 위협하는 즉각적인 처치가 필요한 최우선 순위 상태
- 레벨 2 (응급): 생명이나 신체 기능에 잠재적인 위협이 존재하며, 신속한 처치가 필요한 상태
- 레벨 3 (긴급): 생명이나 기능에 즉각적인 위협은 없으나, 응급 처치가 필요한 상태
- 레벨 4 (준응급): 한두 시간 안에 치료 혹은 재평가가 필요한 상태
- 레벨 5 (비응급): 만성적인 문제이거나 응급실이 아닌 외래 진료로 해결 가능한 비응급 상태

위 목록 중에서 환자의 증상에 가장 적합한 KTAS level을 하나만 선택하여 답변해주세요. 답변은 반드시 1, 2, 3, 4, 5 중 하나의 숫자만 작성하세요.

### 답변:"""


def build_prompt(stage: str, patient_text: str, candidates: list[str], major: str = "", minor: str = "") -> str:
    listed = ", ".join(candidates)
    if stage == "major":
        return MAJOR_TEMPLATE.format(patient_text=patient_text, major_categories=listed)
    if stage == "minor":
        return MINOR_TEMPLATE.format(patient_text=patient_text, major_category=major, minor_categories=listed)
    return KTAS_TEMPLATE.format(
        patient_text=patient_text, major_category=major, minor_category=minor, ktas_levels=listed
    )
