
import streamlit as st
import pandas as pd
import math
import random

st.set_page_config(
    page_title="GRID:MATE",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GRID:MATE")
st.subheader("AI 기반 지능형 전력 관리 플랫폼")
st.caption("INTELLIGENT POWER GRID MANAGEMENT SYSTEM")

st.divider()

# 전력 데이터 설정
st.sidebar.title("⚙️ 전력 데이터 설정")

region = st.sidebar.selectbox(
    "분석 지역",
    ["안산시", "서울특별시", "경기도", "인천광역시"]
)

demand = st.sidebar.number_input(
    "현재 예상 전력 수요 (kW)",
    min_value=0,
    value=820
)

supply = st.sidebar.number_input(
    "현재 예상 전력 공급 (kW)",
    min_value=0,
    value=950
)

# 현재 전력 분석
difference = supply - demand

st.header(f"📊 {region} 전력 현황")

c1, c2, c3 = st.columns(3)

c1.metric("예상 전력 수요", f"{demand:,} kW")
c2.metric("예상 전력 공급", f"{supply:,} kW")
c3.metric("전력 수급 차이", f"{difference:+,} kW")

if difference > 0:
    st.success(f"전력 잉여 예상: {difference:,} kW")
elif difference < 0:
    st.warning(f"전력 부족 예상: {abs(difference):,} kW")
else:
    st.info("전력 수요와 공급이 일치합니다.")

st.divider()

# AI 머신러닝 예측
st.header("🤖 AI 전력 수요 예측")

random.seed(42)

now = pd.Timestamp.now().floor("h")

history_dates = pd.date_range(
    end=now,
    periods=14 * 24,
    freq="h"
)

def make_features(date):
    hour = date.hour
    weekday = date.dayofweek
    temperature = 20 + 7 * math.sin(
        2 * math.pi * (hour - 6) / 24
    )

    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        1 if weekday < 5 else 0,
        (temperature - 20) / 10
    ]

def make_target(date):
    hour = date.hour
    weekday = date.dayofweek
    temperature = 20 + 7 * math.sin(
        2 * math.pi * (hour - 6) / 24
    )

    return (
        650
        + 130 * math.sin(2 * math.pi * (hour - 8) / 24)
        + 45 * math.sin(2 * math.pi * (hour - 17) / 24)
        + 35 * (weekday < 5)
        + 5 * (temperature - 20)
        + random.gauss(0, 15)
    )

# 학습 데이터 생성
X = [make_features(d) for d in history_dates]
Y = [make_target(d) - 650 for d in history_dates]

# 학습 및 테스트 데이터 분리
split = len(X) - 24

train_X = X[:split]
train_Y = Y[:split]

test_X = X[split:]
test_Y = Y[split:]

# 선형회귀 모델 직접 학습
weights = [0.0, 0.0, 0.0, 0.0]

learning_rate = 0.03

for epoch in range(1500):
    gradients = [0.0] * 4

    for features, target in zip(train_X, train_Y):
        prediction = sum(
            w * x for w, x in zip(weights, features)
        )

        error = prediction - target

        for i in range(4):
            gradients[i] += error * features[i]

    for i in range(4):
        gradients[i] /= len(train_X)
        weights[i] -= learning_rate * gradients[i]

def predict(features):
    return max(
        0,
        650 + sum(
            w * x for w, x in zip(weights, features)
        )
    )

# 테스트 데이터 평가
test_predictions = [
    predict(features) for features in test_X
]

mae = sum(
    abs(p - y - 650)
    for p, y in zip(test_predictions, test_Y)
) / len(test_Y)

# 향후 24시간 예측
future_dates = pd.date_range(
    start=now + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

future_predictions = [
    predict(make_features(d))
    for d in future_dates
]

forecast = pd.DataFrame({
    "예측 시간": future_dates,
    "AI 예측 전력 수요": future_predictions
})

# 최대 수요 분석
peak_index = future_predictions.index(
    max(future_predictions)
)

peak_time = future_dates[peak_index]
peak_demand = future_predictions[peak_index]

a, b, c = st.columns(3)

a.metric("예측 기간", "24시간")
b.metric("최대 예상 수요", f"{peak_demand:,.0f} kW")
c.metric(
    "최대 수요 예상 시간",
    peak_time.strftime("%m/%d %H시")
)

st.subheader("📈 향후 24시간 AI 전력 예측")

st.line_chart(
    forecast.set_index("예측 시간")
)

st.subheader("💡 AI 분석 보고서")

st.info(
    f"AI 분석 결과, {peak_time.strftime('%m월 %d일 %H시')}에 "
    f"최대 전력 수요인 약 {peak_demand:,.0f} kW가 "
    "발생할 것으로 예측됩니다."
)

st.write(
    f"테스트 데이터 평균 절대 오차(MAE): {mae:.2f} kW"
)

st.caption(
    "※ 임의 생성한 모의 데이터를 이용해 직접 구현한 "
    "선형회귀 모델의 시연 결과입니다. "
    "실제 전력망 예측값이 아닙니다."
)

st.divider()

st.markdown(
    "### ⚡ GRID:MATE"
    "\nAI 기술로 미래 전력산업의 효율적인 에너지 활용을 지원합니다."
)

st.caption("GRID:MATE | MACHINE LEARNING PROTOTYPE 2.0")


# ==========================================
# GRID:MATE 3단계
# AI 태양광 발전량 예측
# ==========================================

st.divider()

st.header("☀️ AI 태양광 발전량 예측")

st.write(
    "태양광 설비 용량과 구름량을 반영하여 "
    "향후 24시간 발전량을 시뮬레이션합니다."
)

# 사용자 설정
solar_capacity = st.slider(
    "태양광 설비 용량 (kW)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

cloudiness = st.slider(
    "예상 구름량 (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=5
)

# 태양광 발전량 계산용 입력 특성
def solar_features(date, cloud):
    hour = date.hour

    sunlight = max(
        0,
        math.sin(math.pi * (hour - 6) / 12)
    )

    cloud_factor = 1 - cloud / 100

    return [
        sunlight,
        sunlight * cloud_factor
    ]

# 모의 학습 데이터 생성
solar_X = []
solar_Y = []

for date in history_dates:
    simulated_cloud = random.randint(10, 90)

    features = solar_features(
        date,
        simulated_cloud
    )

    target = (
        0.25 * features[0]
        + 0.75 * features[1]
    )

    solar_X.append(features)
    solar_Y.append(target)

# 선형회귀 모델 직접 학습
solar_weights = [0.0, 0.0]

solar_learning_rate = 0.08

for epoch in range(1200):

    gradients = [0.0, 0.0]

    for features, target in zip(solar_X, solar_Y):

        prediction = sum(
            w * x
            for w, x in zip(solar_weights, features)
        )

        error = prediction - target

        for i in range(2):
            gradients[i] += error * features[i]

    for i in range(2):
        gradients[i] /= len(solar_X)

        solar_weights[i] -= (
            solar_learning_rate * gradients[i]
        )

# 향후 태양광 발전량 예측
solar_predictions = []

for date in future_dates:

    features = solar_features(
        date,
        cloudiness
    )

    predicted_ratio = sum(
        w * x
        for w, x in zip(solar_weights, features)
    )

    predicted_generation = (
        solar_capacity
        * max(0, predicted_ratio)
    )

    solar_predictions.append(
        predicted_generation
    )

# 결과 데이터 구성
solar_forecast = pd.DataFrame({
    "예측 시간": future_dates,
    "태양광 발전 예상량": solar_predictions
})

# 주요 결과
max_solar = max(solar_predictions)
total_solar = sum(solar_predictions)

s1, s2, s3 = st.columns(3)

s1.metric(
    "설비 용량",
    f"{solar_capacity:,} kW"
)

s2.metric(
    "최대 예상 발전량",
    f"{max_solar:,.1f} kW"
)

s3.metric(
    "24시간 발전량 합계",
    f"{total_solar:,.1f} kWh"
)

# 태양광 그래프
st.subheader("📈 향후 24시간 태양광 발전 예측")

st.line_chart(
    solar_forecast.set_index("예측 시간")
)

# 전력 수요와 태양광 발전량 통합
st.divider()

st.header("⚡ AI 통합 전력 수급 분석")

combined = pd.DataFrame({
    "예상 전력 수요": future_predictions,
    "태양광 발전량": solar_predictions
}, index=future_dates)

combined["태양광 제외 필요 전력"] = (
    combined["예상 전력 수요"]
    - combined["태양광 발전량"]
)

st.line_chart(combined)

# 가장 전력 부담이 큰 시간
peak_net_index = combined[
    "태양광 제외 필요 전력"
].idxmax()

peak_net = combined.loc[
    peak_net_index,
    "태양광 제외 필요 전력"
]

st.subheader("💡 AI 통합 분석 보고서")

st.info(
    f"현재 설정한 시뮬레이션 조건에서 "
    f"{peak_net_index.strftime('%m월 %d일 %H시')}에 "
    f"태양광 발전량을 제외한 전력 수요가 "
    f"약 {peak_net:,.1f} kW로 가장 높게 예상됩니다."
)

if cloudiness >= 70:
    st.warning(
        "구름량이 높은 조건입니다. "
        "태양광 발전량 감소 가능성에 대비해 "
        "전력 공급 계획을 검토하세요."
    )
elif cloudiness <= 20:
    st.success(
        "구름량이 낮은 조건입니다. "
        "태양광 발전 자원을 활용할 가능성이 높습니다."
    )
else:
    st.info(
        "현재 설정한 구름량 조건에 따라 "
        "태양광 발전량을 예측했습니다."
    )

st.caption(
    "※ 모의 데이터를 학습한 선형회귀 시뮬레이션입니다. "
    "실제 기상 및 태양광 발전소 측정 데이터가 아닙니다."
)


# ==========================================
# GRID:MATE 4단계
# ESS 에너지 저장 최적화 시스템
# ==========================================

st.divider()

st.header("🔋 ESS 에너지 저장 최적화 시스템")

st.write(
    "AI 전력 수요 및 태양광 발전 예측 결과를 활용하여 "
    "배터리의 충전과 방전 시점을 계산합니다."
)

# ESS 설정
st.subheader("⚙️ ESS 배터리 설정")

e1, e2 = st.columns(2)

with e1:
    battery_capacity = st.slider(
        "ESS 배터리 용량 (kWh)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100
    )

    initial_soc = st.slider(
        "초기 배터리 충전 상태 (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )

with e2:
    max_power = st.slider(
        "최대 충전·방전 출력 (kW)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )

    efficiency = st.slider(
        "배터리 충·방전 효율 (%)",
        min_value=70,
        max_value=100,
        value=95,
        step=1
    )


# 배터리 초기 상태
soc = battery_capacity * initial_soc / 100
eff = efficiency / 100

st.write("🔍 ESS 설정 확인")
st.write("배터리 용량:", battery_capacity, "kWh")
st.write("초기 배터리 잔량:", soc, "kWh")
st.write("최대 충전·방전 출력:", max_power, "kW")

# 시간대별 태양광 제외 순수 전력 수요 계산
net_demands = [
    float(future_predictions[i]) - float(solar_predictions[i])
    for i in range(24)
]

# 양수인 전력 수요만 추출
positive_demands = sorted(
    [d for d in net_demands if d > 0]
)

# 상위 25% 수요 시간대를 피크 대응 대상으로 설정
if positive_demands:
    peak_threshold = positive_demands[
        int(0.75 * (len(positive_demands) - 1))
    ]
else:
    peak_threshold = 0

# ESS 최적화 시뮬레이션
ess_results = []

for i in range(24):

    demand_value = float(future_predictions[i])
    solar_value = float(solar_predictions[i])

    net_demand = demand_value - solar_value

    charge = 0.0
    discharge = 0.0
    export = 0.0

    # 태양광 잉여 전력이 있으면 충전
    if net_demand < 0:

        surplus = abs(net_demand)
        available_space = battery_capacity - soc

        charge = min(
            surplus,
            max_power,
            available_space / eff
        )

        charge = max(0, charge)

        soc += charge * eff
        export = surplus - charge

    # 전력 수요가 피크 기준보다 높을 때 우선 방전
    else:

        peak_excess = max(
            0,
            net_demand - peak_threshold
        )

        available_energy = soc * eff

        discharge = min(
            peak_excess,
            max_power,
            available_energy
        )

        discharge = max(0, discharge)

        soc -= discharge / eff
        
        
    
    # ESS 사용 후 외부 전력 필요량
    grid_import = max(
        0,
        net_demand + charge - discharge
    )

    ess_results.append({
        "시간": future_dates[i],
        "예상 전력 수요(kW)": demand_value,
        "태양광 발전(kW)": solar_value,
        "ESS 충전(kW)": charge,
        "ESS 방전(kW)": discharge,
        "배터리 잔량(kWh)": soc,
        "외부 전력 필요량(kW)": grid_import,
        "잉여 전력(kW)": export
    })

# 결과 데이터 생성
ess_df = pd.DataFrame(ess_results)

# 주요 성과 계산
before_peak = max(
    max(0, net_demands[i])
    for i in range(24)
)

after_peak = ess_df[
    "외부 전력 필요량(kW)"
].max()

peak_reduction = before_peak - after_peak

# ==========================================
# GRID:MATE 피크 시간대 자동 탐지
# ==========================================

# ESS 사용 전 최대 전력 수요 시간
peak_before_idx = max(
    range(len(net_demands)),
    key=lambda i: net_demands[i]
)

peak_before_time = future_dates[peak_before_idx]

# ESS 사용 후 최대 외부 전력 필요 시간
peak_after_idx = int(
    ess_df["외부 전력 필요량(kW)"].idxmax()
)

peak_after_time = ess_df.loc[
    peak_after_idx, "시간"
]

# 피크 대응 시간 표시
ess_df["피크 대응 시간"] = [
    "피크 시간" if net_demands[i] >= peak_threshold
    and net_demands[i] > 0 else ""
    for i in range(24)
]

# 피크 시간대 결과 출력
st.subheader("🚨 전력 피크 시간대 자동 탐지")

p1, p2 = st.columns(2)

p1.metric(
    "ESS 사용 전 피크 시간",
    peak_before_time.strftime("%m/%d %H시")
)

p2.metric(
    "ESS 사용 후 피크 시간",
    peak_after_time.strftime("%m/%d %H시")
)

st.info(
    f"전력 수요 상위 25% 기준: {peak_threshold:.1f} kW"
)


total_charge = ess_df["ESS 충전(kW)"].sum()
total_discharge = ess_df["ESS 방전(kW)"].sum()


# 핵심 결과 표시
st.divider()

st.subheader("📊 ESS 최적화 분석 결과")

r1, r2, r3 = st.columns(3)

r1.metric(
    "총 충전 에너지",
    f"{total_charge:.1f} kWh"
)

r2.metric(
    "총 방전 에너지",
    f"{total_discharge:.1f} kWh"
)

r3.metric(
    "최대 전력 부담 감소",
    f"{peak_reduction:.1f} kW"
)


# 피크 시간대 시각화
st.subheader("📈 ESS 사용 후 전력 부담 및 피크 시간")

peak_chart = ess_df.set_index("시간")[
    ["외부 전력 필요량(kW)"]
].copy()

# 최대 전력 부담이 발생한 지점만 표시
peak_chart["최대 부담 발생 지점"] = float("nan")

peak_chart.loc[
    peak_after_time,
    "최대 부담 발생 지점"
] = after_peak

st.line_chart(peak_chart)

# 충전 상태 그래프
st.subheader("🔋 시간대별 배터리 충전 상태")

soc_chart = ess_df.set_index("시간")[
    ["배터리 잔량(kWh)"]
]

st.line_chart(soc_chart)

# 충전 방전 그래프
st.subheader("⚡ 시간대별 ESS 충·방전 계획")

power_chart = ess_df.set_index("시간")[
    ["ESS 충전(kW)", "ESS 방전(kW)"]
]

st.bar_chart(power_chart)

# ESS 사용 전후 비교
st.subheader("📉 ESS 사용 전후 전력 부담 비교")

comparison = pd.DataFrame({
    "ESS 사용 전": [
        before_peak
    ],
    "ESS 사용 후": [
        after_peak
    ]
}, index=["최대 외부 전력 필요량"])

st.bar_chart(comparison)

# 상세 데이터
st.subheader("📋 시간대별 ESS 운영 계획")

display_df = ess_df.copy()

display_df["시간"] = (
    display_df["시간"].dt.strftime("%m/%d %H시")
)

st.dataframe(
    display_df.round(1),
    use_container_width=True
)

# AI 분석 보고서
st.divider()

st.header("💡 GRID:MATE 에너지 최적화 보고서")

if total_charge > 0:
    st.success(
        f"예측 기간 중 총 {total_charge:.1f} kWh를 "
        "ESS에 충전하도록 계획했습니다."
    )
else:
    st.info(
        "예측 기간에 ESS를 충전할 수 있는 "
        "태양광 잉여 전력이 발생하지 않았습니다."
    )

if total_discharge > 0:
    st.info(
        f"총 {total_discharge:.1f} kWh를 방전하도록 "
        "계획했습니다."
    )

if peak_reduction > 0:
    st.success(
        f"ESS 활용으로 최대 외부 전력 필요량이 "
        f"{peak_reduction:.1f} kW 감소하는 "
        "시뮬레이션 결과가 나타났습니다."
    )
else:
    st.warning(
        "현재 설정에서는 최대 전력 부담 감소 효과가 "
        "나타나지 않았습니다. 배터리 설정과 발전량을 확인하세요."
    )

st.caption(
    "※ AI 예측 결과를 활용한 규칙 기반 ESS 운영 시뮬레이션입니다. "
    "1시간 단위 계산이며, 실제 전력망 제어 또는 "
    "실시간 최적화 시스템은 아닙니다."
)

st.caption("GRID:MATE | ESS OPTIMIZATION PROTOTYPE 4.0")

# ==========================================
# GRID:MATE 자동 에너지 분석 보고서
# ==========================================

st.divider()
st.header("🧠 GRID:MATE 자동 에너지 분석")

# 최대 전력 부담 감소율 계산
if before_peak > 0:
    reduction_rate = peak_reduction / before_peak * 100
else:
    reduction_rate = 0

# 최소 배터리 잔량
minimum_soc = ess_df["배터리 잔량(kWh)"].min()

# 피크 시간대 ESS 방전량
peak_discharge = sum(
    ess_df.loc[i, "ESS 방전(kW)"]
    for i in range(24)
    if net_demands[i] >= peak_threshold
    and net_demands[i] > 0
)

# 분석 결과 표시
a1, a2, a3 = st.columns(3)

a1.metric(
    "전력 부담 감소율",
    f"{reduction_rate:.1f}%"
)

a2.metric(
    "최소 배터리 잔량",
    f"{minimum_soc:.1f} kWh"
)

a3.metric(
    "피크 시간대 방전량",
    f"{peak_discharge:.1f} kWh"
)

# 자동 분석 문장 생성
st.subheader("📋 에너지 운영 분석 결과")

st.write(
    f"① 예측된 최대 전력 수요 시간은 "
    f"{peak_before_time.strftime('%m/%d %H시')}입니다."
)

st.write(
    f"② ESS 활용 후 최대 외부 전력 필요량은 "
    f"{after_peak:.1f} kW이며, "
    f"기존보다 {peak_reduction:.1f} kW "
    f"감소했습니다."
)

st.write(
    f"③ 최대 전력 부담 감소율은 "
    f"{reduction_rate:.1f}%입니다."
)

st.write(
    f"④ 피크 시간대 ESS 방전량은 "
    f"{peak_discharge:.1f} kWh입니다."
)

# 배터리 상태 진단
if minimum_soc <= battery_capacity * 0.1:
    st.warning(
        "배터리 잔량이 용량의 10% 이하로 "
        "낮아졌습니다. 추가적인 에너지 확보가 필요합니다."
    )
else:
    st.success(
        "배터리 잔량이 용량의 10%를 초과하여 "
        "시뮬레이션 종료 시점까지 유지되었습니다."
    )

st.caption(
    "※ 본 보고서는 예측 데이터와 규칙 기반 ESS "
    "시뮬레이션 결과를 자동 요약한 것입니다."
)


import streamlit as st
import pandas as pd
import math
import random

st.set_page_config(
    page_title="GRID:MATE",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GRID:MATE")
st.subheader("AI 기반 지능형 전력 관리 플랫폼")
st.caption("INTELLIGENT POWER GRID MANAGEMENT SYSTEM")

st.divider()

# 전력 데이터 설정
st.sidebar.title("⚙️ 전력 데이터 설정")

region = st.sidebar.selectbox(
    "분석 지역",
    ["안산시", "서울특별시", "경기도", "인천광역시"]
)

demand = st.sidebar.number_input(
    "현재 예상 전력 수요 (kW)",
    min_value=0,
    value=820
)

supply = st.sidebar.number_input(
    "현재 예상 전력 공급 (kW)",
    min_value=0,
    value=950
)

# 현재 전력 분석
difference = supply - demand

st.header(f"📊 {region} 전력 현황")

c1, c2, c3 = st.columns(3)

c1.metric("예상 전력 수요", f"{demand:,} kW")
c2.metric("예상 전력 공급", f"{supply:,} kW")
c3.metric("전력 수급 차이", f"{difference:+,} kW")

if difference > 0:
    st.success(f"전력 잉여 예상: {difference:,} kW")
elif difference < 0:
    st.warning(f"전력 부족 예상: {abs(difference):,} kW")
else:
    st.info("전력 수요와 공급이 일치합니다.")

st.divider()

# AI 머신러닝 예측
st.header("🤖 AI 전력 수요 예측")

random.seed(42)

now = pd.Timestamp.now().floor("h")

history_dates = pd.date_range(
    end=now,
    periods=14 * 24,
    freq="h"
)

def make_features(date):
    hour = date.hour
    weekday = date.dayofweek
    temperature = 20 + 7 * math.sin(
        2 * math.pi * (hour - 6) / 24
    )

    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        1 if weekday < 5 else 0,
        (temperature - 20) / 10
    ]

def make_target(date):
    hour = date.hour
    weekday = date.dayofweek
    temperature = 20 + 7 * math.sin(
        2 * math.pi * (hour - 6) / 24
    )

    return (
        650
        + 130 * math.sin(2 * math.pi * (hour - 8) / 24)
        + 45 * math.sin(2 * math.pi * (hour - 17) / 24)
        + 35 * (weekday < 5)
        + 5 * (temperature - 20)
        + random.gauss(0, 15)
    )

# 학습 데이터 생성
X = [make_features(d) for d in history_dates]
Y = [make_target(d) - 650 for d in history_dates]

# 학습 및 테스트 데이터 분리
split = len(X) - 24

train_X = X[:split]
train_Y = Y[:split]

test_X = X[split:]
test_Y = Y[split:]

# 선형회귀 모델 직접 학습
weights = [0.0, 0.0, 0.0, 0.0]

learning_rate = 0.03

for epoch in range(1500):
    gradients = [0.0] * 4

    for features, target in zip(train_X, train_Y):
        prediction = sum(
            w * x for w, x in zip(weights, features)
        )

        error = prediction - target

        for i in range(4):
            gradients[i] += error * features[i]

    for i in range(4):
        gradients[i] /= len(train_X)
        weights[i] -= learning_rate * gradients[i]

def predict(features):
    return max(
        0,
        650 + sum(
            w * x for w, x in zip(weights, features)
        )
    )

# 테스트 데이터 평가
test_predictions = [
    predict(features) for features in test_X
]

mae = sum(
    abs(p - y - 650)
    for p, y in zip(test_predictions, test_Y)
) / len(test_Y)

# 향후 24시간 예측
future_dates = pd.date_range(
    start=now + pd.Timedelta(hours=1),
    periods=24,
    freq="h"
)

future_predictions = [
    predict(make_features(d))
    for d in future_dates
]

forecast = pd.DataFrame({
    "예측 시간": future_dates,
    "AI 예측 전력 수요": future_predictions
})

# 최대 수요 분석
peak_index = future_predictions.index(
    max(future_predictions)
)

peak_time = future_dates[peak_index]
peak_demand = future_predictions[peak_index]

a, b, c = st.columns(3)

a.metric("예측 기간", "24시간")
b.metric("최대 예상 수요", f"{peak_demand:,.0f} kW")
c.metric(
    "최대 수요 예상 시간",
    peak_time.strftime("%m/%d %H시")
)

st.subheader("📈 향후 24시간 AI 전력 예측")

st.line_chart(
    forecast.set_index("예측 시간")
)

st.subheader("💡 AI 분석 보고서")

st.info(
    f"AI 분석 결과, {peak_time.strftime('%m월 %d일 %H시')}에 "
    f"최대 전력 수요인 약 {peak_demand:,.0f} kW가 "
    "발생할 것으로 예측됩니다."
)

st.write(
    f"테스트 데이터 평균 절대 오차(MAE): {mae:.2f} kW"
)

st.caption(
    "※ 임의 생성한 모의 데이터를 이용해 직접 구현한 "
    "선형회귀 모델의 시연 결과입니다. "
    "실제 전력망 예측값이 아닙니다."
)

st.divider()

st.markdown(
    "### ⚡ GRID:MATE"
    "\nAI 기술로 미래 전력산업의 효율적인 에너지 활용을 지원합니다."
)

st.caption("GRID:MATE | MACHINE LEARNING PROTOTYPE 2.0")


# ==========================================
# GRID:MATE 3단계
# AI 태양광 발전량 예측
# ==========================================

st.divider()

st.header("☀️ AI 태양광 발전량 예측")

st.write(
    "태양광 설비 용량과 구름량을 반영하여 "
    "향후 24시간 발전량을 시뮬레이션합니다."
)

# 사용자 설정
solar_capacity = st.slider(
    "태양광 설비 용량 (kW)",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100
)

cloudiness = st.slider(
    "예상 구름량 (%)",
    min_value=0,
    max_value=100,
    value=30,
    step=5
)

# 태양광 발전량 계산용 입력 특성
def solar_features(date, cloud):
    hour = date.hour

    sunlight = max(
        0,
        math.sin(math.pi * (hour - 6) / 12)
    )

    cloud_factor = 1 - cloud / 100

    return [
        sunlight,
        sunlight * cloud_factor
    ]

# 모의 학습 데이터 생성
solar_X = []
solar_Y = []

for date in history_dates:
    simulated_cloud = random.randint(10, 90)

    features = solar_features(
        date,
        simulated_cloud
    )

    target = (
        0.25 * features[0]
        + 0.75 * features[1]
    )

    solar_X.append(features)
    solar_Y.append(target)

# 선형회귀 모델 직접 학습
solar_weights = [0.0, 0.0]

solar_learning_rate = 0.08

for epoch in range(1200):

    gradients = [0.0, 0.0]

    for features, target in zip(solar_X, solar_Y):

        prediction = sum(
            w * x
            for w, x in zip(solar_weights, features)
        )

        error = prediction - target

        for i in range(2):
            gradients[i] += error * features[i]

    for i in range(2):
        gradients[i] /= len(solar_X)

        solar_weights[i] -= (
            solar_learning_rate * gradients[i]
        )

# 향후 태양광 발전량 예측
solar_predictions = []

for date in future_dates:

    features = solar_features(
        date,
        cloudiness
    )

    predicted_ratio = sum(
        w * x
        for w, x in zip(solar_weights, features)
    )

    predicted_generation = (
        solar_capacity
        * max(0, predicted_ratio)
    )

    solar_predictions.append(
        predicted_generation
    )

# 결과 데이터 구성
solar_forecast = pd.DataFrame({
    "예측 시간": future_dates,
    "태양광 발전 예상량": solar_predictions
})

# 주요 결과
max_solar = max(solar_predictions)
total_solar = sum(solar_predictions)

s1, s2, s3 = st.columns(3)

s1.metric(
    "설비 용량",
    f"{solar_capacity:,} kW"
)

s2.metric(
    "최대 예상 발전량",
    f"{max_solar:,.1f} kW"
)

s3.metric(
    "24시간 발전량 합계",
    f"{total_solar:,.1f} kWh"
)

# 태양광 그래프
st.subheader("📈 향후 24시간 태양광 발전 예측")

st.line_chart(
    solar_forecast.set_index("예측 시간")
)

# 전력 수요와 태양광 발전량 통합
st.divider()

st.header("⚡ AI 통합 전력 수급 분석")

combined = pd.DataFrame({
    "예상 전력 수요": future_predictions,
    "태양광 발전량": solar_predictions
}, index=future_dates)

combined["태양광 제외 필요 전력"] = (
    combined["예상 전력 수요"]
    - combined["태양광 발전량"]
)

st.line_chart(combined)

# 가장 전력 부담이 큰 시간
peak_net_index = combined[
    "태양광 제외 필요 전력"
].idxmax()

peak_net = combined.loc[
    peak_net_index,
    "태양광 제외 필요 전력"
]

st.subheader("💡 AI 통합 분석 보고서")

st.info(
    f"현재 설정한 시뮬레이션 조건에서 "
    f"{peak_net_index.strftime('%m월 %d일 %H시')}에 "
    f"태양광 발전량을 제외한 전력 수요가 "
    f"약 {peak_net:,.1f} kW로 가장 높게 예상됩니다."
)

if cloudiness >= 70:
    st.warning(
        "구름량이 높은 조건입니다. "
        "태양광 발전량 감소 가능성에 대비해 "
        "전력 공급 계획을 검토하세요."
    )
elif cloudiness <= 20:
    st.success(
        "구름량이 낮은 조건입니다. "
        "태양광 발전 자원을 활용할 가능성이 높습니다."
    )
else:
    st.info(
        "현재 설정한 구름량 조건에 따라 "
        "태양광 발전량을 예측했습니다."
    )

st.caption(
    "※ 모의 데이터를 학습한 선형회귀 시뮬레이션입니다. "
    "실제 기상 및 태양광 발전소 측정 데이터가 아닙니다."
)


# ==========================================
# GRID:MATE 4단계
# ESS 에너지 저장 최적화 시스템
# ==========================================

st.divider()

st.header("🔋 ESS 에너지 저장 최적화 시스템")

st.write(
    "AI 전력 수요 및 태양광 발전 예측 결과를 활용하여 "
    "배터리의 충전과 방전 시점을 계산합니다."
)

# ESS 설정
st.subheader("⚙️ ESS 배터리 설정")

e1, e2 = st.columns(2)

with e1:
    battery_capacity = st.slider(
        "ESS 배터리 용량 (kWh)",
        min_value=500,
        max_value=5000,
        value=2000,
        step=100
    )

    initial_soc = st.slider(
        "초기 배터리 충전 상태 (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )

with e2:
    max_power = st.slider(
        "최대 충전·방전 출력 (kW)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )

    efficiency = st.slider(
        "배터리 충·방전 효율 (%)",
        min_value=70,
        max_value=100,
        value=95,
        step=1
    )


# 배터리 초기 상태
soc = battery_capacity * initial_soc / 100
eff = efficiency / 100

st.write("🔍 ESS 설정 확인")
st.write("배터리 용량:", battery_capacity, "kWh")
st.write("초기 배터리 잔량:", soc, "kWh")
st.write("최대 충전·방전 출력:", max_power, "kW")

# 시간대별 태양광 제외 순수 전력 수요 계산
net_demands = [
    float(future_predictions[i]) - float(solar_predictions[i])
    for i in range(24)
]

# 양수인 전력 수요만 추출
positive_demands = sorted(
    [d for d in net_demands if d > 0]
)

# 상위 25% 수요 시간대를 피크 대응 대상으로 설정
if positive_demands:
    peak_threshold = positive_demands[
        int(0.75 * (len(positive_demands) - 1))
    ]
else:
    peak_threshold = 0

# ESS 최적화 시뮬레이션
ess_results = []

for i in range(24):

    demand_value = float(future_predictions[i])
    solar_value = float(solar_predictions[i])

    net_demand = demand_value - solar_value

    charge = 0.0
    discharge = 0.0
    export = 0.0

    # 태양광 잉여 전력이 있으면 충전
    if net_demand < 0:

        surplus = abs(net_demand)
        available_space = battery_capacity - soc

        charge = min(
            surplus,
            max_power,
            available_space / eff
        )

        charge = max(0, charge)

        soc += charge * eff
        export = surplus - charge

    # 전력 수요가 피크 기준보다 높을 때 우선 방전
    else:

        peak_excess = max(
            0,
            net_demand - peak_threshold
        )

        available_energy = soc * eff

        discharge = min(
            peak_excess,
            max_power,
            available_energy
        )

        discharge = max(0, discharge)

        soc -= discharge / eff
        
        
    
    # ESS 사용 후 외부 전력 필요량
    grid_import = max(
        0,
        net_demand + charge - discharge
    )

    ess_results.append({
        "시간": future_dates[i],
        "예상 전력 수요(kW)": demand_value,
        "태양광 발전(kW)": solar_value,
        "ESS 충전(kW)": charge,
        "ESS 방전(kW)": discharge,
        "배터리 잔량(kWh)": soc,
        "외부 전력 필요량(kW)": grid_import,
        "잉여 전력(kW)": export
    })

# 결과 데이터 생성
ess_df = pd.DataFrame(ess_results)

# 주요 성과 계산
before_peak = max(
    max(0, net_demands[i])
    for i in range(24)
)

after_peak = ess_df[
    "외부 전력 필요량(kW)"
].max()

peak_reduction = before_peak - after_peak

# ==========================================
# GRID:MATE 피크 시간대 자동 탐지
# ==========================================

# ESS 사용 전 최대 전력 수요 시간
peak_before_idx = max(
    range(len(net_demands)),
    key=lambda i: net_demands[i]
)

peak_before_time = future_dates[peak_before_idx]

# ESS 사용 후 최대 외부 전력 필요 시간
peak_after_idx = int(
    ess_df["외부 전력 필요량(kW)"].idxmax()
)

peak_after_time = ess_df.loc[
    peak_after_idx, "시간"
]

# 피크 대응 시간 표시
ess_df["피크 대응 시간"] = [
    "피크 시간" if net_demands[i] >= peak_threshold
    and net_demands[i] > 0 else ""
    for i in range(24)
]

# 피크 시간대 결과 출력
st.subheader("🚨 전력 피크 시간대 자동 탐지")

p1, p2 = st.columns(2)

p1.metric(
    "ESS 사용 전 피크 시간",
    peak_before_time.strftime("%m/%d %H시")
)

p2.metric(
    "ESS 사용 후 피크 시간",
    peak_after_time.strftime("%m/%d %H시")
)

st.info(
    f"전력 수요 상위 25% 기준: {peak_threshold:.1f} kW"
)


total_charge = ess_df["ESS 충전(kW)"].sum()
total_discharge = ess_df["ESS 방전(kW)"].sum()


# 핵심 결과 표시
st.divider()

st.subheader("📊 ESS 최적화 분석 결과")

r1, r2, r3 = st.columns(3)

r1.metric(
    "총 충전 에너지",
    f"{total_charge:.1f} kWh"
)

r2.metric(
    "총 방전 에너지",
    f"{total_discharge:.1f} kWh"
)

r3.metric(
    "최대 전력 부담 감소",
    f"{peak_reduction:.1f} kW"
)


# 피크 시간대 시각화
st.subheader("📈 ESS 사용 후 전력 부담 및 피크 시간")

peak_chart = ess_df.set_index("시간")[
    ["외부 전력 필요량(kW)"]
].copy()

# 최대 전력 부담이 발생한 지점만 표시
peak_chart["최대 부담 발생 지점"] = float("nan")

peak_chart.loc[
    peak_after_time,
    "최대 부담 발생 지점"
] = after_peak

st.line_chart(peak_chart)

# 충전 상태 그래프
st.subheader("🔋 시간대별 배터리 충전 상태")

soc_chart = ess_df.set_index("시간")[
    ["배터리 잔량(kWh)"]
]

st.line_chart(soc_chart)

# 충전 방전 그래프
st.subheader("⚡ 시간대별 ESS 충·방전 계획")

power_chart = ess_df.set_index("시간")[
    ["ESS 충전(kW)", "ESS 방전(kW)"]
]

st.bar_chart(power_chart)

# ESS 사용 전후 비교
st.subheader("📉 ESS 사용 전후 전력 부담 비교")

comparison = pd.DataFrame({
    "ESS 사용 전": [
        before_peak
    ],
    "ESS 사용 후": [
        after_peak
    ]
}, index=["최대 외부 전력 필요량"])

st.bar_chart(comparison)

# 상세 데이터
st.subheader("📋 시간대별 ESS 운영 계획")

display_df = ess_df.copy()

display_df["시간"] = (
    display_df["시간"].dt.strftime("%m/%d %H시")
)

st.dataframe(
    display_df.round(1),
    use_container_width=True
)

# AI 분석 보고서
st.divider()

st.header("💡 GRID:MATE 에너지 최적화 보고서")

if total_charge > 0:
    st.success(
        f"예측 기간 중 총 {total_charge:.1f} kWh를 "
        "ESS에 충전하도록 계획했습니다."
    )
else:
    st.info(
        "예측 기간에 ESS를 충전할 수 있는 "
        "태양광 잉여 전력이 발생하지 않았습니다."
    )

if total_discharge > 0:
    st.info(
        f"총 {total_discharge:.1f} kWh를 방전하도록 "
        "계획했습니다."
    )

if peak_reduction > 0:
    st.success(
        f"ESS 활용으로 최대 외부 전력 필요량이 "
        f"{peak_reduction:.1f} kW 감소하는 "
        "시뮬레이션 결과가 나타났습니다."
    )
else:
    st.warning(
        "현재 설정에서는 최대 전력 부담 감소 효과가 "
        "나타나지 않았습니다. 배터리 설정과 발전량을 확인하세요."
    )

st.caption(
    "※ AI 예측 결과를 활용한 규칙 기반 ESS 운영 시뮬레이션입니다. "
    "1시간 단위 계산이며, 실제 전력망 제어 또는 "
    "실시간 최적화 시스템은 아닙니다."
)

st.caption("GRID:MATE | ESS OPTIMIZATION PROTOTYPE 4.0")

# ==========================================
# GRID:MATE 자동 에너지 분석 보고서
# ==========================================

st.divider()
st.header("🧠 GRID:MATE 자동 에너지 분석")

# 최대 전력 부담 감소율 계산
if before_peak > 0:
    reduction_rate = peak_reduction / before_peak * 100
else:
    reduction_rate = 0

# 최소 배터리 잔량
minimum_soc = ess_df["배터리 잔량(kWh)"].min()

# 피크 시간대 ESS 방전량
peak_discharge = sum(
    ess_df.loc[i, "ESS 방전(kW)"]
    for i in range(24)
    if net_demands[i] >= peak_threshold
    and net_demands[i] > 0
)

# 분석 결과 표시
a1, a2, a3 = st.columns(3)

a1.metric(
    "전력 부담 감소율",
    f"{reduction_rate:.1f}%"
)

a2.metric(
    "최소 배터리 잔량",
    f"{minimum_soc:.1f} kWh"
)

a3.metric(
    "피크 시간대 방전량",
    f"{peak_discharge:.1f} kWh"
)

# 자동 분석 문장 생성
st.subheader("📋 에너지 운영 분석 결과")

st.write(
    f"① 예측된 최대 전력 수요 시간은 "
    f"{peak_before_time.strftime('%m/%d %H시')}입니다."
)

st.write(
    f"② ESS 활용 후 최대 외부 전력 필요량은 "
    f"{after_peak:.1f} kW이며, "
    f"기존보다 {peak_reduction:.1f} kW "
    f"감소했습니다."
)

st.write(
    f"③ 최대 전력 부담 감소율은 "
    f"{reduction_rate:.1f}%입니다."
)

st.write(
    f"④ 피크 시간대 ESS 방전량은 "
    f"{peak_discharge:.1f} kWh입니다."
)

# 배터리 상태 진단
if minimum_soc <= battery_capacity * 0.1:
    st.warning(
        "배터리 잔량이 용량의 10% 이하로 "
        "낮아졌습니다. 추가적인 에너지 확보가 필요합니다."
    )
else:
    st.success(
        "배터리 잔량이 용량의 10%를 초과하여 "
        "시뮬레이션 종료 시점까지 유지되었습니다."
    )

st.caption(
    "※ 본 보고서는 예측 데이터와 규칙 기반 ESS "
    "시뮬레이션 결과를 자동 요약한 것입니다."
)

