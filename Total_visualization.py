import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 설정값
# =========================
FILE_PATH = "EV6/bms_01241248850_2023-02.csv"
WEEK_IDX = 0

GAP_THRESHOLD = "10s"
INTERPOLATION_GAP_THRESHOLD = "15min"

CHARGING_MIN_DURATION = pd.Timedelta(minutes=5)
REST_MIN_DURATION = pd.Timedelta(minutes=10)
LONG_GAP_AS_REST = pd.Timedelta(minutes=5)
REST_CURRENT_LIMIT = 5  # ±5 A

# y축 범위 계산에서 제외할 state
# 실제 plot에서는 Other를 legend에는 남기되, 선은 투명하게 처리한다.
YLIM_STATES = ["Charging", "Rest"]

STATE_COLORS = {
    "Charging": "red",
    "Rest": "blue",
    "Other": "none",
}

PLOT_SPECS = [
    {
        "y_col": "soc",
        "y_label": "SoC (%)",
        "zero_line": False,
    },
    {
        "y_col": "ext_temp",
        "y_label": "External Temp (°C)",
        "zero_line": False,
    },
    {
        "y_col": "mod_temp_avg",
        "y_label": "Module Avg Temp (°C)",
        "zero_line": False,
    },
    {
        "y_col": "pack_current",
        "y_label": "Current (A)",
        "zero_line": True,
    },
    {
        "y_col": "pack_volt",
        "y_label": "Voltage (V)",
        "zero_line": False,
    },
]


# =========================
# 데이터 전처리
# =========================
def load_bms_data(file_path):
    """CSV 파일을 불러오고 time 기준으로 정렬한다."""
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def parse_temp_list(temp_str):
    """mod_temp_list 문자열에서 숫자만 추출해 float 리스트로 반환한다."""
    if pd.isna(temp_str):
        return []

    nums = re.findall(r"-?\d+\.?\d*", str(temp_str).strip())
    return [float(num) for num in nums]


def calc_avg_temp(temp_str):
    """mod_temp_list의 평균 온도를 계산한다."""
    temps = parse_temp_list(temp_str)

    if len(temps) == 0:
        return np.nan

    return np.mean(temps)


def add_time_gap_column(df):
    """연속성 판단을 위한 시간 간격 컬럼을 추가한다."""
    df = df.copy()
    df["dt"] = df["time"].diff()
    return df


def get_valid_groups(df, group_col, candidate_col, min_duration):
    """candidate 조건이 일정 시간 이상 유지된 group 번호만 반환한다."""
    summary = (
        df.groupby(group_col)
        .agg(
            is_candidate=(candidate_col, "first"),
            start_time=("time", "first"),
            end_time=("time", "last"),
        )
        .reset_index()
    )

    summary["duration"] = summary["end_time"] - summary["start_time"]

    valid_groups = summary[
        (summary["is_candidate"] == True)
        & (summary["duration"] >= min_duration)
    ][group_col]

    return valid_groups


def add_state_columns(df):
    """
    Charging, Rest, Other 상태를 계산한다.

    우선순위:
    1. Charging
    2. Rest
    3. Other
    """
    df = add_time_gap_column(df)
    gap_break = df["dt"] > pd.Timedelta(seconds=10)

    df["charge_candidate"] = (
        (df["pack_current"] < 0)
        & (df["speed"] == 0)
        & (df["acceleration"] == 0)
    )
    df["charge_group"] = (
        (df["charge_candidate"] != df["charge_candidate"].shift()) | gap_break
    ).cumsum()

    charging_groups = get_valid_groups(
        df=df,
        group_col="charge_group",
        candidate_col="charge_candidate",
        min_duration=CHARGING_MIN_DURATION,
    )

    df["rest_candidate"] = df["pack_current"].between(
        -REST_CURRENT_LIMIT,
        REST_CURRENT_LIMIT,
        inclusive="both",
    )
    df["rest_group"] = (
        (df["rest_candidate"] != df["rest_candidate"].shift()) | gap_break
    ).cumsum()

    rest_groups = get_valid_groups(
        df=df,
        group_col="rest_group",
        candidate_col="rest_candidate",
        min_duration=REST_MIN_DURATION,
    )

    df["state"] = "Other"
    df.loc[df["charge_group"].isin(charging_groups), "state"] = "Charging"
    df.loc[
        df["rest_group"].isin(rest_groups) & (df["state"] != "Charging"),
        "state",
    ] = "Rest"

    # 긴 log gap은 Rest로 처리한다.
    df.loc[df["dt"] >= LONG_GAP_AS_REST, "state"] = "Rest"

    return df


def add_derived_columns(df):
    """시각화에 필요한 파생 컬럼을 추가한다."""
    df = df.copy()

    if "mod_temp_list" in df.columns:
        df["mod_temp_avg"] = df["mod_temp_list"].apply(calc_avg_temp)

    return df


def preprocess_data(file_path):
    """전체 전처리 파이프라인."""
    df = load_bms_data(file_path)
    df = add_state_columns(df)
    df = add_derived_columns(df)
    return df


# =========================
# 주간 구간 생성
# =========================
def make_weekly_ranges(df):
    """전체 기간을 달력 기준 7일 단위로 나눈다."""
    start_time = df["time"].min()
    end_time = df["time"].max()

    window_start = start_time.normalize()
    weekly_ranges = []

    while window_start <= end_time:
        window_end = window_start + pd.Timedelta(days=7)
        weekly_ranges.append((window_start, window_end))
        window_start = window_end

    return weekly_ranges


def get_week_data(df, weekly_ranges, week_idx):
    """week_idx에 해당하는 데이터와 시작/종료 시각을 반환한다."""
    if week_idx >= len(weekly_ranges):
        raise ValueError(f"week_idx 범위 초과: 0 ~ {len(weekly_ranges) - 1} 사이로 입력")

    ws, we = weekly_ranges[week_idx]
    week_df = df[(df["time"] >= ws) & (df["time"] < we)].copy()

    return week_df, ws, we


# =========================
# 제목 생성
# =========================
def extract_vehicle_id(file_path):
    """파일명에서 차량/단말 ID를 추출해 EV6_<id> 형식으로 반환한다."""
    stem = Path(file_path).stem
    match = re.search(r"bms_(\d+)_\d{4}-\d{2}", stem)

    if match:
        return f"EV6_{match.group(1)}"

    return stem


def make_week_title(file_path, ws, we):
    """Figure 전체 제목용 차량 ID와 주간 날짜 문자열을 만든다."""
    vehicle_id = extract_vehicle_id(file_path)
    end_date = we - pd.Timedelta(days=1)
    date_range = f"{ws.strftime('%y-%m-%d')} ~ {end_date.strftime('%m-%d')}"
    return f"{vehicle_id}\n{date_range}"


# =========================
# Plot helper 함수
# =========================
def insert_gap_nans(plot_df, time_col, y_col, gap_threshold=GAP_THRESHOLD):
    """큰 시간 공백이 있으면 NaN을 삽입해 matplotlib 선 연결을 끊는다."""
    plot_df = plot_df[[time_col, y_col]].copy()
    plot_df = plot_df.sort_values(time_col).reset_index(drop=True)

    gap_threshold = pd.Timedelta(gap_threshold)
    rows = []

    for i in range(len(plot_df)):
        rows.append(plot_df.iloc[i].to_dict())

        if i >= len(plot_df) - 1:
            continue

        t1 = plot_df.iloc[i][time_col]
        t2 = plot_df.iloc[i + 1][time_col]

        if (t2 - t1) > gap_threshold:
            rows.append({time_col: t1 + pd.Timedelta(seconds=1), y_col: np.nan})

    return pd.DataFrame(rows)


def make_interpolation_segments(
    plot_df,
    time_col,
    y_col,
    gap_threshold=INTERPOLATION_GAP_THRESHOLD,
):
    """큰 log gap 구간을 직선 interpolation 선분으로 만든다."""
    plot_df = plot_df[[time_col, y_col]].copy()
    plot_df = plot_df.sort_values(time_col).reset_index(drop=True)

    gap_threshold = pd.Timedelta(gap_threshold)
    segments = []

    for i in range(len(plot_df) - 1):
        t1 = plot_df.loc[i, time_col]
        t2 = plot_df.loc[i + 1, time_col]
        y1 = plot_df.loc[i, y_col]
        y2 = plot_df.loc[i + 1, y_col]

        if (t2 - t1) >= gap_threshold:
            segments.append(pd.DataFrame({time_col: [t1, t2], y_col: [y1, y2]}))

    return segments


def get_visible_y_values_for_ylim(
    week_df,
    y_col,
    ylim_states=YLIM_STATES,
    interpolation_gap_threshold=INTERPOLATION_GAP_THRESHOLD,
):
    """
    y축 범위 계산용 값만 반환한다.

    원칙:
    - Other state 값은 제외한다.
    - Charging, Rest 값은 포함한다.
    - 실제로 초록색으로 그려지는 interpolation segment의 양 끝 값도 포함한다.
    """
    values = []

    state_df = week_df[week_df["state"].isin(ylim_states)]
    y_state = pd.to_numeric(state_df[y_col], errors="coerce").dropna()
    if not y_state.empty:
        values.append(y_state)

    interpolation_segments = make_interpolation_segments(
        week_df,
        time_col="time",
        y_col=y_col,
        gap_threshold=interpolation_gap_threshold,
    )
    for segment in interpolation_segments:
        y_segment = pd.to_numeric(segment[y_col], errors="coerce").dropna()
        if not y_segment.empty:
            values.append(y_segment)

    if not values:
        return pd.Series(dtype=float)

    return pd.concat(values, ignore_index=True)


def set_data_driven_ylim(
    ax,
    week_df,
    y_col,
    interpolation_gap_threshold=INTERPOLATION_GAP_THRESHOLD,
    padding_ratio=0.12,
):
    """Other를 제외하고 실제로 보고 싶은 데이터에 맞춰 y축 범위를 자동 조정한다."""
    y = get_visible_y_values_for_ylim(
        week_df=week_df,
        y_col=y_col,
        ylim_states=YLIM_STATES,
        interpolation_gap_threshold=interpolation_gap_threshold,
    )

    if y.empty:
        y = pd.to_numeric(week_df[y_col], errors="coerce").dropna()

    if y.empty:
        return

    y_min = y.min()
    y_max = y.max()
    y_range = y_max - y_min

    if y_range == 0:
        padding = max(abs(y_max) * 0.1, 1.0)
    else:
        padding = y_range * padding_ratio

    ax.set_ylim(y_min - padding, y_max + padding)


def plot_interpolation_segments(ax, week_df, y_col, interpolation_gap_threshold):
    """큰 log gap 구간을 초록색 보간 선분으로 표시한다."""
    interpolation_segments = make_interpolation_segments(
        week_df,
        time_col="time",
        y_col=y_col,
        gap_threshold=interpolation_gap_threshold,
    )

    label_added = False
    for segment in interpolation_segments:
        ax.plot(
            segment["time"],
            segment[y_col],
            color="green",
            linewidth=1,
            label="Interpolated gap/Rest" if not label_added else None,
        )
        label_added = True


def plot_state_segments(ax, week_df, y_col, gap_threshold):
    """state별 색상으로 실제 데이터 구간을 표시한다."""
    for state_name, color in STATE_COLORS.items():
        state_df = week_df[week_df["state"] == state_name].copy()

        if state_df.empty:
            continue

        plot_df = insert_gap_nans(
            state_df,
            time_col="time",
            y_col=y_col,
            gap_threshold=gap_threshold,
        )

        ax.plot(
            plot_df["time"],
            plot_df[y_col],
            color=color,
            label=state_name,
        )


def format_week_axis(ax, ws, we):
    """주간 plot의 x축 형식을 설정한다."""
    ax.set_xlim(ws, we)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d (%a)"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)


def plot_week_by_state(
    ax,
    df,
    weekly_ranges,
    week_idx,
    y_col,
    y_label,
    gap_threshold=GAP_THRESHOLD,
    zero_line=False,
    interpolation_gap_threshold=INTERPOLATION_GAP_THRESHOLD,
):
    """특정 week_idx 구간에서 y_col을 state별 색상으로 하나의 ax에 plot한다."""
    week_df, ws, we = get_week_data(df, weekly_ranges, week_idx)

    if week_df.empty:
        ax.set_title(f"Week {week_idx + 1}에는 데이터가 없습니다.")
        return

    if y_col not in week_df.columns:
        ax.set_title(f"'{y_col}' 컬럼 없음")
        return

    plot_interpolation_segments(
        ax=ax,
        week_df=week_df,
        y_col=y_col,
        interpolation_gap_threshold=interpolation_gap_threshold,
    )
    plot_state_segments(
        ax=ax,
        week_df=week_df,
        y_col=y_col,
        gap_threshold=gap_threshold,
    )

    set_data_driven_ylim(
        ax=ax,
        week_df=week_df,
        y_col=y_col,
        interpolation_gap_threshold=interpolation_gap_threshold,
    )

    if zero_line:
        ax.axhline(0, linestyle="--", linewidth=1, color="black")

    format_week_axis(ax, ws, we)

    ax.set_ylabel(y_label, labelpad=8, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True)


def plot_all_weekly_variables(
    df,
    weekly_ranges,
    week_idx,
    plot_specs,
    gap_threshold=GAP_THRESHOLD,
    interpolation_gap_threshold=INTERPOLATION_GAP_THRESHOLD,
):
    """PLOT_SPECS에 정의된 모든 변수를 한 figure 안에 1열 subplot으로 그린다."""
    week_df, ws, we = get_week_data(df, weekly_ranges, week_idx)

    if week_df.empty:
        print(f"Week {week_idx + 1}에는 데이터가 없습니다.")
        return

    n_plots = len(plot_specs)
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(14, 3.0 * n_plots),
        sharex=True,
    )

    if n_plots == 1:
        axes = [axes]

    for ax, spec in zip(axes, plot_specs):
        plot_week_by_state(
            ax=ax,
            df=df,
            weekly_ranges=weekly_ranges,
            week_idx=week_idx,
            y_col=spec["y_col"],
            y_label=spec["y_label"],
            zero_line=spec.get("zero_line", False),
            gap_threshold=gap_threshold,
            interpolation_gap_threshold=interpolation_gap_threshold,
        )

    fig.suptitle(make_week_title(FILE_PATH, ws, we), fontsize=14, y=0.97)
    axes[-1].set_xlabel("Time", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.82, 0.94),
            frameon=True,
            fontsize=9,
        )

    fig.align_ylabels(axes)
    fig.subplots_adjust(
        left=0.13,
        right=0.80,
        top=0.88,
        bottom=0.08,
        hspace=0.18,
    )
    plt.show()


# =========================
# 실행부
# =========================
def main():
    df = preprocess_data(FILE_PATH)
    weekly_ranges = make_weekly_ranges(df)

    plot_all_weekly_variables(
        df=df,
        weekly_ranges=weekly_ranges,
        week_idx=WEEK_IDX,
        plot_specs=PLOT_SPECS,
        gap_threshold=GAP_THRESHOLD,
        interpolation_gap_threshold=INTERPOLATION_GAP_THRESHOLD,
    )


if __name__ == "__main__":
    main()