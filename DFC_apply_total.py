import os
import re

import numpy as np
import pandas as pd


# =========================================================
# 0. User settings
# =========================================================
EV6_DIR = "EV6"
FULL_CHARGE_LIST = "filter_results/full_charge_files.csv"

# DFC 결과 CSV 저장 폴더
OUTPUT_DIR = "EV6_DFC"

# Charging / Rest 판정 기준
Log_gap_threshold = "10s"          # 이보다 큰 log gap은 연속 구간을 끊는 기준
Charg_min_duration = "5min"     # Charging 후보가 이 시간 이상 지속되면 Charging으로 인정
Rest_min_duration = "10min"        # Rest 후보가 이 시간 이상 지속되면 Rest로 인정
Long_gap_as_Rest = "5min"          # 이보다 큰 log gap은 Rest로 처리

# Delayed Full Charge 설정
SOC_standby = 80                # SOC가 이 값에 도달한 이후부터 DFC 적용
T_buffer = "60min"            # 다음 주행 시작 몇 시간/분 전에 full charge 완료할지

# Full Charge 판정 기준
Full_charge_soc = 95


# =========================================================
# 1. Data loading and preprocessing
# =========================================================
def load_bms_csv(file_path):
    """
    BMS CSV를 불러오고 time 기준으로 정렬한다.
    """
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["dt"] = df["time"].diff()
    return df


# =========================================================
# 2. State classification: Charging / Rest / Other
# =========================================================
def classify_state(
    df,
    Log_gap_threshold=Log_gap_threshold,
    Charg_min_duration=Charg_min_duration,
    Rest_min_duration=Rest_min_duration,
    Long_gap_as_Rest=Long_gap_as_Rest,
):
    """
    각 log row를 Charging, Rest, Other 중 하나로 분류한다.

    우선순위:
    1. Charging
    2. Rest
    3. Other

    Charging 조건:
    - pack_current < 0
    - speed == 0
    - acceleration == 0
    - 위 조건이 Charg_min_duration 이상 연속 유지

    Rest 조건:
    - pack_current가 -5A ~ 5A 범위
    - 위 조건이 Rest_min_duration 이상 연속 유지

    추가 처리:
    - Long_gap_as_Rest 이상 log가 끊긴 row는 Rest로 처리한다.
      실제 데이터가 없는 휴지 구간을 plot에서 표시하기 위한 목적이다.
    """
    df = df.copy()

    Log_gap_threshold = pd.Timedelta(Log_gap_threshold)
    Charg_min_duration = pd.Timedelta(Charg_min_duration)
    Rest_min_duration = pd.Timedelta(Rest_min_duration)
    Long_gap_as_Rest = pd.Timedelta(Long_gap_as_Rest)

    gap_break = df["dt"] > Log_gap_threshold

    # ---------- Charging candidate ----------
    df["charge_candidate"] = (
        (df["pack_current"] < 0) &
        (df["speed"] == 0) &
        (df["acceleration"] == 0)
    )

    df["charge_group"] = (
        (df["charge_candidate"] != df["charge_candidate"].shift()) | gap_break
    ).cumsum()

    charge_summary = (
        df.groupby("charge_group")
          .agg(
              is_candidate=("charge_candidate", "first"),
              start_time=("time", "first"),
              end_time=("time", "last")
          )
          .reset_index()
    )
    charge_summary["duration"] = charge_summary["end_time"] - charge_summary["start_time"]

    charging_groups = charge_summary[
        (charge_summary["is_candidate"] == True) &
        (charge_summary["duration"] >= Charg_min_duration)
    ]["charge_group"]

    # ---------- Rest candidate ----------
    df["rest_candidate"] = (
        (df["pack_current"] >= -5) &
        (df["pack_current"] <= 5)
    )

    df["rest_group"] = (
        (df["rest_candidate"] != df["rest_candidate"].shift()) | gap_break
    ).cumsum()

    rest_summary = (
        df.groupby("rest_group")
          .agg(
              is_candidate=("rest_candidate", "first"),
              start_time=("time", "first"),
              end_time=("time", "last")
          )
          .reset_index()
    )
    rest_summary["duration"] = rest_summary["end_time"] - rest_summary["start_time"]

    rest_groups = rest_summary[
        (rest_summary["is_candidate"] == True) &
        (rest_summary["duration"] >= Rest_min_duration)
    ]["rest_group"]

    # ---------- Final state ----------
    df["state"] = "Other"

    df.loc[df["charge_group"].isin(charging_groups), "state"] = "Charging"

    df.loc[
        (df["rest_group"].isin(rest_groups)) &
        (df["state"] != "Charging"),
        "state"
    ] = "Rest"

    df.loc[df["dt"] >= Long_gap_as_Rest, "state"] = "Rest"

    return df


# =========================================================
# 3. Delayed Full Charge
# =========================================================
def apply_delayed_full_charge(
    df,
    SOC_standby,
    T_buffer,
    soc_col="soc",
    time_col="time",
    state_col="state",
    charge_state="Charging",
    output_col="soc_DFC",
    modified_col="DFC_modified",
    Full_charge_soc=Full_charge_soc,
):
    """
    Delayed Full Charge(DFC) 시나리오를 SOC에 적용합니다.

    - Full Charge: 해당 충전 세션의 최대 SOC가 Full_charge_soc 이상일 때만 DFC 적용
    - Partial Charge(Full_charge_soc 미만)는 DFC 미적용
    - 각 충전 세션에서 SOC_standby 이후부터 실제 충전 종료 SOC까지의 원래 프로파일을 이동
    - 이동된 프로파일은 다음 주행 시작 T_buffer(버퍼) 전에 full charge가 완료되도록 재배치
    - SOC_standby 도달 전 구간은 원래 SOC 유지, 이동된 구간 이후도 원래 SOC 유지
    - 결과는 원래 SOC 컬럼에 반영됨
    - DFC_NOTE 컬럼에 적용 여부 및 사유 기록
    - Full Charge 이후부터 다음 주행 직전까지 충전기 연결(chrg_cable_conn=1)이 유지되어야 함
    - 중간에 충전기 분리(chrg_cable_conn=0) 또는 로그 공백이 있으면 DFC를 적용하지 않음
    """
    df_out = df.copy()
    df_out[output_col] = df_out[soc_col]
    df_out[modified_col] = False
    df_out["DFC_NOTE"] = ""
    df_out["DFC_FILLER"] = False
    T_buffer = pd.Timedelta(T_buffer)

    if "charge_group" in df_out.columns:
        session_col = "charge_group"
        remove_temp_session_col = False
    else:
        charge_mask_tmp = df_out[state_col] == charge_state
        gap_break_tmp = df_out[time_col].diff() > pd.Timedelta(Log_gap_threshold)
        session_col = "DFC_charge_group"
        remove_temp_session_col = True
        df_out[session_col] = (
            (charge_mask_tmp != charge_mask_tmp.shift()) | gap_break_tmp
        ).cumsum()

    charge_mask = df_out[state_col] == charge_state

    for group_id, group_df in df_out[charge_mask].groupby(session_col):
        group_df = group_df.sort_values(time_col)

        # ------ Full Charge 판정 ------
        # (Full Charge: 세션 내 SOC 최대값이 Full_charge_soc 이상)
        full_charge = group_df[soc_col].max() >= Full_charge_soc
        if not full_charge:
            # Partial Charge는 적용하지 않음
            df_out.loc[group_df.index, "DFC_NOTE"] = "Partial_charge"
            continue

        # ------ SOC_standby 도달 시점 찾기 ------
        # (SOC_standby 이상인 첫 row 찾기)
        above_standby = group_df[group_df[soc_col] >= SOC_standby]
        if above_standby.empty:
            # SOC_standby에 도달하지 못한 경우
            df_out.loc[group_df.index, "DFC_NOTE"] = "No_soc_standby_point"
            continue
        target_idx = above_standby.index[0]
        target_time = df_out.loc[target_idx, time_col]
        target_value = df_out.loc[target_idx, soc_col]

        # ------ 충전 종료 시점 ------
        original_end_idx = group_df.index[-1]
        original_end_time = df_out.loc[original_end_idx, time_col]
        original_end_soc = df_out.loc[original_end_idx, soc_col]

        # ------ 다음 주행 이벤트 찾기 ------
        # (충전 종료 이후 speed > 0 인 첫 row)
        next_drive_df = df_out[
            (df_out[time_col] > original_end_time) &
            (df_out["speed"] > 0)
        ].sort_values(time_col)
        if next_drive_df.empty:
            # 다음 주행 없음
            df_out.loc[group_df.index, "DFC_NOTE"] = "No_next_drive"
            continue
        next_drive_time = next_drive_df.iloc[0][time_col]

        # ------ Full Charge 이후부터 출발 전까지 충전기 연결 상태 확인 ------
        # 충전 종료 이후 로그 존재 여부 확인
        post_charge_df = df_out[
            (df_out[time_col] > original_end_time)
            &
            (df_out[time_col] < next_drive_time)
        ].copy()

        # 로그가 전혀 없으면 연결 상태를 판단할 수 없음
        if post_charge_df.empty:
            df_out.loc[group_df.index, "DFC_NOTE"] = "Insufficient_post_charge_log"
            continue

        # 충전 종료 이후 큰 로그 공백 확인
        post_charge_gap = post_charge_df[time_col].diff()
        if (post_charge_gap > pd.Timedelta(Long_gap_as_Rest)).any():
            df_out.loc[group_df.index, "DFC_NOTE"] = "Insufficient_post_charge_log"
            continue

        # 충전기 연결 유지 여부 확인
        if "chrg_cable_conn" not in df_out.columns:
            raise ValueError("chrg_cable_conn 컬럼이 필요합니다")

        if (post_charge_df["chrg_cable_conn"] == 0).any():
            df_out.loc[group_df.index, "DFC_NOTE"] = "Cable_disconnected"
            continue

        new_end_time = next_drive_time - pd.Timedelta(T_buffer)

        # ------ DFC 적용을 위한 시간축 확장 ------
        full_range = pd.date_range(
            start=target_time,
            end=next_drive_time,
            freq="1s"
        )

        existing_times = set(df_out[time_col])
        missing_times = [t for t in full_range if t not in existing_times]

        if len(missing_times) > 0:
            filler = pd.DataFrame({time_col: missing_times})

            for col in df_out.columns:
                if col in [time_col, "DFC_FILLER"]:
                    continue
                filler[col] = np.nan

            filler["DFC_FILLER"] = True
            filler[state_col] = "Rest"
            filler["speed"] = 0
            filler["acceleration"] = 0
            filler["chrg_cable_conn"] = 1
            filler[soc_col] = np.nan

            df_out = pd.concat([df_out, filler], ignore_index=False)
            df_out = df_out.sort_values(time_col)

        # ------ SOC_standby 이후 ~ 실제 충전 종료까지 원래 프로파일 tail 추출 ------
        tail_df = group_df[group_df[time_col] >= target_time].copy()
        if len(tail_df) < 2:
            # 이동시킬 프로파일이 충분하지 않음
            df_out.loc[group_df.index, "DFC_NOTE"] = "Too_short_profile"
            continue
        original_tail_duration = original_end_time - target_time
        if original_tail_duration <= pd.Timedelta(0):
            df_out.loc[group_df.index, "DFC_NOTE"] = "Invalid_profile_duration"
            continue

        # ------ 재배치 가능 여부 확인 ------
        shifted_start_time = new_end_time - original_tail_duration
        if shifted_start_time <= target_time:
            # 재배치 불가(충분한 윈도우 없음)
            df_out.loc[group_df.index, "DFC_NOTE"] = "Insufficient_window"
            continue

        # 원래 충전 세션 row
        original_session_mask = df_out.index.isin(group_df.index)

        # 이번 DFC 이벤트에서 생성된 filler row만 사용
        filler_window_mask = (
            (df_out["DFC_FILLER"] == True)
            & (df_out[time_col] >= target_time)
            & (df_out[time_col] < next_drive_time)
        )

        hold_mask = (
            (original_session_mask | filler_window_mask)
            & (df_out[time_col] >= target_time)
            & (df_out[time_col] < shifted_start_time)
        )

        apply_mask = (
            (original_session_mask | filler_window_mask)
            & (df_out[time_col] >= shifted_start_time)
            & (df_out[time_col] <= new_end_time)
        )

        full_hold_mask = (
            (original_session_mask | filler_window_mask)
            & (df_out[time_col] > new_end_time)
            & (df_out[time_col] < next_drive_time)
        )

        # ------ DFC 적용 ------
        # 1) target_time ~ shifted_start_time 전까지 SOC 유지
        df_out.loc[hold_mask, output_col] = target_value
        df_out.loc[hold_mask, modified_col] = True
        df_out.loc[hold_mask, "DFC_NOTE"] = "DFC_applied"

        # 2) 원래 프로파일(충전 종료까지)을 shifted_start_time ~ new_end_time에 재배치
        shifted_times = shifted_start_time + (tail_df[time_col] - target_time)
        x_original = shifted_times.astype("int64")
        y_original = tail_df[soc_col].to_numpy()
        x_apply = df_out.loc[apply_mask, time_col].astype("int64")
        if len(x_apply) > 0:
            df_out.loc[apply_mask, output_col] = np.interp(
                x_apply,
                x_original,
                y_original,
            )
            df_out.loc[apply_mask, modified_col] = True
            df_out.loc[apply_mask, "DFC_NOTE"] = "DFC_applied"

        # 3) full charge 완료 후 다음 주행 전까지는 원래 충전 종료 SOC 유지
        df_out.loc[full_hold_mask, output_col] = original_end_soc
        df_out.loc[full_hold_mask, modified_col] = True
        df_out.loc[full_hold_mask, "DFC_NOTE"] = "DFC_applied"

    # 결과를 원래 SOC 컬럼에 반영
    df_out[soc_col] = df_out[output_col]
    # 임시 컬럼 제거
    df_out = df_out.drop(columns=[output_col, modified_col, "DFC_FILLER"], errors="ignore")
    # DFC_NOTE 컬럼만 추가로 남김
    return df_out, None


# =========================================================
# 4. Saving
# =========================================================
def make_DFC_output_path(file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file))[0]
    return os.path.join(output_dir, f"{base_name}_DFC.csv")


# =========================================================
# 5. Main execution
# =========================================================
def main():
    full_charge_df = pd.read_csv(FULL_CHARGE_LIST)

    if "file" not in full_charge_df.columns:
        raise ValueError("full_charge_files.csv must contain 'file' column")

    summary_rows = []

    files = (
        full_charge_df[
            full_charge_df["full_charge"] == True
        ]["file"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Found {len(files)} files")

    for i, file in enumerate(files, start=1):

        input_file = os.path.join(EV6_DIR, file)

        if not os.path.exists(input_file):
            print(f"[{i}] Missing file: {file}")
            continue

        print(f"[{i}/{len(files)}] Processing {file}")

        try:
            df = load_bms_csv(input_file)
            df = classify_state(df)

            df_out, _ = apply_delayed_full_charge(
                df,
                SOC_standby=SOC_standby,
                T_buffer=T_buffer,
                soc_col="soc",
                output_col="soc_DFC",
                modified_col="DFC_modified",
                Full_charge_soc=Full_charge_soc,
            )

            output_file = make_DFC_output_path(
                file=file,
                output_dir=OUTPUT_DIR,
            )

            df_out.to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig",
            )

            applied_count = (
                (df_out["DFC_NOTE"] == "DFC_applied")
                .sum()
            )

            if applied_count == 0:
                print(f"WARNING: No DFC applied -> {file}")

            summary_rows.append(
                {
                    "file": file,
                    "rows": len(df_out),
                    "DFC_applied_rows": int(applied_count),
                    "DFC_applied": bool(applied_count > 0),
                }
            )

        except Exception as e:
            print(f"Failed: {file} -> {e}")

    summary_df = pd.DataFrame(summary_rows)

    print(
        f"DFC applied files: "
        f"{summary_df['DFC_applied'].sum()} / {len(summary_df)}"
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "DFC_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nDone")
    print(f"Saved DFC files to: {OUTPUT_DIR}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
