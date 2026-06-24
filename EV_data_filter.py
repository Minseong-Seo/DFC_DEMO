import os
import glob
from pathlib import Path

import pandas as pd


# =========================
# 설정
# =========================
DATA_DIR = Path("/Volumes/T7/Data study/EV6")
OUTPUT_DIR = Path("/Volumes/T7/Data study/filter_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_FILES_CSV = OUTPUT_DIR / "valid_files.csv"
INVALID_FILES_CSV = OUTPUT_DIR / "invalid_files.csv"
VALIDATION_REPORT_CSV = OUTPUT_DIR / "validation_report.csv"


# =========================
# 기본 파일 탐색
# =========================
def find_bms_files(data_dir: Path) -> list[Path]:
    """
    EV6 폴더 안의 bms_*.csv 파일만 찾는다.
    macOS 숨김 파일인 ._bms_*.csv는 제외한다.
    """
    files = sorted(data_dir.glob("bms_*.csv"))
    files = [f for f in files if not f.name.startswith("._")]
    return files


# =========================
# 개별 검사 함수들
# =========================
def check_file_readable(file_path: Path) -> tuple[bool, str]:
    """
    CSV 파일을 정상적으로 읽을 수 있는지 확인한다.
    """
    try:
        pd.read_csv(file_path, nrows=5)
        return True, "readable"
    except Exception as e:
        return False, f"read_error: {e}"


def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> tuple[bool, str]:
    """
    분석에 필요한 컬럼들이 모두 존재하는지 확인한다.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        return False, f"missing_columns: {missing_cols}"

    return True, "required_columns_ok"


def check_minimum_rows(df: pd.DataFrame, min_rows: int = 100) -> tuple[bool, str]:
    """
    데이터 행 개수가 너무 적은 파일을 제외한다.
    기준값은 나중에 조정 가능하다.
    """
    if len(df) < min_rows:
        return False, f"too_few_rows: {len(df)} < {min_rows}"

    return True, "row_count_ok"


def check_time_column(df: pd.DataFrame, time_col: str = "time") -> tuple[bool, str]:
    """
    time 컬럼이 datetime으로 변환 가능한지 확인한다.
    """
    if time_col not in df.columns:
        return False, f"missing_time_column: {time_col}"

    time_series = pd.to_datetime(df[time_col], errors="coerce")
    invalid_ratio = time_series.isna().mean()

    if invalid_ratio > 0.05:
        return False, f"invalid_time_ratio_too_high: {invalid_ratio:.3f}"

    return True, "time_column_ok"


# =========================
# SOC 점프 + 긴 gap 검사 함수
# =========================
def check_soc_or_time_gap_problem(
    df: pd.DataFrame,
    time_col: str = "time",
    soc_col: str = "soc",
    soc_threshold: float = 10.0,
    time_threshold_hours: float = 12.0,
) -> tuple[bool, str]:
    """
    연속한 두 샘플 사이에서 아래 둘 중 하나라도 만족하면 해당 월 파일을 제외한다.

    1. ΔSOC >= 10%
    2. Δtime >= 12 h

    즉, SOC가 비정상적으로 크게 튀었거나, 데이터 missing gap이 너무 길면 unusable로 판단한다.
    """
    if time_col not in df.columns:
        return False, f"missing_time_column: {time_col}"

    if soc_col not in df.columns:
        return False, f"missing_soc_column: {soc_col}"

    temp_df = df[[time_col, soc_col]].copy()
    temp_df[time_col] = pd.to_datetime(temp_df[time_col], errors="coerce")
    temp_df[soc_col] = pd.to_numeric(temp_df[soc_col], errors="coerce")

    temp_df = temp_df.dropna(subset=[time_col, soc_col])
    temp_df = temp_df.sort_values(time_col).reset_index(drop=True)

    if len(temp_df) < 2:
        return False, "not_enough_valid_time_soc_samples"

    dt_hours = temp_df[time_col].diff().dt.total_seconds() / 3600
    dsoc = temp_df[soc_col].diff().abs()

    soc_bad_mask = dsoc >= soc_threshold
    time_bad_mask = dt_hours >= time_threshold_hours

    # OR 조건: 둘 중 하나라도 문제가 있으면 invalid 처리
    bad_mask = soc_bad_mask | time_bad_mask

    if bad_mask.any():
        first_bad_idx = bad_mask[bad_mask].index[0]

        prev_time = temp_df.loc[first_bad_idx - 1, time_col]
        curr_time = temp_df.loc[first_bad_idx, time_col]
        prev_soc = temp_df.loc[first_bad_idx - 1, soc_col]
        curr_soc = temp_df.loc[first_bad_idx, soc_col]
        bad_dt = dt_hours.loc[first_bad_idx]
        bad_dsoc = dsoc.loc[first_bad_idx]

        is_soc_bad = bool(soc_bad_mask.loc[first_bad_idx])
        is_time_bad = bool(time_bad_mask.loc[first_bad_idx])

        if is_soc_bad and is_time_bad:
            failure_type = "BOTH_SOC_AND_TIME"
        elif is_soc_bad:
            failure_type = "SOC_ONLY"
        elif is_time_bad:
            failure_type = "TIME_ONLY"
        else:
            failure_type = "UNKNOWN"

        reasons = []
        if is_soc_bad:
            reasons.append(f"soc_reason: ΔSOC={bad_dsoc:.2f}% >= {soc_threshold:.2f}%")
        if is_time_bad:
            reasons.append(f"time_reason: Δtime={bad_dt:.2f}h >= {time_threshold_hours:.2f}h")

        detail = (
            f"ΔSOC={bad_dsoc:.2f}%, Δtime={bad_dt:.2f}h, "
            f"from {prev_time} SOC={prev_soc:.2f} "
            f"to {curr_time} SOC={curr_soc:.2f}"
        )

        return (
            False,
            f"{failure_type} | " + " | ".join(reasons) + " | " + detail,
        )

    return True, "soc_or_time_gap_ok"


# =========================
# 파일 하나에 대한 전체 검사
# =========================
def validate_single_file(file_path: Path) -> dict:
    """
    하나의 CSV 파일을 검사하고 결과를 dict로 반환한다.
    나중에 unusable 조건이 추가되면 이 함수 안에 check 함수를 추가하면 된다.
    """
    result = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "is_valid": True,
        "failed_reason": "",
        "failure_type": "VALID",
        "num_rows": None,
        "num_columns": None,
    }

    # 1. 파일 읽기 가능 여부 확인
    ok, message = check_file_readable(file_path)
    if not ok:
        result["is_valid"] = False
        result["failed_reason"] = message
        return result

    # 2. 실제 데이터 읽기
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        result["is_valid"] = False
        result["failed_reason"] = f"read_error: {e}"
        return result

    result["num_rows"] = len(df)
    result["num_columns"] = len(df.columns)

    # 3. 기본 검사 조건들
    required_cols = [
        "time",
        "soc",
        "pack_current",
        "pack_volt",
        "speed",
        "acceleration",
    ]

    checks = [
        check_required_columns(df, required_cols),
        check_minimum_rows(df, min_rows=100),
        check_time_column(df, time_col="time"),
        check_soc_or_time_gap_problem(
            df,
            time_col="time",
            soc_col="soc",
            soc_threshold=10.0,
            time_threshold_hours=12.0,
        ),
    ]

    failed_messages = [message for ok, message in checks if not ok]

    if failed_messages:
        result["is_valid"] = False
        result["failed_reason"] = " | ".join(failed_messages)

        if any("BOTH_SOC_AND_TIME" in message for message in failed_messages):
            result["failure_type"] = "BOTH_SOC_AND_TIME"
        elif any("SOC_ONLY" in message for message in failed_messages):
            result["failure_type"] = "SOC_ONLY"
        elif any("TIME_ONLY" in message for message in failed_messages):
            result["failure_type"] = "TIME_ONLY"
        else:
            result["failure_type"] = "OTHER"

    return result


# =========================
# 전체 파일 검사
# =========================
def validate_all_files(data_dir: Path) -> pd.DataFrame:
    files = find_bms_files(data_dir)

    print(f"Found {len(files)} BMS files")

    results = []
    for i, file_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Checking {file_path.name}")
        result = validate_single_file(file_path)
        results.append(result)

    return pd.DataFrame(results)


# =========================
# 실행부
# =========================
def main():
    report_df = validate_all_files(DATA_DIR)

    valid_df = report_df[report_df["is_valid"]].copy()
    invalid_df = report_df[~report_df["is_valid"]].copy()

    soc_only_df = invalid_df[invalid_df["failure_type"] == "SOC_ONLY"].copy()
    time_only_df = invalid_df[invalid_df["failure_type"] == "TIME_ONLY"].copy()
    both_soc_time_df = invalid_df[invalid_df["failure_type"] == "BOTH_SOC_AND_TIME"].copy()
    other_invalid_df = invalid_df[invalid_df["failure_type"] == "OTHER"].copy()

    report_df.to_csv(VALIDATION_REPORT_CSV, index=False, encoding="utf-8-sig")
    valid_df.to_csv(VALID_FILES_CSV, index=False, encoding="utf-8-sig")
    invalid_df.to_csv(INVALID_FILES_CSV, index=False, encoding="utf-8-sig")

    soc_only_df.to_csv(OUTPUT_DIR / "invalid_soc_only_files.csv", index=False, encoding="utf-8-sig")
    time_only_df.to_csv(OUTPUT_DIR / "invalid_time_only_files.csv", index=False, encoding="utf-8-sig")
    both_soc_time_df.to_csv(OUTPUT_DIR / "invalid_both_soc_time_files.csv", index=False, encoding="utf-8-sig")
    other_invalid_df.to_csv(OUTPUT_DIR / "invalid_other_files.csv", index=False, encoding="utf-8-sig")

    print("\nDone")
    print(f"Valid files: {len(valid_df)}")
    print(f"Invalid files: {len(invalid_df)}")
    print(f"  - SOC only invalid files: {len(soc_only_df)}")
    print(f"  - Time only invalid files: {len(time_only_df)}")
    print(f"  - Both SOC and time invalid files: {len(both_soc_time_df)}")
    print(f"  - Other invalid files: {len(other_invalid_df)}")
    print(f"Saved report: {VALIDATION_REPORT_CSV}")
    print(f"Saved valid list: {VALID_FILES_CSV}")
    print(f"Saved invalid list: {INVALID_FILES_CSV}")


if __name__ == "__main__":
    main()