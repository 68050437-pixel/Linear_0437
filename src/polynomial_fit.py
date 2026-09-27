from __future__ import annotations
import csv
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "drone_mission.csv"
OUTPUT_DIR = ROOT / "outputs"


def load_dataset(csv_path: Path) -> tuple[list[float], list[float], list[str]]:
    xs: list[float] = []
    ys: list[float] = []
    events: list[str] = []
    with csv_path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            xs.append(float(row["time_min"]))
            ys.append(float(row["altitude_m"]))
            events.append(row.get("event", ""))
    if len(xs) < 2:
        raise ValueError("ต้องมีอย่างน้อย 2 จุดข้อมูล จึงจะสร้างสมการได้")
    if len(set(xs)) != len(xs):
        raise ValueError("ค่า X ซ้ำกัน ทำให้เมทริกซ์แวนเดอร์มอนด์หาตัวผกผันไม่ได้")
    return xs, ys, events


def vandermonde(xs: list[float], degree: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for x in xs:
        matrix.append([x**power for power in range(degree + 1)])
    return matrix


def clone_matrix(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def format_matrix(matrix: list[list[float]], digits: int = 4) -> str:
    lines = []
    for row in matrix:
        cells = "  ".join(f"{value:10.{digits}f}" for value in row)
        lines.append(f"[ {cells} ]")
    return "\n".join(lines)


def gaussian_elimination(coeff: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    if any(len(row) != n for row in coeff):
        raise ValueError("เมทริกซ์สัมประสิทธิ์ต้องเป็นสี่เหลี่ยมจัตุรัส")

    aug = [coeff[i][:] + [rhs[i]] for i in range(n)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("ระบบสมการเอกฐาน หาคำตอบเดียวไม่ได้")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = aug[i][n]
        for j in range(i + 1, n):
            total -= aug[i][j] * solution[j]
        solution[i] = total / aug[i][i]
    return solution


def evaluate_polynomial(coeffs: list[float], x: float) -> float:
    value = 0.0
    for coeff in reversed(coeffs):
        value = value * x + coeff
    return value


def format_equation(coeffs: list[float]) -> str:
    parts: list[str] = []
    for power, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-12:
            continue
        sign = " + " if coeff >= 0 and parts else (" - " if parts else ("-" if coeff < 0 else ""))
        abs_coeff = abs(coeff)
        if power == 0:
            term = f"{abs_coeff:.6f}"
        elif power == 1:
            term = f"{abs_coeff:.6f}x" if abs(abs_coeff - 1) > 1e-12 else "x"
        else:
            term = f"{abs_coeff:.6f}x^{power}" if abs(abs_coeff - 1) > 1e-12 else f"x^{power}"
        parts.append(f"{sign}{term}")
    return "p(x) = " + ("".join(parts) if parts else "0")


def format_exact_hint(coeffs: list[float]) -> str:
    return (
        "สำหรับชุดข้อมูลภารกิจสายฟ้าแดง ค่าที่แม่นยำในรูปเศษส่วนคือ\n"
        "p(x) = 5 + (121/12)x + (31/16)x^2 - (43/192)x^3"
        if _is_sample_mission(coeffs)
        else ""
    )


def _is_sample_mission(coeffs: list[float]) -> bool:
    expected = [5.0, 121 / 12, 31 / 16, -43 / 192]
    if len(coeffs) != 4:
        return False
    return all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8) for a, b in zip(coeffs, expected))


def print_dataset(xs: list[float], ys: list[float], events: list[str]) -> None:
    print("ชุดข้อมูลภารกิจ")
    print("-" * 72)
    print(f"{'จุด':<6}{'เวลา x (นาที)':<18}{'ความสูง y (เมตร)':<20}เหตุการณ์")
    for i, (x, y, event) in enumerate(zip(xs, ys, events)):
        print(f"{i:<6}{x:<18.2f}{y:<20.2f}{event}")
    print()


def verify_fit(xs: list[float], ys: list[float], coeffs: list[float]) -> None:
    print("ตรวจสอบว่าเส้นโค้งผ่านจุดข้อมูลเดิม")
    print("-" * 72)
    print(f"{'x':<12}{'y จริง':<14}{'p(x)':<14}{'คลาดเคลื่อน'}")
    for x, y in zip(xs, ys):
        pred = evaluate_polynomial(coeffs, x)
        print(f"{x:<12.2f}{y:<14.4f}{pred:<14.4f}{abs(pred - y):.4e}")
    print()


def save_plot(xs: list[float], ys: list[float], coeffs: list[float], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("ยังไม่ได้ติดตั้ง matplotlib กรุณาใช้คำสั่ง: pip install -r requirements.txt") from exc

    start, end = min(xs), max(xs)
    span = end - start
    grid = [start + span * i / 200 for i in range(201)]
    curve = [evaluate_polynomial(coeffs, x) for x in grid]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid, curve, color="#1f4e79", linewidth=2.2, label="polynomial curve p(x)")
    ax.scatter(xs, ys, color="#c0392b", s=70, zorder=3, label="GPS data points")
    for x, y in zip(xs, ys):
        ax.annotate(f"({x:g}, {y:g})", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=8)
    ax.set_title("Saifa Daeng drone altitude versus time")
    ax.set_xlabel("time after launch x (minutes)")
    ax.set_ylabel("altitude y (meters)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def predict_loop(coeffs: list[float], xmin: float, xmax: float) -> None:
    print("โหมดพยากรณ์ความสูง")
    print("พิมพ์ตัวเลขเวลา (นาที) หรือพิมพ์ q เพื่อออก")
    print(f"ช่วงข้อมูลที่บันทึกคือ {xmin:g} ถึง {xmax:g} นาที")
    print("ถ้าใส่ค่านอกช่วง โปรแกรมยังคำนวณได้ แต่เป็นการประมาณนอกช่วง (extrapolation)")
    while True:
        raw = input("\nเวลา x = ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        try:
            x = float(raw)
        except ValueError:
            print("กรุณาใส่ตัวเลข เช่น 6 หรือ 9.5")
            continue
        y = evaluate_polynomial(coeffs, x)
        print(f"p({x:g}) = {y:.4f} เมตร")
        if x < xmin or x > xmax:
            print("หมายเหตุ: ค่านี้อยู่นอกช่วงข้อมูล ควรใช้ด้วยความระวัง")


def run(csv_path: Path = DEFAULT_CSV, predict: bool = True) -> list[float]:
    xs, ys, events = load_dataset(csv_path)
    degree = len(xs) - 1
    v = vandermonde(xs, degree)
    coeffs = gaussian_elimination(clone_matrix(v), ys[:])

    print("=" * 72)
    print("โปรแกรมสร้างสมการพหุนามด้วยเมทริกซ์แวนเดอร์มอนด์")
    print("=" * 72)
    print()
    print_dataset(xs, ys, events)

    print(f"ดีกรีของพหุนาม = จำนวนจุด - 1 = {len(xs)} - 1 = {degree}")
    print("รูปแบบสมการ: p(x) = a0 + a1 x + a2 x^2 + ... + an x^n")
    print()
    print("เมทริกซ์แวนเดอร์มอนด์ V")
    print(format_matrix(v))
    print()
    print("เวกเตอร์ y")
    print(format_matrix([[y] for y in ys]))
    print()
    print("แก้ระบบ V a = y ด้วยการกำจัดเกาส์ ได้สัมประสิทธิ์")
    for i, coeff in enumerate(coeffs):
        print(f"  a{i} = {coeff:.10f}")
    print()
    print(format_equation(coeffs))
    hint = format_exact_hint(coeffs)
    if hint:
        print(hint)
    print()
    verify_fit(xs, ys, coeffs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "drone_polynomial.png"
    save_plot(xs, ys, coeffs, plot_path)
    print(f"บันทึกกราฟที่ {plot_path}")
    print()

    sample_x = 6.0
    sample_y = evaluate_polynomial(coeffs, sample_x)
    print(f"ตัวอย่างพยากรณ์: ที่เวลา x = {sample_x:g} นาที ความสูงประมาณ {sample_y:.4f} เมตร")
    print()

    if predict:
        predict_loop(coeffs, min(xs), max(xs))
    return coeffs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="สร้างสมการพหุนามจากชุดข้อมูลด้วยเมทริกซ์")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="ไฟล์ชุดข้อมูล CSV")
    parser.add_argument("--no-predict", action="store_true", help="รันครั้งเดียว ไม่เข้าโหมดถามค่า")
    args = parser.parse_args()
    run(csv_path=args.csv, predict=not args.no_predict)
