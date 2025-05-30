import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any

GRAVITY = 9.81


def goda_test(wave_period: float, water_depth: float,
              probe1: np.ndarray, probe2: np.ndarray, time: np.ndarray,
              delta_L: float, start_time: float, end_time: float,
              step: int = 1) -> Tuple[float, float, float]:
    # 确保输入为numpy数组
    probe1 = np.asarray(probe1, dtype=float)
    probe2 = np.asarray(probe2, dtype=float)
    time = np.asarray(time, dtype=float)

    # 数据过滤
    mask = (time >= start_time) & (time <= end_time)
    time_segment = time[mask][::step] - time[mask][0]
    probe1_segment = probe1[mask][::step]
    probe2_segment = probe2[mask][::step]

    # 傅里叶拟合
    coeff1, _ = fit_fourier_series(time_segment, probe1_segment, wave_period)
    coeff2, _ = fit_fourier_series(time_segment, probe2_segment, wave_period)

    # 波长计算
    wavelength = calculate_wavelength(water_depth, wave_period)
    wave_number = 2 * np.pi / wavelength

    # Goda分析
    return goda_analysis(coeff1, coeff2, delta_L, wave_number)


def fit_fourier_series(time: np.ndarray, signal: np.ndarray,
                       period: float, n_harmonics: int = 8) -> Tuple[Dict[str, Any], float]:
    time = np.asarray(time, dtype=float)
    signal = np.asarray(signal, dtype=float)

    def fourier_model(t, a0, *coefficients):
        w = 2 * np.pi / float(period)  # 确保period是float
        t = np.asarray(t, dtype=float)
        y = np.full_like(t, a0)  # 用a0初始化数组
        for k in range(1, n_harmonics + 1):
            a_k = coefficients[k - 1]
            b_k = coefficients[n_harmonics + k - 1]
            y += a_k * np.cos(k * w * t) + b_k * np.sin(k * w * t)
        return y

    initial_guess = [np.mean(signal)] + [0.1] * (2 * n_harmonics)

    try:
        params, _ = curve_fit(fourier_model, time, signal, p0=initial_guess, maxfev=10000)
    except Exception as e:
        print(f"Fitting failed: {e}, using initial guess")
        params = initial_guess

    predicted = fourier_model(time, *params)
    ss_res = np.sum((signal - predicted) ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        'a0': float(params[0]),
        'a1': float(params[1]),
        'b1': float(params[n_harmonics + 1]),
        'all_coeffs': params,
        'model': fourier_model
    }, r_squared


def goda_analysis(probe1_result: Dict[str, float],
                  probe2_result: Dict[str, float],
                  probe_distance: float,
                  wave_number: float) -> Tuple[float, float, float]:
    # 提取系数并确保为float类型
    A1 = float(probe1_result['a1'])
    B1 = float(probe1_result['b1'])
    A2 = float(probe2_result['a1'])
    B2 = float(probe2_result['b1'])

    # 计算三角函数值
    kd = float(wave_number * probe_distance)
    cos_kd = np.cos(kd)
    sin_kd = np.sin(kd)

    # 计算各项
    term1 = A2 - A1 * cos_kd - B1 * sin_kd
    term2 = B2 + A1 * sin_kd - B1 * cos_kd
    term3 = A2 - A1 * cos_kd + B1 * sin_kd
    term4 = B2 - A1 * sin_kd - B1 * cos_kd

    # 计算振幅
    Ai = float(np.sqrt(term1 ** 2 + term2 ** 2) / (2 * abs(sin_kd)))
    Ar = float(np.sqrt(term3 ** 2 + term4 ** 2) / (2 * abs(sin_kd)))
    Cr = float(Ar / Ai) if Ai != 0 else 0.0

    print(f"Cr = {Cr:.4f}, Ai = {Ai:.4f}, Ar = {Ar:.4f}")
    return Cr, Ai, Ar


def calculate_wavelength(depth: float, period: float, tolerance: float = 1e-6) -> float:
    omega = 2 * np.pi / float(period)
    k = omega ** 2 / GRAVITY  # 初始猜测(深水)

    # 深水情况
    if depth > (np.pi / k):  # 更保守的深水判断
        return 2 * np.pi / k

    # 迭代求解
    for _ in range(1000):
        kh = k * depth
        tanh_kh = np.tanh(kh)
        f = omega ** 2 - GRAVITY * k * tanh_kh
        df = -GRAVITY * (tanh_kh + k * depth * (1 - tanh_kh ** 2))

        delta_k = f / df
        k -= delta_k

        if abs(delta_k) < tolerance:
            break

    return 2 * np.pi / k


# 主程序
if __name__ == "__main__":
    # 参数设置
    CONFIG = {
        "probe_distance": 0.3,
        "wave_period": 5.0,
        "water_depth": 0.85,
        "analysis_start": 30,
        "analysis_end": 60
    }

    # 加载数据
    try:
        data = pd.read_excel('gauges.xlsx', header=None)
        time = np.asarray(data.iloc[1:, 0], dtype=float)
        probe1 = np.asarray(data.iloc[1:, 1], dtype=float) - CONFIG["water_depth"]
        probe2 = np.asarray(data.iloc[1:, 2], dtype=float) - CONFIG["water_depth"]
    except Exception as e:
        print(f"Data loading failed: {e}")
        raise

    # 执行分析
    try:
        Cr, Ai, Ar = goda_test(
            wave_period=CONFIG["wave_period"],
            water_depth=CONFIG["water_depth"],
            probe1=probe1,
            probe2=probe2,
            time=time,
            delta_L=CONFIG["probe_distance"],
            start_time=CONFIG["analysis_start"],
            end_time=CONFIG["analysis_end"]
        )
        print(f"Final Results - Cr: {Cr:.4f}, Ai: {Ai:.4f}, Ar: {Ar:.4f}")
    except Exception as e:
        print(f"Analysis failed: {e}")