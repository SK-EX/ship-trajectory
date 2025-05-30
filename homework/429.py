import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def goda_test(wave_period, water_depth, probe1,probe2, time,deltaL, goda_start, goda_end, step):
    probe1 = np.array(probe1)
    probe2 = np.array(probe2)
    time = np.array(time)

    start_flag = 0
    total_data_ct = len(probe1)
    probe1f = []
    probe2f = []
    time_filtered = []
    goda_start_ct = None
    goda_end_ct = None
    ct_ft = 0

    for count in range(0, total_data_ct-step , step):
        probe1f.append(probe1[count])
        probe2f.append(probe2[count])
        time_filtered.append(time[count])
        current_time = time[count]

        if current_time >= goda_start and start_flag == 0:
            goda_start_ct = ct_ft
            start_flag = 1

        if current_time >= goda_end and start_flag == 1:
            goda_end_ct = ct_ft
            start_flag = 2
        ct_ft += 1
        probe1f = np.array(probe1f)
        probe2f = np.array(probe2f)
        time_filtered = np.array(time_filtered)

        mask = (time_filtered >= goda_start) & (time_filtered <= goda_end)
        index = np.where(mask)[0]
        # if len(index) == 0 :
        #     raise ValueError("No data found within the specified time range.")

        time_segment = time_filtered[index] - time_filtered[index[0]]
        probe1_segment = probe1f[index]
        probe2_segment = probe2f[index]
        coeff1, _ = neo_fit(time_segment, probe1_segment, wave_period)
        coeff2, _ = neo_fit(time_segment, probe2_segment, wave_period)

        # neo_coeff1 ={'A': coeff1['coeffs'][1],'B': coeff1['coeffs'][9]}
        # neo_coeff2 ={'A': coeff2['coeffs'][1],'B': coeff2['coeffs'][9]}

        waveNumber = 2 * np.pi / linear_wave_dispersion(water_depth, wave_period, 1e-6)

        Cr, Ai, Ar = goda(coeff1,coeff2, deltaL,waveNumber)
        print("Cr = ", Cr)
        print("Ai = ", Ai)
        print("Ar = ", Ar)
        return Cr, Ai, Ar


def neo_fit(x, y, Tw):
    def fourier_model(t, a0, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, w):
        y_vals = a0
        for k in range(1, 9):
            a = locals()[f'a{k}']
            b = locals()[f'b{k}']
            y_vals += a * np.cos(k * w * t) + b * np.sin(k * w * t)
        return y_vals

    initial_guess = [np.mean(y)] + [0.1] * 8 + [0.1] * 8 + [2 * np.pi / Tw]

    try:
        popt, pcov = curve_fit(fourier_model, x, y, p0=initial_guess, maxfev=10000)
    except Exception as e:
        print(f"Error fitting data: {e}")
        popt = initial_guess
        pcov = None

    y_pred = fourier_model(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Store coefficients with proper labels
    fresult = {
        'coeffs': popt,
        'A': popt[1],  # a1 coefficient
        'B': popt[9],  # b1 coefficient (index 10 because of a0-a8 + b1)
        'model': fourier_model,
        'pcov': pcov
    }
    return fresult, r_squared

def goda(result1, result2, deltaL, waveNumber):
    A1 = result1['A']
    B1 = result1['B']
    A2 = result2['A']
    B2 = result2['B']

    kd = waveNumber * deltaL
    cos_kd = np.cos(kd)
    sin_kd = np.sin(kd)

    term1 = A2 - A1 * cos_kd - B1 * sin_kd
    term2 = B2 + A1 * sin_kd - B1 * cos_kd
    term3 = A2 - A1 * cos_kd + B1 * sin_kd
    term4 = B2 - A1 * sin_kd - B1 * cos_kd

    Ai = (1 / (2 * np.abs(sin_kd))) * np.sqrt(term1 ** 2 + term2 ** 2)
    Ar = (1 / (2 * np.abs(sin_kd))) * np.sqrt(term3 ** 2 + term4 ** 2)
    Cr = Ar / Ai if Ai != 0 else 0
    return  Cr, Ai, Ar

def linear_wave_dispersion( h, T, al):
    g = 9.81  # gravitational acceleration [m/s^2]
    omega = 2 * np.pi / T  # angular frequency

    # Initial guess using deep water approximation (k = ω²/g)
    k = omega ** 2 / g
    L = 2 * np.pi / k

    # For deep water, return immediately
    if h > L / 2:
        return L

    # For shallow/intermediate water, solve iteratively
    max_iter = 1000
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = np.tanh(kh)
        f = omega ** 2 - g * k * tanh_kh
        df = -g * tanh_kh - g * kh * (1 - tanh_kh ** 2)

        delta_k = f / df
        k -= delta_k

        if abs(delta_k) < al:
            break
    L = 2 * np.pi / k
    return L


L1 = 0.3
L2 = 0.3
T = 5.0
dt = 0.1
h = 0.85
alTarget = 0.1
omega = 2 * np.pi / T
deltaT  = 0
order = 2
analysisStart = 1
nPeriods =  5
analysisEnd = analysisStart + nPeriods* T *(1/dt)

df =  pd.read_excel('gauges.xlsx', header= None)
tt = df.iloc[1:,0].values
g1 = df.iloc[1:,1].values - h
g2 = df.iloc[1:,2].values - h
g3 = df.iloc[1:,3].values - h
g4 = df.iloc[1:,4].values - h

waveLength = linear_wave_dispersion( h, T, 1e-6)

k = 2 * np.pi / waveLength
Cr, Ar, Ai = goda_test(T,h, g1,g2, tt,  L1, 30, 60, 1)
print(Cr)
print(Ar)
print(Ai)