import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional


@dataclass
class WaveParameters:
    """存储波浪相关参数的类"""
    height: float  # 波高 (m)
    period: float  # 波周期 (s)
    water_depth: float  # 水深 (m)


@dataclass
class DeviceParameters:
    """存储设备相关参数的类"""
    chamber_diameter: float  # 气室直径 (m)
    sampling_rate: int = 50  # 采样率 (Hz)


class WaveEnergyAnalyzer:
    """波浪能转换系统分析器"""

    def __init__(self, wave_params: WaveParameters, device_params: DeviceParameters):
        """初始化分析器"""
        self.wave = wave_params
        self.device = device_params
        self.rho_water = 1025  # 海水密度 (kg/m³)
        self.g = 9.81  # 重力加速度 (m/s²)

        # 计算气室截面积
        self.chamber_area = np.pi * (self.device.chamber_diameter / 2) ** 2

    def simulate_test_data(self, duration: float = 100) -> None:
        """生成模拟测试数据"""
        num_samples = int(duration * self.device.sampling_rate)
        self.time = np.linspace(0, duration, num_samples)
        self.pressure = np.random.randn(num_samples) * 1000  # 模拟压力数据 (Pa)
        self.water_level = np.random.randn(num_samples) * 0.5  # 模拟水位变化 (m)

    def calculate_pneumatic_power(self, start_time: float = 30, window_duration: float = 8.4) -> float:
        """
        计算气动功率
        Args:
            start_time: 分析窗口开始时间 (s)
            window_duration: 分析窗口持续时间 (s)
        Returns:
            气动功率 (W)
        """
        dt = self.time[1] - self.time[0]

        # 计算每个时间步的功
        work_per_step = self.pressure[1:] * self.chamber_area * dt

        # 选择分析窗口
        start_idx = int(start_time * self.device.sampling_rate)
        end_idx = int((start_time + window_duration) * self.device.sampling_rate)
        selected_work = work_per_step[start_idx:end_idx]

        # 数值积分计算平均功率
        total_work = np.sum(selected_work)
        actual_duration = (end_idx - start_idx) * dt
        pneumatic_power = total_work / actual_duration

        # 绘制结果
        self._plot_work_per_step(work_per_step)

        return pneumatic_power

    def _plot_work_per_step(self, work_per_step: np.ndarray) -> None:
        """绘制每个时间步的功"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time[1:], work_per_step)
        plt.title('Work per timestep')
        plt.xlabel('Time (s)')
        plt.ylabel('Work (J)')
        plt.grid(True)
        plt.show()

    def calculate_wave_power(self) -> Optional[float]:
        """
        计算波浪功率
        Returns:
            波浪功率 (W/m) 或 None (如果计算失败)
        """
        try:
            wavelength = self._solve_dispersion_equation()
            wave_number = 2 * np.pi / wavelength

            # 计算波浪功率
            term = (1 + (2 * wave_number * self.wave.water_depth) /
                    np.sinh(2 * wave_number * self.wave.water_depth))
            wave_power = (self.rho_water * self.g ** 2 * self.wave.height ** 2 *
                          self.wave.period) / (64 * np.pi) * term

            return wave_power

        except ValueError as e:
            print(f"Error calculating wave power: {str(e)}")
            return None

    def _solve_dispersion_equation(self, tolerance: float = 1e-6, max_iterations: int = 1000) -> float:
        """
        牛顿迭代法求解色散方程
        Args:
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
        Returns:
            波长 (m)
        Raises:
            ValueError: 如果不收敛
        """
        # 初始猜测（深水波长）
        wavelength = self.g * self.wave.period ** 2 / (2 * np.pi)

        for _ in range(max_iterations):
            wave_number = 2 * np.pi / wavelength
            residual = (2 * np.pi / self.wave.period) ** 2 - self.g * wave_number * np.tanh(
                wave_number * self.wave.water_depth)

            # 计算导数
            derivative = (self.g * self.wave.water_depth * (wave_number ** 2) /
                          np.cosh(wave_number * self.wave.water_depth) ** 2) - (
                                 self.g * np.tanh(wave_number * self.wave.water_depth) / wavelength)

            delta = residual / derivative
            wavelength -= delta  # 牛顿迭代

            if abs(delta) < tolerance:
                return wavelength

        raise ValueError(f"Solution did not converge after {max_iterations} iterations")

    def calculate_water_velocity(self) -> np.ndarray:
        """计算气室内水面速度"""
        dt = self.time[1] - self.time[0]
        velocity = np.zeros_like(self.water_level)

        # 使用中心差分法计算速度（更精确）
        velocity[1:-1] = (self.water_level[2:] - self.water_level[:-2]) / (2 * dt)

        # 处理边界点
        velocity[0] = (self.water_level[1] - self.water_level[0]) / dt
        velocity[-1] = (self.water_level[-1] - self.water_level[-2]) / dt

        self._plot_water_velocity(velocity)

        return velocity

    def _plot_water_velocity(self, velocity: np.ndarray) -> None:
        """绘制水面速度"""
        plt.figure(figsize=(12, 4))
        plt.plot(self.time, velocity)
        plt.title('Water Surface Velocity in Chamber')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)
        plt.show()


def main():
    # 初始化参数
    wave_params = WaveParameters(height=1.5, period=8.0, water_depth=2.0)
    device_params = DeviceParameters(chamber_diameter=0.15)

    # 创建分析器实例
    analyzer = WaveEnergyAnalyzer(wave_params, device_params)

    # 生成模拟数据
    analyzer.simulate_test_data(duration=100)

    # 计算气动功率
    pneumatic_power = abs(analyzer.calculate_pneumatic_power())
    print(f"Pneumatic Power: {pneumatic_power:.2f} W")

    # 计算波浪功率
    wave_power = analyzer.calculate_wave_power()
    if wave_power is not None:
        print(f"Wave Power: {wave_power:.2f} W/m")

        # 计算效率
        efficiency = abs(pneumatic_power / wave_power * 100)
        print(f"Pneumatic Efficiency: {efficiency:.2f}%")
    else:
        print("Cannot calculate efficiency due to invalid wave power")

    # 计算水面速度
    analyzer.calculate_water_velocity()


if __name__ == "__main__":
    main()