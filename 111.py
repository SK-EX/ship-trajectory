import numpy as np
import matplotlib.pyplot as plt


def fourier_lowpass(signal, cutoff_freq, sample_rate):
    """
    Apply a Fourier-based low pass filter to a signal.

    Parameters:
    signal (array): Input signal to be filtered
    cutoff_freq (float): Cutoff frequency in Hz
    sample_rate (float): Sampling rate of the signal in Hz

    Returns:
    array: Filtered signal
    """
    # Compute the Fourier Transform of the signal
    fft_signal = np.fft.fft(signal)
    n = len(signal)

    # Create the frequency axis
    freq = np.fft.fftfreq(n, d=1 / sample_rate)

    # Create the filter (1 for frequencies below cutoff, 0 otherwise)
    filter_mask = np.abs(freq) < cutoff_freq

    # Apply the filter
    filtered_fft = fft_signal * filter_mask

    # Inverse Fourier Transform to get back to time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    # Return the real part (imaginary part should be very small)
    return np.real(filtered_signal)


# Example usage
if __name__ == "__main__":
    # Create a test signal: combination of 5Hz and 20Hz sine waves
    sample_rate = 1000  # Hz
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)

    # Apply low pass filter with cutoff at 10Hz
    cutoff_freq = 10  # Hz
    filtered_signal = fourier_lowpass(signal, cutoff_freq, sample_rate)

    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, filtered_signal, label=f'Filtered (cutoff {cutoff_freq}Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time Domain')
    plt.legend()
    plt.grid()

    # Plot frequency spectrum
    plt.subplot(2, 1, 2)
    fft_original = np.abs(np.fft.fft(signal)[:n // 2])
    fft_filtered = np.abs(np.fft.fft(filtered_signal)[:n // 2])
    freq_axis = np.fft.fftfreq(n, d=1 / sample_rate)[:n // 2]

    plt.plot(freq_axis, fft_original, label='Original Spectrum')
    plt.plot(freq_axis, fft_filtered, label='Filtered Spectrum')
    plt.axvline(cutoff_freq, color='r', linestyle='--', label='Cutoff Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()