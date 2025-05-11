#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile

def preprocess_audio(cover_file):
    """Tiền xử lý file âm thanh: đọc, chuẩn hóa và kiểm tra."""
    try:
        sample_rate, audio_data = wavfile.read(cover_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file '{cover_file}'!")
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file âm thanh: {str(e)}")

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)  # Chuyển sang mono
    audio_data = audio_data / np.max(np.abs(audio_data))  # Chuẩn hóa

    return {
        "audio_data": audio_data,
        "sample_rate": sample_rate,
        "num_samples": len(audio_data),
        "duration_seconds": len(audio_data) / sample_rate
    }

if __name__ == "__main__":
    audio_info = preprocess_audio("cover.wav")
    print(f"Hoàn tất xử lý file âm thanh đầu vào")
    