#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import json
from preprocess_text import preprocess_text
from preprocess_audio import preprocess_audio

def calculate_snr(original, modified):
    """Tính SNR giữa âm thanh gốc và âm thanh đã giấu tin."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - modified) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def dsss_embed(cover_file, message_file, output_file='stego.wav', scaling_factor=0.05):
    # Đọc thông điệp từ file
    try:
        with open(message_file, 'r', encoding='utf-8') as f:
            message = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file '{message_file}'!")
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc file thông điệp: {str(e)}")

    # Tiền xử lý thông điệp
    bits = preprocess_text(message)

    # Tiền xử lý âm thanh
    audio_info = preprocess_audio(cover_file)
    audio_data = audio_info['audio_data']
    sample_rate = audio_info['sample_rate']

    bit_array = np.array([1 if b == '1' else -1 for b in bits])

    # DSSS parameters
    bit_rate = 1e6
    chip_rate = 11e6
    chips_per_bit = int(chip_rate / bit_rate)
    samples_per_chip = max(1, int(sample_rate / chip_rate))
    samples_per_bit = chips_per_bit * samples_per_chip

    # Kiểm tra độ dài âm thanh
    required_samples = len(bits) * chips_per_bit * samples_per_chip
    if required_samples > len(audio_data):
        raise ValueError(f"Âm thanh cần ít nhất {required_samples} mẫu, nhưng chỉ có {len(audio_data)}!")

    # Trải phổ DSSS
    dsss_signal = []
    pn_bits = np.random.choice([1, -1], size=len(bit_array) * chips_per_bit)
    for i, bit in enumerate(bit_array):
        chips = pn_bits[i * chips_per_bit:(i + 1) * chips_per_bit]
        signal = bit * chips
        signal_samples = np.repeat(signal, samples_per_chip)
        dsss_signal.extend(signal_samples)

    dsss_signal = np.array(dsss_signal[:len(audio_data)])  # Cắt theo độ dài audio

    # Giấu tin
    stego = audio_data.copy()
    stego[:len(dsss_signal)] += scaling_factor * dsss_signal
    stego = stego / np.max(np.abs(stego))  # Tránh tràn biên độ

    # Tính SNR
    snr = calculate_snr(audio_data, stego)

    # Ghi ra file âm thanh
    wavfile.write(output_file, sample_rate, stego.astype(np.float32))
    np.savez('embed_params.npz', pn_bits=pn_bits, samples_per_chip=samples_per_chip, bits=len(bits))
    print(f"✅ Giấu tin vào '{output_file}' thành công.")

    # Tạo file thông tin với các khóa tiếng Việt
    stego_info = {
        "tep_dau_ra": output_file,
        "tan_so_lay_mau": int(sample_rate),
        "so_mau": len(audio_data),
        "thoi_gian_giay": audio_info['duration_seconds'],
        "do_dai_thong_diep_bit": len(bits),
        "snr_db": float(snr),
        "he_so_ti_le": scaling_factor,
        "chip_tren_bit": chips_per_bit,
        "mau_tren_chip": samples_per_chip,
        "tep_thong_diep": message_file
    }
    with open('stego_info.json', 'w', encoding='utf-8') as f:
        json.dump(stego_info, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã tạo file 'stego_info.json' chứa thông tin cơ bản.")

if __name__ == "__main__":
    dsss_embed('cover.wav', 'text.txt')