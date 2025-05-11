#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile
import json

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def calculate_snr(original, modified):
    """Tính SNR (Signal-to-Noise Ratio) giữa âm thanh gốc và âm thanh đã giấu tin."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - modified) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def dsss_embed(cover_file, output_file='stego.wav', scaling_factor=0.1):
    # Đọc thông điệp từ file message.txt
    try:
        with open('message.txt', 'r', encoding='utf-8') as f:
            message = f.read().strip()
        if not message:
            raise ValueError("File message.txt rỗng!")
        # Giới hạn ở ký tự ASCII
        message = ''.join(c for c in message if ord(c) < 128)
        if not message:
            raise ValueError("Thông điệp chỉ chứa ký tự không hợp lệ!")
    except FileNotFoundError:
        raise FileNotFoundError("Không tìm thấy file message.txt!")

    # In thông điệp và chuỗi bit để gỡ lỗi
    bits = text_to_bits(message)
    print(f"📝 Thông điệp nhúng: {message}")

    # Đọc file âm thanh
    try:
        sample_rate, audio_data = wavfile.read(cover_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file '{cover_file}'!")
    
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data / np.max(np.abs(audio_data))  # Chuẩn hóa

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

    # Tạo file parameter chứa thông tin cơ bản
    duration = len(audio_data) / sample_rate  # Thời gian (giây)
    stego_info = {
        "output_file": output_file,
        "sample_rate": int(sample_rate),
        "num_samples": len(audio_data),
        "duration_seconds": duration,
        "message_length_bits": len(bits),
        "snr_db": float(snr),
        "scaling_factor": scaling_factor,
        "chips_per_bit": chips_per_bit,
        "samples_per_chip": samples_per_chip
    }
    with open('stego_info.json', 'w', encoding='utf-8') as f:
        json.dump(stego_info, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã tạo file 'stego_info.json' chứa thông tin cơ bản.")

if __name__ == "__main__":
    dsss_embed('cover.wav')
