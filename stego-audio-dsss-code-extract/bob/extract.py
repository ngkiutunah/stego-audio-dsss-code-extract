#!/usr/bin/env python3
import numpy as np
from scipy.io import wavfile

def bits_to_text(bits):
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    text = ''
    for b in chars:
        if len(b) == 8:
            try:
                char = chr(int(b, 2))
                if 32 <= ord(char) <= 126:  # Chỉ giữ ký tự ASCII in được
                    text += char
                else:
                    print(f"⚠️ Bỏ qua ký tự không hợp lệ từ bit: {b}")
            except ValueError:
                print(f"⚠️ Lỗi: Chuỗi bit '{b}' không thể chuyển thành ký tự.")
        else:
            print(f"⚠️ Lỗi: Chuỗi bit '{b}' không đủ 8 bit.")
    return text

def dsss_extract(stego_file, param_file='embed_params.npz'):
    # Load audio
    try:
        sample_rate, audio_data = wavfile.read(stego_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file '{stego_file}'!")
    
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Load params
    try:
        data = np.load(param_file)
        pn_bits = data['pn_bits']
        samples_per_chip = int(data['samples_per_chip'])
        num_bits = int(data['bits'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Không tìm thấy file '{param_file}'!")
    except KeyError as e:
        raise KeyError(f"File '{param_file}' thiếu tham số: {e}")

    chips_per_bit = len(pn_bits) // num_bits
    samples_per_bit = samples_per_chip * chips_per_bit

    extracted_bits = ''
    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        if end > len(audio_data):
            print(f"⚠️ Lỗi: Âm thanh không đủ dài để trích xuất bit thứ {i+1}.")
            break
        segment = audio_data[start:end]
        chips = np.mean(segment.reshape(-1, samples_per_chip), axis=1)
        pn_segment = pn_bits[i * chips_per_bit:(i + 1) * chips_per_bit]
        bit_val = np.sum(chips * pn_segment)
        extracted_bits += '1' if bit_val > 0 else '0'

    # In chuỗi bit trích xuất để gỡ lỗi
    if len(extracted_bits) != num_bits:
        print(f"⚠️ Cảnh báo: Số bit trích xuất ({len(extracted_bits)}) không khớp với số bit nhúng ({num_bits})!")

    message = bits_to_text(extracted_bits)
    print(f"✅ Thông điệp trích xuất: {message}")

    # Lưu thông điệp vào file extracted_message.txt
    try:
        with open('extracted_message.txt', 'w', encoding='utf-8') as f:
            f.write(message)
        print(f"✅ Đã lưu thông điệp vào 'extracted_message.txt'.")
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu file: {e}")

    return message

if __name__ == "__main__":
    dsss_extract('stego.wav')