#!/usr/bin/env python3

def text_to_bits(text):
    """Chuyển văn bản thành chuỗi bit nhị phân."""
    if not text:
        raise ValueError("Thông điệp không được rỗng!")
    return ''.join(format(ord(c), '08b') for c in text)

def preprocess_text(message, output_file=None):
    """Tiền xử lý thông điệp và trả về chuỗi bit. Lưu vào file nếu cần."""
    bits = text_to_bits(message)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(bits)
        print(f"✅ Đã lưu chuỗi bit vào '{output_file}'")
    return bits

if __name__ == "__main__":
    with open("text.txt", 'r', encoding='utf-8') as f:
        message = f.read().strip()
    bits = preprocess_text(message, output_file="message_bits.txt")
    print(f"Chuỗi bit: {bits}")