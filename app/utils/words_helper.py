import re

def split_by_byte(text: str, max_bytes: int) -> list[str]:
    chunks = []
    current = ''
    bytes_count = 0

    for char in text:
        char_bytes = len(char.encode('utf-8'))
        if bytes_count + char_bytes > max_bytes:
            chunks.append(current)
            current = char
            bytes_count = char_bytes
        else:
            current += char
            bytes_count += char_bytes

    if current:
        chunks.append(current)

    return chunks

def chunk_text_by_sentence(text: str, max_bytes: int) -> list[str]:
    if len(text.encode('utf-8')) <= max_bytes:
        return [text]

    # Chia câu dựa trên dấu câu và khoảng trắng
    sentences = re.split(r'(?<=[.?!。])\s+', text.strip())
    chunks = []
    current_chunk = ''
    current_bytes = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        test_chunk = sentence if not current_chunk else f'{current_chunk} {sentence}'
        test_bytes = len(test_chunk.encode('utf-8'))

        if test_bytes > max_bytes:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Nếu câu đơn quá dài, fallback dùng split_by_byte
            if len(sentence.encode('utf-8')) > max_bytes:
                chunks.extend(split_by_byte(sentence, max_bytes))
                current_chunk = ''
                current_bytes = 0
            else:
                current_chunk = sentence
                current_bytes = len(sentence.encode('utf-8'))
        else:
            current_chunk = test_chunk
            current_bytes = test_bytes

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks