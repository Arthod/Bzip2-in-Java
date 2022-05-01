# Huffman from code lengths test


def huffman_from_code_lengths(code_lengths: list[int]):
    # https://www.ietf.org/rfc/rfc1951.txt

    bl_count = [0 for _ in range(len(code_lengths))]
    next_code = [0 for _ in range(len(code_lengths))]
    codes = [0 for _ in range(len(code_lengths))]

    # 1)
    for cl in code_lengths:
        if (cl >= 1):
            bl_count[cl] += 1
    print(bl_count)

    # 2)
    code = 0
    bl_count[0] = 0
    for bits in range(1, max(code_lengths) + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    print(next_code)

    # 3)
    for n in range(len(code_lengths)):
        l = code_lengths[n]

        if (l != 0):
            codes[n] = next_code[l]
            next_code[l] += 1
    print(codes)

    print({i: format(code, 'b') for i, code in enumerate(codes)})


if __name__ == "__main__":
    code_lengths = [3,3,3,3,3,2,4,4]

    huffman_from_code_lengths(code_lengths)