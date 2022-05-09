# huffman from code lengths test

"""
void BZ2_hbAssignCodes ( Int32 *code,
                         UChar *length,
                         Int32 minLen,
                         Int32 maxLen,
                         Int32 alphaSize )
{
   Int32 n, vec, i;

   vec = 0;
   for (n = minLen; n <= maxLen; n++) {
      for (i = 0; i < alphaSize; i++)
         if (length[i] == n) { code[i] = vec; vec++; };
      vec <<= 1;
   }
}

"""

if __name__ == "__main__":
    lens = {"a": 2, "b": 4, "c": 2, "d": 4, "e": 2, "f": 3}
    chars = lens.keys()
    #chars = sorted(lens.keys(), key=lambda k: lens[k])
    print(chars)
    codes = {}

    vec = 0


    for l in range(min(lens.values()), max(lens.values()) + 1):
        for char in chars:
            if l == lens[char]:
                print(char, vec)
                codes[char] = format(vec, 'b').zfill(lens[char])
                vec += 1
        vec *= 2

    print(codes)

