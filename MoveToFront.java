import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class MoveToFront {
    private static int[] recentlyUsedSymbols = new int[256];
    private static int runA = 0;
    private static int runB = 1;
    private static int offset = 0;


    public static void main(String[] args) {
        for (int i = 0; i < 10000; i++) {
            int len = (int) 10000;
            int[] inArr = new int[len];
            Random random = new Random();
            for (int j = 0; j < len; j++) {
                inArr[j] = Math.floorMod(random.nextInt(), 3) * 256/3;
            }
            test(inArr);
        }
    }

    private static void test(int[] inArr) {
        int[] encoded = encode(inArr, false);
        //System.out.println(Arrays.toString(encoded));
        int[] outArr = decode(encoded, false);
        //System.out.println(Arrays.toString(inArr));
        //System.out.println(Arrays.toString(outArr));

        /// Check correctness
        // Check is same length
        if (inArr.length != outArr.length) {
            System.out.println("Error: Not same length");
            return;
        }
        // Check all elements are the same
        for (int i = 0; i < inArr.length; i++) {
            if (inArr[i] != outArr[i]) {
                System.out.println("Error: Not equal at index " + i);
                return;
            }
        }

        /// Print space saved
        System.out.println("Ints difference: " + (inArr.length - encoded.length));
    }
    
    
    public static int[] encode(int[] inArr, Boolean RLE) {
        // Init out array
        offset = 0;
        int[] outArr = new int[inArr.length];

        // Reset array
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols[i] = i;
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteRead;
        int indexOfByte = -1;
        for (int i = 0; i < inArr.length; i++) {
            // Read from array (in)
            byteRead = inArr[i];

            // DEBUG
            if (byteRead > 255 || byteRead < 0) {
                System.out.println("MTF can't handle non 8-bit values");
            }

            // Search the array for the byte
            for (int j = 0; j < 256; j++) {
                if (recentlyUsedSymbols[j] == byteRead) {
                    indexOfByte = j;
                    break;
                }
            }

            // If index of byte is 0, we can run-length encode it
            // Count amount of consecutive 0's, this will be k
            if (indexOfByte == 0 && RLE) {
                int k = 1;
                while (i + k < inArr.length && byteRead == inArr[i + k]) {
                    k++;
                }

                String s = Integer.toBinaryString(k - 1);
                for (int j = 0; j < s.length(); j++) {
                    if (s.charAt(j) == '0') outArr[i + j - offset] = runA;
                    if (s.charAt(j) == '1') outArr[i + j - offset] = runB;
                }
                offset += k - s.length();
                i += k - 1;

            } else {
                // Shift all indicies to the right, until element
                for (int j = indexOfByte; j > 0; j--) {
                    recentlyUsedSymbols[j] = recentlyUsedSymbols[j - 1];
                }
    
                // Replace first index with element
                recentlyUsedSymbols[0] = byteRead;
                
                // Write to array (out)
                outArr[i - offset] = indexOfByte + 1;
            }
        }

        // Cut array
        int[] outArr2 = new int[inArr.length - offset];
        for (int i = 0; i < outArr2.length; i++) {
            outArr2[i] = outArr[i];
        }
        
        return outArr2;
    }

    public static int[] decode(int[] inArr, Boolean RLE) {
        // Init out array
        ArrayList<Integer> outList = new ArrayList<Integer>(inArr.length);

        // Reset array
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols[i] = i;
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteRead;
        int charOfByte;
        for (int i = 0; i < inArr.length; i++) {
            // Read from array (in)
            byteRead = inArr[i];
            
            if ((byteRead == runA || byteRead == runB) && RLE) {
                String s;
                if (byteRead == runA) {
                    s = "0";
                } else {
                    s = "1";
                }
                int k = 1;
                while (i + k < inArr.length && ((byteRead = inArr[i + k]) == runA || byteRead == runB)) {
                    if (byteRead == runA) {
                        s += "0";
                    } else {
                        s += "1";
                    }
                    k++;
                }

                int zeroCount = Integer.parseInt(s, 2) + 1;
                for (int j = 0; j < zeroCount; j++) {
                    // Write to array (out)
                    outList.add(recentlyUsedSymbols[0]);
                }

                i += k - 1;

            } else {
                // Get the byte in that index position of the byte read
                byteRead = byteRead - 1;
                charOfByte = recentlyUsedSymbols[byteRead];

                // Shift all indicies to the right, until element
                for (int j = byteRead; j > 0; j--) {
                    recentlyUsedSymbols[j] = recentlyUsedSymbols[j - 1];
                }

                // Replace first index with element
                recentlyUsedSymbols[0] = charOfByte;

                // Write to array (out)
                outList.add(charOfByte);
            }
        }

        return outList.stream().mapToInt(i -> i).toArray(); // clean up
    }
}
