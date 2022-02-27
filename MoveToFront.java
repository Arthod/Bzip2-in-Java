public class MoveToFront {
    private static int[] recentlyUsedSymbols = new int[256];
    
    public static int[] encode(int[] inArr) {
        // Init out array
        int[] outArr = new int[inArr.length];

        // Reset array
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols[i] = 255 - i;
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteRead;
        int indexOfByte = -1;
        for (int i = 0; i < inArr.length; i++) {
            // Read from array (in)
            byteRead = inArr[i];

            // Search the array (from right to left) for the byte
            for (int j = 0; j < 256; j++) {
                if (recentlyUsedSymbols[j] == byteRead) {
                    indexOfByte = j;
                    break;
                }
            }

            // Shift all indicies to the left, until element
            for (int j = indexOfByte; j < 255; j++) {
                recentlyUsedSymbols[j] = recentlyUsedSymbols[j + 1];
            }

            // Replace last index with element
            recentlyUsedSymbols[255] = byteRead;
            
            // Write to array (out)
            outArr[i] = indexOfByte;
        }
        
        return outArr;
    }

    public static int[] decode(int[] inArr) {
        // Init out array
        int[] outArr = new int[inArr.length];

        // Reset array
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols[i] = 255 - i;
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteRead;
        int charOfByte;
        for (int i = 0; i < inArr.length; i++) {
            // Read from array (in)
            byteRead = inArr[i];
            
            // Get the byte in that index position of the byte read
            charOfByte = recentlyUsedSymbols[byteRead];

            // Shift all indicies to the left, until element
            for (int j = byteRead; j < 255; j++) {
                recentlyUsedSymbols[j] = recentlyUsedSymbols[j + 1];
            }

            // Replace last index with element
            recentlyUsedSymbols[255] = charOfByte;

            // Write to array (out)
            outArr[i] = charOfByte;
        }

        return outArr;
    }
}
