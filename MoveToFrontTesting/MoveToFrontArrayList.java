
import java.util.ArrayList;

public class MoveToFrontArrayList {
    static long startTime;
    static long durationTime;
    private static ArrayList<Integer> recentlyUsedSymbols = new ArrayList<Integer>();
    
    public static long testEncode(int[] inArr, int[] outArr) {
        // Get start time
        startTime = System.nanoTime();

        // Reset array
        recentlyUsedSymbols.clear();
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols.add(i);
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteRead;
        int indexOfByte;
        for (int i = 0; i < inArr.length; i++) {
            byteRead = inArr[i];

            indexOfByte = recentlyUsedSymbols.indexOf(byteRead);
            recentlyUsedSymbols.remove(indexOfByte);
            recentlyUsedSymbols.add(0, byteRead);

            outArr[i] = indexOfByte;
        }
        
        // Get duration
        durationTime = System.nanoTime() - startTime;
        return durationTime;
    }

    public static long testDecode(int[] inArr, int[] outArr) {
        // Get start time
        startTime = System.nanoTime();

        // Reset array
        recentlyUsedSymbols.clear();
        for (int i = 0; i < 256; i++) {
            recentlyUsedSymbols.add(i);
        }
        
        // Read inArr array char by char, find the index and insert it into the outArr array
		int byteIndexRead;
        int charOfByteIndex;
        for (int i = 0; i < inArr.length; i++) {
            byteIndexRead = inArr[i];
            
            charOfByteIndex = recentlyUsedSymbols.remove(byteIndexRead);
            recentlyUsedSymbols.add(0, charOfByteIndex);

            outArr[i] = charOfByteIndex;
        }

        // Get duration
        durationTime = System.nanoTime() - startTime;
        return durationTime;
    }
}
