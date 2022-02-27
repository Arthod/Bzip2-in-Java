import java.util.LinkedList;

public class MoveToFrontLinkedList {
    static long startTime;
    static long durationTime;
    private static LinkedList<Integer> recentlyUsedSymbols = new LinkedList<Integer>();
    
    public static long testEncode(int[] inArr, int[] outArr) {
        // Get start time
        startTime = System.nanoTime();

        // Reset linked list
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
            recentlyUsedSymbols.addFirst(byteRead);

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
		int byteRead;
        int charOfByte;
        for (int i = 0; i < inArr.length; i++) {
            byteRead = inArr[i];
            
            charOfByte = recentlyUsedSymbols.get(byteRead);
            recentlyUsedSymbols.remove(byteRead);
            recentlyUsedSymbols.addFirst(charOfByte);

            outArr[i] = charOfByte;
        }

        // Get duration
        durationTime = System.nanoTime() - startTime;
        return durationTime;
    }
}
