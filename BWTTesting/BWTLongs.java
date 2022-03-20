import java.util.Arrays;
import java.util.Random;

public class BWTLongs {
    private final static int unusedByte = -1;
    private final static Random random = new Random();
    
    public static int[] transform(int[] S, int[] rowId) {
        // Create new array with k=6 EOF characters at the end
        int[] sArr = new int[S.length + 6];
        for (int i = 0; i < S.length; i++) {
            sArr[i] = S[i];
        }
        for (int i = 0; i < 6; i++) {
            sArr[S.length + i] = unusedByte;
        }

        // Create array W of N words. Pack 2 bytes into 1 long    Q2
        long[] W = new long[S.length];
        for (int i = 0; i < S.length - 5; i++) {
            if (sArr[i] == unusedByte || sArr[i + 1] == unusedByte) {
                System.out.println("error");
            }
            W[i] = bytesToLong(sArr[i], sArr[i + 1], sArr[i + 2], sArr[i + 3], sArr[i + 4], sArr[i + 5], 255);
        }
        W[S.length - 5] = bytesToLong(sArr[S.length - 5], sArr[S.length - 4], sArr[S.length - 3], sArr[S.length - 2], sArr[S.length - 1], 0, 0);
        W[S.length - 4] = bytesToLong(sArr[S.length - 4], sArr[S.length - 3], sArr[S.length - 2], sArr[S.length - 1], 0, 0, 0);
        W[S.length - 3] = bytesToLong(sArr[S.length - 3], sArr[S.length - 2], sArr[S.length - 1], 0, 0, 0, 0);
        W[S.length - 2] = bytesToLong(sArr[S.length - 2], sArr[S.length - 1], 0, 0, 0, 0, 0);
        W[S.length - 1] = bytesToLong(sArr[S.length - 1], 0, 0, 0, 0, 0, 0);
        
        // Array V  Q4
        // Sort by first two characters using radix sort using counting sort
        int[] V_temp = new int[W.length];
        int[] V = new int[W.length];
        int k;
        int[] count = new int[257];
        
        // Radix sort
        // Count the characters
        for (int i = 0; i < S.length; i++) {
            count[sArr[i + 1] + 1]++;
        }

        for (int i = 1; i < count.length; i++) {
            count[i] += count[i - 1];
        }        
        
        // Go in reverse, and write the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = sArr[i + 1] + 1;
            V_temp[count[k] - 1] = i;
            count[k]--;
        }
        
        // Test V_temp array is sorted by 2nd char
        for (int i = 0; i < V.length - 2; i++) {
            int c = V_temp[i];
            int l = V_temp[i + 1];
            if (sArr[c + 1] > sArr[l + 1]) {
                System.out.println("Fejl not sorted by 2nd char: " + c);
                for (int j = 0; j < 10; j++) {
                    System.out.print(new String(new byte[] { (byte) sArr[c + j] }));
                }
            }
        }

        /// Sort by first character
        // Count characters
        count = new int[count.length];
        for (int i = 0; i < S.length; i++) {
            count[sArr[i] + 1]++;
        }

        for (int i = 1; i < count.length; i++) {
            count[i] += count[i - 1];
        }

        // Go in reverse, and write the index of the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = sArr[V_temp[i]] + 1;
            V[count[k] - 1] = V_temp[i];
            count[k]--;
        }

        // Test V array is sorted by first two chars
        for (int i = 0; i < V.length - 2; i++) {
            int c = V[i];
            int l = V[i + 1];
            if (sArr[c] > sArr[l]) {
                System.out.println("Fail");
            }
            if (sArr[c] == sArr[l]) {
                if (sArr[c + 1] > sArr[l + 1]) {
                    System.out.println("Fail");
                }
            }
        }

        // Q5
    
        int amountComparedEqualTotal = 0;  // number of characters that have been compared equal
        int first = 0;
        for (int ch1 = -1; ch1 < 256 && amountComparedEqualTotal < S.length; ch1++) {
            // Q6
            for (int ch2 = -1; ch2 < 256 && amountComparedEqualTotal < S.length; ch2++) {
                // We know that V is sorted by the first two characters
                first = amountComparedEqualTotal;
                

                while (ch1 == sArr[V[amountComparedEqualTotal]] && ch2 == sArr[V[amountComparedEqualTotal] + 1]) {
                    amountComparedEqualTotal++;
                    
                    // If reached end of input, stop all
                    if (amountComparedEqualTotal == S.length) {
                        break;
                    }
                }

                // If there's atleast two that are compared equal we need to sort them.
                if (amountComparedEqualTotal - first >= 2) {
                    //System.out.println(first + " -> " + (amountComparedEqualTotal - first));
                    quicksortIndexArray(V, W, first, amountComparedEqualTotal - 1);
                }
            }
            /*
            for (int i = first; i < amountComparedEqualTotal; i++) {
                if (sArr[V[i]] != ch1) {
                    System.out.println("qwerty");
                }
            }*/
        }

        // Test V array is sorted
        for (int i = 0; i < V.length - 2; i++) {
            int c = V[i];
            int l = V[i + 1];
            while (sArr[c] == sArr[l]) {
                c++;
                l++;
            }
            if (sArr[c] > sArr[l]) {
                for (int j = -1; j < 50; j++) {
                    System.out.print(new String(new byte[] { (byte) sArr[c + j] }));
                }
                System.out.println();
                System.out.println("Fejl: " + c);
            }
        }
        
        // Now our V has the correctly sorted indicies of the square
        // Now we fetch the last column of the BWT square
        // We also write the row index to the last index of the out array
        int[] outArr = new int[W.length];
        int zeroSeen = 1;
        for (int i = 0; i < outArr.length; i++) {
            if (V[i] == 0) {
                // i is the index of the row of the original string
                rowId[0] = i + 1;
                zeroSeen = 0;
            } else {
                outArr[i + zeroSeen] = S[V[i] - 1];
            }
        }
        outArr[0] = S[S.length - 1];

        return outArr;
    }
    
    private static long bytesToLong(int a, int b, int c, int d, int e, int f, int g) {
        return (((long) 0 << 56) + ((long) a << 48) + ((long)b << 40) + ((long)c << 32) + ((long)d << 24) + ((long)e << 16) + ((long)f << 8) + ((long)g << 0));
    }

    private static void quicksortIndexArray(int[] indexArray, long[] comparedArray, int startIndex, int endIndex) {
        // In-place implementation of quicksort that sorts an index 
        // array based on the comparable values of a comparable array
        if (startIndex >= endIndex) {
            return;
        }
        //System.out.println("going in " + startIndex + " -> " + endIndex);
        int q = partition(indexArray, comparedArray, startIndex, endIndex);
        quicksortIndexArray(indexArray, comparedArray, startIndex, q - 1);
        quicksortIndexArray(indexArray, comparedArray, q + 1, endIndex);
    }

    private static void randomizedQuicksortIndexArray(int[] indexArray, long[] comparedArray, int startIndex, int endIndex) {
        // In-place implementation of quicksort that sorts an index 
        // array based on the comparable values of a comparable array
        if (startIndex >= endIndex) {
            return;
        }
        //System.out.println("going in " + startIndex + " -> " + endIndex);
        int q = randomizedPartition(indexArray, comparedArray, startIndex, endIndex);
        randomizedQuicksortIndexArray(indexArray, comparedArray, startIndex, q - 1);
        randomizedQuicksortIndexArray(indexArray, comparedArray, q + 1, endIndex);
    }

    private static int randomizedPartition(int[] indexArray, long[] comparedArray, int startIndex, int endIndex) {
        int randomIndex = random.nextInt(endIndex - startIndex + 1) + startIndex;
        if (randomIndex > endIndex || randomIndex < startIndex) {
            System.out.println("error");
        }
        swap(indexArray, endIndex, randomIndex);
        return partition(indexArray, comparedArray, startIndex, endIndex);
    }

    private static int partition(int[] indexArray, long[] comparedArray, int startIndex, int endIndex) {
        int pivot = indexArray[endIndex];
        int i = startIndex - 1;

        for (int j = startIndex; j <= endIndex - 1; j++) {

            // TODO: can optimize.. We know first two characters of the word are already sorted, no need to recheck them
            int indexLimit = Math.max(indexArray[j], pivot);
            for (int k = 0; k < comparedArray.length - indexLimit; k += 6) {

                // Check if they are equal
                if (comparedArray[indexArray[j] + k] != comparedArray[pivot + k]) {
                // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[indexArray[j] + k] <= comparedArray[pivot + k]) {
                        i++;
                        swap(indexArray, i, j);
                    }
                    break;
                }
            }
            /*
            if (comparedArray[indexArray[j]] > comparedArray[pivot]) {
                i++;
                // Swap
                swap(indexArray, j, i);
            }*/
        }
        swap(indexArray, i + 1, endIndex);

        return i + 1;
    }
    private static void swap(int[] arr, int index1, int index2) {
        int temp = arr[index1];
        arr[index1] = arr[index2];
        arr[index2] = temp;
    }
}
