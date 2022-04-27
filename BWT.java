import java.util.Arrays;
import java.util.Random;

// TODO see if sArr can be removed.. I assume it can
// TODO convert input from int to short, and all other places

public class BWT {
    private final static int unusedByte = -1;
    private final static Random random = new Random();
    
    public static int[] transform(int[] S, int[] rowId) {
        // Create new array with k=2 EOF characters at the end
        short[] sArr = new short[S.length + 2];
        for (int i = 0; i < S.length; i++) {
            sArr[i] = (short) S[i];
        }
        for (int i = 0; i < 2; i++) {
            sArr[S.length + i] = unusedByte;
        }

        // Create array W of N words. Pack 2 bytes into 1 integer    Q2
        int[] W = new int[S.length];
        for (int i = 0; i < S.length - 2; i++) {
            if (sArr[i] == unusedByte || sArr[i + 1] == unusedByte) {
                System.out.println("error");
                return null;
            }
            W[i] = bytesToInt(sArr[i], sArr[i + 1], 255);
        }
        W[S.length - 2] = bytesToInt(sArr[S.length - 2], sArr[S.length - 1], 1);
        W[S.length - 1] = bytesToInt(sArr[S.length - 1], 0, 0);
        
        // Array V  Q4
        // Sort by first two characters using radix sort using counting sort
        int[] V_temp = new int[W.length];
        int[] V = new int[W.length];
        int k;
        int[] count = new int[257];
        int[] countCurrent;
        
        // Radix sort
        // Count the characters (skip the first character)
        for (int i = 1; i < S.length + 1; i++) {
            count[sArr[i] + 1]++;
        }

        countCurrent = new int[257];
        countCurrent[0] = 1;
        for (int i = 1; i < count.length; i++) {
            countCurrent[i] = count[i] + countCurrent[i - 1];
        }
        
        // Go in reverse, and write the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = sArr[i + 1] + 1;
            V_temp[countCurrent[k] - 1] = i;
            countCurrent[k]--;
        }

        /// Sort by first character
        count[0]--;
        count[sArr[0] + 1]++;
        for (int i = 1; i < count.length; i++) {
            count[i] += count[i - 1];
        }

        // Go in reverse, and write the index of the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = sArr[V_temp[i]] + 1;
            V[count[k] - 1] = V_temp[i];
            count[k]--;
        }

        /*
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
        }*/

        // Q5
        int amountComparedEqualTotal = 0;  // number of characters that have been compared equal
        int first = 0;
        for (int ch1 = 0; ch1 < 256; ch1++) {
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
                    randomizedQuicksortIndexArray(V, W, first, amountComparedEqualTotal - 1);
                }
            }
        }
        /*
        // Test V array is sorted
        for (int i = 0; i < V.length - 2; i++) {
            int c = V[i];
            int l = V[i + 1];
            while (sArr[c] == sArr[l]) {
                c++;
                l++;
            }
            if (sArr[c] > sArr[l]) {
                System.out.println("Fejl: " + c);
                for (int j = -1; j < 50; j++) {
                    System.out.print(new String(new byte[] { (byte) sArr[c + j] }));
                }
                System.out.println();
            }
        }*/
        
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
    

    /*
     *  Reverses the BWT transformation. Assumes the int in inArr is the row id of the original string
     */
    public static int[] reverseTransform(int[] inArr, int rowId) {
        // Add -1 to a new last index of the array
        int[] arr = new int[inArr.length + 1];
        for (int i = 0; i < rowId; i++) {
            arr[i] = inArr[i];
        }
        arr[rowId] = -1;
        for (int i = rowId + 1; i < arr.length; i++) {
            arr[i] = inArr[i - 1];
        }
        inArr = arr;

        // P[i] is the number of instances of character L[i] in L[0, 1, ..., i-1]
        int[] P = new int[inArr.length];

        // count[ch] is the total number of instances in L, of characters preceding 
        // character ch in the alphabet.
        int[] count = new int[257];
        

        // First pass - 
        //  count[ch] is the amount of times ch appears in L.
        //  P[i] is the number of
        for (int i = 0; i < inArr.length; i++) {
            P[i] = count[inArr[i] + 1];
            count[inArr[i] + 1]++;
        }

        int sum = 0;
        for (int i = 0; i < count.length; i++) {
            sum += count[i];
            count[i] = sum - count[i];
        }

        int[] outArr = new int[inArr.length - 1];
        int i = rowId;
        i = P[i];
        for (int j = outArr.length - 1; j >= 0; j--) {
            outArr[j] = inArr[i];
            i = P[i] + count[inArr[i] + 1];
        }

        return outArr;
    }

    private static int bytesToInt(int a, int b, int c) {
        return ((0 << 24) | (a << 16) | (b << 8) | (c << 0));
    }

    private static void randomizedQuicksortIndexArray(int[] indexArray, int[] comparedArray, int startIndex, int endIndex) {
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

    private static int randomizedPartition(int[] indexArray, int[] comparedArray, int startIndex, int endIndex) {
        int randomIndex = random.nextInt(endIndex - startIndex + 1) + startIndex;
        if (randomIndex > endIndex || randomIndex < startIndex) {
            System.out.println("error");
        }
        swap(indexArray, endIndex, randomIndex);
        return partition(indexArray, comparedArray, startIndex, endIndex);
    }

    private static int partition(int[] indexArray, int[] comparedArray, int startIndex, int endIndex) {
        int pivot = indexArray[endIndex];
        int i = startIndex - 1;

        for (int j = startIndex; j < endIndex; j++) {

            // TODO: can optimize.. We know first two characters of the word are already sorted, no need to recheck them
            int k = 0;
            while (true) {
                if (comparedArray[indexArray[j] + k] != comparedArray[pivot + k]) {
                    // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[indexArray[j] + k] <= comparedArray[pivot + k]) {
                        i++;
                        swap(indexArray, i, j);
                    }
                    break;
                }
                k += 2;
            }
            /*
            for (int k = 0; k < comparedArray.length - indexLimit; k += 2) {
                // Check if they are equal
                if (comparedArray[indexArray[j] + k] != comparedArray[pivot + k]) {
                    // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[indexArray[j] + k] <= comparedArray[pivot + k]) {
                        i++;
                        swap(indexArray, i, j);
                    }
                    break;
                }
            }*/
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
