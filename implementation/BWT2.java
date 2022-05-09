import java.util.Arrays;
import java.util.Random;

// TODO see if sArr can be removed.. I assume it can
// TODO convert input from int to short, and all other places

public class BWT2 {
    private final static Random random = new Random();
    
    public static int[] transform(int[] S, int[] rowId) {
        

        
        // Array V  Q4
        // Sort by first two characters using radix sort using counting sort
        int[] V_temp = new int[S.length];
        int[] V = new int[S.length];
        int k;
        int[] count = new int[256];
        int[] countCurrent;
        
        // Radix sort
        // Count the characters (skip the first character)
        for (int i = 0; i < S.length; i++) {
            count[S[i]]++;
        }

        countCurrent = new int[count.length];
        countCurrent[0] = count[0];
        for (int i = 1; i < count.length; i++) {
            countCurrent[i] = count[i] + countCurrent[i - 1];
        }
        
        // Go in reverse, and write the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = S[(i + 1) % S.length];
            V_temp[(countCurrent[k] - 1) % S.length] = i;
            countCurrent[k]--;
        }

        /// Sort by first character
        for (int i = 1; i < count.length; i++) {
            count[i] += count[i - 1];
        }

        // Go in reverse, and write the index of the index
        for (int i = S.length - 1; i >= 0; i--) {
            k = S[V_temp[i]];
            V[(count[k] - 1) % S.length] = V_temp[i];
            count[k]--;
        }



        // Test V array is sorted by first two chars
        for (int i = 0; i < V.length - 2; i++) {
            int c = V[i];
            int l = V[i + 1];
            if (S[c] > S[l]) {
                System.out.println("Fail");
            }
            if (S[c] == S[l]) {
                if (S[(c + 1) % S.length] > S[(l + 1) % S.length]) {
                    System.out.println("Fail");
                }
            }
        }

        // Q5
        int amountComparedEqualTotal = 0;  // number of characters that have been compared equal
        int first = 0;
        for (int ch1 = 0; ch1 < 256; ch1++) {
            // Q6
            for (int ch2 = 0; ch2 < 256 && amountComparedEqualTotal < S.length; ch2++) {
                // We know that V is sorted by the first two characters
                first = amountComparedEqualTotal;

                while (ch1 == S[V[amountComparedEqualTotal]] && ch2 == S[(V[amountComparedEqualTotal] + 1) % S.length]) {
                    amountComparedEqualTotal++;
                    
                    // If reached end of input, stop all
                    if (amountComparedEqualTotal == S.length) {
                        break;
                    }
                }

                // If there's atleast two that are compared equal we need to sort them.
                if (amountComparedEqualTotal - first >= 2) {
                    randomizedQuicksortIndexArray(V, S, first, amountComparedEqualTotal - 1);
                }
            }
        }

        // Now our V has the correctly sorted indicies of the square
        // Now we fetch the last column of the BWT square
        // We also write the row index to the last index of the out array
        int[] outArr = new int[S.length];
        for (int i = 0; i < S.length; i++) {
            //outArr[i] = S[V[i] - 1];
            outArr[i] = S[Math.floorMod(V[i] - 1, S.length)];
        }
        //outArr[0] = S[S.length - 1];

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

            for (int k = 2; k < indexArray.length - 2; k++) {
                if (comparedArray[(indexArray[j] + k) % indexArray.length] != comparedArray[(pivot + k) % indexArray.length]) {
                    // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[(indexArray[j] + k) % indexArray.length] <= comparedArray[(pivot + k) % indexArray.length]) {
                        i++;
                        swap(indexArray, i, j);
                    }
                    break;
                }
            }
            /*
            for (int k = 0; k < comparedArray.length; k += 2) {
                // Check if they are equal
                if (comparedArray[indexArray[j] + k + indexLimit] != comparedArray[pivot + k]) {
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
