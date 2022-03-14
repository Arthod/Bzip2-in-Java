import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class BWT {
    private static final int unusedByte = 0;

    public static int[] naiveTransform(int[] S) {
        int N = S.length + 1;
        // TODO.. make this more efficient
        int[] SS = new int[S.length + 1];
        for (int i = 0; i < S.length; i++) {
            SS[i] = S[i];
        }
        SS[S.length] = unusedByte;
        S = SS;


        int[][] arrNew = new int[S.length][S.length];
        
        for (int i = 0; i < S.length; i++) {
            for (int j = 0; j < S.length; j++) {
                arrNew[i][j] = S[Math.floorMod(i + j, S.length)];
            }
        }

        Arrays.sort(arrNew, (arr1,arr2) -> {
            for (int i = 0; i < arr1.length; i++) {
                if (arr1[i] != arr2[i]) {
                    return arr1[i] - arr2[i];
                }
            }
            return 0;
        });

        
        int[] outArr = new int[S.length + 1];
        int rowId = 0;
        for (int i = 0; i < outArr.length - 1; i++) {
            outArr[i] = arrNew[i][S.length - 1];
            if (outArr[i] == unusedByte) {
                rowId = i;
            }
        }
        outArr[outArr.length - 1] = rowId;
        
        return outArr;
    }

    public static int[] transform(int[] S) {
        // Create new array with k=5 EOF characters at the end
        int[] sArr = new int[S.length + 5];
        for (int i = 0; i < S.length; i++) {
            sArr[i] = S[i];
        }
        // Append 5 characters to end of string
        for (int i = 0; i < 5; i++) {
            sArr[sArr.length - 5 + i] = unusedByte;
        }
        System.out.println(Arrays.toString(sArr));

        // Create array W of N words. Pack 4 bytes into 1 word (integer)    Q2
        int[] W = new int[S.length + 1];
        for (int i = 0; i < W.length; i++) {
            W[i] = bytesToWords(sArr[i], sArr[i + 1], sArr[i + 2], sArr[i + 3]);
        }
        
        // Array V  Q4
        // Sort by first two characters using radix sort using counting sort
        int[] count = new int[256];
        int[] V_temp = new int[W.length];
        int[] V = new int[W.length];
        int k;
        
        /// Sort by second character
        // Count amount of characters
        for (int i = 0; i < W.length; i++) {
            count[sArr[i + 1]]++;
        }
        // For each index, add the previous index
        for (int i = 1; i < 256; i++) {
            count[i] = count[i] + count[i - 1];
        }
        // Go in reverse, and write the index
        for (int i = W.length - 1; i >= 0; i--) {
            k = sArr[i + 1];
            V_temp[count[k] - 1] = i;
            count[k]--;
        }

        // Sort by first characterv
        // Count amount of characters
        count = new int[256];
        for (int i = 0; i < W.length; i++) {
            count[sArr[i]]++;
        }
        // For each index, add the previous index
        for (int i = 1; i < 256; i++) {
            count[i] = count[i] + count[i - 1];
        }
        // Go in reverse, and write the index of the index
        for (int i = W.length - 1; i >= 0; i--) {
            k = sArr[V_temp[i]];
            V[count[k] - 1] = V_temp[i];
            count[k]--;
        }
        

        /*
        ArrayList<Integer> indexList = new ArrayList<Integer>(W.length);// new int[S.length];
        for (int i = 0; i < W.length; i++) {
            indexList.add(i);
        }
        int[] V3 = sortIndexArray(sArr, indexList);
        System.out.println("V:  " + Arrays.toString(V3));
        */

        // Q5
        int amountComparedEqualTotal = 0;  // number of characters that have been compared equal
        int first = 0;
        for (int ch1 = 0; ch1 < 256 && amountComparedEqualTotal < W.length - 1; ch1++) {
            // Q6
            for (int ch2 = 0; ch2 < 256 && amountComparedEqualTotal < W.length - 1; ch2++) {
                // We know that V is sorted by the first two characters
                first = amountComparedEqualTotal;
                
                while (ch1 == sArr[V[amountComparedEqualTotal]] && ch2 == sArr[V[amountComparedEqualTotal] + 1]) {
                    amountComparedEqualTotal++;
                    
                    // If reached end of input, stop all
                    if (amountComparedEqualTotal == W.length - 1) {
                        break;
                    }
                }

                // If there's atleast two that are compared equal we need to sort them.
                if (amountComparedEqualTotal - first >= 2) {
                    quicksortIndexArray(V, W, first, amountComparedEqualTotal - 1);
                }
            }
        }
        
        // Now our V has the correctly sorted indicies of the square
        // Now we fetch the last column of the BWT square
        // We also write the row index to the last index of the out array
        int[] outArr = new int[W.length + 1];
        for (int i = 0; i < outArr.length - 1; i++) {
            if (V[i] == unusedByte) {
                // i is the index of the row of the original string
                outArr[outArr.length - 1] = i;
                outArr[i] = sArr[W.length - 1];
            } else {
                outArr[i] = sArr[V[i] - 1];
            }
        }

        return outArr;
    }

    /*
     *  Reverses the BWT transformation. Assumes the int in inArr is the row id of the original string
     */
    public static int[] reverseTransform(int[] inArr) {
        // P[i] is the number of instances of character L[i] in L[0, 1, ..., i-1]
        int[] P = new int[inArr.length - 1];

        // count[ch] is the total number of instances in L, of characters preceding 
        // character ch in the alphabet.
        int[] count = new int[256];

        // First pass - 
        //  count[ch] is the amount of times ch appears in L.
        //  P[i] is the number of 
        for (int i = 0; i < inArr.length - 1; i++) {
            P[i] = count[inArr[i]];
            count[inArr[i]]++;
        }

        int sum = 0;
        for (int i = 0; i < 256; i++) {
            int j = count[i];
            count[i] = sum;
            sum += j;
        }

        int[] outArr = new int[inArr.length - 2];
        int i = inArr[inArr.length - 1];
        i = P[i] + count[inArr[i]];
        for (int j = outArr.length - 1; j >= 0; j--) {
            outArr[j] = inArr[i];
            i = P[i] + count[inArr[i]];
        }

        return outArr;
    }

    private static int bytesToWords(int a, int b, int c, int d) {
        // Big endian
        return ((a << 24) + (b << 16) + (c << 8) + (d << 0));
    }

    /*
    private static int[] sortIndexArray(int[] comparedArray, ArrayList<Integer> indexList) {
        // Q4 - Create index array V and sort it using
        // the first two characters of S as sort keys

        // Sort V based on the characters    TODO, implement comparison of first two characters in a better way (or correctly).
        Collections.sort(indexList, (i1, i2) -> 255 * (comparedArray[i1] - comparedArray[i2]) + (comparedArray[i1 + 1] - comparedArray[i2 + 1]));


        // Insert the into int array
        int[] indexArray = new int[indexList.size()];
        for (int i = 0; i < indexArray.length; i++) {
            indexArray[i] = indexList.get(i);
        }

        return indexArray;
    }
    */

    private static void quicksortIndexArray(int[] indexArray, int[] comparedArray, int startIndex, int endIndex) {
        // In-place implementation of quicksort that sorts an index 
        // array based on the comparable values of a comparable array
        if (startIndex >= endIndex) {
            return;
        }
        int q = partition(indexArray, comparedArray, startIndex, endIndex);
        quicksortIndexArray(indexArray, comparedArray, startIndex, q - 1);
        quicksortIndexArray(indexArray, comparedArray, q + 1, endIndex);
    }

    private static int partition(int[] indexArray, int[] comparedArray, int startIndex, int endIndex) {
        int pivot = indexArray[endIndex];
        int i = startIndex - 1;

        for (int j = startIndex; j <= endIndex - 1; j++) {

            // TODO: can optimize.. We know first two characters of the word are already sorted, no need to recheck them
            int indexLimit = Math.max(indexArray[j], pivot);
            for (int k = 0; k < comparedArray.length - indexLimit; k += 4) {

                // Check if they are equal
                if (comparedArray[indexArray[j] + k] != comparedArray[pivot + k]) {
                // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[indexArray[j] + k] <= comparedArray[pivot + k]) {
                        i++;
                        swap(indexArray, j, i);
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
