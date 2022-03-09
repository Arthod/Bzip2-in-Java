import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class BWT {

    public static int[] transform(int[] S) {
        int N = S.length;

        // Create new array with k EOF characters at the end
        int[] sArr = new int[N + 4];
        for (int i = 0; i < N; i++) {
            sArr[i] = S[i];
        }

        // Append 4 characters to end of string
        for (int i = 0; i < 4; i++) {
            sArr[N + i] = 0;
        }

        // Create array W of N words. Pack 4 bytes into 1 word (integer)    Q2
        int[] W = new int[N];
        for (int i = 0; i < N; i++) {
            W[i] = bytesToWords(sArr[i], sArr[i + 1], sArr[i + 2], sArr[i + 3]);
        }
        
        // Array V  Q4
        ArrayList<Integer> indexList = new ArrayList<Integer>(W.length);// new int[S.length];
        for (int i = 0; i < W.length; i++) {
            indexList.add(i);
        }
        int[] V = sortIndexArray(sArr, indexList);

        // Q5
        int amountComparedEqualTotal = 0;  // number of characters that have been compared equal
        int first = 0;
        for (int ch1 = 0; ch1 < 256 && amountComparedEqualTotal < S.length - 1; ch1++) {
            // Q6
            for (int ch2 = 0; ch2 < 256 && amountComparedEqualTotal < S.length - 1; ch2++) {
                // We know that V is sorted by the first two characters
                first = amountComparedEqualTotal;
                
                while (ch1 == sArr[V[amountComparedEqualTotal]] && ch2 == sArr[V[amountComparedEqualTotal] + 1]) {
                    amountComparedEqualTotal++;
                    
                    // If reached end of input, stop all
                    if (amountComparedEqualTotal == S.length - 1) {
                        break;
                    }
                }

                // If there's atleast two that are compared equal we need to sort them.
                if (amountComparedEqualTotal - first >= 2) {
                    quicksortIndexArray(V, W, first, amountComparedEqualTotal - 1);
                }
            }
        }
        System.out.println(Arrays.toString(V));
        
        // Now our V has the correctly sorted indicies of the square
        // Now we fetch the last column of the BWT square
        int[] outArr = new int[N];
        for (int i = 0; i < outArr.length; i++) {
            if (V[i] == 0) {
                // i is the index of the row of the original string
                outArr[i] = S[S.length - 1];
            } else {
                outArr[i] = S[V[i] - 1];
            }
        }


        return outArr;
    }

    private static int bytesToWords(int a, int b, int c, int d) {
        // Big endian
        return ((a << 24) + (b << 16) + (c << 8) + (d << 0));
    }

    private static int[] sortIndexArray(int[] comparedArray, ArrayList<Integer> indexList) {
        // Q4 - Create index array V and sort it using
        // the first two characters of S as sort keys   

        // Sort V based on the characters    TODO, implement comparison of first two characters in a better way (or correctly).
        Collections.sort(indexList, (i1, i2) -> 255 * (comparedArray[i1] - comparedArray[i2]) + (comparedArray[i1 + 1] - comparedArray[i2 + 1]));

        System.out.println(indexList.toString());

        // Insert the into int array
        int[] indexArray = new int[indexList.size()];
        for (int i = 0; i < indexArray.length; i++) {
            indexArray[i] = indexList.get(i);
        }

        System.out.println(Arrays.toString(indexArray));
        return indexArray;
    }

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
            // TODO comparison should not only be on 1 word, but on all words until biggest one found.

            for (int k = 0; k < comparedArray.length; k += 4) {
                // Check if they are equal
                if (comparedArray[indexArray[j] + k] != comparedArray[pivot + k]) {
                    // If they are not equal, check comparison, and swap if greater than pivot
                    if (comparedArray[indexArray[j] + k] > comparedArray[pivot + k]) {
                        i++;
                        swap(indexArray, j, i);
                        break;
                    }
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
