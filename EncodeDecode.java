import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;

// Issue: when BWT transforming, if there are more than 256 rows, the original rowId
//      can exceed 255, which when used in MTF results in error and inconsistent behavior
// Issue: Huffman encoding and decoding. Should be from file to file. Error
//      happens when I use arrays.

class EncodeDecode {
	public static void main(String[] args) throws Exception {
        // Args
        String inFileName = args[0];
        String encodedFileName = "encoded.txt";
        String outFileName = "decoded.txt";
        int TREES_IMPROVE_ITER = Integer.parseInt(args[1]);    // Amount of times to improve the huffman trees, default 3
        int TREES_COUNT = Integer.parseInt(args[2]); // Amount of huffman trees, default 6
        int BLOCK_SIZE = Integer.parseInt(args[3]); // Bytes block size, default 50
        Boolean RLE = Boolean.parseBoolean(args[4]);


        // Compression
        int[] inArr = readFile(inFileName);
        int[] tempArr = inArr.clone();
        int[] rowId = new int[1];

        tempArr = BWT.transform(tempArr, rowId);
        tempArr = MoveToFront.encode(tempArr, RLE);
        tempArr = MultipleHuffman.encode(tempArr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE);

        System.out.println(tempArr.length);
        //writeToFile(encodedFileName, tempArr);
        //tempArr = readFile(encodedFileName);

        // Decompression
        //tempArr = MultipleHuffman.decode(tempArr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE);
        //tempArr = MoveToFront.decode(tempArr, RLE);
        //tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        
        // Write out to file
        //writeToFile(outFileName, tempArr);

        
        //if (!isEqual(tempArr, inArr)) {
        //    System.out.println("ERROR, NOT EQUAL");
        //    return;
        //}
	}

    private static int[] readFile(String inFileName) throws Exception {
        FileInputStream inFile = new FileInputStream(inFileName);

        // Turn inFile into array
        int bytesAmount = (int) inFile.getChannel().size();
        int[] arr = new int[bytesAmount + 1];
        
        // Iterate through all bytes in file and write to array
        int byteRead = 0;
        int k = 0;
        while ((byteRead = inFile.read()) != -1) {
            arr[k] = byteRead;
            k++;
        }
        inFile.close();
        arr[arr.length - 1] = 255;

        return arr;
    }

    public static void writeToFile(String fileName, int[] arr) throws IOException {
        FileOutputStream outFileStream = new FileOutputStream(fileName);

        // Write int array to file
        for (int i = 0; i < arr.length; i++) {
            outFileStream.write(arr[i]);
        }

        // Close file
        outFileStream.close();
    }

    private static boolean isEqual(int[] arr1, int[] arr2) {
        // Check that compression/decompression returns same string
        if (arr1.length != arr2.length) {
            System.out.println("Not same length");
            return false;
        }

        for (int i = 0; i < arr1.length; i++) {
            if (arr1[i] != arr2[i]) {
                System.out.print("Not equal at index " + i + ", where \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) arr1[i + j]}));
                }
                System.out.print("\" =/= \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) arr2[i + j]}));
                }
                System.out.println("\"");
                return false;
            }
        }

        return true;
    }

    private static int countRuns(int[] arr) {
        return countRuns(arr, 0);
    }

    private static int countRuns(int[] arr, int index) {
        int runs = 0;
        for (int i = index; i < arr.length + index; i++) {
            if (arr[i % arr.length] > arr[(i + 1) % arr.length]) {
                runs++;
            }
        }

        return runs;
    }

    private static int countEquals(int[] arr) {
        int eqs = 0;
        for (int i = eqs; i < arr.length + eqs; i++) {
            if (arr[i % arr.length] == arr[(i + 1) % arr.length]) {
                eqs++;
            }
        }

        return eqs;
    }

    private static int sumArr(int[] arr) {
        int sum = 0;
        for (int el : arr) {
            sum += el;
        }
        return sum;
    }
}
