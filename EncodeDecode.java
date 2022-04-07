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

        int[] S = {1,1,1,1,1,1,1,1,1,1,3};
        int[] rowId = {0};
        
        System.out.println("arrIn: " + Arrays.toString(S) + ", runs: " + countRuns(S) + ", sum: " + sumArr(S));
        int[] outArr = S.clone();
        for (int i = 0; i < 10; i++) {
            outArr = BWT2.transform(outArr, rowId);
            System.out.println("arrOut: " + Arrays.toString(outArr) + ", runs: " + countRuns(outArr) + ", sum: " + sumArr(outArr));
        }
        

        //encodeDecodeFile(args[0]);

	}

    private static void encodeDecodeFile(String inFileName) throws Exception {

		// File input and bit output
        String encodedFileName = "encoded.txt";
        String outFileName = "decoded.txt";

        int[] inArr = readFile(inFileName);

        // Compression
        int[] tempArr = inArr.clone();
        int[] rowId = new int[1];

        System.out.println("BWT transforming");
        tempArr = BWT2.transform(tempArr, rowId);
        System.out.println("MTF encoding");
        tempArr = MoveToFront.encode(tempArr);
        System.out.println("Huffman encoding");
        tempArr = Huffman.encode(tempArr);

        // Write encoded to file, and read again from it (this step is required for Huffman encoding to work (for some reason...))
        System.out.println("Writing encoded to file");
        writeToFile(encodedFileName, tempArr);
        tempArr = readFile(encodedFileName);

        // Decompression
        System.out.println("Huffman decoding");
        tempArr = Huffman.decode(tempArr);
        System.out.println("MTF decoding");
        tempArr = MoveToFront.decode(tempArr);
        System.out.println("BWT reverse transforming");
        tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        
        // Write out to file
        System.out.println("Writing decoded to file");
        writeToFile(outFileName, tempArr);

        
        isEqual(tempArr, inArr);
    }
    
    private static int[] readFile(String inFileName) throws Exception {
        FileInputStream inFile = new FileInputStream(inFileName);

        // Turn inFile into array
        int bytesAmount = (int) inFile.getChannel().size();
        int[] arr = new int[bytesAmount];
        
        // Iterate through all bytes in file and write to array
        int byteRead = 0;
        int k = 0;
        while ((byteRead = inFile.read()) != -1) {
            arr[k] = byteRead;
            k++;
        }
        inFile.close();

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

    private static int sumArr(int[] arr) {
        int sum = 0;
        for (int el : arr) {
            sum += el;
        }
        return sum;
    }
}
