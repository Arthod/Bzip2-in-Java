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
    private static int TREES_IMPROVE_ITER = 3; // Amount of times to improve the huffman trees, default 3
    private static int TREES_COUNT = 6; // Amount of huffman trees, default 6
    private static int BLOCK_SIZE = 50; // Bytes block size, default 50
    private static Boolean RLE = true;
    private static int DEBUG_LEVEL = 0; // 0..5
    private static Boolean showSelectorFrequencies = false;

	public static void main(String[] args) throws Exception {
        // Args
        String inFileName = args[0];

        if (args.length > 1) {
            String flag = args[1];
            if (flag.equals("-it")) {
                TREES_IMPROVE_ITER = Integer.parseInt(args[2]);

            } else if (flag.equals("-tc")) {
                TREES_COUNT = Integer.parseInt(args[2]);

            } else if (flag.equals("-bs")) {
                BLOCK_SIZE = Integer.parseInt(args[2]);

            } else if (flag.equals("-rle")) {
                RLE = Boolean.parseBoolean(args[2]);

            } else if (flag.equals("-ssf")) {
                showSelectorFrequencies = Boolean.parseBoolean(args[2]);

            } else {
                //System.out.println("Read no flags");
            }
        } else {
            //System.out.println("Read no flags");
        }
        if (DEBUG_LEVEL >= 1) {
            System.out.println("Input: " + inFileName);
            System.out.println("Trees Improve Iter: " + TREES_IMPROVE_ITER);
            System.out.println("Trees Count: " + TREES_COUNT);
            System.out.println("Block Size: " + BLOCK_SIZE);
            System.out.println("RLE: " + RLE);
            System.out.println("SSF: " + showSelectorFrequencies);
        }

        if (true) {
            testCorrectnessSteps(inFileName);
            return;
        } else {


            // Read raw file
            int[] inArr = readFile(inFileName);
            int[] rowId = new int[1];

            // Compress raw file
            int[] arr = compress(inArr, rowId);
            
            if (!showSelectorFrequencies) {
                System.out.println(arr.length);
            }
            
            String encodedFileName = "encoded.txt";
            String outFileName = "decoded.txt";
            if (DEBUG_LEVEL >= 1) System.out.println("Saving to file encoding");

            // Write encoded file and read encoded file
            writeToFile(encodedFileName, arr);
            arr = readFile(encodedFileName);

            // Decompres encoded file
            arr = decompress(arr, rowId);
            
            // Write decompressed file
            writeToFile(outFileName, arr);

            // Check for errors        
            if (!isEqual(arr, inArr)) {
                System.out.println("ERROR, NOT EQUAL");
                return;
            }
        }
	}

    private static void testCorrectnessSteps(String inFileName) throws Exception {
        System.out.println("Testing correctness all");
        // Read raw file
        int[] inArr = readFile(inFileName);
        int[] rowId = new int[1];
        int[] tempArr = inArr.clone();

        // Test BWT
        tempArr = BWT.transform(tempArr, rowId);
        tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        if (!isEqual(tempArr, inArr)) {
            System.out.println("ERROR, NOT EQUAL bwt not working");
            return;
        }

        tempArr = BWT.transform(tempArr, rowId);
        tempArr = MoveToFront.encode(tempArr, RLE);
        tempArr = MoveToFront.decode(tempArr, RLE);
        tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        if (!isEqual(tempArr, inArr)) {
            System.out.println("ERROR, NOT EQUAL mtf not working");
            return;
        }

        tempArr = BWT.transform(tempArr, rowId);
        tempArr = MoveToFront.encode(tempArr, RLE);
        tempArr = MultipleHuffman.encode(tempArr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE, showSelectorFrequencies);
        
        writeToFile("TESTDELETEME.txt", tempArr);
        tempArr = readFile("TESTDELETEME.txt");
        
        tempArr = MultipleHuffman.decode(tempArr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE);
        tempArr = MoveToFront.decode(tempArr, RLE);
        tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        if (!isEqual(tempArr, inArr)) {
            System.out.println("ERROR, NOT EQUAL mtf not working");
            return;
        }

        System.out.println("All working");
    }


    private static int[] compress(int[] arr, int[] rowId) throws IOException {
        // Compression
        if (DEBUG_LEVEL >= 1) System.out.println("Burrows-Wheeler transform");
        arr = BWT.transform(arr, rowId);

        if (DEBUG_LEVEL >= 1) System.out.println("MoveToFront & Run-Length encoding");
        arr = MoveToFront.encode(arr, RLE);

        if (DEBUG_LEVEL >= 1) System.out.println("Multiple Huffman encoding");
        arr = MultipleHuffman.encode(arr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE, showSelectorFrequencies);
        
        return arr;
    }
    private static int[] decompress(int[] arr, int[] rowId) throws IOException {
        // Decompression
        if (DEBUG_LEVEL >= 1) System.out.println("Multiple Huffman decoding");
        int[] tempArr = MultipleHuffman.decode(arr, TREES_IMPROVE_ITER, TREES_COUNT, BLOCK_SIZE);

        if (DEBUG_LEVEL >= 1) System.out.println("MoveToFront & Run-Length decoding");
        tempArr = MoveToFront.decode(tempArr, RLE);

        if (DEBUG_LEVEL >= 1) System.out.println("Burrows-Wheeler reverse transform");
        tempArr = BWT.reverseTransform(tempArr, rowId[0]);
        
        return tempArr;
    }

    private static int[] readFile(String inFileName) throws Exception {
        FileInputStream inFile = new FileInputStream(inFileName);

        // Turn inFile into array
        int bytesAmount = (int) inFile.getChannel().size();
        int[] arr = new int[bytesAmount];
        
        // Iterate through all bytes in file and write to array
        for (int i = 0; i < bytesAmount; i++) {
            arr[i] = inFile.read();
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

}
