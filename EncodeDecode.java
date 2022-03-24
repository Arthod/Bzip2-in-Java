import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

// Issue: when BWT transforming, if there are more than 256 rows, the original rowId
//      can exceed 255, which when used in MTF results in error and inconsistent behavior
// Issue: Huffman encoding and decoding. Should be from file to file. Error
//      happens when I use arrays.

class EncodeDecode {
	public static void main(String[] args) throws Exception {        
		// File input and bit output
		String inFileName = args[0];
        String encodedFileName = "encoded.txt";
        String outFileName = "decoded.txt";

        int[] inArr = readFile(inFileName);

        // Compression
        int[] tempArr = inArr.clone();
        int[] rowId = new int[1];

        System.out.println("BWT transforming");
        tempArr = BWT.transform(tempArr, rowId);
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

        // Check that compression/decompression returns same string
        if (tempArr.length != inArr.length) {
            System.out.println("Not same length");
            return;
        }

        for (int i = 0; i < tempArr.length; i++) {
            if (tempArr[i] != inArr[i]) {
                System.out.print("Not equal at index " + i + ", where \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) tempArr[i + j]}));
                }
                System.out.print("\" =/= \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) inArr[i + j]}));
                }
                System.out.println("\"");
                return;
            }
        }
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
}
