import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

class EncodeDecode {
	public static void main(String[] args) throws Exception {        
		// File input and bit output
		String inFileName = args[0];
        String encodedFileName = "encoded.txt";
        String outFileName = "decoded.txt";

        FileInputStream inFile = new FileInputStream(inFileName);

        // Turn inFile into array
        int bytesAmount = (int) inFile.getChannel().size();
        int[] original = new int[bytesAmount];
        
        // Iterate through all bytes in file and write to array
        int byteRead;
        int k = 0;
        while ((byteRead = inFile.read()) != -1) {
            original[k] = byteRead;
            if (byteRead == 0) {
                System.out.println("zero found");
            }
            k++;
        }
        inFile.close();

        // Compression
        int[] tempArr = original.clone();
        //tempArr = BWT.transform(tempArr);
        tempArr = MoveToFront.encode(tempArr);
        tempArr = Huffman.encode(tempArr);

        // Write encoded to file
        writeToFile(encodedFileName, tempArr);

        // Decompression
        tempArr = Huffman.decode(tempArr);
        tempArr = MoveToFront.decode(tempArr);
        //tempArr = BWT.reverseTransform(tempArr);


        // Write out to file
        writeToFile(outFileName, tempArr);

        // Check that compression/decompression returns same string
        if (tempArr.length != original.length) {
            System.out.println("Not same length");
            return;
        }

        for (int i = 0; i < tempArr.length; i++) {
            if (tempArr[i] != original[i]) {
                System.out.print("Not equal at index " + i + ", where \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) tempArr[i + j]}));
                }
                System.out.print("\" =/= \"");
                for (int j = 0; j < 5; j++) {
                    System.out.print(new String(new byte[] {(byte) original[i + j]}));
                }
                System.out.println("\"");
                return;
            }
        }
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
