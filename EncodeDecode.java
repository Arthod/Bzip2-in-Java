import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
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
            k++;
        }
        inFile.close();

        // Compression
        int[] tempArr = original.clone();
        System.out.println(original.length);
        tempArr = BWT.transform(tempArr);
        System.out.println(tempArr.length);
        //tempArr = MoveToFront.encode(tempArr);
        //tempArr = Huffman.encode(tempArr);

        // Write encoded to file
        writeToFile(encodedFileName, tempArr);

        // Decompression
        //tempArr = Huffman.decode(tempArr);
        //tempArr = MoveToFront.decode(tempArr);
        tempArr = BWT.reverseTransform(tempArr);


        // Write out to file
        writeToFile(outFileName, tempArr);

        // Check that compression/decompression returns same string
        System.out.println(original.length);
        System.out.println(tempArr.length);
        if (tempArr.length != original.length) {
            System.out.println("Not same length");
            return;
        }
        for (int i = 0; i < tempArr.length; i++) {
            if (tempArr[i] != original[i]) {
                System.out.println(i);
                System.out.println("Not equal");
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
