import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

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
        int i = 0;
        while ((byteRead = inFile.read()) != -1) {
            original[i] = byteRead;
            i++;
        }
        inFile.close();


        int[] tempArr = original;
        tempArr = BWT.transform(tempArr);
        writeToFile(outFileName, tempArr);
        
        /*
        // Encode
        int[] tempArr = original;
        tempArr = MoveToFront.encode(tempArr);
        tempArr = Huffman.encode(tempArr);

        // Write encoded to file
        writeToFile(encodedFileName, tempArr);

        // Decode
        tempArr = Huffman.decode(tempArr);
        tempArr = MoveToFront.decode(tempArr);

        // Write out to file
        writeToFile(outFileName, tempArr);
        */
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
