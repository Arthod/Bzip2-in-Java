import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

class EncodeDecode {
	public static void main(String[] args) throws Exception {        
		// File input and bit output
		String inFileName = args[0];
		String outFileName = args[1];
        String tempFileName = "temp.txt";
        String encodedFileName = "encoded.txt";
        int[] out;
        int[] encoded;

        FileInputStream inFile = new FileInputStream(inFileName);
        FileOutputStream tempFile = new FileOutputStream(tempFileName);

        // Turn inFile into array
        int bytesAmount = (int) inFile.getChannel().size();

        // Init array with fixed length
        int[] original = new int[bytesAmount];
        
        // Iterate through all bytes in file and write to array
        int byteRead;
        int i = 0;
        while ((byteRead = inFile.read()) != -1) {
            original[i] = byteRead;
            i++;
        }

        // Start timer
        long timeStart = System.nanoTime();
        
        // Encode
        encoded = MoveToFront.encode(original);
        for (i = 0; i < encoded.length; i++) {
            tempFile.write(encoded[i]);
        }
        Huffman.encode(tempFileName, encodedFileName);

        // Decode
        Huffman.decode(encodedFileName, tempFileName);
        FileInputStream inTempFile = new FileInputStream(tempFileName);
        i = 0;
        while ((byteRead = inTempFile.read()) != -1) {
            encoded[i] = byteRead;
            i++;
        }


        MoveToFront.decode(tempFileName, outFileName);

        // End timer
        long duration = System.nanoTime() - timeStart;

        // Print compression rate and time
        File encodedFile = new File("encoded.txt"); 
        System.out.println("--LL Compression finished--");
        System.out.println("Time: " + duration/10e+8 + " s");
        System.out.println("In File: " + inFile.getChannel().size() + " bytes");
        System.out.println("Out File: " + encodedFile.length() + " bytes");
        System.out.println("Compression: " + (float) encodedFile.length() / inFile.length());
        
        // Delete temp file
        File tempFile = new File(tempFileName); 
        tempFile.delete();
	}
    
}
