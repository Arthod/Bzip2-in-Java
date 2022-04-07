import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;

class BWTHuffmanTesting {
    public static void main(String[] args) throws IOException, Exception {
        
        File folder = new File("cantrbry/");
        File[] testFiles = folder.listFiles();
        String[] testFileNames = new String[testFiles.length];
        for (int i = 0; i < testFiles.length; i++) {
            testFileNames[i] = testFiles[i].getName();
        }
        System.out.println(Arrays.toString(testFileNames));

        // Run and save files with BWT-MTF-Huffman
        System.out.println("BWT-MTF-Huffman");
        for (String fileName : testFileNames) {
            System.out.println(fileName);
            int[] inArr = readFile("cantrbry/" + fileName);
            int[] rowId = new int[1];
            
            int[] tempArr = inArr.clone();
            tempArr = BWT.transform(tempArr, rowId);
            tempArr = MoveToFront.encode(tempArr);
            tempArr = Huffman.encode(tempArr);

            writeToFile("cantrbryBWT/" + fileName, tempArr);
        }

        // Run and save files with Huffman
        System.out.println("Pure Huffman");
        for (String fileName : testFileNames) {
            System.out.println(fileName);
            int[] inArr = readFile("cantrbry/" + fileName);
            
            inArr = Huffman.encode(inArr);

            writeToFile("cantrbryPureHuffman/" + fileName, inArr);
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