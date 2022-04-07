import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;

class BWTKTesting {
    public static void main(String[] args) throws IOException, Exception {
        
        File folder = new File("cantrbry");
        File[] testFiles = folder.listFiles();
        String[] testFileNames = new String[testFiles.length];
        for (int i = 0; i < testFiles.length; i++) {
            testFileNames[i] = testFiles[i].getName();
        }
        System.out.println(Arrays.toString(testFileNames));

        for (int i = 1; i <= 10; i++) {
            System.out.println(i);
            for (String fileName : testFileNames) {
                System.out.println(fileName);
                int[] inArr = readFile("cantrbry" + "/" + fileName);
                int[] rowId = new int[1];
                
                int[] tempArr = inArr.clone();
                for (int j = 0; j < i; j++) {
                    tempArr = BWT.transform(tempArr, rowId);
                }
                tempArr = MoveToFront.encode(tempArr);
                tempArr = Huffman.encode(tempArr);

                writeToFile("cantrbry" + i + "/" + fileName, tempArr);
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