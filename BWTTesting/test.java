import java.io.FileInputStream;
import java.util.Arrays;
import java.util.Random;

public class test {
    private static int stringLength = (int) 2_000_000;
    private static int testAmountPer = 3;
    
    
    public static void main(String[] args) {

        for (int i = 0; i < 10000; i++) {
            int[] inArr = generateRandomString(1000);

            System.out.println(i);
            BWTInts.transform(inArr, new int[] {0});
        }
    }
    /*
    public static void main(String[] args) throws Exception {
        int[] inArr;

        long startTime = 0;
        long sumTime = 0;

        sumTime = 0;
        for (int i = 0; i < testAmountPer; i++) {
            inArr = getString();

            startTime = System.nanoTime();
            BWTInts.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTInts kingjamesbible avg. time: " + (sumTime / testAmountPer) / 10e8 + "s");
        
        sumTime = 0;
        for (int i = 0; i < testAmountPer; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTInts.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTInts random avg. time: " + (sumTime / testAmountPer) / 10e8 + "s");
    }*/

    private static int[] getString() throws Exception {
        return readFile("KingJamesBible.txt");//generateRandomString(stringLength);
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

    private static int[] generateRandomString(int length) {
        int[] arr = new int[length];
        Random random = new Random();

        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextInt(8) * 35;
        }

        /*
        for (int i = 100; i < 200; i += 1) {
            arr[i] = 0;
        }
        for (int i = 200; i < 300; i += 2) {
            arr[i] = 0;
        }
        for (int i = 500; i < 800; i += 2) {
            arr[i] = 15;
            arr[i + 1] = 16;
        }
        for (int i = 1100; i < 1400; i += 1) {
            arr[i] = 200;
        }
*/
        return arr;
    }
}