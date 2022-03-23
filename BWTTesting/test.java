import java.io.FileInputStream;
import java.util.Arrays;
import java.util.Random;

public class test {
    private static int stringLength = (int) 2_000_000;
    private static int testAmountPer = 1;
    
    public static void main(String[] args) throws Exception {
        //testKorrekthed();
        testTime();
    }

    private static void testKorrekthed() throws Exception {
        int[] inArr;

        System.out.println("KingJamesBible test");
        inArr = getString();
        BWTInts.transform(inArr, new int[] {0});

        System.out.println("Random tests");
        for (int i = 0; i < 100000; i++) {
            inArr = generateRandomString(1000);

            System.out.println(i);
            BWTInts.transform(inArr, new int[] {0});
        }        
    }

    private static void testTime() throws Exception {
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
            inArr = getString();

            startTime = System.nanoTime();
            BWTIntsDev.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTIntsDev kingjamesbible avg. time: " + (sumTime / testAmountPer) / 10e8 + "s");
        
        sumTime = 0;
        for (int i = 0; i < testAmountPer * 10; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTInts.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTInts random avg. time: " + (sumTime / testAmountPer) / 10e8 + "s");
        sumTime = 0;
        for (int i = 0; i < testAmountPer * 10; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTIntsDev.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTIntsDev random avg. time: " + (sumTime / testAmountPer) / 10e8 + "s");
    }

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