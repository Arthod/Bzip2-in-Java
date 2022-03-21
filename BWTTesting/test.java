import java.util.Arrays;
import java.util.Random;

public class test {
    
    public static void main(String[] args) {
        int stringLength = (int) 1_000_000;
        int[] inArr;

        long startTime = 0;
        long sumTime = 0;

        sumTime = 0;
        for (int i = 0; i < 10; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTLongs.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTLongs avg. time: " + (sumTime / 10) / 10e8 + "s");

        for (int i = 0; i < 10; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTInts.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTInts avg. time: " + (sumTime / 10) / 10e8 + "s");

        sumTime = 0;
        for (int i = 0; i < 10; i++) {
            inArr = generateRandomString(stringLength);

            startTime = System.nanoTime();
            BWTBytes.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTBytes avg. time: " + (sumTime / 10) / 10e8 + "s");
        sumTime = 0;
    }

    private static int[] generateRandomString(int length) {
        int[] arr = new int[length];
        Random random = new Random();

        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextInt(8) * 35;
        }

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

        return arr;
    }
}