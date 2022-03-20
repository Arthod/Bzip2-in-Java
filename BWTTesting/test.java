import java.util.Arrays;
import java.util.Random;

public class test {
    
    public static void main(String[] args) {
        
        for (int i = 0; i < 10000; i++) {
            int[] inArr = generateRandomString(10000);
            //int[] inArr = new int[] {210, 140, 210, 210, 105, 70, 0, 0, 0, 0};
            //System.out.println(Arrays.toString(inArr));

            BWTBytes.transform(inArr, new int[] {0});
        }
    }
    /*
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
    }*/

    private static int[] generateRandomString(int length) {
        int[] arr = new int[length];
        Random random = new Random();

        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextInt(8) * 35;
        }

        return arr;
    }
}