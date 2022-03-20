import java.util.Arrays;
import java.util.Random;

public class test {
    
    public static void main(String[] args) {
        Random random = new Random();
        int[] inArr;

        long startTime = 0;
        long sumTime = 0;

        for (int i = 0; i < 10; i++) {
            inArr = new int[2000000];
            for (int j = 0; j < inArr.length; j++) {
                inArr[j] = random.nextInt(255);
            }

            startTime = System.nanoTime();
            BWTInts.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTInts avg. time: " + (sumTime / 10) / 10e8 + "s");

        sumTime = 0;
        for (int i = 0; i < 10; i++) {
            inArr = new int[2000000];
            for (int j = 0; j < inArr.length; j++) {
                inArr[j] = random.nextInt(255);
            }

            startTime = System.nanoTime();
            BWTLongs.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTLongs avg. time: " + (sumTime / 10) / 10e8 + "s");

        sumTime = 0;
        for (int i = 0; i < 10; i++) {
            inArr = new int[2000000];
            for (int j = 0; j < inArr.length; j++) {
                inArr[j] = random.nextInt(255);
            }

            startTime = System.nanoTime();
            BWTBytes.transform(inArr, new int[] {0});
            sumTime += System.nanoTime() - startTime;
        }
        System.out.println("BWTBytes avg. time: " + (sumTime / 10) / 10e8 + "s");
        sumTime = 0;
    }
}