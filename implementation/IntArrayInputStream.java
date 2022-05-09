import java.io.IOException;
import java.io.InputStream;

public class IntArrayInputStream extends InputStream {
    int[] arr;
    int p = 0;

    public IntArrayInputStream(int[] inArr) {
        this.arr = inArr;
    }

    @Override
    public int read() throws IOException {
        if (p >= this.arr.length) {
            return -1;
        }
        return arr[p++];
    }

}
