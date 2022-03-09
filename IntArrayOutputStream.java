import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;

public class IntArrayOutputStream extends OutputStream {
    private ArrayList<Integer> arrayList;

    public IntArrayOutputStream(int initialCapacity) {
        this.arrayList = new ArrayList<Integer>(initialCapacity);
    }

    @Override
    public void write(int e) throws IOException {
        arrayList.add(e);
    }

    public int[] toIntArray() {
        // TODO: Optimize later?
        // https://stackoverflow.com/questions/718554/how-to-convert-an-arraylist-containing-integers-to-primitive-int-array
        return arrayList.stream().mapToInt(i -> i).toArray();
    }
}
