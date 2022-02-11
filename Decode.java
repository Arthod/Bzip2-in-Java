// Victor Halgaard Kristensen - vikri19@student.sdu.dk
// Ahmad Mahir Sadaldin Othman - ahoth19@student.sdu.dk 

import java.io.FileInputStream;
import java.io.FileOutputStream;

class Decode { 
    private static int[] frequency = new int[256];

    public static void main(String[] args) throws Exception {
        // Bit input and file output
        FileInputStream inFile = new FileInputStream(args[0]);
        BitInputStream inBit = new BitInputStream(inFile);
        FileOutputStream outFile = new FileOutputStream(args[1]);

        // Read and sum the frequencies from file
        int sum = 0;
        for (int i = 0; i < frequency.length; i++) {
            frequency[i] = inBit.readInt();
            sum += frequency[i];
        }

        // Generate Huffman tree
        Element root = Encode.Huffman(frequency);

        // Decoding the passwords and writing data to the out-file.
        Element elementAt = root;
        int leafReachedCount = 0;
        // The code goes through the tree starting at the top and then descending left or
        // right depending on the next bit. When it finds a leaf it writes the data to the
        // out-file and repeats. This loop ends when the sum of frequencies is reached.
        while (leafReachedCount < sum) {
            if (elementAt.getData() instanceof Node) {
                int nextBit = inBit.readBit(); 
                if (nextBit == 0) {
                    elementAt = ((Node) elementAt.getData()).getLeft();
                } else {
                    elementAt = ((Node) elementAt.getData()).getRight();
                }
            } else {
                outFile.write((int) elementAt.getData());
                elementAt = root;
                leafReachedCount++;
            }
        }
        
        // Closing the files.
        inBit.close();
        outFile.close();
    }
}
