
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class Huffman {
	private static String[] codes = new String[256];
    private static int[] frequency = new int[256];

    public static void encode(String inFileName, String outFileName) throws IOException {
		// File input and bit output
		FileInputStream inFile = new FileInputStream(inFileName);
		FileOutputStream outFile = new FileOutputStream(outFileName);
		BitOutputStream outBit = new BitOutputStream(outFile);

		// Reading input file
		int inputRead;
		while ((inputRead = inFile.read()) != -1) {
			frequency[inputRead]++;
		}

		// Generating the Huffman tree and then the codes
		Element root = huffmanTree(frequency);
		generateCode(root, "");

		// Writing frequency to file
		for (int i = 0; i < 256; i++) {
			outBit.writeInt(frequency[i]);
		}

		// Reading input file again, for every byte write its code bits to file
		inFile.close();
		inFile = new FileInputStream(inFileName);
		while ((inputRead = inFile.read()) != -1) {
			String code = codes[inputRead];
			for (int j = 0; j < code.length(); j++) {
				outBit.writeBit(Character.getNumericValue(code.charAt(j)));
			}
		}
		outBit.close();
		inFile.close();
    }
    
    public static void decode(String inFileName, String outFileName) throws Exception {
        // Bit input and file output
        FileInputStream inFile = new FileInputStream(inFileName);
        BitInputStream inBit = new BitInputStream(inFile);
        FileOutputStream outFile = new FileOutputStream(outFileName);

        // Read and sum the frequencies from file
        int sum = 0;
        for (int i = 0; i < frequency.length; i++) {
            frequency[i] = inBit.readInt();
            sum += frequency[i];
        }

        // Generate Huffman tree
        Element root = huffmanTree(frequency);

        // Decoding the codes and writing data to the out-file.
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

	/*
	 * Generates a Huffman tree from an integer array of frequencies. Node objects are
 	 * used to store elements in a binary tree.
	 */
	public static Element huffmanTree(int[] frequency) {
		int l = frequency.length;
		PQHeap pq = new PQHeap();
		for (int i = 0; i < l; i++) {
			pq.insert(new Element(frequency[i], i));
		}
		for (int i = 0; i < l-1; i++) {
			Element left = pq.extractMin();
			Element right = pq.extractMin();
			pq.insert(new Element(left.getKey() + right.getKey(), new Node(left, right)));
		}
		return pq.extractMin();
	}

	/*
	 * Recursive inorder tree walk, sets a code for all leaves of the tree. The
 	 * code of a specific leaf is the boolean left and right traverses needed to 
 	 * reach it from the root.
	 */
	private static void generateCode(Element e, String tempcode) {
		if (e.getData() instanceof Node) {
			Node n = (Node) e.getData();
			generateCode(n.getLeft(), tempcode + "0");
			generateCode(n.getRight(), tempcode + "1");
		} else {
			codes[(int) e.getData()] = tempcode;
		}
	}
}
