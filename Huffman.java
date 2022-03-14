
import java.io.IOException;
import java.util.ArrayList;

public class Huffman {
	private static String[] codes = new String[256];
    private static int[] frequency = new int[256];

    public static int[] encode(int[] inArr) throws IOException {
		// Bit output
		IntArrayOutputStream outArrayStream = new IntArrayOutputStream(inArr.length);
		BitOutputStream outBit = new BitOutputStream(outArrayStream);

		// Reading input array
        for (int i = 0; i < inArr.length; i++) {
            frequency[inArr[i]]++;
        }

		// Generating the Huffman tree and then the codes
		Element root = huffmanTree(frequency);
		generateCode(root);

		// Writing frequency to file
		for (int i = 0; i < frequency.length; i++) {
			outBit.writeInt(frequency[i]);
		}

		// Reading input file again, for every byte write its code bits to file
        for (int i = 0; i < inArr.length; i++) {
			String code = codes[inArr[i]];
			for (int j = 0; j < code.length(); j++) {
				outBit.writeBit(Character.getNumericValue(code.charAt(j)));
			}            
        }

        // Close out bit stream
		outBit.close();

        // Return byte array of outArrayStream
        return outArrayStream.toIntArray();
    }
    
    public static int[] decode(int[] inArr) throws Exception {
        // Bit input
        IntArrayInputStream inArrayStream = new IntArrayInputStream(inArr);
        BitInputStream inBit = new BitInputStream(inArrayStream);
        ArrayList<Integer> outList = new ArrayList<Integer>();

        // Read and sum the frequencies from file
        int sum = 0;
        for (int i = 0; i < frequency.length; i++) {
            frequency[i] = inBit.readInt();
            sum += frequency[i];
        }

        // Generate Huffman tree
        Element root = huffmanTree(frequency);

        // DEBUG
        /*
        for (int i = 0; i < frequency.length; i++) {
            if (frequency[i] > 0) {
                System.out.println(i + ", " + codes[i] + ": " + frequency[i]);
            }
        }
        */

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
                outList.add((int) elementAt.getData());
                elementAt = root;
                leafReachedCount++;
            }
        }
        
        // Close in bit reader
        inBit.close();

        // TODO: Optimize later?
        // https://stackoverflow.com/questions/718554/how-to-convert-an-arraylist-containing-integers-to-primitive-int-array
        return outList.stream().mapToInt(i -> i).toArray();
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
    private static void generateCode(Element e) {
		generateCode(e, "");
    }
}
