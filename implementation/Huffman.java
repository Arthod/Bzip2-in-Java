
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Huffman {
    private static int CHAR_MAX = 256;

    public static int[] encode(int[] inArr) throws IOException {
		// Bit output
		IntArrayOutputStream outArrayStream = new IntArrayOutputStream(inArr.length);
		BitOutputStream outBit = new BitOutputStream(outArrayStream);

		// Reading input array
        int[] frequency = new int[CHAR_MAX + 1];
        for (int i = 0; i < inArr.length; i++) {
            frequency[inArr[i]]++;
        }

		// Generating the Huffman tree and then the codes
		Element root = huffmanTree(frequency);
		String[] codes = generateCode(root);

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
        int[] frequency = new int[CHAR_MAX + 1];
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
        int nodesCount = 0;
		PQHeap pq = new PQHeap();
		for (int i = 0; i < l; i++) {
            if (frequency[i] > 0) {
                nodesCount++;
			    pq.insert(new Element(frequency[i], i));
            }
		}
		for (int i = 0; i < nodesCount - 1; i++) {
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
	private static void generateCode(Element e, String[] codes, String tempcode) {
		if (e.getData() instanceof Node) {
			Node n = (Node) e.getData();
			generateCode(n.getLeft(), codes, tempcode + "0");
			generateCode(n.getRight(), codes, tempcode + "1");
		} else {
			codes[(int) e.getData()] = tempcode;
		}
	}
    public static String[] generateCode(Element root) {
        String[] codes = new String[CHAR_MAX + 1];
		generateCode(root, codes, "");

        return codes;
    }

    public static String[] generateCodesFromLengths(int[] codeLengths) {
        int minLength = 1000;
        int maxLength = 0;

        // Get min and max code lengths
        for (int i = 0; i < codeLengths.length; i++) {
            int codeLength = codeLengths[i];
            
            minLength = Math.max(Math.min(codeLength, minLength), 1);
            maxLength = Math.max(codeLength, maxLength);
        }

        // Codes int
        int code = 0;
        String[] codes = new String[codeLengths.length];
        for (int len = minLength; len <= maxLength; len++) {
            for (int c = 0; c <= CHAR_MAX; c++) {
                if (len == codeLengths[c]) {
                    codes[c] = String.format("%" + codeLengths[c] + "s", Integer.toBinaryString(code))
                        .replace(' ', '0');
                    code++;
                }
            }
            code = code << 1;
        }

        // Return
        return codes;
    }
}
