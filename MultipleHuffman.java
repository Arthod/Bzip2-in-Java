
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class MultipleHuffman {
    private static int CHAR_MAX = 256;
    private static int CODE_LENGTH_MAX = 15;    // Initial max code length, default 15
    private static int CODE_LENGTH_MIN = 0;


    public static int[] encode(int[] inArr, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE) throws IOException {
        // Count frequency of each character in the text
        int[] frequency = new int[CHAR_MAX + 1];
        for (int i = 0; i < inArr.length; i++) {
            frequency[inArr[i]]++;
        }

        // Initialize initial trees
        // Initial intervals of symbols where (100/TREES_COUNT)% frequency is covered
        int[][] codeLengths = new int[TREES_COUNT][CHAR_MAX + 1];
        int low = 0;
        int frequencyRemaining = inArr.length;
        for (int i = 0; i < TREES_COUNT; i++) {
            int frequencyTreeTotal = frequencyRemaining / (TREES_COUNT - i);
            int frequencyTreeCounter = 0;
            int high = low - 1;

            // Get initial highs and lows
            while (frequencyTreeCounter < frequencyTreeTotal && high <= CHAR_MAX) {
                high++;
                frequencyTreeCounter += frequency[high];
            }

            // Remove last high if not needed
            if (high > low && i != 0 && i % 2 == 1) {
                frequencyTreeCounter -= frequency[high];
                high--;
            }

            // Print
            String s = String.format("initial group %d, [%d .. %d], has %d syms (%4.1f%%)",
                i, low, high, frequencyTreeCounter, (100.0 * (float) frequencyTreeCounter) / (float) (inArr.length));
            System.out.println(s);

            // Set code lengths for the interval. 0 if in interval, 15 otherwise.
            for (int j = 0; j <= CHAR_MAX; j++) {
                if (j >= low && j <= high) {
                    codeLengths[i][j] = CODE_LENGTH_MIN;    // is alrady per default zero, TODO: Remove
                } else {
                    codeLengths[i][j] = CODE_LENGTH_MAX;
                }
            }

            low = high + 1;
            frequencyRemaining -= frequencyTreeCounter;
        }

        /// Iterate TREES_IMPROVE_ITER amount of times to improve the huffman code lengths
        /// Each time we regenerate the code lengths for each huffman tree by picking the one
        /// that has the lowest cost for each group of the length 50.
        
        // Selectors table, which table per block_size, calculate cheapest tree
        int[] selectors = new int[(int) Math.ceil(inArr.length / (double) BLOCK_SIZE)];
        int[][] treeFrequencies = new int[TREES_COUNT][CHAR_MAX + 1];
        for (int i = 0; i < TREES_IMPROVE_ITER; i++) {
            treeFrequencies = new int[TREES_COUNT][CHAR_MAX + 1];
            // Go through each block_size
            for (int blockCurrent = 0; blockCurrent < selectors.length; blockCurrent++) {
                int treeCheapest = -1;
                int treeCheapestCost = Integer.MAX_VALUE;

                // Iterate through all trees
                for (int j = 0; j < TREES_COUNT; j++) {
                    int treeCost = 0;

                    // Get cost of this tree in this block
                    for (int k = 0; k < BLOCK_SIZE; k++) {
                        int idx = blockCurrent * BLOCK_SIZE + k;
                        if (idx == inArr.length)
                            break;
                        int byteRead = inArr[idx];
                        treeCost += codeLengths[j][byteRead];
                    }

                    // If it is cheaper, set it for this selector
                    if (treeCost < treeCheapestCost) {
                        treeCheapest = j;
                        treeCheapestCost = treeCost;
                    }
                }

                // Set frequencies for the cheapest tree
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    int idx = blockCurrent * BLOCK_SIZE + k;
                    if (idx == inArr.length)
                        break;
                    int byteRead = inArr[idx];
                    treeFrequencies[treeCheapest][byteRead]++;
                }

                // Set cheapest for this selector
                selectors[blockCurrent] = treeCheapest;

            }

            // Now we have new frequencies (and selectors)
            // If not last improve iteration, regenerate code lengths
            if (i != TREES_IMPROVE_ITER - 1) {
                // Regenerate Huffman code lengths
                for (int j = 0; j < TREES_COUNT; j++) {
                    Element root = Huffman.huffmanTree(treeFrequencies[j]);
                    codeLengths[j] = generateCodeLengths(root);
                }
            }
        }

		// Bit output
		IntArrayOutputStream outArrayStream = new IntArrayOutputStream(inArr.length);
		BitOutputStream outBit = new BitOutputStream(outArrayStream);

        // Write frequencies of to out
        for (int i = 0; i < TREES_COUNT; i++) {
            for (int j = 0; j <= CHAR_MAX; j++) {
                outBit.writeInt(treeFrequencies[i][j]);
            }
        }

        // Generate codes for the huffman trees with their frequencies
        String[][] treeCodes = new String[TREES_COUNT][CHAR_MAX + 1];
        for (int i = 0; i < TREES_COUNT; i++) {
            Element root = Huffman.huffmanTree(treeFrequencies[i]);
            treeCodes[i] = Huffman.generateCode(root);
        }

        // if trees amount is 6, we need 3 bits to represent. 001, 010, 011, etc.
        int bitsNeeded = (int) Math.ceil(Math.log(TREES_COUNT) / Math.log(2));

        // Go through each block_size
        for (int blockCurrent = 0; blockCurrent < selectors.length; blockCurrent++) {
            // Get codes from the tree of current selector
            String[] codes = treeCodes[selectors[blockCurrent]];

            // Write constant bit representation of which Huffman tree is used now 
            String treeSelectedBits = Integer.toBinaryString(selectors[blockCurrent]);
            while (treeSelectedBits.length() < bitsNeeded) {
                treeSelectedBits = "0" + treeSelectedBits;
            }
            for (int i = 0; i < treeSelectedBits.length(); i++) {
                outBit.writeBit(Character.getNumericValue(treeSelectedBits.charAt(i)));
            }

            // Iterate through all bytes in block
            for (int i = 0; i < BLOCK_SIZE; i++) {
                int idx = blockCurrent * BLOCK_SIZE + i;
                if (idx == inArr.length)
                    break;
                int byteRead = inArr[idx];
                String code = codes[byteRead];

                // write the bit representation of that byte
                for (int j = 0; j < code.length(); j++) {
                    outBit.writeBit(Character.getNumericValue(code.charAt(j)));
                }
            }
        }

        // DEBUG: print selectors frequencies
        int[] selectorsFrequencies = new int[TREES_COUNT];
        for (int i = 0; i < selectors.length; i++) {
            selectorsFrequencies[selectors[i]]++;
        }
        System.out.println("Selectors Frequencies: " + Arrays.toString(selectorsFrequencies));

        // Close out bit stream
		outBit.close();

        // Return byte array of outArrayStream
        return outArrayStream.toIntArray();
    }
    
    public static int[] decode(int[] inArr, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE) throws IOException {
        // Bit input
        IntArrayInputStream inArrayStream = new IntArrayInputStream(inArr);
        BitInputStream inBit = new BitInputStream(inArrayStream);
        ArrayList<Integer> outList = new ArrayList<Integer>();

        // Read frequencies from array
        int[][] treeFrequencies = new int[TREES_COUNT][CHAR_MAX + 1];
        int sum = 0;
        for (int i = 0; i < TREES_COUNT; i++) {
            for (int j = 0; j <= CHAR_MAX; j++) {
                treeFrequencies[i][j] = inBit.readInt();
                sum += treeFrequencies[i][j];
            }
        }

        // Generate Huffman trees
        Element[] roots = new Element[TREES_COUNT];
        for (int i = 0; i < TREES_COUNT; i++) {
            roots[i] = Huffman.huffmanTree(treeFrequencies[i]);
        }

        // if trees amount is 6, we need 3 bits to represent. 001, 010, 011, etc.
        int bitsNeeded = (int) Math.ceil(Math.log(TREES_COUNT) / Math.log(2));


        // Read bitsNeeded which is what tree for this current block, then read block and decompress it
        int leafReachedCountTotal = 0;
        while (leafReachedCountTotal < sum) {
            // Read bitsNeeded amount of bits
            String treeSelectedBits = "";
            for (int i = 0; i < bitsNeeded; i++) {
                treeSelectedBits = treeSelectedBits + inBit.readBit();
            }
            int treeSelected = Integer.parseInt(treeSelectedBits, 2);

            Element elementAt = roots[treeSelected];
            int leafReachedCount = 0;

            while (leafReachedCount < BLOCK_SIZE && leafReachedCountTotal < sum) {
                if (elementAt.getData() instanceof Node) {
                    int nextBit = inBit.readBit();
                    if (nextBit == 0) {
                        elementAt = ((Node) elementAt.getData()).getLeft();
                    } else {
                        elementAt = ((Node) elementAt.getData()).getRight();
                    }
                } else {
                    outList.add((int) elementAt.getData());
                    elementAt = roots[treeSelected];
                    leafReachedCount++;
                    leafReachedCountTotal++;
                }
            }
            //leafReachedCountTotal += leafReachedCount;
        }
        
        // Close in bit reader
        inBit.close();

        // TODO: Optimize later?
        // https://stackoverflow.com/questions/718554/how-to-convert-an-arraylist-containing-integers-to-primitive-int-array
        return outList.stream().mapToInt(i -> i).toArray();
    }

	/*
	 * Recursive inorder tree walk, sets a code for all leaves of the tree. The
 	 * code of a specific leaf is the boolean left and right traverses needed to 
 	 * reach it from the root.
	 */
	private static void generateCodeLengths(Element e, int counter, int[] codeLength) {
		if (e.getData() instanceof Node) {
			Node n = (Node) e.getData();
			generateCodeLengths(n.getLeft(), counter + 1, codeLength);
			generateCodeLengths(n.getRight(), counter + 1, codeLength);
		} else {
			codeLength[(int) e.getData()] = counter;
		}
	}

    private static int[] generateCodeLengths(Element e) {
        int[] codeLengths = new int[CHAR_MAX + 1];
		generateCodeLengths(e, 0, codeLengths);

        return codeLengths;
    }
}
