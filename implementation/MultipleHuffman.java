
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MultipleHuffman {
    private static int CHAR_MAX = 257;  // 0 = runA, 1 = runB, 2 = 1, 3 = 2, ..., 256 = 255, 257 = EOF
    private static int CODE_LENGTH_MAX = 15;    // Initial max code length, default 15
    private static int CODE_LENGTH_MIN = 0;


    public static int[] encode(int[] inArr, int TREES_IMPROVE_ITER, int TREES_COUNT, int BLOCK_SIZE, boolean showSelectorFrequencies) throws IOException {
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
            //System.out.println(s);

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

            // Regenerate Huffman code lengths
            for (int j = 0; j < TREES_COUNT; j++) {
                treeFrequencies[j][0]++;    // TODO, this is hardcoding to avoid too long code lengths later on
                Element root = Huffman.huffmanTree(treeFrequencies[j]);
                codeLengths[j] = generateCodeLengths(root);
            }
        }

		// Bit output
		IntArrayOutputStream outArrayStream = new IntArrayOutputStream(inArr.length);
		BitOutputStream outBit = new BitOutputStream(outArrayStream);

        // Write sparse SymMap 16 bits representing if there is a 16 bits, which says if a certain symbols length is written.
        // Go through tree frequencies 16 bits per
        for (int i = 0; i < TREES_COUNT; i++) {
            // TODO is it 16 x 16?? or 16 x 16 + (1 or 2)
            boolean[] symMapLevel1 = new boolean[16];
            boolean[] symMapLevel2 = new boolean[CHAR_MAX - 1];

            // Write symmap level 1
            for (int j = 0; j < 16; j++) {
                boolean writeZeroBit = true;
                for (int k = 0; k < 16; k++) {
                    int idx = j * 16 + k + 2;
                    
                    if (treeFrequencies[i][idx] > 0) {
                        outBit.writeBit(1);
                        writeZeroBit = false;
                        symMapLevel1[j] = true;
                        break;
                    }
                }
                if (writeZeroBit) {
                    outBit.writeBit(0);
                }
            }

            // If symMapLevel1[i] is true, write the bit of symMapLevel2[j] 
            for (int j = 0; j < 16; j++) {
                if (symMapLevel1[j]) {
                    for (int k = 0; k < 16; k++) {
                        int idx = j * 16 + k;
                        if (treeFrequencies[i][idx + 2] > 0) {
                            outBit.writeBit(1);
                            symMapLevel2[idx] = true;
                        } else {
                            outBit.writeBit(0);
                        }
                    }
                }
            }

            // Write initial 5 bit, which is the bit length of char 0
            int bitLength = codeLengths[i][0];
            assert bitLength <= 32;
            String bitLengthStr = String.format("%5s", Integer.toBinaryString(bitLength))
                .replace(' ', '0');
            for (char c : bitLengthStr.toCharArray()) {
                if (c == '0')   outBit.writeBit(0);
                else            outBit.writeBit(1);
            }
            
            // runB is assumed to be there
            int diff = bitLength - codeLengths[i][1];

            if (diff == 0) {
                outBit.writeBit(0);
            } else {
                outBit.writeBit(1);
                bitLength = bitLength - diff;
                if (diff > 0) { // Need to increment value
                    outBit.writeBit(1);
                } else {
                    outBit.writeBit(0);
                }
                for (int k = 0; k < Math.abs(diff) - 1; k++) {
                    outBit.writeBit(1);
                }
                outBit.writeBit(0);
            }   
            // Delta encode starting with char 1 from initial bit length.
            // 0 for same length, 110 for -1, 100 for +1, 1010 for +2, 1110 for -2, etc.         
            for (int j = 2; j < codeLengths[i].length; j++) {
                if (symMapLevel2[j - 2]) {
                    diff = bitLength - codeLengths[i][j];

                    if (diff == 0) {
                        outBit.writeBit(0);
                    } else {
                        outBit.writeBit(1);
                        bitLength = bitLength - diff;
                        if (diff > 0) { // Need to increment value
                            outBit.writeBit(1);
                        } else {
                            outBit.writeBit(0);
                        }
                        for (int k = 0; k < Math.abs(diff) - 1; k++) {
                            outBit.writeBit(1);
                        }
                        outBit.writeBit(0);
                    }
                }
            }
        }
  

        // Generate codes for the huffman trees from their code lengths
        String[][] treeCodes = new String[TREES_COUNT][CHAR_MAX + 1];
        for (int i = 0; i < TREES_COUNT; i++) {
            treeCodes[i] = Huffman.generateCodesFromLengths(codeLengths[i]);
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

        if (showSelectorFrequencies) {
            // DEBUG: print selectors frequencies
            int[] selectorsFrequencies = new int[TREES_COUNT];
            for (int i = 0; i < selectors.length; i++) {
                selectorsFrequencies[selectors[i]]++;
            }
            for (int i = 0; i < TREES_COUNT; i++) {
                System.out.print(selectorsFrequencies[i] + " ");
            }
            //System.out.println("Selectors Frequencies: " + Arrays.toString(selectorsFrequencies));
        }

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

        int[][] codeLengths = new int[TREES_COUNT][CHAR_MAX + 1];
        for (int i = 0; i < TREES_COUNT; i++) {
            // Read symmap level1, is 16 bits
            boolean[] symMapLevel1 = new boolean[16];
            for (int j = 0; j < 16; j++) {
                int readBit = inBit.readBit();
                symMapLevel1[j] = readBit == 1;
            }

            // Read symmap level2, is 16 bits for each true in symMapLevel1
            boolean[] symMapLevel2 = new boolean[CHAR_MAX - 1];
            for (int j = 0; j < 16; j++) {
                if (symMapLevel1[j]) {
                    for (int k = 0; k < 16; k++) {
                        int idx = j * 16 + k;
                        symMapLevel2[idx] = inBit.readBit() == 1;
                    }
                }
            }

            // Read code lengths
            // Read initial code length, 5 bits
            int codeLengthMax = 0;

            String codeLengthStr = "";
            for (int j = 0; j < 5; j++) {
                codeLengthStr = codeLengthStr + Integer.toString(inBit.readBit());   
            }
            int codeLength = Integer.parseInt(codeLengthStr, 2);
            codeLengths[i][0] = codeLength;
            codeLengthMax = Math.max(codeLengthMax, codeLength);
            // If readBit is zero, we stay same code length
            if (inBit.readBit() == 0) {
                codeLengths[i][1] = codeLength;
            } else {
                // Read next bit, if 1 then negative, 0 then positive
                int diffSign = inBit.readBit();
                int count = 1;

                // Continue reading (unary encoded) until we read a zero
                while (inBit.readBit() == 1) {
                    count++;
                }

                // Increment/decrement code length according to this delta
                if (diffSign == 1)
                    codeLength -= count;
                else
                    codeLength += count;

                // Set
                codeLengths[i][1] = codeLength;
                codeLengthMax = Math.max(codeLengthMax, codeLength);
            }

            // Read delta encoded code lengths
            for (int j = 2; j < codeLengths[i].length; j++) {
                if (symMapLevel2[j - 2]) {
                    // If readBit is zero, we stay same code length
                    if (inBit.readBit() == 0) {
                        codeLengths[i][j] = codeLength;
                    } else {
                        // Read next bit, if 1 then negative, 0 then positive
                        int diffSign = inBit.readBit();
                        int count = 1;

                        // Continue reading (unary encoded) until we read a zero
                        while (inBit.readBit() == 1) {
                            count++;
                        }

                        // Increment/decrement code length according to this delta
                        if (diffSign == 1)
                            codeLength -= count;
                        else
                            codeLength += count;

                        // Set
                        codeLengths[i][j] = codeLength;
                        codeLengthMax = Math.max(codeLengthMax, codeLength);
                    }
                }
            }
        }

        // Generate Huffman trees
        String[][] treeCodes = new String[TREES_COUNT][CHAR_MAX + 1];
        for (int i = 0; i < TREES_COUNT; i++) {
            treeCodes[i] = Huffman.generateCodesFromLengths(codeLengths[i]);
        }

        // TODO TODO TODO Improve this, dont use hashmap.... also.. what a mess
        List<Map<String, Integer>> treeCodesToVal = new ArrayList<Map<String, Integer>>();
        for (int i = 0; i < TREES_COUNT; i++) {
            HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
            for (int j = 0; j < treeCodes[i].length; j++) {
                hashMap.put(treeCodes[i][j], j);
            }
            treeCodesToVal.add(hashMap);
        }


        // if trees amount is 6, we need 3 bits to represent. 001, 010, 011, etc.
        int bitsNeeded = (int) Math.ceil(Math.log(TREES_COUNT) / Math.log(2));

        // Read bitsNeeded which is what tree for this current block, then read block and decompress it
        while (true) {
            // Read bitsNeeded amount of bits
            String treeSelectedBits = "";
            int nextBit = 0;
            for (int i = 0; i < bitsNeeded; i++) {
                nextBit = inBit.readBit();
                if (nextBit == -1) break;
                treeSelectedBits = treeSelectedBits + nextBit;
            }
            if (nextBit == -1) break;
            int treeSelected = Integer.parseInt(treeSelectedBits, 2);

            String readBits = "";
            int charsRead = 0;
            int byteRead = -1;

            while (charsRead < BLOCK_SIZE) {
                while (!treeCodesToVal.get(treeSelected).containsKey(readBits)) {
                    nextBit = inBit.readBit();
                    if (nextBit == -1) break;
                    readBits = readBits + nextBit;
                }
                
                if (nextBit == -1) break;
                byteRead = treeCodesToVal.get(treeSelected).get(readBits);
                if (byteRead == 257) {
                    break;
                } else {
                    outList.add(byteRead);
                }
                readBits = "";
                charsRead++;
            }                
            if (byteRead == 257) {
                break;
            }
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
