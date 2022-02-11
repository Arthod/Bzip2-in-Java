// Victor Halgaard Kristensen - vikri19@student.sdu.dk
// Ahmad Mahir Sadaldin Othman - ahoth19@student.sdu.dk 

import java.io.FileInputStream;
import java.io.FileOutputStream;

class Encode {
	private static int[] frequency = new int[256];
	private static String[] passwords = new String[256];

	/*
	 * Main.
	 */
	public static void main(String[] args) throws Exception {
		// File input and bit output
		FileInputStream inFile = new FileInputStream(args[0]);
		FileOutputStream outFile = new FileOutputStream(args[1]);
		BitOutputStream outBit = new BitOutputStream(outFile);
		
		// Reading input file
		int inputRead;
		while ((inputRead = inFile.read()) != -1) {
			frequency[inputRead]++;
		}

		// Writing frequency to file
		for (int i = 0; i < 256; i++) {
			outBit.writeInt(frequency[i]);
		}

		// Generating the Huffman tree and then the passwords
		Element root = Huffman(frequency);
		generatePassword(root, "");
		
		// Reading input file again, for every byte write its password bits to file
		inFile.close();
		inFile = new FileInputStream(args[0]);
		while ((inputRead = inFile.read()) != -1) {
			String password = passwords[inputRead];
			for (int j = 0; j < password.length(); j++) {
				outBit.writeBit(Character.getNumericValue(password.charAt(j)));
			}
		}
		outBit.close();
		inFile.close();
	}

	/*
	 * Generates a Huffman tree from an integer array of frequencies. Node objects are
 	 * used to store elements in a binary tree.
	 */
	public static Element Huffman(int[] frequency) {
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
	 * Recursive inorder tree walk, sets a password for all leaves of the tree. The
 	 * password of a specific leaf is the boolean left and right traverses needed to 
 	 * reach it from the root.
	 */
	private static void generatePassword(Element e, String tempPassword) {
		if (e.getData() instanceof Node) {
			Node n = (Node) e.getData();
			generatePassword(n.getLeft(), tempPassword + "0");
			generatePassword(n.getRight(), tempPassword + "1");
		} else {
			passwords[(int) e.getData()] = tempPassword;
		}
	}
}
