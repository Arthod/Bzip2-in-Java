// Victor Halgaard Kristensen - vikri19@student.sdu.dk
// Ahmad Mahir Sadaldin Othman - ahoth19@student.sdu.dk 

class Node {
	private Element left;
	private Element right;
	
	/*
	 * Constructor. Stores elements in left and right.
	 */
	public Node(Element left, Element right) {
	   	this.left = left;
	   	this.right = right;
   	}

	/*
	 * Left element getter.
	 */
   	public Element getLeft() {
	   	return this.left;
   	}

	/*
	 * Right element getter.
	 */
   	public Element getRight() {
	   	return this.right;
	}

	/*
	 * Left element setter.
	 */
   	public void setLeft(Element left) {
	   	this.left = left;
	}

	/*
	 * Right element setter.
	 */
	public void setRight(Element right) {
	   	this.right = right;
   	}
}
