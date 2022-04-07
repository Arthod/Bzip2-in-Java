// Victor Halgaard Kristensen - vikri19@student.sdu.dk
// Ahmad Mahir Sadaldin Othman - ahoth19@student.sdu.dk 

import java.util.ArrayList;

class PQHeap implements PQ {
    private ArrayList<Element> heap;

    /*
     * Constructor. Returns an empty priority queue.
     */
    public PQHeap() {
        heap = new ArrayList<>();
    }

    /*
     * Method from PQ interface. Inserts Element e into the priority queue.
     */
    public void insert(Element e) {
        int i = heap.size();
        heap.add(e);
        while(i > 0 && heap.get(parent(i)).getKey() > heap.get(i).getKey()) {
            exchange(i, parent(i));
            i = parent(i);
        }
    }

    /*
     * Method from PQ interface. Removes and returns the element with lowest 
     * priority.
     */
    public Element extractMin() {
        Element min = heap.get(0);
        heap.set(0, heap.get(heap.size()-1));
        heap.remove(heap.size() - 1);
        minHeapify(0);
        return min;
    }

    /*
     * Exchanges child and parent at i if the heap order is violated. Run
     * recursively down the tree to maintain total heap order.
     */
    private void minHeapify(int i) {
        int l = left(i);
        int r = right(i);
        int smallest;
        if(l <= heap.size()-1 && heap.get(l).getKey() < heap.get(i).getKey())
            smallest = l;
        else
            smallest = i;
        if(r <= heap.size()-1 && heap.get(r).getKey() < heap.get(smallest).getKey())
            smallest = r;
        if(smallest != i) {
            exchange(i, smallest);
            minHeapify(smallest);
        }
    }

    /*
     * Exchanges the element of node i with the element of node j.
     */
    private void exchange(int i, int j) {
        Element temp = heap.get(i);
        heap.set(i, heap.get(j));
        heap.set(j, temp);
    }

    /*
     * Returns the index of the parent of node with index i.
     */
    private int parent(int i) {
        return (int) Math.floor((i - 1) / 2);
    }

    /*
     * Returns the index of the left child of node with index i.
     */
    private int left(int i) {
        return 2 * i + 1;
    }

    /*
     * Returns the index of the right child of node with index i.
     */
    private int right(int i) {
        return 2 * i + 2;
    }
    
}