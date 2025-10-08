import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedList;
import java.util.PriorityQueue;

public class Memory {
    public static void main(String[] args) {
        // 单调队列
        Deque<Integer> deque = new LinkedList<>();

        // 优先队列
        PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] m, int[] n) {
                return m[1] - n[1];
            }
        });
    }
}
