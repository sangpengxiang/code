import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
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

        // list 快速添加
        List<List<Integer>> res = new ArrayList<>();
        res.add(Arrays.asList(nums[i], nums[left], nums[right]));

        
        // 数组转List
        String[] arr = {"Java", "Python", "Go", "C++"};
        // 方式一：推荐（可修改）
        List<String> list1 = new ArrayList<>(Arrays.asList(arr));
        // 方式二：直接使用 asList（返回固定长度 List，不推荐直接修改）
        List<String> list2 = Arrays.asList(arr);   // ⚠️ 不能 add/remove

        // List转数组
        List<String> list = new ArrayList<>();
        // 方式一：推荐（指定类型）
        String[] arr1 = list.toArray(new String[0]);        // 推荐，JDK8+ 自动扩容
        String[] arr2 = list.toArray(new String[list.size()]); // 传统写法

        // 包装类
        Map<Character, Integer> widow = new HashMap<>();
        Character;
        Integer;


    }
}

/*


*/
