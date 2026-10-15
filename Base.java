import java.util.*;

import org.w3c.dom.Node;

public class Base {
    public static void main(String[] args) {
1、数组
    int[] nums = new int[10];
    boolean[][] visited = new boolean[10][10];
    int[] arr3 = new int[]{1,2,3};
    int[] arr4 = {1,2,3};        // 简化写法
    //函数开头要做非空检验
    if(nums.length==0){
        return;
    }

    // 数组转List（注意：返回的是固定长度列表）
    List<Integer> list = Arrays.asList(1, 2, 3);
    // 基本类型需要包装类
    Integer[] arr = {1,2,3};
    List<Integer> list2 = Arrays.asList(arr);

    // List转数组，注意这里入参
    String[] arr3 = list.toArray(new String[0]);

    // 按第一列升序（默认）
    int[][] arr = {{3, 8}, {1, 5}, {2, 4}};
    Arrays.sort(arr, (a, b) -> a[0] - b[0]);
    // 结果：[[1,5], [2,4], [3,8]]

    // 按第一列降序
    Arrays.sort(arr, (a, b) -> b[0] - a[0]);
    // 结果：[[3,8], [2,4], [1,5]]
2、字符串String
    String s1 = "ABC";
    int size = s1.length();
    //字符串不支持向数组一样修改其中的字符，但是可以访问，需要转成char[]才能修改
    //访问
    char c = s1.charAt(2);
    //修改
    char[] chars = s1.toCharArray();
    chars[1] = 'a';
    String str = new String(chars);
    str.substring(-1);
    str.substring(0,1);
    //频繁使用拼接，使用StringBuilder.append('a')，最后再从StringBuilder转为String
    StringBuilder sb = new StringBuilder(str);
    String str1 = sb.toString();
    //使用equals比较两个字符串内容是否相等
3、动态数组ArrayList
    ArrayList<String> list = new ArrayList<>();
    list.isEmpty();
    list.size();
    list.add("A");
    list.remove(1); list.remove("B");
    list.set(1, "X");
    list.get(1);
    list.contains("B");
    list.indexOf("C"); list.lastIndexOf("C");
4、双链表LinkedList
    LinkedList<Integer> linkedlist = new LinkedList<>();
    linkedlist.isEmpty();
    linkedlist.size();
    linkedlist.add(1);
    linkedlist.addFirst(1); //链表头部添加元素
    linkedlist.addLast(1); //链表尾部添加元素
    linkedlist.remove(1);
    linkedlist.removeFirst(); //删除链表头部第一个元素
    linkedlist.removeLast(); //删除链表尾部最后一个元素
    linkedlist.set(1, 1);
    linkedlist.get(1);
    linkedlist.contains(1);
5、哈希表HashMap
    HashMap<Integer, String> map = new HashMap<>();
    map.isEmpty();
    map.size();
    map.put(1, "A");
    map.remove(1);
    map.putIfAbsent(1, "A"); //如果key不存在，则将键值对存入哈希表，如果key存在，则什么都不做
    map.get(1);
    map.getOrDefault(1, "B"); //获得key的值，如果key不存在，则返回defaultValue
    map.containsKey(1);
    //获得哈希表中所有的key
    map.keySet();

    // 没有改
6、哈希集合
    Set<String> set = new HashSet<>();
    set.isEmpty();
    set.size();
    set.add("A");
    set.remove("A");
    set.contains("A");
7、队列Queue
    Queue<String> q = new LinkedList<>();
    q.isEmpty();
    q.size();
    q.add("A"); // 如果已满则抛出异常
    q.remove(); // 抛出异常
    q.peek();
    q.contains("A"); // 返回null
        
    // 操作	抛出异常的方法	返回特殊值的方法
    // 插入	add(e)	offer(e)
    // 移除	remove()	poll()
    // 检查	element()	peek()
8、堆栈Stack
    Stack<Integer> s = new Stack<>();
    s.isEmpty();
    s.size();
    s.push(1);	//将元素压入栈顶
    s.pop();
    s.peek();
    //删除并返回栈顶元素
    s.contains(1);
    }
}

class Solution {
    public static void main(String[] args) {
        // 硬币组合
        int[] coins = { 1, 2, 5 };
        int amount = 5;
        System.out.println(coinChange(coins, amount));
    }
}

// dp递归
class Solution {
    int res = Integer.MAX_VALUE;

    public int coinChange(int[] coins, int amount) {
        if (coins.length == 0) {
            return -1;
        }

        dp(coins, amount, 0);

        // 如果没有任何一种硬币组合能组成总金额，返回 -1。
        if (res == Integer.MAX_VALUE) {
            return -1;
        }
        return res;
    }

    public void dp(int[] coins, int amount, int count) {
        if (amount < 0) {
            return;
        }
        if (amount == 0) {
            res = Math.min(res, count);
        }

        for (int i = 0; i < coins.length; i++) {
            dp(coins, amount - coins[i], count + 1);
        }
    }
}

// dp递归+备忘录
class Solution {
    int[] memo;

    public int coinChange(int[] coins, int amount) {
        if (coins.length == 0) {
            return -1;
        }
        memo = new int[amount];

        return dp(coins, amount);
    }

    // memo[n] 表示钱币n可以被换取的最少的硬币数，不能换取就为-1
    // dp函数的目的是为了找到 amount数量的零钱可以兑换的最少硬币数量，返回其值int
    public int dp(int[] coins, int amount) {
        if (amount < 0) {
            return -1;
        }
        if (amount == 0) {
            return 0;
        }
        // 记忆化的处理，memo[n]用赋予了值，就不用继续下面的循环
        // 直接的返回memo[n] 的最优值
        if (memo[amount - 1] != 0) {
            return memo[amount - 1];
        }
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < coins.length; i++) {
            int res = dp(coins, amount - coins[i]);
            if (res >= 0 && res < min) {
                min = res + 1; // 加1，是为了加上得到res结果的那个步骤中，兑换的一个硬币
            }
        }
        memo[amount - 1] = (min == Integer.MAX_VALUE ? -1 : min);
        return memo[amount - 1];
    }
}

// dp+至低向上
class Solution {
    public int coinChange(int[] coins, int amount) {
        // 自底向上的动态规划
        if (coins.length == 0) {
            return -1;
        }

        // dp[n]的值： 表示的凑成总金额为n所需的最少的硬币个数
        int[] dp = new int[amount + 1];
        // 给dp赋初值，最多的硬币数就是全部使用面值1的硬币进行换
        // amount + 1 是不可能达到的换取数量，于是使用其进行填充
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (i - coins[j] >= 0) {
                    // dp[i]有两种实现的方式，
                    // 一种是包含当前的coins[i],那么剩余钱就是 i-coins[i],这种操作要兑换的硬币数是 dp[i-coins[j]] + 1
                    // 另一种就是不包含，要兑换的硬币数是dp[i]
                    dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }

        return dp[amount] == (amount + 1) ? -1 : dp[amount];
    }
}

// 回溯
class Solution {
    ArrayList<Integer> result;

    public int[] backtrack(int[] nums) {
        // 满足条件
        if (nums.length == 0) {
            result.add(nums[0]);
            return result;
        }

        // 做选择
        for (int num : nums) {
            result.add(num);
            backtrack(nums);
            result.remove(num);
        }
    }
}

// 全排列 DFS
class Solution {
    List<List<Integer>> result;

    public List<List<Integer>> permute(int[] nums) {
        // 记录路径
        List<Integer> path = new ArrayList<>();
        result = new ArrayList<>();
        backtrack(nums, path);
        return result;
    }

    public void backtrack(int[] nums, List<Integer> path) {
        if (path.size() == nums.length) {
            result.add(new ArrayList<>(path));
            return;
        }
        for (int num : nums) {
            if (path.contains(num))
                continue;
            path.add(num);
            backtrack(nums, path);
            path.remove(path.size() - 1);
        }
    }
}

// BFS
class Solution {
    public int BFS(Node start, Node target) {
        Queue<Node> q = new LinkedList<>();
        HashSet<Node> visited = new HashSet<>();

        q.offer(start);
        visited.add(start);
        int step = 1;

        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                Node cur = q.poll();
                if (cur == target) {
                    return step;
                }

                for (Node next : cur.neighbors) {
                    if (!visited.contains(next)) {
                        q.offer(next);
                        visited.add(next);
                    }
                }
            }
            step++;
        }
    }
}

// 二分搜索
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0)
            return -1;
        // 二分查找
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return -1;
    }
}

// 左侧边界
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0)
            return -1;
        // 二分查找
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                right = mid - 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        if (left >= nums.length || nums[left] != target) {
            return -1;

        }
        return left;
    }
}

// 右侧边界
class Solution {
    public int search(int[] nums, int target) {
        if (nums.length == 0)
            return -1;
        // 二分查找
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        if (right < 0 || nums[right] != target) {
            return -1;

        }
        return right;
    }
}

// 滑动窗
// 最小覆盖子串
class Solution {
    public int minWindow(String s, String t) {
        Map<Character, Integer> need = new HashMap<>();
        Map<Character, Integer> window = new HashMap<>();
        for (char c : t.toCharArray()) {
            need.put(c, need.getOrDefault(c, 0) + 1);
        }
        int left = 0, right = 0;
        int valid = 0;

        int start = 0, end = Integer.MAX_VALUE;
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (window.get(c).equals(need.get(c))) {
                    valid++;
                }
            }

            while (valid == need.size()) {
                if (right - left < end - start) {
                    start = left;
                    end = right;
                }
                char c1 = s.charAt(left);
                left++;
                if (need.containsKey(c1)) {
                    if (window.get(c1).equals(need.get(c1))) {
                        valid--;
                    }
                    window.put(c1, window.get(c1) - 1);
                }
            }
        }
        return length == s.length() + 1 ? "" : s.substring(start, end);
    }
}

// LRU Cache（最近最少使用缓存）实现
// 使用双向链表 + HashMap 实现 O(1) 时间复杂度的 get 和 put 操作
class LRUCache {
    // 双向链表节点，存储 key 和 value
    class Node {
        int key, value;
        Node pre, next;  // 前驱和后继指针

        Node(int k, int v) {
            key = k;
            value = v;
        }
    }

    private final int captical;          // 缓存容量（注意：原变量名拼写为 captical，正确应为 capacity）
    private final Node head, tail;       // 哨兵节点：head 为虚拟头节点，tail 为虚拟尾节点
    private final HashMap<Integer, Node> map;  // 哈希表：key -> Node，实现 O(1) 查找

    /**
     * 构造函数：初始化容量为 size 的 LRU 缓存
     * @param size 缓存最大容量
     */
    LRUCache(int size) {
        captical = size;
        head = new Node(0, 0);   // 虚拟头节点（最常使用的一端）
        tail = new Node(0, 0);   // 虚拟尾节点（最少使用的一端）
        map = new HashMap<>();
        // 初始化空链表：head <-> tail
        head.next = tail;
        tail.pre = head;
    }

    /**
     * 将节点添加到链表头部（表示最近使用）
     * @param node 要添加的节点
     */
    private void addToHead(Node node) {
        node.next = head.next;   // 新节点指向原来的第一个真实节点
        node.pre = head;         // 新节点的前驱指向 head
        head.next.pre = node;    // 原来第一个节点的前驱指向新节点
        head.next = node;        // head 的后继指向新节点
    }

    /**
     * 从链表中删除指定节点
     * @param node 要删除的节点
     */
    private void removeNode(Node node) {
        node.pre.next = node.next;  // 前驱节点的后继指向当前节点的后继
        node.next.pre = node.pre;   // 后继节点的前驱指向当前节点的前驱
    }

    /**
     * 删除并返回尾部节点（最少使用的节点）
     * @return 被删除的尾部节点
     */
    private Node removeTail() {
        Node lastNode = tail.pre;   // 尾部真实节点（tail 的前驱）
        removeNode(lastNode);       // 从链表中移除
        return lastNode;
    }

    /**
     * 将节点移动到链表头部（标记为最近使用）
     * @param node 要移动的节点
     */
    private void moveToHead(Node node) {
        removeNode(node);   // 先从原位置删除
        addToHead(node);    // 再添加到头部
    }

    /**
     * 获取 key 对应的 value
     * 如果存在，将其移动到头部（表示最近使用）；如果不存在返回 -1
     * @param key 要查询的键
     * @return 对应的值，不存在则返回 -1
     */
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node node = map.get(key);
        moveToHead(node);   // 访问后移动到头部
        return node.value;
    }

    /**
     * 插入或更新 key-value
     * 如果 key 已存在，更新 value 并移动到头部；
     * 如果 key 不存在，创建新节点并插入头部；
     * 如果超过容量，删除尾部最少使用的节点
     * @param key   键
     * @param value 值
     */
    public void put(int key, int value) {
        if (map.containsKey(key)) {
            // key 已存在：更新 value 并移动到头部
            Node node = map.get(key);
            node.value = value;
            moveToHead(node);
        } else {
            // key 不存在：创建新节点并插入头部
            Node node = new Node(key, value);
            addToHead(node);
            map.put(key, node);
            // 超过容量时，删除尾部节点（最少使用）
            if (map.size() > captical) {
                Node last = removeTail();
                map.remove(last.key);  // 同时从哈希表中移除
            }
        }
    }
}

public class AlternatePrintSyncABC {
    private static final Object lock = new Object();
    private static int state = 0; // 0 -> A, 1 -> B, 2 -> C
    private static final int ROUNDS = 10;

    public static void main(String[] args) {
        new Thread(() -> print("A", 0), "Thread-A").start();
        new Thread(() -> print("B", 1), "Thread-B").start();
        new Thread(() -> print("C", 2), "Thread-C").start();
    }

    private static void print(String name, int targetState) {
        for (int i = 0; i < ROUNDS; i++) {
            // 核心重点：synchronized 必须在 for 循环内部！
            synchronized (lock) {
                try { 		//1. 为什么要 try-catch？ 因为 lock.wait()
                    while (state % 3 != targetState) { // 2. 为什么要用 while，而不是 if？ 防止虚假唤醒（Spurious Wakeup）或者错误唤醒
                        lock.wait(); // 不满足条件，释放锁并进入等待池
                    }
                    System.out.print(name);
                    state++;
                    lock.notifyAll(); // 必须 notifyAll，防止假死死锁！
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            } // 出代码块，立刻释放 lock 锁，给其他就绪线程竞争机会
        }
    }
}

// Lambda 本质上就是在写 run() 的方法体
new Thread(() -> print("A", 0), "Thread-A").start();

// 等价于：
new Thread(new Runnable() {
    @Override
    public void run() {
        print("A", 0);   // 箭头函数的内容，就是 run() 里的代码
    }
}, "Thread-A").start();


public class ProducerConsumerSimple {

    private static final Queue<Integer> queue = new LinkedList<>();
    private static final int CAPACITY = 5;          // 缓冲区容量
    private static final Object lock = new Object();

    public static void main(String[] args) {
        // 1个生产者
        new Thread(new Producer(), "生产者").start();
        // 1个消费者
        new Thread(new Consumer(), "消费者").start();
    }

    // ==================== 生产者 ====================
    static class Producer implements Runnable {
        private int productId = 0;

        @Override
        public void run() {
            while (true) { // 一直不停地生产/消费
                try {
                    Thread.sleep(800);  // 模拟生产耗时

                    synchronized (lock) {
                        // 缓冲区满 → 等待
                        while (queue.size() == CAPACITY) {
                            System.out.println("【生产者】缓冲区满了，等待消费者...");
                            lock.wait();
                        }

                        // 生产
                        int product = ++productId;
                        queue.offer(product);
                        System.out.println("【生产者】生产了 " + product + "，当前数量：" + queue.size());

                        // 唤醒消费者
                        lock.notifyAll();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }

    // ==================== 消费者 ====================
    static class Consumer implements Runnable {
        @Override
        public void run() {
            while (true) {
                try {
                    Thread.sleep(1200);  // 模拟消费耗时

                    synchronized (lock) {
                        // 缓冲区空 → 等待
                        while (queue.isEmpty()) {
                            System.out.println("【消费者】缓冲区空了，等待生产者...");
                            lock.wait();
                        }

                        // 消费
                        int product = queue.poll();
                        System.out.println("【消费者】消费了 " + product + "，当前数量：" + queue.size());

                        // 唤醒生产者
                        lock.notifyAll();
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
            }
        }
    }
}

// 快排
public class QuickSort {

    public static void quickSort(int[] arr) {
        if (arr == null || arr.length <= 1) {
            return;
        }
        quickSort(arr, 0, arr.length - 1);
    }

    private static void quickSort(int[] arr, int left, int right) {
        if (left >= right) {
            return;
        }

        // 获取分区点
        int pivotIndex = partition(arr, left, right);

        // 递归排序左半部分
        quickSort(arr, left, pivotIndex - 1);
        // 递归排序右半部分
        quickSort(arr, pivotIndex + 1, right);
    }

    /**
     * 分区函数（以最右边的元素作为基准）
     * 返回基准最终所在的位置
     */
    private static int partition(int[] arr, int left, int right) {
        int randomIndex = left + random.nextInt(right - left + 1);
        swap(arr, randomIndex, right);
        
        return lomutoPartition(arr, left, right);
    }
    
    private static int lomutoPartition(int[] arr, int left, int right) { 
        int pivot = arr[right]; // 选择最后一个元素作为基准
        int i = left;

        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                swap(arr, i, j);
                i++;
            }
        }

        swap(arr, i, right);
        return i;
    }
    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // 测试
    public static void main(String[] args) {
        int[] arr = {5, 2, 9, 1, 7, 6, 3, 8, 4};

        System.out.println("排序前：");
        printArray(arr);

        quickSort(arr);

        System.out.println("排序后：");
        printArray(arr);
    }

    private static void printArray(int[] arr) {
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}

// 堆排
public class HeapSort {

    public static void heapSort(int[] arr) {
        if (arr == null || arr.length <= 1) {
            return;
        }

        int n = arr.length;

        // 1. 构建大顶堆（从最后一个非叶子节点开始，自底向上调整）
        for (int i = n / 2 - 1; i >= 0; i--) {
            maxHeapify(arr, n, i);
        }

        // 2. 逐步将堆顶元素（最大值）交换到末尾，并重新调整堆
        for (int i = n - 1; i > 0; i--) {
            // 将根节点（当前最大值）与当前末尾元素交换
            swap(arr, 0, i);
            // 对剩余的元素重新调整为大顶堆，堆大小减 1
            maxHeapify(arr, i, 0);
        }
    }

    /**
     * 维护大顶堆性质（下沉操作 / Sift Down）
     * @param arr 数组
     * @param heapSize 当前堆的有效大小
     * @param i 当前需要调整的节点索引
     */
    private static void maxHeapify(int[] arr, int heapSize, int i) {
        int largest = i;          // 假设当前节点最大
        int left = 2 * i + 1;     // 左子节点索引
        int right = 2 * i + 2;    // 右子节点索引

        // 如果左子节点存在且大于当前节点
        if (left < heapSize && arr[left] > arr[largest]) {
            largest = left;
        }

        // 如果右子节点存在且大于目前最大的节点
        if (right < heapSize && arr[right] > arr[largest]) {
            largest = right;
        }

        // 如果最大值不是当前节点，说明需要交换并递归向下调整
        if (largest != i) {
            swap(arr, i, largest);
            maxHeapify(arr, heapSize, largest);
        }
    }

    private static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // 测试
    public static void main(String[] args) {
        int[] arr = {5, 2, 9, 1, 7, 6, 3, 8, 4};

        System.out.println("排序前：");
        printArray(arr);

        heapSort(arr);

        System.out.println("排序后：");
        printArray(arr);
    }

    private static void printArray(int[] arr) {
        for (int num : arr) {
            System.out.print(num + " ");
        }
        System.out.println();
    }
}

public class Singleton {

    // volatile 保证可见性，禁止指令重排序
    private static volatile Singleton instance;

    // 私有构造方法，防止外部实例化
    private Singleton() {}

    public static Singleton getInstance() {
        // 第一次检查：避免每次获取实例都加锁
        if (instance == null) {
            synchronized (Singleton.class) {
                // 第二次检查：防止多个线程重复创建实例
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}

/*
    为什么需要两次判空？
    - 第一次判空：提高性能。实例创建后，后续调用直接返回，无须进入同步代码块。
    - 第二次判空：保证只创建一个实例。多个线程可能同时通过第一次判空，获得锁后必须再次检查。

    2. 为什么必须使用 volatile？
    instance = new Singleton() 并非原子操作，可以分为三个步骤：
    1. 分配对象内存空间。
    2. 初始化对象。
    3. 将内存地址赋值给 instance。
    在没有 volatile 的情况下，JVM 可能发生指令重排序：
    正常执行顺序
    分配内存
    初始化对象
    赋值引用


    可能发生的重排序
    分配内存
    赋值引用
    初始化对象

    其他线程可能读到非 null、但尚未完成初始化的对象。

    volatile 可以禁止这里有风险的指令重排序，并保证实例安全发布，让其他线程看到正确初始化的对象。
    
    3. synchronized 和 volatile 分别负责什么？
    关键字	作用
    synchronized	保证创建实例过程的互斥性，避免重复创建
    volatile	保证可见性及安全发布，防止读取未正确初始化的对象
    4. 面试如何快速回答？
    双重校验锁主要解决三个问题：
    1. 线程安全：通过 synchronized 保证只有一个线程创建实例。
    2. 性能优化：通过第一次判空避免不必要的加锁。
    3. 指令重排序：通过 volatile 防止其他线程获得未初始化完成的实例。

*/