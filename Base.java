import java.util.*;

public class Base {
    public static void main(String[] args) {
1、数组
    int[] nums = new int[10];
    boolean[][] visited = new boolean[10][10];
    //函数开头要做非空检验
    if(nums.length==0){
        return;
    }
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

        // memo[n]的值： 表示的凑成总金额为n所需的最少的硬币个数
        int[] memo = new int[amount + 1];
        // 给memo赋初值，最多的硬币数就是全部使用面值1的硬币进行换
        // amount + 1 是不可能达到的换取数量，于是使用其进行填充
        Arrays.fill(memo, amount + 1);
        memo[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.length; j++) {
                if (i - coins[j] >= 0) {
                    // memo[i]有两种实现的方式，
                    // 一种是包含当前的coins[i],那么剩余钱就是 i-coins[i],这种操作要兑换的硬币数是 memo[i-coins[j]] + 1
                    // 另一种就是不包含，要兑换的硬币数是memo[i]
                    memo[i] = Math.min(memo[i], memo[i - coins[j]] + 1);
                }
            }
        }

        return memo[amount] == (amount + 1) ? -1 : memo[amount];
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
