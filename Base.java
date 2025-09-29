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
