public class Base {
    
}
/*
1、数组
    int[] nums = new int[n];
    boolean[] [] visited = new boolean[] [];
    //函数开头要做非空检验
    if(nums.length==0){
        return;
    }
2、字符串String
    //字符串不支持向数组一样修改其中的字符，但是可以访问，需要转成char[]才能修改
    //访问
    char c = s1.charAt(2);
    //修改
    char[] chars = s1.toCharArray();
    char[1] = 'a';
    String s2 = new String(chars);
    //频繁使用拼接，使用StringBuilder.append('a')，最后再从StringBuilder转为String
    StringBuilder sb = new StringBuilder(str);
    String str = sb.toString();
    //使用equals比较两个字符串内容是否相等
3、动态数组ArrayList
    ArrayList<String> nums = new ArrayList<>();
    boolean isEmpty();
    lits.size();
    lits.add(E e);
    list.remove(1); list.remove("B");
    lits.set(1, "X")
    lits.get(int index);
    list.contains("B");
    list.indexOf("C"); list.lastIndexOf("C")
4、双链表LinkedList
    LinkedList<Integer> nums = new LinkedList<>();
    boolean isEmpty();
    int size();
    //链表头部添加元素
    boolean addFirst(E e);
    //链表尾部添加元素
    boolean add(E e);
    //删除链表头部第一个元素
    E removeFirst();
    //删除链表尾部最后一个元素
    E removeLast();
    boolean contains(Object 0);
5、哈希表HashMap
    HashMap<Integer, String> map = new HashMap<>();
    //获得哈希表中所有的key
    Set<k> keySet();
    V put(K key, V value);
    V remove(Object key);
    //如果key不存在，则将键值对存入哈希表，如果key存在，则什么都不做
    V putIfAbsent(K key, V value);
    V get(Object key);
    //获得key的值，如果key不存在，则返回defaultValue
    V getOrDefault(Object key, V defaultValue);
    boolean containsKey(Object key);
6、哈希集合
    Set<String> set = new HashSet<>();
    boolean add(E e);
    boolean remove(Object o);
    boolean contains(Object o);
7、队列Queue
    Queue<String> q = new LinkedList<>();
    boolean isEmpty();
    int size();
    queue.add("A"); // 如果已满则抛出异常
    queue.offer(E e); // 如果队列已满则返回false
    E peek();
    //返回队头元素并弹出
    queue.remove(); // 抛出异常
    E poll(); // 返回null
        
    操作	抛出异常的方法	返回特殊值的方法
    插入	add(e)	offer(e)
    移除	remove()	poll()
    检查	element()	peek()
8、堆栈Stack
    Stack<Integer> s = new Stack<>();
    boolean isEmpty();
    int size();
    //将元素压入栈顶
    E push(E item);			
    //返回栈顶元素
    E peek();
    //删除并返回栈顶元素
    E pop();
*/