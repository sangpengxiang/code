import java.util.*;

import javax.swing.tree.TreeNode;

import org.w3c.dom.Node;

class Index {
    public Index() {

    }
}


// 二叉树遍历
// 递归写法
// 二叉树节点定义
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

public class BinaryTreePreorderTraversal {
    
    // 递归实现前序遍历
    public List<Integer> preorderRecursive(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        traverse(root, result);
        return result;
    }
    
    private void traverse(TreeNode node, List<Integer> result) {
        if (node == null) return;
        result.add(node.val);      // 访问根节点
        traverse(node.left, result);  // 遍历左子树
        traverse(node.right, result); // 遍历右子树
    }
    
    // 迭代实现前序遍历
    /*
     * 迭代方法：
        使用栈来模拟递归过程
        先将根节点压入栈
        循环直到栈为空：弹出栈顶节点并访问，然后先压入右子节点，再压入左子节点
     */
    public List<Integer> preorderIterative(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            result.add(node.val);  // 访问当前节点
            
            // 先压入右子节点，再压入左子节点（栈是LIFO，所以先右后左）
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        
        return result;
    }

    // 中序遍历 - 迭代
    /*
     * 使用栈来模拟递归过程
        从根节点开始，将所有左子节点压入栈中
        弹出栈顶节点并访问
        处理该节点的右子树
     */
    public List<Integer> inorderIterative(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        
        while (curr != null || !stack.isEmpty()) {
            // 将所有左子节点压入栈中
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            
            // 弹出栈顶节点并访问
            curr = stack.pop();
            result.add(curr.val);
            
            // 处理右子树
            curr = curr.right;
        }
        
        return result;
    }

    // 后序遍历 - 迭代（使用两个栈）
    public List<Integer> postorderIterative(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;
        
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        stack1.push(root);
        
        while (!stack1.isEmpty()) {
            TreeNode node = stack1.pop();
            stack2.push(node);
            
            // 先压入左子节点，再压入右子节点
            if (node.left != null) {
                stack1.push(node.left);
            }
            if (node.right != null) {
                stack1.push(node.right);
            }
        }
        
        // stack2中的节点顺序就是后序遍历结果
        while (!stack2.isEmpty()) {
            result.add(stack2.pop().val);
        }
        
        return result;
    }

    public static void main(String[] args) {
        // 构建示例二叉树
        //       1
        //      / \
        //     2   3
        //    / \
        //   4   5
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.left.right = new TreeNode(5);
        
        BinaryTreePreorderTraversal solution = new BinaryTreePreorderTraversal();
        
        System.out.println("递归前序遍历: " + solution.preorderRecursive(root));
        System.out.println("迭代前序遍历: " + solution.preorderIterative(root));
    }

    // 二叉树层序遍历 BFS
    public class BinaryTreeLevelOrderTraversal {
    
        // 层序遍历 - 基本实现（返回所有节点的值）
        public List<Integer> levelOrder(TreeNode root) {
            List<Integer> result = new ArrayList<>();
            if (root == null) return result;
            
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                result.add(node.val);
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            return result;
        }
    }
}
    // 二叉树的最大深度
public class MaximumDepthOfBinaryTree {
    
    // 方法一：递归实现（DFS）
    public int maxDepthRecursive(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        int leftDepth = maxDepthRecursive(root.left);
        int rightDepth = maxDepthRecursive(root.right);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
    
    // 方法二：迭代实现（BFS层序遍历）
    public int maxDepthBFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            depth++;
        }
        
        return depth;
    }
    
    // 方法三：迭代实现（DFS使用栈）
    public int maxDepthDFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        Stack<TreeNode> stack = new Stack<>();
        Stack<Integer> depthStack = new Stack<>();
        stack.push(root);
        depthStack.push(1);
        int maxDepth = 0;
        
        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            int currentDepth = depthStack.pop();
            maxDepth = Math.max(maxDepth, currentDepth);
            
            if (node.right != null) {
                stack.push(node.right);
                depthStack.push(currentDepth + 1);
            }
            if (node.left != null) {
                stack.push(node.left);
                depthStack.push(currentDepth + 1);
            }
        }
        
        return maxDepth;
    }
}

// 二叉树的最大深度
/*
 * 二叉树的最大深度可以通过递归或迭代（使用层序遍历）的方式求解。
    递归：最大深度等于左子树的最大深度和右子树的最大深度中的较大值加1。
    迭代：使用层序遍历，记录层数，每遍历一层，深度加1。
 */
public class MaximumDepthOfBinaryTree {
    
    // 方法一：递归实现（DFS）
    public int maxDepthRecursive(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        int leftDepth = maxDepthRecursive(root.left);
        int rightDepth = maxDepthRecursive(root.right);
        
        return Math.max(leftDepth, rightDepth) + 1;
    }
    
    // 方法二：迭代实现（BFS层序遍历）
    public int maxDepthBFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int depth = 0;
        
        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            
            depth++;
        }
        
        return depth;
    }
}

// 翻转二叉树
// 递归
class Solution {
	public TreeNode invertTree(TreeNode root) {
		//递归函数的终止条件，节点为空时返回
		if(root==null) {
			return null;
		}
		//下面三句是将当前节点的左右子树交换
		TreeNode tmp = root.right;
		root.right = root.left;
		root.left = tmp;
		//递归交换当前节点的 左子树
		invertTree(root.left);
		//递归交换当前节点的 右子树
		invertTree(root.right);
		//函数返回时就表示当前这个节点，以及它的左右子树
		//都已经交换完了
		return root;
	}
}
// 迭代
class Solution {
	public TreeNode invertTree(TreeNode root) {
		if(root==null) {
			return null;
		}
		//将二叉树中的节点逐层放入队列中，再迭代处理队列中的元素
		LinkedList<TreeNode> queue = new LinkedList<TreeNode>();
		queue.add(root);
		while(!queue.isEmpty()) {
			//每次都从队列中拿一个节点，并交换这个节点的左右子树
			TreeNode tmp = queue.poll();
			TreeNode left = tmp.left;
			tmp.left = tmp.right;
			tmp.right = left;
			//如果当前节点的左子树不为空，则放入队列等待后续处理
			if(tmp.left!=null) {
				queue.add(tmp.left);
			}
			//如果当前节点的右子树不为空，则放入队列等待后续处理
			if(tmp.right!=null) {
				queue.add(tmp.right);
			}
			
		}
		//返回处理完的根节点
		return root;
	}
}

// 对称二叉树
// 递归判断
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return root == null || recur(root.left, root.right);
    }
    boolean recur(TreeNode L, TreeNode R) {
        if (L == null && R == null) return true;
        if (L == null || R == null || L.val != R.val) return false;
        return recur(L.left, R.right) && recur(L.right, R.left);
    }
}


// 二叉树的直径
// 分别求左右最远距离
class Solution {
    int ans;
    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }
    public int depth(TreeNode node) {
        if (node == null) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L+R+1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }
}

// 将有序数组转换为二叉搜索树
// 中序遍历，每次都选mid
class Solution {
    public TreeNode sortedArrayToBST(int[] nums) {
        return helper(nums, 0, nums.length - 1);
    }

    public TreeNode helper(int[] nums, int left, int right) {
        if (left > right) {
            return null;
        }

        // 总是选择中间位置左边的数字作为根节点
        int mid = (left + right) / 2;

        TreeNode root = new TreeNode(nums[mid]);
        root.left = helper(nums, left, mid - 1);
        root.right = helper(nums, mid + 1, right);
        return root;
    }
}

// 验证二叉搜索树
// 需要辅助函数，参数中药塞mid 和 max
class Solution {
    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean isValidBST(TreeNode node, long left, long right) {
        if (node == null) {
            return true;
        }
        long x = node.val;
        return left < x && x < right &&
               isValidBST(node.left, left, x) &&
               isValidBST(node.right, x, right);
    }
}

// 二叉搜索树中第 K 小的元素
// 使用栈的中序遍历
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Deque<TreeNode> stack = new ArrayDeque<TreeNode>();
        while (root != null || !stack.isEmpty()) {
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
            root = stack.pop();
            --k;
            if (k == 0) {
                break;
            }
            root = root.right;
        }
        return root.val;
    }
}

// 二叉树的右视图
/**
思路：层次遍历，记录下每一层最后一个元素
*/
public List<Integer> rightSideView(TreeNode root) {
    List<Integer> ans = new ArrayList<>();
    if(root==null){
        return ans;
    }

    Queue<TreeNode> queue = new LinkedList<>();
    queue.offer(root);

    while(!queue.isEmpty()){
        int size = queue.size();

        for(int i=0; i<size; i++){
            TreeNode node = queue.poll();
            if(node.left!=null){
                queue.offer(node.left);
            }

            if(node.right!=null){
                queue.offer(node.right);
            }

            if(i == size-1){
                // 当前层最后一个元素
                ans.add(node.val);
            }
        }
    }

    return ans;
}

// 二叉树展开为链表
// 迭代实现（O(1)空间复杂度）
/*
 * 对于当前节点，如果其左子节点不为空
    找到左子树的最右节点
    将右子树接到左子树的最右节点
    将左子树移到右子树位置，左子树置空
    移动到下一个节点
 */
class Solution {
    public void flatten(TreeNode root) {
        TreeNode current = root;
        
        while (current != null) {
            if (current.left != null) {
                // 找到左子树的最右节点
                TreeNode rightmost = current.left;
                while (rightmost.right != null) {
                    rightmost = rightmost.right;
                }
                
                // 将右子树接到左子树的最右节点
                rightmost.right = current.right;
                
                // 将左子树移到右子树位置
                current.right = current.left;
                current.left = null;
            }
            
            // 移动到下一个节点
            current = current.right;
        }
    }
}

// 从前序与中序遍历序列构造二叉树
/*
 * 前序遍历的第一个元素总是树的根节点
    在中序遍历中找到这个根节点，根节点左边是左子树的中序遍历，右边是右子树的中序遍历
    根据左子树的节点数量，可以在前序遍历中确定左子树的前序遍历和右子树的前序遍历
    递归地构建左子树和右子树
 */
class Solution {
    private Map<Integer, Integer> inorderMap;
    private int preIndex;
    
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        // 创建中序遍历值到索引的映射
        inorderMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderMap.put(inorder[i], i);
        }
        
        preIndex = 0;
        return buildTreeHelper(preorder, 0, inorder.length - 1);
    }
    
    private TreeNode buildTreeHelper(int[] preorder, int inStart, int inEnd) {
        // 递归终止条件
        if (inStart > inEnd) {
            return null;
        }
        
        // 前序遍历的第一个元素是当前子树的根节点
        int rootVal = preorder[preIndex];
        TreeNode root = new TreeNode(rootVal);
        preIndex++;
        
        // 在中序遍历中找到根节点的位置
        int inIndex = inorderMap.get(rootVal);
        
        // 递归构建左子树和右子树
        root.left = buildTreeHelper(preorder, inStart, inIndex - 1);
        root.right = buildTreeHelper(preorder, inIndex + 1, inEnd);
        
        return root;
    }
}

///////////////////////////////
// 环形链表
/*
 * 快慢指针
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast,slow;
        fast = slow = head;
        while(fast !=null && fast.next!= null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast==slow) return true;
        }
        return false;
    }
}
// 环形链表II 找到入口
/*
 * 相遇后，slow从头开始
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode slow, fast;
        slow = head;
        fast = head;
//        int mres = 0;
        while (fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
            if (slow==fast) {
                break;
            }
        }
        if (fast==null||fast.next==null) return null; 
        slow = head;
        while (slow!=fast){
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }
}
// 合并两个有序链表
/*
 * 可以使用递归或迭代，这里使用迭代
 */

class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode res = new ListNode(0);
        ListNode mlist = res;

        while (list1!=null&&list2!=null){
            if (list1.val<list2.val){
                res.next = list1;
                res = res.next;
                list1 = list1.next;
            }else {
                res.next = list2;
                res = res.next;
                list2 = list2.next;
            }
        }
        if (list1==null){
            res.next=list2;
        }
        if (list2==null){
            res.next=list1;
        }
        return mlist.next;
    }
}
// 合并N个有序链表
/*
 * 中间for一下，不过这个时间比较慢
 */
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        // 合并n个升序链表
        ListNode mres = new ListNode(-1);
        ListNode res = mres;
        ListNode temnode = new ListNode(0);
        while (temnode.val!=Integer.MAX_VALUE){
            temnode = new ListNode(Integer.MAX_VALUE);
            int num = -1;
            for (int i=0; i<lists.length; i++){
                if (lists[i]!=null&&temnode.val>lists[i].val){
                    temnode = lists[i];
                //    lists[i] = lists[i].next;
                    num = i;
                }
            }
            if (temnode.val!=Integer.MAX_VALUE){
                mres.next = temnode;
                lists[num] = lists[num].next;
                mres = mres.next;
            }
        }
        return res.next;
    }
}
// 两数相加
/*
 * 最后也可能进位
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;
        while(l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;
            
            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);

            cur = cur.next;
            if(l1 != null)
                l1 = l1.next;
            if(l2 != null)
                l2 = l2.next;
        }
        if(carry == 1) {
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }
}

// 删除链表的倒数第 N 个结点
// 先过N步，但是感觉这里声明可以简略点
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode node1 = new ListNode(0);
        ListNode node2 = new ListNode(0);
        ListNode node3 ;
        node1.next = head;
        node2.next = head;
        node3 = node2;
        if (head.next==null) return null;
        
        for (int i=0; i<n; i++){
            node1 = node1.next;
        }
        while (node1!=null&&node1.next!=null){
            node1 = node1.next;
            node2 = node2.next;
        }
        node2.next = node2.next.next;
        return node3.next;
    }
}

// 复制带随机指针的链表
// 1、使用HashMap 2、先next存一半，再走一遍指向之前存的
class Solution {
    public Node copyRandomList(Node head) {
        if(head == null) return null;
        Node cur = head;
        Map<Node, Node> map = new HashMap<>();
        // 3. 复制各节点，并建立 “原节点 -> 新节点” 的 Map 映射
        while(cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        // 4. 构建新链表的 next 和 random 指向
        while(cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        // 5. 返回新链表的头节点
        return map.get(head);
    }
}

// 排序链表
// 使用归并，但是1题顶3题
class Solution {
    public ListNode sortList(ListNode head) {
        // 1、递归结束条件
        if (head == null || head.next == null) {
            return head;
        }

        // 2、找到链表中间节点并断开链表 & 递归下探
        ListNode midNode = middleNode(head);
        ListNode rightHead = midNode.next;
        midNode.next = null;

        ListNode left = sortList(head);
        ListNode right = sortList(rightHead);

        // 3、当前层业务操作（合并有序链表）
        return mergeTwoLists(left, right);
    }
    
    //  找到链表中间节点（876. 链表的中间结点）
    private ListNode middleNode(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head.next.next;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        return slow;
    }

    // 合并两个有序链表（21. 合并两个有序链表）
    private ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode sentry = new ListNode(-1);
        ListNode curr = sentry;

        while(l1 != null && l2 != null) {
            if(l1.val < l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }

            curr = curr.next;
        }

        curr.next = l1 != null ? l1 : l2;
        return sentry.next;
    }
}

// 两数之和
// HashMap
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[0];
    }
}

// 
// string排序后用Hash存下
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> lists = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            String change = change(strs[i]);
            if (map.containsKey(change)) {
                map.get(change).add(strs[i]);
            } else {
                List<String> l = new ArrayList<>();
                l.add(strs[i]);
                map.put(change, l);
            }
        }
        for (Map.Entry<String, List<String>> entry : map.entrySet()) {
            lists.add(entry.getValue());
        }
        return lists;
    }

    public String change(String s) {
        char[] chars = s.toCharArray();
        Arrays.sort(chars);
        return String.valueOf(chars);
    }
}

// 最长连续序列
// 1、存到HashSet里
// 2、通过x-1判断是不是起点，是while循环，不是continue
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> st = new HashSet<>();
        for (int num : nums) {
            st.add(num); // 把 nums 转成哈希集合
        }

        int ans = 0;
        for (int x : st) { // 遍历哈希集合
            if (st.contains(x - 1)) { // 如果 x 不是序列的起点，直接跳过
                continue;
            }
            // x 是序列的起点
            int y = x + 1;
            while (st.contains(y)) { // 不断查找下一个数是否在哈希集合中
                y++;
            }
            // 循环结束后，y-1 是最后一个在哈希集合中的数
            ans = Math.max(ans, y - x); // 从 x 到 y-1 一共 y-x 个数
        }
        return ans;
    }
}

// 移动零（简单）
// 双指针
class Solution {
	public void moveZeroes(int[] nums) {
		if(nums==null) {
			return;
		}
		//第一次遍历的时候，j指针记录非0的个数，只要是非0的统统都赋给nums[j]
		int j = 0;
		for(int i=0;i<nums.length;++i) {
			if(nums[i]!=0) {
				nums[j++] = nums[i];
			}
		}
		//非0元素统计完了，剩下的都是0了
		//所以第二次遍历把末尾的元素都赋为0即可
		for(int i=j;i<nums.length;++i) {
			nums[i] = 0;
		}
	}
}	

// 盛最多水的容器
/*
 * 双指针
 * 每次都移动最小一侧那个
 * 不用判断坐移或者右移是否比原来大，直接用math去比较就行
 */
public class Solution {
    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int ans = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            ans = Math.max(ans, area);
            if (height[l] <= height[r]) {
                ++l;
            }
            else {
                --r;
            }
        }
        return ans;
    }
}
// 三数之和
// 先for，装双指针，挺麻烦的其实，中间细节听偶，而且双指针有结果后还需要left++， right--
// 下面这个思路还是挺清晰的
/*
 *  特判，对于数组长度 n，如果数组为 null 或者数组长度小于 3，返回 []。
    对数组进行排序。
    遍历排序后数组：
        若 nums[i]>0：因为已经排序好，所以后面不可能有三个数加和等于 0，直接返回结果。
        对于重复元素：跳过，避免出现重复解
        令左指针 L=i+1，右指针 R=n−1，当 L<R 时，执行循环：
            当 nums[i]+nums[L]+nums[R]==0，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将 L,R 移到下一位置，寻找新的解
            若和大于 0，说明 nums[R] 太大，R 左移
            若和小于 0，说明 nums[L] 太小，L 右移
 */
class Solution {
    //定义三个指针，保证遍历数组中的每一个结果
    //画图，解答
    public List<List<Integer>> threeSum(int[] nums) {
        //定义一个结果集
        List<List<Integer>> res = new ArrayList<>();
        //数组的长度
        int len = nums.length;
        //当前数组的长度为空，或者长度小于3时，直接退出
        if(nums == null || len <3){
            return res;
        }
        //将数组进行排序
        Arrays.sort(nums);
        //遍历数组中的每一个元素
        for(int i = 0; i<len;i++){
            //如果遍历的起始元素大于0，就直接退出
            //原因，此时数组为有序的数组，最小的数都大于0了，三数之和肯定大于0
            if(nums[i]>0){
                break;
            }
            //去重，当起始的值等于前一个元素，那么得到的结果将会和前一次相同
            if(i > 0 && nums[i] == nums[i-1]) continue;
            int l = i +1;
            int r = len-1;
            //当 l 不等于 r时就继续遍历
            while(l<r){
                //将三数进行相加
                int sum = nums[i] + nums[l] + nums[r];
                //如果等于0，将结果对应的索引位置的值加入结果集中
                if(sum==0){
                    // 将三数的结果集加入到结果集中
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    //在将左指针和右指针移动的时候，先对左右指针的值，进行判断
                    //如果重复，直接跳过。
                    //去重，因为 i 不变，当此时 l取的数的值与前一个数相同，所以不用在计算，直接跳
                    while(l < r && nums[l] == nums[l+1]) {
                        l++;
                    }
                    //去重，因为 i不变，当此时 r 取的数的值与前一个相同，所以不用在计算
                    while(l< r && nums[r] == nums[r-1]){
                        r--;
                    } 
                    //将 左指针右移，将右指针左移。
                    l++;
                    r--;
                    //如果结果小于0，将左指针右移
                }else if(sum < 0){
                    l++;
                    //如果结果大于0，将右指针左移
                }else if(sum > 0){
                    r--;
                }
            }
        }
        return res;
    }
}

// 接雨水
/*
 * 类似动态规划，没什么好讲的，需要记住就行
 * 三个数组：
 *  从左往右的左侧最大
 *  从右往左的右侧最大
 *  在算当前最大面积
 */
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0) {
            return 0;
        }

        int[] leftMax = new int[n];
        leftMax[0] = height[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i]);
        }

        int[] rightMax = new int[n];
        rightMax[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i]);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans += Math.min(leftMax[i], rightMax[i]) - height[i];
        }
        return ans;
    }
}

// 无重复字符的最长子串
// 滑动窗
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> widow = new HashMap<>();
        int left = 0,right = 0;
        int length = 0;
        while(right<s.length()) {
            char c = s.charAt(right);
            right++;
            widow.put(c,widow.getOrDefault(c,0)+1);
            while(widow.get(c)>1){
                char c1 = s.charAt(left);
                left++;
                widow.put(c1,widow.get(c1)-1);
            }
            if(right-left>length)
                length = right-left;
        }
        return length;
    }
}

// 找到字符串中所有字母异位词
// 套模板
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> widow = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:p.toCharArray()){
            need.put(c,need.getOrDefault(c,0)+1);
        }
        //注意这里初始化
        int left = 0,right = 0;
        int vilid = 0;
        ArrayList<Integer> marry = new ArrayList<>();
        while(right<s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                widow.put(c, widow.getOrDefault(c, 0) + 1);
                //注意这里不能用==，？但是char不是常量吧
                if (need.get(c).equals(widow.get(c)) ) {
                    vilid++;
                }
            }
            while(vilid==need.size()){
                if(right-left==p.length()){
                    marry.add(left);
                }
                char c1 = s.charAt(left);
                left++;
                if(need.containsKey(c1)){
                    if(need.get(c1).equals(widow.get(c1))){
                        vilid--;
                        //break;
                    }
                    widow.put(c1,widow.get(c1)-1);
                }
            }
        }
        return marry;
    }
}

// 和为K的子数组
/*
 * 前缀和 + 哈希表优化
 * 
 * 当我们在数组中向前移动时，我们逐步增加 pre（当前的累积和）。对于每个新的 pre 值，我们检查 pre - k 是否在 Map 中：
    pre - k 的意义：这个检查的意义在于，如果 pre - k 存在于 Map 中，说明之前在某个点的累积和是 pre - k。由于当前的累积和是 pre，这意味着从那个点到当前点的子数组之和恰好是 k（因为 pre - (pre - k) = k）。
    如何使用这个信息：如果 pre - k 在 Map 中，那么 pre - k 出现的次数表示从不同的起始点到当前点的子数组和为 k 的不同情况。这是因为每一个 pre - k 都对应一个起点，使得从那个起点到当前点的子数组和为 k。
    因此，每当我们找到一个 pre - k 存在于 Map 中时，我们就把它的计数（即之前这种情况发生的次数）加到 count 上，因为这表示我们又找到了相应数量的以当前元素结束的子数组，其和为 k。
 */
class Solution {
    public int subarraySum(int[] nums, int k) {
        //假设索引i处的前缀和为preSum(i)(前缀和包含nums[i]的值)
        //那么如果存在preSum(i)-k=preSum(j)--索引j处的前缀和(j < i）
        //则说明存在一个 [j-1,... i]的子数组符合条件
        //那么获取子数组的个数，等价于，获取preSum(i)-k=preSum(j) 出现的次数

        //定义HashMap, key=preSum, value=preSum 出现的次数
        Map<Integer, Integer> map = new HashMap<>();
        //针对特殊case:nums=[1],k=1;nums=[1,2,3],k=6,map插入初始值0->1
        //记录第一次出现某处索引preSum(i) = k,preSum(i) - k = 0,count=1
        map.put(0, 1);
        int preSum = 0;
        int res = 0;
        for (int i = 0; i < nums.length; i++) {
            preSum += nums[i];
            if (map.containsKey(preSum - k)) {
                //如果存在满足：preSum(i)-k=preSum(j) 对应的preSum(j)，将preSum(j)出现的次数累加到res
                res += map.get(preSum - k);
            }
            //将preSum加入map,preSum已经存在count就+1
            map.put(preSum, map.getOrDefault(preSum, 0) + 1);
        }
        return res;
    }
}

// 滑动窗口最大值
/*
 * 单调队列
 * 分成形成窗口前和窗口后
 */
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        // 未形成窗口
        for(int i = 0; i < k; i++) {
            while(!deque.isEmpty() && deque.peekLast() < nums[i])
                deque.removeLast();
            deque.addLast(nums[i]);
        }
        res[0] = deque.peekFirst();
        // 形成窗口后
        for(int i = k; i < nums.length; i++) {
            if(deque.peekFirst() == nums[i - k])
                deque.removeFirst();
            while(!deque.isEmpty() && deque.peekLast() < nums[i])
                deque.removeLast();
            deque.addLast(nums[i]);
            res[i - k + 1] = deque.peekFirst();
        }
        return res;
    }
}

// 最小覆盖子串
// 滑动窗套路
class Solution {
    public String minWindow(String s, String t) {
        Map<Character, Integer> widow = new HashMap<>();
        Map<Character, Integer> need = new HashMap<>();
        for(char c:t.toCharArray()){
            need.put(c,need.getOrDefault(c,0)+1);
        }
        //注意这里初始化
        int left = 0,right = 0;
        int vilid = 0;

        int start = 0;
        int end = 0;
        int length = s.length()+1;
        while(right<s.length()) {
            char c = s.charAt(right);
            right++;
            if (need.containsKey(c)) {
                widow.put(c, widow.getOrDefault(c, 0) + 1);
                //注意这里不能用==，？但是char不是常量吧
                if (need.get(c).equals(widow.get(c)) ) {
                    vilid++;
                }
            }
            while(vilid==need.size()){
                if(right-left<length){
                    start = left;
                    end = right;
                    length = right-left;
                }
                char c1 = s.charAt(left);
                left++;
                if(need.containsKey(c1)){
                    if(need.get(c1).equals(widow.get(c1))){
                        vilid--;
                        //break;
                    }
                    widow.put(c1,widow.get(c1)-1);
                }
            }
        }
        return length == s.length()+1?"":s.substring(start,end);
    }

}

// 最大子数组和（经典动态规划）
// 要不然我，要不然我和前面一起
// 第 53 题（最大子序和）是第 124 题（二叉树的最大路径和）的线性版本
class Solution {
    public int maxSubArray(int[] nums) {
        int[] memo = new int[nums.length];
        int sum = nums[0];
        memo[0] = nums[0];
        for (int i=1;i<nums.length; i++){
            // 要不然我，要不然和我
            if (nums[i]+memo[i-1]>nums[i]){
                memo[i] = nums[i]+memo[i-1];
            }else {
                memo[i] = nums[i];
            }
            if (sum<memo[i]) sum=memo[i];
        }
        return sum;
    }
}

// 最长递增子序列（经典动态规划）
// 动态规划，子序列不要求连续
class Solution {
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int max = 0;
        Arrays.fill(dp,1);
        for(int i=0; i<nums.length; i++){
            for(int j=0; j<i; j++){
                if (nums[i]>nums[j]){
                    if(dp[i]<dp[j]+1){
                        dp[i]=dp[j]+1;
                    }
                }
            }
            if(max<dp[i])
                max=dp[i];
        }
       return max;
    }
}

// 合并区间
// 1、排序 
// 2、比较当前是否比上一个右侧小，是加进数组去，不是，更新当前结果最后一个数组右侧值，取当前和上一个右侧最大值
class Solution {
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, new Comparator<int[]>() {
            public int compare(int[] interval1, int[] interval2) {
                return interval1[0] - interval2[0];
            }
        });
        List<int[]> merged = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; ++i) {
            int L = intervals[i][0], R = intervals[i][1];
            if (merged.size() == 0 || merged.get(merged.size() - 1)[1] < L) {
                merged.add(new int[]{L, R});
            } else {
                merged.get(merged.size() - 1)[1] = Math.max(merged.get(merged.size() - 1)[1], R);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}

// 旋转数组
/*
 * 翻转三次数组
 *  原始数组	1 2 3 4 5 6 7
    翻转所有元素	7 6 5 4 3 2 1
    翻转 [0,kmodn−1] 区间的元素	5 6 7 4 3 2 1
    翻转 [kmodn,n−1] 区间的元素	5 6 7 1 2 3 4
 */
class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start += 1;
            end -= 1;
        }
    }
}

// 除自身以外数组的乘积
// 接雨水 + 前缀积 后缀积
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;

        // L 和 R 分别表示左右两侧的乘积列表
        int[] L = new int[length];
        int[] R = new int[length];

        int[] answer = new int[length];

        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {
            L[i] = nums[i - 1] * L[i - 1];
        }

        // R[i] 为索引 i 右侧所有元素的乘积
        // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            R[i] = nums[i + 1] * R[i + 1];
        }

        // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < length; i++) {
            answer[i] = L[i] * R[i];
        }

        return answer;
    }
}

// 缺失的第一个正数
// 类似两数之和 和为K的子数组，存到Hash里，前缀和
public class Solution {

    public int firstMissingPositive(int[] nums) {
        int len = nums.length;

        Set<Integer> hashSet = new HashSet<>();
        for (int num : nums) {
            hashSet.add(num);
        }

        for (int i = 1; i <= len ; i++) {
            if (!hashSet.contains(i)){
                return i;
            }
        }

        return len + 1;
    }
}

// 矩阵置零
// 使用两个数组标注下该行和该列有没有0
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        boolean[] row = new boolean[m];
        boolean[] col = new boolean[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    row[i] = col[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (row[i] || col[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}

// 螺旋矩阵
/*
 *  从左到右遍历上边界（top行），遍历完成后top++。
    从上到下遍历右边界（right列），遍历完成后right--。
    如果top<=bottom，则从右到左遍历下边界（bottom行），遍历完成后bottom--。
    如果left<=right，则从下到上遍历左边界（left列），遍历完成后left++。
    重复上述步骤直到top>bottom或left>right。
 */
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }
        
        int m = matrix.length;    // 行数
        int n = matrix[0].length; // 列数
        int top = 0, bottom = m - 1;
        int left = 0, right = n - 1;
        
        while (top <= bottom && left <= right) {
            // 从左到右遍历上边界
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            top++;
            
            // 从上到下遍历右边界
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;
            
            // 检查是否还有行需要遍历
            if (top <= bottom) {
                // 从右到左遍历下边界
                for (int i = right; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
                bottom--;
            }
            
            // 检查是否还有列需要遍历
            if (left <= right) {
                // 从下到上遍历左边界
                for (int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
        }
        
        return result;
    }
}

// 翻转图像
/*
 * 水平翻转
 * 主对角线翻转
 */
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - i - 1][j];
                matrix[n - i - 1][j] = temp;
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
}

// 搜索二维矩阵 II
// 从右上角或左下角，确保两个沿两个方向一个+，一个-
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int x = 0, y = n - 1;
        while (x < m && y >= 0) {
            if (matrix[x][y] == target) {
                return true;
            }
            if (matrix[x][y] > target) {
                --y;
            } else {
                ++x;
            }
        }
        return false;
    }
}


// 课程表
/*
 *  根据依赖关系，构建邻接表、入度数组。
    选取入度为 0 的数据，根据邻接表，减小依赖它的数据的入度。
    找出入度变为 0 的数据，重复第 2 步。
    直至所有数据的入度为 0，得到排序，如果还有数据的入度不为 0，说明图中存在环。
 */
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        // 构建邻接表
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        
        // 记录每个节点的入度
        int[] inDegree = new int[numCourses];
        
        // 构建图
        for (int[] pre : prerequisites) {
            int course = pre[0];
            int prerequisite = pre[1];
            graph.get(prerequisite).add(course);
            inDegree[course]++;
        }
        
        // 使用队列进行BFS
        Queue<Integer> queue = new LinkedList<>();
        
        // 将所有入度为0的节点加入队列
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0) {
                queue.offer(i);
            }
        }
        
        int count = 0; // 记录已处理的课程数量
        
        while (!queue.isEmpty()) {
            int current = queue.poll();
            count++;
            
            // 遍历当前节点的所有邻居
            for (int neighbor : graph.get(current)) {
                inDegree[neighbor]--;
                if (inDegree[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        
        return count == numCourses;
    }
}

// 实现 Trie (前缀树)
/*
 * 创建一个TrieNode类，里面有isEnd和child两个成员变量
 * 
 */
class Trie {
    // 新建一个TrieNode类，为什么不直接用数组呢，是多了个isEnd，减少花销
    class TrieNode{
        private boolean isEnd;
        // 注意这里声明的数组,一般不会在构造方法中调用自己的
        TrieNode[] child;
        TrieNode(){
            isEnd = false;
            // 一般是不会在在这里new自己，这里是new了一个数组，并不是new自己
            // 可以参考这篇文章中间下面解释很好 https://bbs.csdn.net/topics/370163563
            child = new TrieNode[26];
        }
    }

    // 成员变量，根节点
    TrieNode root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new TrieNode();
    }

    /** Inserts a word into the trie. */
    public void insert(String word) {
        // 注意各个方法的root应该是最原始的，不应该相互影响
        TrieNode root = this.root;
        for (char c : word.toCharArray()){
            if (root.child[c-'a']==null){
                root.child[c-'a'] = new TrieNode();
            }
            root = root.child[c-'a'];
        }
        root.isEnd = true;
    }

    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        TrieNode root = this.root;
        for (char c : word.toCharArray()){
            if (root.child[c-'a']==null){
                return false;
            }
            root = root.child[c-'a'];
        }
        return root.isEnd;
    }

    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        TrieNode root = this.root;
        for (char c : prefix.toCharArray()){
            if (root.child[c-'a']==null){
                return false;
            }
            root = root.child[c-'a'];
        }
        return true;
    }
}

// 全排列
// DFS
class Solution {
    List<List<Integer>> mres = new LinkedList<>();
    public List<List<Integer>> permute(int[] nums) {
        LinkedList<Integer> temlist = new LinkedList<>();
        trackback(nums, temlist);
        return mres;
    }

    private void trackback(int[] nums, LinkedList<Integer> temlist) {
        // return case
        if (temlist.size() == nums.length){
            mres.add(new LinkedList<>(temlist));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (temlist.contains(nums[i])) {
                continue;
            }
            temlist.add(nums[i]);
            trackback(nums, temlist);
            temlist.removeLast();
        }
    }
}

// 子集
// 输入：nums = [1,2,3]
// 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    public List<List<Integer>> subsets(int[] nums) {
        LinkedList<Integer> marr = new LinkedList<>();
        tracback(nums,0, marr);
        return res;
    }
    //这里有问题还必须引入一个i,不然就全排列了，子集和全排列应该就这里的区别
    public void tracback(int[] nums, int start, LinkedList<Integer> marr){
        res.add(new LinkedList<>(marr));
        for (int i=start; i<nums.length; i++){
            marr.add(nums[i]);
            tracback(nums, i+1, marr);
            marr.removeLast();
        }
    }
}

// 电话号码的字母组合
// 生成一个数组DFS,这个不用回溯
class Solution {
    String[] memo = {"0","0","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    List<String> mres = new LinkedList<>();
    public List<String> letterCombinations(String digits) {
        // dfs
        if(digits.length()==0) return new LinkedList<>();
        String s = new String();
        dfs(digits, 0, s);
        return mres;
    }

    private void dfs(String digits, int deep, String s) {
        if (deep==digits.length()){
            mres.add(s);
            return;
        }
        String schar = memo[digits.charAt(deep)-'0'];
        for (int i=0; i<schar.length(); i++){
            dfs(digits, deep+1, s+schar.charAt(i));
        }
    }
}

// 组合总和
// DFS+回溯
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // 其实是一个子集问题，用子集的套路
        LinkedList<Integer> marr = new LinkedList<>();
        tracebace(candidates, target, marr, 0);
        return res;

    }

    public void tracebace(int[] candidates, int target, LinkedList<Integer> marr, int sum){
        // 不符合的情况
        if (sum>target) return;
        if (sum==target){
            res.add(new LinkedList<>(marr));
            return;
        }
        // 尝试添加下个数
        for (int i=0; i<candidates.length; i++){
            if (!marr.isEmpty()&&candidates[i]<marr.get(marr.size()-1)) continue;
            marr.add(candidates[i]);
            tracebace(candidates, target, marr, sum+candidates[i]);
            marr.removeLast();
        }
    }
}

// 括号生成
// 剪枝 + DFS + 尝试添加左右括号
class Solution {
    ArrayList<String> res = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        StringBuilder mstr = new StringBuilder();
        tracebace(n, n, mstr);
        return res;
    }

    public void tracebace(int left, int right, StringBuilder mstr){
        // 不合法的跳过
        if (left<0||right<0) return;
        if (left>right) return;
        if (left==0 && right==0){
            res.add(new String(mstr));
            return;
        }

        //尝试添加左括号
        mstr.append('(');
        tracebace(left-1, right, mstr);
        mstr.deleteCharAt(mstr.length()-1);

        //尝试添加右括号
        mstr.append(')');
        tracebace(left, right-1, mstr);
        mstr.deleteCharAt(mstr.length()-1);
    }
}

// 单词搜索
// 遍历入口+dfs回溯
class Solution {
    int[][] memo;
    public boolean exist(char[][] board, String word) {
        // 回溯
        memo = new int[board.length][board[0].length];
        for (int i=0; i<board.length; i++){
            for (int j=0; j<board[0].length; j++){
                if (board[i][j]==word.charAt(0)){
                    if (trackback(board, word, 0, i, j)){
                        return true;
                    }
                }
            }
        }
        return false;
    }
    public boolean trackback(char[][] board, String word, int index, int x, int y){
        // return case
        if (x<0||x>=board.length||y<0||y>=board[0].length||board[x][y]!=word.charAt(index)||memo[x][y]==1){
            return false;
        }
        if (index==word.length()-1){
            return true;
        }
        memo[x][y] = 1;
        
        boolean mres =  trackback(board, word, index+1, x+1, y)||
                trackback(board,word, index+1, x,y+1)||
                trackback(board,word,index+1,x-1,y)||
                trackback(board,word,index+1, x, y-1);
        if (mres){
            return true;
        }else {
            // 回溯注意这里啊，得恢复原来的状态
            memo[x][y] = 0;
        }
        return false;
    }
}

// 分割回文串
// 动态规划+DFS

// N皇后
//还是需要注意中间加1减1的问题，但是需要自己过一遍吧，不能光debug
class Solution {
    private List<List<String>> msolution = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        //这里初始化不太好像，用二维数组的话，如何转成string或者用具体的forfor具体数组[][]赋值
        char[][] mchar = new char[n][n];
        for(char[] mchar1 : mchar){
            Arrays.fill(mchar1,'.');
            //注意这里，对于基本数据类型不能用增强for进行赋值
//            for(char mmchar : mchar1){
//                mmchar='.';
//            }
        }
        backtrack(mchar,0);
        return msolution;
    }
    public void backtrack(char[][] mchar, int row){
        if(row==(mchar.length)){
            msolution.add(toList(mchar));
            return;
        }
        for(int col=0; col<mchar.length; col++){
            if(isValid(mchar, row, col)){
                mchar[row][col]='Q';
                //这里应该用new string还是tostring，还有这个写的小list太啰嗦了吧
                //更新 tostring是地址不行,
                //mrowsolution.add(new String(mchar[row]));
                backtrack(mchar,row+1);
                //mrowsolution.remove(new String(mchar[row]));
                mchar[row][col]='.';
            };
        }
    }
    public boolean isValid(char[][] mchar1, int row, int col){
        //判断三个方向，左上，上，右上
        //判断左上
        //注意for循环中间是可以定义多个参数的
        if(row == 0) return true;
        for(int i=row, j=col; i>0&&j>0; i--,j--){
            if(mchar1[i-1][j-1]=='Q') return false;
        };
        //判断右上
        for(int i=row, j=col; i>0&&j<mchar1.length-1; i--,j++){
            if(mchar1[i-1][j+1]=='Q') return false;
        };
        //判断正上
        for(int i = row; i>0; i--){
            if(mchar1[i-1][col]=='Q') return false;
        };
        return true;
    };
    //将二维数组转为List<String>
    private List<String> toList(char[][] mchar){
        List<String> mlist = new ArrayList<>();
        for(char[] mmchar : mchar){
            //使用append加入到mmchar
            //StringBuilder msb = new StringBuilder();
            //msb.append(mmchar);
            mlist.add(new String(mmchar));
        }
        return mlist;
    }

}

// 搜索插入位置
// 二分搜素
class Solution {
    public int searchInsert(int[] nums, int target) {
        int n = nums.length;
        int left = 0, right = n - 1, ans = n;
        while (left <= right) {
            int mid = ((right - left) >> 1) + left;
            if (target <= nums[mid]) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;
    }
}

// 搜索二维矩阵
// 一次二分搜索就行
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length;
        int low = 0, high = m * n - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            int x = matrix[mid / n][mid % n];
            if (x < target) {
                low = mid + 1;
            } else if (x > target) {
                high = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
}

// 在排序数组中查找元素的第一个和最后一个位置
// 两次二分搜索
class Solution {
    public static void main(String[] args) {
        int[] arr ={0,0,0,1,2,3};
        searchRange(arr, 0);
    }
    public static int[] searchRange(int[] nums, int target) {
        if (nums.length==0) return new int[]{-1, -1};
        // 二分查找
        int left = 0;
        int right = nums.length-1;
        int mresleft = 0;
        int mresright = 0;
        while (left<=right){
            int mid = left + (right-left)/2;
            if (nums[mid]==target){
                right = mid-1;
            }else if (nums[mid]>target){
                right = mid-1;
            }else if (nums[mid]<target){
                left = mid+1;
            }
        }
        if (left>=nums.length||nums[left]!=target){
            return new int[]{-1, -1};
        }
        mresleft = left;

        left = 0;
        right = nums.length-1;
        while (left<=right){
            int mid = left+(right-left)/2;
            if (nums[mid]==target){
                left = mid+1;
            }else if (nums[mid]>target){
                right = mid-1;
            }else if (nums[mid]<target){
                left = mid+1;
            }
        }
        if (right<0||nums[right]!=target){
            return new int[]{-1,-1};
        }
        mresright = right;
        return new int[]{mresleft, mresright};
    }
}

// 搜索旋转排序数组
/*
 * 判断是否有序，再选择搜索区间
 * 
 * 如果 [l, mid - 1] 是有序数组，且 target 的大小满足 [nums[l],nums[mid])，则我们应该将搜索范围缩小至 [l, mid - 1]，否则在 [mid + 1, r] 中寻找。
如果 [mid, r] 是有序数组，且 target 的大小满足 [nums[mid+1],nums[r]]，则我们应该将搜索范围缩小至 [mid + 1, r]，否则在 [l, mid - 1] 中寻找。
 */
class Solution {
    public int search(int[] nums, int target) {
        int n = nums.length;
        if (n == 0) {
            return -1;
        }
        if (n == 1) {
            return nums[0] == target ? 0 : -1;
        }
        int l = 0, r = n - 1;
        while (l <= r) {
            int mid = (l + r) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[0] <= nums[mid]) {
                if (nums[0] <= target && target < nums[mid]) {
                    r = mid - 1;
                } else {
                    l = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[n - 1]) {
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }
}

// 寻找两个正序数组的中位数 （hard）

// 有效的括号
// 栈存一下
class Solution {
    public boolean isValid(String s) {
        int n = s.length();
        Stack<Character> mque = new Stack<>();
        for (int i=0; i<n; i++){
            if (s.charAt(i)=='('||s.charAt(i)=='{'||s.charAt(i)=='['){
                mque.push(s.charAt(i));
            }else if (mque.isEmpty()){
                return false;
            } else {
                if (s.charAt(i)==')'&&!mque.peek().equals('(')){
                    return false;
                }else if (s.charAt(i)=='}'&&!mque.peek().equals('{')){
                    return false;
                }else if (s.charAt(i)==']'&&!mque.peek().equals('[')){
                    return false;
                }else{
                    mque.pop();
                }
            }
        }
        return mque.size() == 0;
    }
}

// 最小栈
// 使用辅助栈
/*
 *  push() 方法： 每当push()新值进来时，如果 小于等于 min_stack 栈顶值，则一起 push() 到 min_stack，即更新了栈顶最小值；
    pop() 方法： 判断将 pop() 出去的元素值是否是 min_stack 栈顶元素值（即最小值），如果是则将 min_stack 栈顶元素一起 pop()，这样可以保证 min_stack 栈顶元素始终是 stack 中的最小值。
    getMin()方法： 返回 min_stack 栈顶即可。
 */
class MinStack {
    private Stack<Integer> stack;
    private Stack<Integer> min_stack;
    public MinStack() {
        stack = new Stack<>();
        min_stack = new Stack<>();
    }
    public void push(int x) {
        stack.push(x);
        if(min_stack.isEmpty() || x <= min_stack.peek())
            min_stack.push(x);
    }
    public void pop() {
        if(stack.pop().equals(min_stack.peek()))
            min_stack.pop();
    }
    public int top() {
        return stack.peek();
    }
    public int getMin() {
        return min_stack.peek();
    }
}