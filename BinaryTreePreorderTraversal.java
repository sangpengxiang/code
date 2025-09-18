import java.util.*;

import javax.swing.tree.TreeNode;

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