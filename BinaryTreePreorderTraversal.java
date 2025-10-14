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
// 上面这个赋值1是表示node数，直接赋0表示边数也可以
class Solution {
    int res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        help(root);
        return res;
    }
    public int help(TreeNode root) {
        if(root == null) return 0;
        int len_l = help(root.left);
        int len_r = help(root.right);
        res = Math.max(res, len_l+len_r);
        return Math.max(len_l, len_r) + 1 ;
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

// 路径总和 III
// 前缀和 + 哈希表（优化解法）
class Solution {
    public int pathSum(TreeNode root, int targetSum) {
        // 使用HashMap记录前缀和出现的次数
        Map<Long, Integer> prefixSumCount = new HashMap<>();
        // 初始化：前缀和为0的路径有1条（空路径）
        prefixSumCount.put(0L, 1);
        return dfs(root, 0L, targetSum, prefixSumCount);
    }
    
    private int dfs(TreeNode node, long currentSum, int targetSum, Map<Long, Integer> prefixSumCount) {
        if (node == null) {
            return 0;
        }
        
        // 计算当前路径和
        currentSum += node.val;
        
        // 查找是否存在前缀和使得 currentSum - prefixSum = targetSum
        // 即 prefixSum = currentSum - targetSum
        int count = prefixSumCount.getOrDefault(currentSum - targetSum, 0);
        
        // 更新当前前缀和的出现次数
        prefixSumCount.put(currentSum, prefixSumCount.getOrDefault(currentSum, 0) + 1);
        
        // 递归处理左右子树
        count += dfs(node.left, currentSum, targetSum, prefixSumCount);
        count += dfs(node.right, currentSum, targetSum, prefixSumCount);
        
        // 回溯，移除当前前缀和（因为要返回上一层）
        prefixSumCount.put(currentSum, prefixSumCount.get(currentSum) - 1);
        
        return count;
    }
}

// 二叉树的最近公共祖先
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // 递归终止条件
        if (root == null || root == p || root == q) {
            return root;
        }
        
        // 在左子树中查找
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        // 在右子树中查找
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        // 如果左右子树都找到了，说明当前节点就是最近公共祖先
        if (left != null && right != null) {
            return root;
        }
        
        // 如果只有一边找到了，返回找到的那边
        return left != null ? left : right;
    }
}

// 二叉树中的最大路径和
// 递归实现
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private int maxSum = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
    
    private int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }
        
        // 递归计算左右子树的最大贡献值
        // 如果贡献值为负，则不计入（取0）
        int leftGain = Math.max(maxGain(node.left), 0);
        int rightGain = Math.max(maxGain(node.right), 0);
        
        // 计算当前节点的路径和（可以包含左右子树）
        int priceNewPath = node.val + leftGain + rightGain;
        
        // 更新全局最大路径和
        maxSum = Math.max(maxSum, priceNewPath);
        
        // 返回当前节点的最大贡献值（只能选择一条路径）
        return node.val + Math.max(leftGain, rightGain);
    }
}


///////////////////////////////
// 相交链表
// 遍历完，换对方
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) {
            return null;
        }
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }
}

// 反转链表
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null;
        while(cur != null) {
            ListNode tmp = cur.next; // 暂存后继节点 cur.next
            cur.next = pre;          // 修改 next 引用指向
            pre = cur;               // pre 暂存 cur
            cur = tmp;               // cur 访问下一节点
        }
        return pre;
    }
}

// 回文链表
class Solution {
    public boolean isPalindrome(ListNode head) {
        List<Integer> vals = new ArrayList<Integer>();

        // 将链表的值复制到数组中
        ListNode currentNode = head;
        while (currentNode != null) {
            vals.add(currentNode.val);
            currentNode = currentNode.next;
        }

        // 使用双指针判断是否回文
        int front = 0;
        int back = vals.size() - 1;
        while (front < back) {
            if (!vals.get(front).equals(vals.get(back))) {
                return false;
            }
            front++;
            back--;
        }
        return true;
    }
}

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

// 两两交换链表中的节点
// 迭代
class Solution {
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode current = dummy;
        
        while (current.next != null && current.next.next != null) {
            ListNode node1 = current.next;
            ListNode node2 = current.next.next;
            
            // 交换节点
            node1.next = node2.next;
            node2.next = node1;
            current.next = node2;
            
            // 移动到下一对的前一个位置
            current = node1;
        }
        
        return dummy.next;
    }
}

//  K 个一组翻转链表
class SolutionKGroup {
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        
        while (prev.next != null) {
            // 检查剩余节点是否足够k个
            ListNode end = prev;
            for (int i = 0; i < k && end != null; i++) {
                end = end.next;
            }
            if (end == null) break;
            
            // 反转k个节点
            ListNode start = prev.next;
            ListNode nextGroup = end.next;
            end.next = null;
            prev.next = reverse(start);
            start.next = nextGroup;
            
            prev = start;
        }
        
        return dummy.next;
    }
    
    private ListNode reverse(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
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

///////////////////////////////
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

// 字母异位词分组
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

// 轮转数组
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

////////////////////////
// 岛屿数量
// DFS
class Solution {
    public int numIslands(char[][] grid) {
        int mres = 0;
        for (int row=0; row<grid.length; row++){
            for (int loc=0; loc<grid[0].length; loc++){
                if (grid[row][loc]=='1'){
                    mres++;
                    // dfs去着色
                    dfs(grid, row, loc);
                }
            }
        }
        return mres;
    }

    private void dfs(char[][] grid, int row, int loc) {
        if (row<0||row>=grid.length||loc<0||loc>=grid[0].length){
            return;
        }
        if (grid[row][loc]=='0'){
            return;
        }
        grid[row][loc] = '0';
        dfs(grid, row-1,loc);
        dfs(grid,row+1, loc);
        dfs(grid,row,loc-1);
        dfs(grid,row,loc+1);
    }
}

// 腐烂的橘子
/*
 * 
 *  初始化队列，记录腐烂橘子的位置。

    初始化分钟数=0，新鲜橘子计数=0。

    遍历网格，统计新鲜橘子数量，并将腐烂橘子加入队列。

    如果新鲜橘子数量为0，直接返回0。

    开始BFS：
    当队列不为空且还有新鲜橘子时，进行扩散。
    记录当前队列的大小（即当前分钟的腐烂橘子数），然后逐个处理。
    对于每个腐烂橘子，检查其上下左右四个方向，如果有新鲜橘子，则将其腐烂，并加入队列，同时新鲜橘子数量减1。
    当前分钟的所有腐烂橘子处理完后，分钟数加1。

    最后，检查新鲜橘子数量，如果为0，返回分钟数；否则返回-1。
 */
class Solution {
    public int orangesRotting(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int freshOranges = 0;
        
        // 初始化：统计新鲜橘子，腐烂橘子入队
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) {
                    freshOranges++;
                } else if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                }
            }
        }
        
        // 如果没有新鲜橘子，直接返回0
        if (freshOranges == 0) return 0;
        
        int minutes = 0;
        int[][] directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        
        // BFS遍历
        while (!queue.isEmpty()) {
            int size = queue.size();
            boolean infected = false;
            
            for (int i = 0; i < size; i++) {
                int[] cell = queue.poll();
                
                for (int[] dir : directions) {
                    int x = cell[0] + dir[0];
                    int y = cell[1] + dir[1];
                    
                    if (x >= 0 && x < m && y >= 0 && y < n && grid[x][y] == 1) {
                        grid[x][y] = 2;
                        freshOranges--;
                        queue.offer(new int[]{x, y});
                        infected = true;
                    }
                }
            }
            
            if (infected) minutes++;
        }
        
        return freshOranges == 0 ? minutes : -1;
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
// DFS+判断回文
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<>();
        if (s == null || s.length() == 0) return result;
        
        // 记忆化：存储已经计算过的回文判断
        Boolean[][] memo = new Boolean[s.length()][s.length()];
        backtrackWithMemo(s, 0, new ArrayList<>(), result, memo);
        
        return result;
    }
    
    private void backtrackWithMemo(String s, int start, List<String> path,
                                  List<List<String>> result, Boolean[][] memo) {
        if (start == s.length()) {
            result.add(new ArrayList<>(path));
            return;
        }
        
        for (int end = start; end < s.length(); end++) {
            if (isPalindromeWithMemo(s, start, end, memo)) {
                path.add(s.substring(start, end + 1));
                backtrackWithMemo(s, end + 1, path, result, memo);
                path.remove(path.size() - 1);
            }
        }
    }
    
    private boolean isPalindromeWithMemo(String s, int left, int right, Boolean[][] memo) {
        if (memo[left][right] != null) {
            return memo[left][right];
        }
        
        int l = left, r = right;
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) {
                memo[left][right] = false;
                return false;
            }
            l++;
            r--;
        }
        
        memo[left][right] = true;
        return true;
    }
}

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
// l 最后指向mid + 1，如果不存在target元素，那么就是插入位置
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while(left<=right) {
            int mid = left + (right - left)/2;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target) {
                left = mid +1;
            } else if (nums[mid] > target) {
                right = mid -1;
            }
        }
        return left;
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
 * 旋转数组特性：旋转后的数组由两个有序部分组成

    关键观察：每次二分时，至少有一半是有序的

    判断有序部分：
        如果 nums[left] <= nums[mid]，左半部分有序
        否则，右半部分有序
    目标值定位：
        在有序部分中判断目标值是否存在
        如果存在，在有序部分继续搜索
        如果不存在，在另一部分搜索
 * 
 * */
class Solution {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) return -1;
        
        int left = 0;
        int right = nums.length - 1;
        
        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            // 找到目标值
            if (nums[mid] == target) {
                return mid;
            }
            
            // 判断左半部分是否有序
            if (nums[left] <= nums[mid]) {
                // 左半部分有序
                if (target >= nums[left] && target < nums[mid]) {
                    // target 在有序的左半部分
                    right = mid - 1;
                } else {
                    // target 在右半部分
                    left = mid + 1;
                }
            } else {
                // 右半部分有序
                if (target > nums[mid] && target <= nums[right]) {
                    // target 在有序的右半部分
                    left = mid + 1;
                } else {
                    // target 在左半部分
                    right = mid - 1;
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

// 字符串解码
// @b 双栈，主要是嵌套，【 时，两个都入栈
/*
 *  使用两个栈，一个存储数字（重复次数），一个存储字符串

    遍历字符串：
        如果当前字符是数字，解析数字（注意可能有多位数字）并存入数字栈
        如果当前字符是字母，将其添加到当前字符串中
        如果当前字符是'['，将当前字符串和数字分别入栈，然后重置当前字符串和数字（因为进入新的嵌套）
        如果当前字符是']'，则从数字栈中弹出重复次数，从字符串栈中弹出之前保存的字符串，将当前字符串重复指定次数后拼接到弹出的字符串后面，作为当前字符串
 */
class Solution {
    public String decodeString(String s) {
        // 使用两个栈：一个存数字，一个存字符串
        Stack<Integer> countStack = new Stack<>();
        Stack<StringBuilder> stringStack = new Stack<>();
        
        StringBuilder currentString = new StringBuilder();
        int currentNum = 0;
        
        for (char c : s.toCharArray()) {
            if (Character.isDigit(c)) {
                // 处理数字（可能有多位数）
                currentNum = currentNum * 10 + (c - '0');
            } else if (c == '[') {
                // 遇到左括号，将当前状态压栈
                countStack.push(currentNum);
                stringStack.push(currentString);
                
                // 重置当前状态
                currentNum = 0;
                currentString = new StringBuilder();
            } else if (c == ']') {
                // 遇到右括号，弹出栈顶状态进行计算
                int repeatCount = countStack.pop();
                StringBuilder decodedString = stringStack.pop();
                
                // 重复当前字符串 repeatCount 次
                for (int i = 0; i < repeatCount; i++) {
                    decodedString.append(currentString);
                }
                
                currentString = decodedString;
            } else {
                // 普通字符，直接添加到当前字符串
                currentString.append(c);
            }
        }
        
        return currentString.toString();
    }
}

// 每日温度
// 其实就是for两边，中间用栈可以退出优化了一下，但每次都得压入
// 栈存的是索引
class Solution {
    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int[] answer = new int[n];
        Stack<Integer> stack = new Stack<>(); // 存储索引的单调栈
        
        for (int i = 0; i < n; i++) {
            // 当栈不为空且当前温度大于栈顶索引对应的温度
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int prevIndex = stack.pop();
                answer[prevIndex] = i - prevIndex; // 计算等待天数
            }
            // 将当前索引入栈
            stack.push(i);
        }
        
        // 栈中剩余的元素默认就是0（没有更高温度）
        return answer;
    }
}

// 柱状图中最大的矩形（hard）
// 和接雨水感觉有点像，这里用栈优化，去找左边第一个小于当前高度的位置，右边第一个小于当前高度的位置
// 注意数组存的是索引，后面再计算一遍
class Solution {
    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) return 0;
        
        int n = heights.length;
        int[] leftBound = new int[n];  // 左边第一个小于当前高度的位置
        int[] rightBound = new int[n]; // 右边第一个小于当前高度的位置
        
        // 初始化右边界数组
        Arrays.fill(rightBound, n);
        
        Stack<Integer> stack = new Stack<>();
        
        // 计算左边界
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && heights[i] <= heights[stack.peek()]) {
                stack.pop();
            }
            leftBound[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }
        
        stack.clear();
        
        // 计算右边界
        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && heights[i] <= heights[stack.peek()]) {
                stack.pop();
            }
            rightBound[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }
        
        // 计算最大面积
        int maxArea = 0;
        for (int i = 0; i < n; i++) {
            int width = rightBound[i] - leftBound[i] - 1;
            maxArea = Math.max(maxArea, heights[i] * width);
        }
        
        return maxArea;
    }
}


// 数组中的第K个最大元素
// 堆排+建堆+从栈顶移除k个，其实就是堆排最后少移几个
// 大顶堆不保证前序遍历递增，所以后面需要取几个往后移
// 看这个直接用堆排，就是最后一点不一样，这里直接交换k次就行了，不用像堆排全部遍历一遍
class Solution {
    public int findKthLargest(int[] nums, int k) {
        if(nums.length == 1) {
            return nums[0];
        }
        for (int i=nums.length/2-1; i>=0; i--) {
            heapfiy(nums, i, nums.length);
        }
        // 这里直接交换k次就行了
        for(int i=nums.length-1; i>=nums.length-k; i--){
            swap(nums, i, 0);
            heapfiy(nums, 0, i);
        }
        return nums[nums.length-k];
    }

    public void heapfiy(int[] nums, int cur, int size) {
        int left = cur *2 +1;
        int right = cur*2 +2;
        int max = cur;
        if(left<size && nums[left]>nums[max]) {
            max = left;
        }
        if(right<size && nums[right]>nums[max]){
            max = right;
        }
        if (max != cur) {
            swap(nums, cur, max);
            heapfiy(nums, max, size);
        }
    }

    public void swap(int[] arr, int a, int b) {
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }
}
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int heapSize = nums.length;
        buildMaxHeap(nums, heapSize);
        for (int i = nums.length - 1; i >= nums.length - k + 1; --i) {
            swap(nums, 0, i);
            --heapSize;
            maxHeapify(nums, 0, heapSize);
        }
        return nums[0];
    }

    public void buildMaxHeap(int[] a, int heapSize) {
        for (int i = heapSize / 2 - 1; i >= 0; --i) {
            maxHeapify(a, i, heapSize);
        } 
    }

    public void maxHeapify(int[] a, int i, int heapSize) {
        int l = i * 2 + 1, r = i * 2 + 2, largest = i;
        if (l < heapSize && a[l] > a[largest]) {
            largest = l;
        } 
        if (r < heapSize && a[r] > a[largest]) {
            largest = r;
        }
        if (largest != i) {
            swap(a, i, largest);
            maxHeapify(a, largest, heapSize);
        }
    }

    public void swap(int[] a, int i, int j) {
        int temp = a[i];
        a[i] = a[j];
        a[j] = temp;
    }
}

// 前 K 个高频元素
// 思想上不难，但是构建小顶堆和遍历Map
class Solution {
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> occurrences = new HashMap<Integer, Integer>();
        for (int num : nums) {
            occurrences.put(num, occurrences.getOrDefault(num, 0) + 1);
        }

        // int[] 的第一个元素代表数组的值，第二个元素代表了该值出现的次数
        PriorityQueue<int[]> queue = new PriorityQueue<int[]>(new Comparator<int[]>() {
            public int compare(int[] m, int[] n) {
                return m[1] - n[1];
            }
        });
        for (Map.Entry<Integer, Integer> entry : occurrences.entrySet()) {
            int num = entry.getKey(), count = entry.getValue();
            if (queue.size() == k) {
                if (queue.peek()[1] < count) {
                    queue.poll();
                    queue.offer(new int[]{num, count});
                }
            } else {
                queue.offer(new int[]{num, count});
            }
        }
        int[] ret = new int[k];
        for (int i = 0; i < k; ++i) {
            ret[i] = queue.poll()[0];
        }
        return ret;
    }
}

// 数据流的中位数（hard）

// 买卖股票的最佳时机
// 记 从左往右，更新最小和最大利润
public class Solution {
    public int maxProfit(int prices[]) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            } else if (prices[i] - minprice > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;
    }
}

// 跳跃游戏
// 这个看能不能跳过去 maxPosition
/*
 * 贪心算法，maxJump，for一边看能不能调到最远
 * 
 * 贪心和动态规划的区别：贪心算法是每次都选择局部最优，动态规划是选择全局最优
 */

class Solution {
    public boolean canJump(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return true;
        }
        int maxJump = 0;
        for(int i=0; i<nums.length; i++) {
            if (i<=maxJump) {
                maxJump = Math.max(maxJump, nums[i]+i);
            } else {
                return false;
            }
        }
        return true;
    }
}
/*
    动态规划算法：
        定义子问题：dp[i]表示前i个节点的能跳跃到的最大距离;
    更新公式：
        dp[i] = max{ dp[i - 1] , i + nums[i] }
        max{ 前i - 1个节点的最大距离, 当前节点能跳到的距离 }
    初始化：
        dp[0] = nums[0];
        从下表1的节点开始执行：
    执行过程：
        1.进入第i个节点前:dp[i - 1] >= i ?
            判断当前节点是否可达：根据能否从前i - 1个节点跳跃过来
        2.根据公式更新dp[i]
        返回：
            dp[len - 1] >= len - 1;

    然后很显然，dp[i]只和dp[i-1]有关, 所以可以优化空间，只需要一个变量rightMax就行
*/
class Solution {
    public boolean canJump(int[] nums) {
        int len = nums.size();
        vector<int> dp(len);
        dp[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (i > dp[i -1]) {
                return false; // 这里中间退出
            }
            dp[i] = max(dp[i - 1], i + nums[i]);
        }
        return dp[len - 1] >= len - 1;

    }
}


// 跳跃游戏II @b
// 这个一定能跳过去，看最少跳几次
// 贪心，maxPosition end steps
// 注意最后一个不跳，第二种写法好理解一点
public int jump(int[] nums) {
    int end = 0;
    int maxPosition = 0; 
    int steps = 0;
    for(int i = 0; i < nums.length - 1; i++){ // @z i < nums.length - 1
        //找能跳的最远的
        maxPosition = Math.max(maxPosition, nums[i] + i); 
        if( i == end){ //遇到边界，就更新边界，并且步数加一
            end = maxPosition;
            steps++;
        }
    }
    return steps;
}
// 或者按下面的写法
class Solution {
    public int jump(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return 0;
        }
        int end = 0;
        int maxJump = 0;
        int step = 0;
        for (int i=0; i<nums.length; i++) {
            maxJump = Math.max(maxJump, i+nums[i]);
            // 更新end, 如果是边界就不用再跳了
            if (i==end && i!=nums.length-1) {
                end = maxJump;
                step++;
            }
        }
        return step;
    }
}

// 划分字母区间（没有）

// 爬楼梯
// 动态规划，当前是 n-1 + n-2
class Solution {
    public int climbStairs(int n) {
        
        if (n==0) return 0;
        if (n==1) return 1;
        if (n==2) return 2;
        int[] mres = new int[n];
        //base case
        mres[0] = 1;
        mres[1] = 2;
        for (int i=2; i<n; i++){
            mres[i] = mres[i-1]+mres[i-2];
        }
        return mres[n-1];
    }
}
// 杨辉三角
/*
 * 这样看
 * 
    [1]
    [1,1]
    [1,2,1]
    [1,3,3,1]
    [1,4,6,4,1]
    [1,5,10,10,5,1]
 */
class Solution {
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i=0; i<numRows; i++){
            ArrayList<Integer> list = new ArrayList<>();
            // 注意这里
            for (int j=0; j<=i; j++){
                if (j==0 || j==i){
                    list.add(1);
                }else{
                    list.add(res.get(i-1).get(j-1)+res.get(i-1).get(j));
                }
            }
            res.add(list);
        }
        return res;
    }
}

// 杨辉三角
class Solution {
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int dp[] = new int[nums.length];
        Arrays.fill(dp, 0);
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i=2; i<nums.length; i++) {
            dp[i] = Math.max(dp[i-1], dp[i-2]+nums[i]);
        }
        return dp[nums.length-1];
    }
}

// 完全平方数
// @b 类似凑硬币，但是凑的是i^2
// 转移方程内部从n^2开始
class Solution {
    public int numSquares(int n) {
        var f = new int[n + 1];
        Arrays.fill(f, 10001);
        f[0] = 0;
        for (int i = 1; i * i <= n; i++) {
            for (int j = i * i; j <= n; j++) {
                f[j] = Math.min(f[j], f[j - i * i] + 1);
            }
        }
        return f[n];
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

// 单词拆分
// dp 又有点类似于前缀和
// dp 当前为true，并且剩余包含在wordDict中，那么dp[i] = true
// 每个都for
public class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}

// 最长递增子序列
// for 所有，for所有一般去每个的最大，关键还是这个dp数组的定义
/*
 *  定义 dp[i] 为以第 i 个元素结尾的最长递增子序列的长度。对于每个 i，我们遍历所有 j < i，如果 nums[j] < nums[i]，那么 dp[i] 可以取 dp[j] + 1 的最大值。

    状态转移方程：
        dp[i] = max(dp[j]) + 1, 对于所有 j < i 且 nums[j] < nums[i]

    初始化：每个位置的 dp 值至少为1（即自身作为一个序列）。

    最后，整个数组的最长递增子序列就是 dp 数组中的最大值。

 */
class Solution {
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int n = nums.length;
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        int maxLength = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxLength = Math.max(maxLength, dp[i]);
        }
        return maxLength;
    }
}

// 乘积最大子数组
/*
 * 这题是求数组中子区间的最大乘积，对于乘法，我们需要注意，
 * 负数乘以负数，会变成正数，所以解这题的时候我们需要维护两个变量，
 * 当前的最大值，以及最小值，最小值可能为负数，但没准下一步乘以一个负数，
 * 当前的最大值就变成最小值，而最小值则变成最大值了。
 */
class Solution {
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        
        int n = nums.length;
        int[] maxDP = new int[n]; // 以i结尾的最大乘积子数组
        int[] minDP = new int[n]; // 以i结尾的最小乘积子数组
        
        maxDP[0] = nums[0];
        minDP[0] = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < n; i++) {
            // 三种可能：当前数本身、当前数乘以前面的最大值、当前数乘以前面的最小值
            maxDP[i] = Math.max(nums[i], 
                               Math.max(maxDP[i - 1] * nums[i], minDP[i - 1] * nums[i]));
            minDP[i] = Math.min(nums[i], 
                               Math.min(maxDP[i - 1] * nums[i], minDP[i - 1] * nums[i]));
            
            result = Math.max(result, maxDP[i]);
        }
        
        return result;
    }
}

// 背包问题
/*
 * 0-1背包问题
 * 问题描述：给定一组物品，每种物品都有自己的重量和价值，在限定的总重量内，我们如何选择，才能使得物品的总价值最大。每种物品只有一件，可以选择放或不放。

基本思路：
使用动态规划，定义一个二维数组dp[i][j]，表示前i件物品恰好放入一个容量为j的背包可以获得的最大价值。

状态转移方程：
如果不放第i件物品，那么问题转化为前i-1件物品放入容量为j的背包，即dp[i][j] = dp[i-1][j]
如果放第i件物品，那么问题转化为前i-1件物品放入容量为j - weight[i]的背包，此时的最大价值为dp[i-1][j - weight[i]] + value[i]
所以状态转移方程为：dp[i][j] = max(dp[i-1][j], dp[i-1][j - weight[i]] + value[i])
注意：初始化时，dp[0][j]表示前0件物品，所以价值为0。dp[i][0]表示容量为0，所以价值也为0。

完全背包问题
问题描述：与0-1背包问题类似，唯一不同的地方是每种物品有无限件。

基本思路：
同样使用动态规划，状态定义与0-1背包相同。

状态转移方程：
如果不放第i件物品，那么dp[i][j] = dp[i-1][j]
如果放第i件物品，那么问题转化为前i件物品（因为可以重复选取）放入容量为j - weight[i]的背包，即dp[i][j] = dp[i][j - weight[i]] + value[i]
所以状态转移方程为：dp[i][j] = max(dp[i-1][j], dp[i][j - weight[i]] + value[i])

空间优化
对于背包问题，我们通常可以使用一维数组进行空间优化，将二维数组降为一维数组。注意在0-1背包中，内层循环需要从大到小遍历，而在完全背包中，内层循环需要从小到大遍历。

0-1背包空间优化
使用一维数组dp[j]表示容量为j的背包所能获得的最大价值。
状态转移方程：dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
注意：内层循环从大到小遍历，防止同一件物品被多次加入。

完全背包空间优化
同样使用一维数组dp[j]。
状态转移方程：dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
注意：内层循环从小到大遍历，允许同一件物品被多次加入。

0-1背包问题：
Partition Equal Subset Sum（分割等和子集）
Target Sum（目标和）
Ones and Zeroes（一和零）

完全背包问题：
Coin Change（零钱兑换）
Coin Change 2（零钱兑换 II）
Perfect Squares（完全平方数）

0-1 背包问题
    特点：每种物品只有一件，要么选，要么不选。
    状态转移方程：
    dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    关键点：内层循环遍历容量时，需要从大到小遍历，防止物品被重复使用。
    for (int i = 0; i < n; i++) {
        // 必须从后往前遍历，避免重复选择同一物品
        for (int j = capacity; j >= weights[i]; j--) {
            dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

完全背包问题
    特点：每种物品有无限件。
    状态转移方程：
    dp[j] = max(dp[j], dp[j - weight[i]] + value[i])
    关键点：内层循环遍历容量时，需要从小到大遍历，允许物品被重复使用。
    for (int i = 0; i < n; i++) {
        // 必须从前往后遍历，允许重复选择同一物品
        for (int j = weights[i]; j <= capacity; j++) {
            dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
        }
    }

遍历顺序（核心）：
    0-1背包：物品正序，容量倒序（防止重复使用）。

完全背包：
    物品正序，容量正序（允许重复使用）。
    求组合数：先遍历物品，再遍历容量。
    求排列数：先遍历容量，再遍历物品。

完全背包：
注意：为什么先遍历物品，再遍历容量是求组合数？
    因为这种顺序保证了在考虑一种新的硬币面额时，我们是在之前所有硬币组合的基础上添加这种新硬币，
    不会产生 (1,2) 和 (2,1) 这种顺序不同的重复情况。
 */
public class Knapsack01 {
    
    /**
     * 0-1背包问题 - 二维DP数组解法
     * @param capacity 背包容量
     * @param weights 物品重量数组
     * @param values 物品价值数组
     * @return 最大价值
     */
    public static int knapsack2D(int capacity, int[] weights, int[] values) {
        int n = weights.length;
        // dp[i][j] 表示前i个物品，背包容量为j时的最大价值
        int[][] dp = new int[n + 1][capacity + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= capacity; j++) {
                if (j < weights[i - 1]) {
                    // 当前物品重量大于背包容量，不能放入
                    dp[i][j] = dp[i - 1][j];
                } else {
                    // 选择：不放当前物品 或 放当前物品
                    dp[i][j] = Math.max(
                        dp[i - 1][j], 
                        dp[i - 1][j - weights[i - 1]] + values[i - 1]
                    );
                }
            }
        }
        return dp[n][capacity];
    }
    
    /**
     * 0-1背包问题 - 一维DP数组优化
     * 空间复杂度从O(n*capacity)优化到O(capacity)
     */
    public static int knapsack1D(int capacity, int[] weights, int[] values) {
        int n = weights.length;
        // dp[j] 表示背包容量为j时的最大价值
        int[] dp = new int[capacity + 1];
        
        for (int i = 0; i < n; i++) {
            // 必须从后往前遍历，避免重复选择同一物品
            for (int j = capacity; j >= weights[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
            }
        }
        return dp[capacity];
    }
    
    public static void main(String[] args) {
        int capacity = 10;
        int[] weights = {2, 3, 4, 5};
        int[] values = {3, 4, 5, 6};
        
        System.out.println("二维DP解法: " + knapsack2D(capacity, weights, values));
        System.out.println("一维DP解法: " + knapsack1D(capacity, weights, values));
    }
}

public class CompleteKnapsack {
    
    /**
     * 完全背包问题 - 二维DP数组解法
     */
    public static int completeKnapsack2D(int capacity, int[] weights, int[] values) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= capacity; j++) {
                // 最多能放 k 个当前物品
                int maxK = j / weights[i - 1];
                for (int k = 0; k <= maxK; k++) {
                    dp[i][j] = Math.max(
                        dp[i][j],
                        dp[i - 1][j - k * weights[i - 1]] + k * values[i - 1]
                    );
                }
            }
        }
        return dp[n][capacity];
    }
    
    /**
     * 完全背包问题 - 一维DP数组优化
     */
    public static int completeKnapsack1D(int capacity, int[] weights, int[] values) {
        int n = weights.length;
        int[] dp = new int[capacity + 1];
        
        for (int i = 0; i < n; i++) {
            // 必须从前往后遍历，允许重复选择同一物品
            for (int j = weights[i]; j <= capacity; j++) {
                dp[j] = Math.max(dp[j], dp[j - weights[i]] + values[i]);
            }
        }
        return dp[capacity];
    }
    
    public static void main(String[] args) {
        int capacity = 10;
        int[] weights = {2, 3, 4, 5};
        int[] values = {3, 4, 5, 6};
        
        System.out.println("完全背包二维DP: " + completeKnapsack2D(capacity, weights, values));
        System.out.println("完全背包一维DP: " + completeKnapsack1D(capacity, weights, values));
    }
}

// 分割等和子集
class Solution {
    public boolean canPartition(int[] nums) {
        if (nums == null || nums.length < 2) {
            return false;
        }
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum % 2 != 0) {
            return false;
        }
        sum = sum / 2;
        // 0-1背包问题
        // i是第i个物品，j是weight
        boolean[][] dp = new boolean[nums.length + 1][sum + 1];
        // 初始化：和为0总是可以达成（不选任何数字）
        for (int i = 0; i <= nums.length; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i <= nums.length; i++) {
            for (int j = 1; j <= sum; j++) {
                if (nums[i - 1] > j) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j - nums[i - 1]] || dp[i - 1][j];
                }
            }
        }
        return dp[nums.length][sum];
    }
}

// 不同路径
/*
 * 方法：动态规划
    我们使用一个二维数组 dp，其中 dp[i][j] 表示从起点 (0,0) 到达 (i,j) 的不同路径数目。
    状态转移方程：由于机器人只能向右或向下移动，所以到达 (i,j) 的路径数等于从左边 (i-1,j) 和上边 (i,j-1) 的路径数之和。
    即：dp[i][j] = dp[i-1][j] + dp[i][j-1]

    初始化：对于第一行和第一列，由于只能一直向右或向下，所以只有一条路径，即 dp[0][j] = 1, dp[i][0] = 1。
 * 
 */
class Solution {
    public int uniquePaths(int m, int n) {
        // dp[i][j] 表示到达(i,j)位置的不同路径数
        int[][] dp = new int[m][n];
        
        // 初始化第一行和第一列
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }
        
        // 填充dp数组
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        
        return dp[m-1][n-1];
    }
}

// 最小路径和
/*
 * 动态规划，和上面类似，但是初始条件不一样
 */
class Solution {
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        
        int m = grid.length;
        int n = grid[0].length;
        
        // dp[i][j] 表示从(0,0)到(i,j)的最小路径和
        int[][] dp = new int[m][n];
        
        // 初始化起点
        dp[0][0] = grid[0][0];
        
        // 初始化第一列（只能从上边来）
        for (int i = 1; i < m; i++) {
            dp[i][0] = dp[i-1][0] + grid[i][0];
        }
        
        // 初始化第一行（只能从左边来）
        for (int j = 1; j < n; j++) {
            dp[0][j] = dp[0][j-1] + grid[0][j];
        }
        
        // 填充dp数组
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
        
        return dp[m-1][n-1];
    }
}

// 最长回文子串
/*
 *  回文串围绕中心对称。
    中心可以是一个字符（奇数长度）或两个字符之间（偶数长度）。
    枚举每个中心，然后向两边扩展，找到以这个中心为轴的最长回文。

    注意：start = i - (len - 1) / 2;
        return right - left - 1;
 */
public class LongestPalindrome {
    public static String longestPalindrome(String s) {
        if (s == null || s.length() < 2) {
            return s;
        }

        int start = 0, maxLen = 1;

        for (int i = 0; i < s.length(); i++) {
            // 奇数长度回文
            int len1 = expandFromCenter(s, i, i);
            // 偶数长度回文
            int len2 = expandFromCenter(s, i, i + 1);
            int len = Math.max(len1, len2);

            if (len > maxLen) {
                maxLen = len;
                start = i - (len - 1) / 2;
            }
        }

        return s.substring(start, start + maxLen);
    }

    // 向两边扩展，返回回文长度
    private static int expandFromCenter(String s, int left, int right) {
        while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }

    // 测试
    public static void main(String[] args) {
        System.out.println(longestPalindrome("babad")); // 输出 "bab" 或 "aba"
        System.out.println(longestPalindrome("cbbd"));  // 输出 "bb"
        System.out.println(longestPalindrome("a"));     // 输出 "a"
        System.out.println(longestPalindrome("ac"));    // 输出 "a" 或 "c"
    }
}

// 最长公共子序列
public class LongestCommonSubsequence {

    /**
     * 计算两个字符串的最长公共子序列长度
     * @param text1 第一个字符串
     * @param text2 第二个字符串
     * @return 最长公共子序列长度
     */
    public static int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length();  // text1 长度
        int n = text2.length();  // text2 长度

        // 创建 DP 表，dp[i][j] 表示：
        // text1 前 i 个字符 与 text2 前 j 个字符 的最长公共子序列长度
        int[][] dp = new int[m + 1][n + 1];

        // 从 1 开始遍历，因为 dp[0][*] 和 dp[*][0] 表示空串，对应的 LCS 长度为 0
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 如果当前字符相等，则 LCS 长度 = 上一个状态的长度 + 1
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    // 如果字符不相等，LCS 长度 = 两种可能的最大值：
                    // ① 舍弃 text1 当前字符的 LCS
                    // ② 舍弃 text2 当前字符的 LCS
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        // dp[m][n] 即为整个字符串的 LCS 长度
        return dp[m][n];
    }

    public static void main(String[] args) {
        System.out.println(longestCommonSubsequence("abcde", "ace"));  // 输出 3 ("ace")
        System.out.println(longestCommonSubsequence("abc", "abc"));    // 输出 3 ("abc")
        System.out.println(longestCommonSubsequence("abc", "def"));    // 输出 0 (没有公共子序列)
    }
}

// 编辑距离
// 这个中间记住吧，增删改没啥好说的确实不太好明确的解释
class Solution {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        
        // dp[i][j] 表示将 word1 的前 i 个字符转换为 word2 的前 j 个字符所需的最少操作次数
        int[][] dp = new int[m+1][n+1];
        
        // 初始化
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i; // 删除 i 个字符
        }
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j; // 插入 j 个字符
        }
        
        // 动态规划填充dp数组
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i-1) == word2.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = Math.min(
                        Math.min(
                            dp[i-1][j] + 1,
                            dp[i][j-1] + 1), 
                            dp[i-1][j-1] + 1);
                }
            }
        }
        
        return dp[m][n];
    }
}


// 只出现一次的数字
// 异或运算，没啥可讲的，记住吧
class Solution {
    /*
     考异或运算
     任何数和 00 做异或运算，结果仍然是原来的数，即 a \oplus 0=aa⊕0=a。
     任何数和其自身做异或运算，结果是 00，即 a \oplus a=0a⊕a=0。
     异或运算满足交换律和结合律
     */
    public int singleNumber(int[] nums) {
        int mres = 0;
        for (int i=0; i<nums.length; i++){
            mres ^= nums[i];
        }
        return mres;
    }
}

// 多数元素
// HashMap 统计，什么摩尔投票法太麻烦了
class Solution {
    public int majorityElement(int[] nums) {
        // 使用哈希表统计每个数字的出现次数
        Map<Integer, Integer> countMap = new HashMap<>();
        
        for (int num : nums) {
            countMap.put(num, countMap.getOrDefault(num, 0) + 1);
            // 如果某个数字的出现次数超过 n/2，立即返回
            if (countMap.get(num) > nums.length / 2) {
                return num;
            }
        }
        
        return -1; // 根据题目假设，这里不会执行
    }
}

// 颜色分类
// 三指针，一个就三个元素，一个从前往后排，一个从后往前排就可以了
class Solution {
    public void sortColors(int[] nums) {
        // 三指针法：p0指向0的右边界，p2指向2的左边界，curr当前指针
        int p0 = 0;           // 指向下一个0应该放置的位置
        int curr = 0;          // 当前遍历的指针
        int p2 = nums.length - 1; // 指向下一个2应该放置的位置
        
        while (curr <= p2) {
            if (nums[curr] == 0) {
                // 遇到0，交换到前面
                swap(nums, curr, p0);
                p0++;
                curr++;
            } else if (nums[curr] == 2) {
                // 遇到2，交换到后面
                swap(nums, curr, p2);
                p2--;
                // 注意：这里curr不增加，因为从后面交换过来的元素还需要检查
            } else {
                // 遇到1，直接跳过
                curr++;
            }
        }
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}

// 下一个排列
/*
 * 思路：
    从后向前查找第一个相邻元素对 (i, i+1)，满足 nums[i] < nums[i+1]。这样，nums[i] 就是需要被替换的数字。
    如果找不到这样的相邻元素对，说明整个数组是降序排列的，也就是最大的排列，那么将其反转得到最小排列（升序）。
    如果找到了这样的 i，然后从后向前查找第一个大于 nums[i] 的元素 nums[j]。
    交换 nums[i] 和 nums[j]。
    将位置 i+1 到末尾的元素反转（因为原来 i+1 到末尾是降序的，反转后变为升序，这样才是最小的下一个排列）。

举例：
    假设 nums = [1,2,3,8,5,7,6,4]
步骤：
    从后往前找，找到第一个 nums[i] < nums[i+1] 的位置，即 5<7，此时 i=4（值为5）。
    然后从后往前找第一个大于5的数，即6（位置6）。
    交换5和6，得到 [1,2,3,8,6,7,5,4]
    然后将位置5到末尾（即7,5,4）反转，得到 [1,2,3,8,6,4,5,7]
    注意：反转是因为我们想要下一个排列，而且我们知道从 i+1 到末尾是降序的，所以反转后变为升序，这样才是最小的比当前大的排列。
 */
class Solution {
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length <= 1) return;
        
        // 步骤1：从后向前找到第一个降序的位置
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            i--;
        }
        
        // 步骤2：如果找到了这样的位置
        if (i >= 0) {
            // 从后向前找到第一个大于nums[i]的数
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            // 交换这两个数
            swap(nums, i, j);
        }
        
        // 步骤3：反转i+1到末尾的序列
        reverse(nums, i + 1, nums.length - 1);
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
    
    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            swap(nums, start, end);
            start++;
            end--;
        }
    }
}

// 寻找重复数
// 链表双指针找环的入口
public int findDuplicate(int[] nums) {
    // 将数组视为链表：索引 i 指向 nums[i]
    // 由于有重复数字，链表会形成环
    
    // 阶段1：找到相遇点
    int slow = nums[0];
    int fast = nums[0];
    
    do {
        slow = nums[slow];       // 慢指针：每次走一步
        fast = nums[nums[fast]]; // 快指针：每次走两步
    } while (slow != fast);
    
    // 阶段2：找到环的入口
    slow = nums[0];              // 慢指针回到起点
    while (slow != fast) {
        slow = nums[slow];       // 两个指针都每次走一步
        fast = nums[fast];
    }
    
    return slow;                 // 环的入口就是重复数字
}