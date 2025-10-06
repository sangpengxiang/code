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


// 跳跃游戏II
// 贪心，end maxPosition steps
public int jump(int[] nums) {
    int end = 0;
    int maxPosition = 0; 
    int steps = 0;
    for(int i = 0; i < nums.length - 1; i++){
        //找能跳的最远的
        maxPosition = Math.max(maxPosition, nums[i] + i); 
        if( i == end){ //遇到边界，就更新边界，并且步数加一
            end = maxPosition;
            steps++;
        }
    }
    return steps;
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
