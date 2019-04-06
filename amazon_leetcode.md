<!-- GFM-TOC -->
* [973. K Closest Points to Origin](#973-k-closest-points-to-origin)
* [146. LRU Cache](#146-lru-cache)
* [1. Two Sum](#1-two-sum)
* [200. Number of Islands](#200-number-of-islands)
* [5. Longest Palindromic Substring](#5-longest-palindromic-substring)
* [819. Most Common Word](#819-most-common-word)
* [138. Copy List with Random Pointer](#138-copy-list-with-random-pointer)
* [21. Merge Two Sorted Lists](#21-merge-two-sorted-lists)
* [103. Binary Tree Zigzag Level Order Traversal](#103-binary-tree-zigzag-level-order-traversal)
* [2. Add Two Numbers](#2-add-two-numbers)
* [763. Partition Labels](#763-partition-labels)
* [127. Word Ladder](#127-word-ladder)
* [139. Word Break](#139-word-break)
<!-- GFM-TOC -->

# 973. K Closest Points to Origin

### 法1: O(nlogn) 排序

```java
public int[][] kClosest(int[][] points, int K) {
    Arrays.sort(points, Comparator.comparing(p -> p[0] * p[0] + p[1] * p[1]));
    return Arrays.copyOfRange(points, 0, K);
}
```

### 法2: O(nlogn) [priority queue](#http://www.cnblogs.com/vamei/archive/2013/03/20/2966612.html)

```java
public int[][] kClosest(int[][] points, int K) {
    if (K == 0 || points == null || points.length == 0 || K > points.length) {
        return null;
    }
    int[][] ans = new int[K][2];
    PriorityQueue<int[]> pq = new PriorityQueue<>(K, new Comparator<int[]>(){
        public int compare(int[] o1, int[] o2){
            return (o1[0]*o1[0] + o1[1]*o1[1]) - (o2[0]*o2[0] + o2[1]*o2[1]);
        }
    });
    for (int[] i : points) {
        pq.add(i);
    }
    for (int j = 0; j < K; j++) {
        ans[j] = pq.poll();
    }
    return ans;
}
```

### 法3: O(nlogn) TreeMap
```java
public int[][] kClosest(int[][] points, int K) {
    Map<Integer,Integer> orderMap = new TreeMap<Integer,Integer>();
    int[][] arr = new int[K][2];
    for (int i=0; i<points.length; i++){
        int sum = Math.abs(points[i][0]) * Math.abs(points[i][0]) +Math.abs(points[i][1]) * Math.abs(points[i][1]);
        orderMap.put(sum,i);
    }
    int count=0;
    for (Integer i: orderMap.keySet()) {
        arr[count] = points[orderMap.get(i)];
        count++;
        if(count==K){break;}
    }
    return arr;
}
```

### 法4: O(n) quick selection

```java
class Solution {
    private Random random = new Random();
    public int[][] kClosest(int[][] points, int K) {
        int start = 0, end = points.length - 1;
        int index = 0;
        while (start <= end) {
            index = partition(points, start, end);
            if (index == K) {
                break;
            }
            if (index > K) {
                end = index - 1;
            } else {
                start = index + 1;
            }
        }
        int[][] result = new int[index][2];
        for (int i = 0; i < index; i++) {
            result[i] = points[i];
        }
        return result;
    }
    
    private int partition(int[][] points, int start, int end) {
        int rd = start + random.nextInt(end - start + 1);
        int[] target = points[rd];
        swap(points, rd, end);
        int left = start, right = end - 1;
        while (left <= right) {
            while (left <= right && !isLarger(points[left], target)) left++;
            while (left <= right && isLarger(points[right], target)) right--;
            if (left <= right) {
                swap(points, left, right);
                left++;
                right--;
            }
        }
        swap(points, left, end);
        return left;
    }
    
    private void swap(int[][] points, int i1, int i2) {
        int[] temp = points[i1];
        points[i1] = points[i2];
        points[i2] = temp;
    }
    
    // return true if p1 dist is larger than p2
    private boolean isLarger(int[] p1, int[] p2) {
        return p1[0] * p1[0] + p1[1] * p1[1] > p2[0] * p2[0] + p2[1] * p2[1];
    }
}
```

# 146. LRU Cache

### 法1: 采用LinkedHashMap

```java
public class LRUCache extends LinkedHashMap<Integer, Integer> {
    private int capacity;

    public LRUCache(int capacity) {
        super(capacity, 0.75f, true);
        this.capacity = capacity;
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity;
    }

    public int get(int key) {
        return getOrDefault(key, -1);
    }

    public void set(int key, int value) {
        put(key, value);
    }
}
```

### 法2: 自己写一个，HashMap with Entry<K, V> being linked.

```java
Map<Integer, Node> map = new HashMap<>();
Node head = new Node(-1, -1);
Node tail = new Node(-1, -1);
int capacity;

public LRUCache(int capacity) {
    join(head, tail);
    this.capacity = capacity;
}

public int get(int key) {
    if (!map.containsKey(key)) {
        return -1;
    }
    Node node = map.get(key);
    remove(node);
    moveToHead(node);
    return node.val;
}

public void set(int key, int value) {
    if (map.containsKey(key)) {
        Node node = map.get(key);
        node.val = value;
        remove(node);
        moveToHead(node);
    } else {
        if (map.size() == capacity) {
            if (tail.prev != head) {
                map.remove(tail.prev.key);
                remove(tail.prev);
            }
        }
        Node node = new Node(key, value);
        moveToHead(node); 
        map.put(key, node);
    }       
}   
    
public void join(Node n1, Node n2) {
    n1.next = n2;
    n2.prev = n1;
}

public void remove(Node node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
}

public void moveToHead(Node node) {
    Node next = head.next; 
    join(head, node);
    join(node, next);
}

class Node {
    Node prev;
    Node next;
    int key;
    int val;
    public Node(int key, int val) {
        this.key = key;
        this.val = val;
    }
}
```

# 1. Two Sum

### 法1: Hashmap
```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    int[] ret = new int[2];
    for (int i = 0; i< nums.length; i++) {
        if (map.containsKey(target-nums[i])) {
            ret[0] = map.get(target-nums[i]);
            ret[1] = i;
            break;
        }
        map.put(nums[i], i);
    }
    return ret;
}
```

# 200. Number of Islands
给一个0/1组成的地图，找到1组成的区域的个数，比如
```
Input:
11110
11010
11000
00000

Output:
1
```

### 法1: DFS
```java
public class NumberofIslands {
    static int[] dx = {-1,0,0,1};
    static int[] dy = {0,1,-1,0};
    public static int numIslands(char[][] grid) {
        if(grid==null || grid.length==0) return 0;
        int islands = 0;
        for(int i=0;i<grid.length;i++) {
            for(int j=0;j<grid[i].length;j++) {
                if(grid[i][j]=='1') {
                    explore(grid,i,j);
                    islands++;
                }
            }
        }
        return islands;
    }
    public static void explore(char[][] grid, int i, int j) {
        grid[i][j]='x';
        for(int d=0;d<dx.length;d++) {
            if(i+dy[d]<grid.length && i+dy[d]>=0 && j+dx[d]<grid[0].length && j+dx[d]>=0 && grid[i+dy[d]][j+dx[d]]=='1') {
                explore(grid,i+dy[d],j+dx[d]);
            }
        }
    }
}
```

# 5. Longest Palindromic Substring
找到一个字符串里的最长的、对称的substring
比如"babad"的结果是"aba"

### 法1: dynamic programming 动态规划，dp[i][j] = dp[i + 1][j - 1] + 0/1

```java
public String longestPalindrome(String s) {
  int n = s.length();
  String res = null;
    
  boolean[][] dp = new boolean[n][n];
    
  for (int i = n - 1; i >= 0; i--) {
    for (int j = i; j < n; j++) {
      dp[i][j] = s.charAt(i) == s.charAt(j) && (j - i < 3 || dp[i + 1][j - 1]);
            
      if (dp[i][j] && (res == null || j - i + 1 > res.length())) {
        res = s.substring(i, j + 1);
      }
    }
  }
    
  return res;
}
```

# 819. Most Common Word
给一个句子，找到这个句子里出现最多的、且没有被ban掉的词
举例
```
Input: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
```
### 法1: HashMap
```java
public String mostCommonWord(String paragraph, String[] banned) {
    Map<String, Integer> wordAndCount = new HashMap<String, Integer>();
    Set<String> bansSet = new HashSet<>(Arrays.asList(banned));
    //[!?,';] is the possible punctuations for this input, can also use '\\pP' instead for all of the punctuations.
    String[] words = paragraph.replaceAll("[!?',;.]","").toLowerCase().split(" ");
    int max = 0;
    String res = "";
    for(String word:words){
        if(bansSet.contains(word)) continue;
    // use getOrDefault - Java8 new default function from Map interface
        wordAndCount.put(word, wordAndCount.getOrDefault(word, 0) + 1);
        int count = wordAndCount.get(word);
        if( count > max) {
            max = count;
            res = word;
        }
    }

    return res;
}
```

# 138. Copy List with Random Pointer
链表的每个点多了一个random pointer，现在想完全复制这一个链表

### 法1: HashMap
```java
public RandomListNode copyRandomList(RandomListNode head) {
  if (head == null) return null;
  
  Map<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();
  
  // loop 1. copy all the nodes
  RandomListNode node = head;
  while (node != null) {
    map.put(node, new RandomListNode(node.label));
    node = node.next;
  }
  
  // loop 2. assign next and random pointers
  node = head;
  while (node != null) {
    map.get(node).next = map.get(node.next);
    map.get(node).random = map.get(node.random);
    node = node.next;
  }
  
  return map.get(head);
}
```

# 21. Merge Two Sorted Lists

### 法1: 迭代
```java
public ListNode mergeTwoLists1(ListNode l1, ListNode l2) {
    ListNode p, dummy = new ListNode(0);
    p = dummy;
    while (l1 != null && l2 != null) {
        if (l1.val < l2.val) {
            p.next = l1;
            l1 = l1.next;
        } else {
            p.next = l2;
            l2 = l2.next;
        }
        p = p.next;
    }
    p.next = (l1==null)?l2:l1;
    return dummy.next;
}
```

### 法2: 递归
```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null || l2 == null) {
        return l1==null?l2:l1;
    }
    if (l1.val > l2.val) {
        return mergeTwoLists(l2, l1);
    }
    l1.next = mergeTwoLists(l1.next, l2);
    return l1;
}
```

# 103. Binary Tree Zigzag Level Order Traversal
输入
```
    3
   / \
  9  20
    /  \
   15   7
```
输出
```
[
  [3],
  [20,9],
  [15,7]
]
```
### 法1: 用LinkedList实现Queue来做BFS，方便两头插入
```java
public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
    List<List<Integer>> list = new LinkedList<>();
    if(root == null) return list;
    Queue<TreeNode> q = new LinkedList<>();
    q.add(root);
    while(!q.isEmpty()){
        int size = q.size();
        LinkedList<Integer> curr_level = new LinkedList<>();
        for(int i=0; i < size; i++){
            TreeNode curr = q.remove();
            // 考虑单数层还是奇数层
            if(list.size() % 2 == 0)
                curr_level.add(curr.val);
            else       
                curr_level.addFirst(curr.val);
            
            if(curr.left != null) q.add(curr.left);
            if(curr.right != null) q.add(curr.right);
                
        }
        list.add(curr_level);
    }
    return list;
}
```

# 2. Add Two Numbers
给两个反序的链表，求和，如342 + 465 = 807:
```
(2 -> 4 -> 3) + (5 -> 6 -> 4) = 7 -> 0 -> 8
```

### 法1: 迭代
```java
public ListNode addTwoNumbers(ListNode L1, ListNode L2) {
    ListNode dummy = new ListNode(0), current = dummy;
    int carry = 0;
    
    while(L1 != null || L2 != null || carry > 0) {
        int sum = carry;
        
        if(L1 != null) {sum += L1.val; L1 = L1.next;}
        if(L2 != null) {sum += L2.val; L2 = L2.next;}
        
        current.next = new ListNode(sum % 10);
        current = current.next;
        carry = sum / 10;
    }
    return dummy.next;
}
```

### 法2: 递归
```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    return helper(l1,l2,0);
}
private ListNode helper(ListNode l1, ListNode l2, int carry){
    if(l1 == null && l2 == null){
        if(carry == 1)  return new ListNode(1);
        else    return null;
    }
    
    int sum = carry;
    sum = (l1 == null)?sum:sum+l1.val;
    sum = (l2 == null)?sum:sum+l2.val;
    
    carry = sum / 10;
    sum = sum % 10;
    
    ListNode node = new ListNode(sum);
    if(l1 == null)  
        node.next = helper(l1,l2.next,carry);
    else if(l2 == null)
        node.next = helper(l1.next,l2,carry);
    else
        node.next = helper(l1.next,l2.next,carry);
    return node;
    
}
```

# 763. Partition Labels
给定一个字符串，尽量分成不同的部分，使得每个部分里的字符不在别的部分出现，比如
```
Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8] ("ababcbaca", "defegde", "hijhklij")
```

### 法1: 用一个array记录每个字符最后一次出现的位置
```java
public List<Integer> partitionLabels(String S) {
    List<Integer> res = new ArrayList();
    int[] rightMostPos = new int[26];
    Arrays.fill(rightMostPos, -1);
    
    for (int i = 0; i < S.length(); ++i) {
        rightMostPos[S.charAt(i) - 'a'] = i;
    }
    
    int currRight = -1, count = 0;
    for (int i = 0; i < S.length(); ++i) {
        count++;
        currRight = Math.max(currRight, rightMostPos[S.charAt(i) - 'a']);
         if (i == currRight) {
            res.add(count);
            count = 0;
        }
    }
    
    return res;
}
```

# 127. Word Ladder
给定两个词和一个字典，每次只可以变词的一个单词且变后的词要在词典出现过，问最少几次可以把第一个词变到第二个词，比如
```
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5 ("hit" -> "hot" -> "dot" -> "dog" -> "cog")
```

### 法1: BFS
```java
public int ladderLength(String beginWord, String endWord, List<String> wordList) {
    if(beginWord==null || endWord==null || beginWord.equals(endWord) || !wordList.contains(endWord)) return 0;
    int len = 1;
    Set<String> dict = new HashSet<>(wordList);
    Queue<String> queue = new LinkedList<>();
    Set<String> visited = new HashSet<>();
    queue.offer(beginWord); visited.add(beginWord);
    while(!queue.isEmpty()){
        int levelSize = queue.size();
        for(int s=0;s<levelSize;s++){
            String cur = queue.poll();
            for(int i=0;i<cur.length();i++){
                for(char c='a';c<='z';c++){
                    char[] carr = cur.toCharArray();
                    char c1 = carr[i];
                    carr[i] = c;
                    String temp = new String(carr);
                   
                    if(temp.equals(endWord)){
                         System.out.println("temp "+temp +" endWord  "+endWord);
                         return len+1;
                    } 
                    if(!visited.contains(temp) && dict.contains(temp)){
                        visited.add(temp); queue.offer(temp);
                    }
                    carr[i] = c1;
                }
            }               
        }
        len++;
    }
    
    return 0;
}
```

# 139. Word Break
给定一个字符串和一个字典，回答是否字符串能用字典里的词组成

### 法1：这是一个完全背包问题，可以用DP
```java
public boolean wordBreak(String s, Set<String> wordDict) {
    if (s == null || s.isEmpty()) {
        return false;
    }
    
    int n = s.length();
    boolean[] breakable = new boolean[n + 1];
    breakable[0] = true;
    for (int i = 1; i <= n; i++) {
        for (int j = i - 1; j >= 0; j--) {
            if (breakable[j] && wordDict.contains(s.substring(j, i))) {
                breakable[i] = true;
                break;
            }
        }
    }
    return breakable[n];
}
```
