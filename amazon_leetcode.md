# 973. K Closest Points to Origin

## 法1: O(nlogn) 排序

```java
public int[][] kClosest(int[][] points, int K) {
    Arrays.sort(points, Comparator.comparing(p -> p[0] * p[0] + p[1] * p[1]));
    return Arrays.copyOfRange(points, 0, K);
}
```

## 法2: O(nlogk) priority queue

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
## 法3: O(n) quick selection

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

## 法1: 采用LinkedHashMap

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

## 法2: 自己写一个，HashMap with Entry<K, V> being linked.

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

## 法1: Hashmap
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

## 法1: DFS
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

