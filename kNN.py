import heapq
import numpy as np
import matplotlib.pyplot as plt
import sys

class Node(object):
    def __init__(self, axis=None, val=[], left=None, right=None):
        self.axis = axis
        self.point = val
        self.left = left
        self.right = right
        self.dflag = 0

class KDtree(object):
    def __init__(self, point, k):
        self.dim = len(point[0])
        self.k = k
        self.Kbest = []
        self.alpha = 0.4
        self.root = self.fit(point)

    def fit(self, point, depth=0):
        if len(point) <= 0 or len(point[0]) == 0:
            return None
        axis = depth % self.dim
        array = [p[axis] for p in point]
        # print(len(point))
        median = self.find_median(0, len(array)-1, array, len(array)//2)
        median = median[len(array)//2]
        index = array.index(median)
        left = [i for i in point if i[axis] < median]
        right = [i for i in point if i[axis] > median]
        return Node(axis = axis,
                    val = point[index],
                    left = self.fit(point=left, depth=depth+1),
                    right = self.fit(point=right, depth=depth+1))

    def find_median(self, l, r, a, index):
        i,j = l,r
        mid = l + (r-l)//2
        x = a[mid]
        while (i<=j):
            while a[i] < x:
                i+=1
            while x < a[j]:
                j-=1
            if i<=j:
                a[i],a[j] = a[j],a[i]
                i +=1
                j -=1
        if l<j and index <= j:
            a = self.find_median(l, j, a, index)
        elif i<r and i <= index:
            a = self.find_median(i, r, a, index)
        return a

    def calc(self, root):
        if root is None:
            return 0
        return self.calc(root.left) + self.calc(root.right) + 1

    def calcD(self, root):
        if root is None:
            return 0
        return self.calcD(root.left) + self.calcD(root.right) + root.dflag

    def flatten(self, root, flat_a=[]):
        if root is None:
            return flat_a
        flat_a = self.flatten(root.left)
        if root.dflag == 0:
            flat_a.append(root.point)
        flat_a = self.flatten(root.right)
        return flat_a

    def FLAG(self, root):
        if root is None:
            return 0
        calc = self.calc(root)
        return root.dflag and (self.alpha * calc <= max(self.calc(root.left), self.calc(root.right))
                               or self.calcD(root) <= self.alpha * calc)

    def add(self, root, point, depth=0):
        axis = depth % self.dim
        if root is None:
            root = Node(axis=axis, val=point, left=None, right=None)
        else:
            if point[root.axis] < root.point[root.axis]:
                root.left = self.add(root.left, point, depth=depth+1)
            elif root.point[axis] < point[root.axis]:
                root.right = self.add(root.right, point, depth=depth+1)
            if self.FLAG(root):
                root = self.rebuild(root, depth)
        return root

    def delete(self, root, point, depth=0):
        if root is None:
            return None
        else:
            if all(root.point == point):
                root.dflag = 1
            elif point[root.axis] < root.point[root.axis]:
                root.left = self.delete(root.left, point, depth=depth+1)
            elif root.point[root.axis] < point[root.axis]:
                root.right = self.delete(root.right, point, depth=depth+1)
            if self.FLAG(root):
                root = self.rebuild(root, depth)
        return root

    def rebuild(self, root, depth):
        flat_a = self.flatten(root=root)
        return self.fit(flat_a, depth=depth)

    def find_Knearest(self, point, root=None):
        if root is None:
            self.Kbest = []
            root = self.root

        if root.left is not None or root.right is not None:
            if point[root.axis] < root.point[root.axis] and root.left is not None:
                self.find_Knearest(point, root=root.left)
            elif root.right is not None:
                self.find_Knearest(point, root=root.right)

        dis = sum(list(map(lambda x,y : abs(x-y), root.point, point)))
        if len(self.Kbest) < self.k:
            heapq.heappush(self.Kbest,(-dis, root.point))
        if dis < abs(heapq.nsmallest(1, self.Kbest, lambda d:d[0])[0][0]):
            heapq.heappop(self.Kbest)
            heapq.heappush(self.Kbest, (-dis, root.point))

        if abs(root.point[root.axis]-point[root.axis]) < abs(heapq.nsmallest(1, self.Kbest, lambda d:d[0])[0][0]):
            if root.right is not None and point[root.axis] < root.point[root.axis]:
                self.find_Knearest(point, root=root.right)
            elif root.left is not None and point[root.axis] >= root.point[root.axis]:
                self.find_Knearest(point, root=root.left)
        return self.Kbest


def gen_data(x1, x2):
    y = np.sin(x1) * 1 / 2 + np.cos(x2) * 1 / 2 + 0.1 * x1
    return y


def load_data():
    x1_train = np.linspace(0, 50, 1000)
    x2_train = np.linspace(-10, 10, 1000)
    data_train = [[x1, x2, gen_data(x1, x2) + np.random.random(1)[0] - 0.5] for x1, x2 in zip(x1_train, x2_train)]
    x1_test = np.linspace(0, 50, 100) + np.random.random(100) * 0.5
    x2_test = np.linspace(-10, 10, 100) + 0.02 * np.random.random(100)
    data_test = [[x1, x2, gen_data(x1, x2)] for x1, x2 in zip(x1_test, x2_test)]
    return np.array(data_train), np.array(data_test)

def main():
    train, test = load_data()
    x_train, y_train = train[:, :], train[:, 2]
    x_test, y_test = test[:, :], test[:, 2]  # 同上，但这里的y没有噪声
    t = KDtree([x_train[0]], k=13)
    for point in x_train[1:]:
        t.root=t.add(t.root, point, 0)
        # t.root=t.delete(t.root, point, 0)
    t.root=t.rebuild(t.root, 0)

    result = [t.find_Knearest(i)[-1][-1] for i in x_test]
    result = [i[-1] for i in result]
    result = [np.average(i) for i in result]
    #
    print(len(result))
    plt.plot(result)
    plt.plot(y_test)
    plt.show()


    return 0
if __name__ == "__main__":
    main()
    # t = KDtree([[]],1)
    # a = [3,2,1,5,4,3,2] # 1,2,2,3,3,4,5
    # a = [4,5,6,3,6,8,4] # 3,4,4,5,6,6,8
    # a = [0,2,1,1,5,6,7] # 0,1,1,2,5,6,7
    # s = t.find_median(0, len(a)-1, a, len(a)//2)
    # s = s[len(a)//2]
    # print(s)