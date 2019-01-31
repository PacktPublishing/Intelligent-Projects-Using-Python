import numpy as np

class sum_tree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity,dtype=object )

    def _propagate_(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate_(parent, change)

    def _retrieve_(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve_(left, s)
        else:
            return self._retrieve_(right, s-self.tree[left])

    def _total_(self):
        return self.tree[0]

    def _update_(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate_(idx, change)


    def _add_(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self._update_(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    
    def _get_(self, s):
        idx = self._retrieve_(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
