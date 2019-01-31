from sum_tree import sum_tree
import random 

class Memory:
   
    # stored as tuple (state,action,reward,next_state) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = sum_tree(capacity)

    def get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self.get_priority(error)
        self.tree._add_(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree._total_() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree._get_(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self.get_priority(error)
        self.tree._update_(idx, p)
