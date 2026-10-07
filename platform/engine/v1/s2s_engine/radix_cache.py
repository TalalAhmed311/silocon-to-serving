"""radix_cache.py — RadixAttention-style prefix cache over token ids (P6.4; SGLang `mem_cache/radix_cache.py`).

A compressed trie: each edge holds a run of tokens and, in parallel, the KV-cache slot of each token (`values`).
  insert(tokens, values)        add a sequence's KV; shares any existing prefix (returns how much already existed)
  match_prefix(tokens)          longest cached prefix → (length, slots, last node); splits an edge on partial match
  lock(node) / unlock(node)     reference counts along the root path: locked tokens can't be evicted
  evict(n)                      free ≥ n tokens from least-recently-used, UNLOCKED leaves (returns their slots)
Token-level granularity (any prefix length can hit) vs P6.3's block hashing (only full blocks hit) is the trade-off
exercise 3 measures.
"""
from __future__ import annotations

import heapq
import itertools

_clock = itertools.count()


class Node:
    __slots__ = ("children", "parent", "key", "value", "ref", "last_access")

    def __init__(self, parent: "Node | None" = None, key: tuple = (), value: tuple = ()):
        self.children: dict[int, Node] = {}      # first token of the child's edge → child
        self.parent = parent
        self.key = key                           # tokens on the edge from parent to this node
        self.value = value                       # KV slots of those tokens (same length)
        self.ref = 0
        self.last_access = next(_clock)


def _common(a: tuple, b, start: int = 0) -> int:
    n = 0
    while n < len(a) and start + n < len(b) and a[n] == b[start + n]:
        n += 1
    return n


class RadixCache:
    def __init__(self):
        self.root = Node()
        self.root.ref = 1                        # never evicted
        self.size = 0                            # cached tokens
        self.locked = 0                          # tokens with ref > 0 (excluding the root)

    # ---- lookup ------------------------------------------------------------------------------------------------
    def match_prefix(self, tokens) -> tuple[int, list, Node]:
        node, i, slots = self.root, 0, []
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            n = _common(child.key, tokens, i)
            child.last_access = next(_clock)
            if n < len(child.key):               # partial edge match: split so the matched part is its own node
                child = self._split(child, n)
            slots.extend(child.value)
            i += n
            node = child
        return i, slots, node

    def _split(self, child: Node, n: int) -> Node:
        """Split child's edge after n tokens; returns the new upper node (holding key[:n])."""
        upper = Node(child.parent, child.key[:n], child.value[:n])
        upper.ref = child.ref
        upper.last_access = child.last_access
        upper.parent.children[child.key[0]] = upper
        child.parent, child.key, child.value = upper, child.key[n:], child.value[n:]
        upper.children[child.key[0]] = child
        return upper

    # ---- insert -----------------------------------------------------------------------------------------------
    def insert(self, tokens, values) -> int:
        """Insert tokens with their KV slots. Returns the length of the prefix that was ALREADY cached — the caller
        owns (and should free) its own duplicate slots for that prefix."""
        assert len(tokens) == len(values)
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                leaf = Node(node, tuple(tokens[i:]), tuple(values[i:]))
                node.children[tokens[i]] = leaf
                self.size += len(leaf.key)
                return i
            n = _common(child.key, tokens, i)
            child.last_access = next(_clock)
            if n < len(child.key):
                child = self._split(child, n)
            i += n
            node = child
        return i

    # ---- reference counting ----------------------------------------------------------------------------------
    def lock(self, node: Node) -> None:
        while node is not self.root:
            if node.ref == 0:
                self.locked += len(node.key)
            node.ref += 1
            node = node.parent

    def unlock(self, node: Node) -> None:
        while node is not self.root:
            node.ref -= 1
            assert node.ref >= 0
            if node.ref == 0:
                self.locked -= len(node.key)
            node = node.parent

    # ---- eviction ---------------------------------------------------------------------------------------------
    def _leaves(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n is not self.root:
                yield n
            stack.extend(n.children.values())

    def evict(self, num_tokens: int) -> list:
        """Remove LRU unlocked leaves until ≥ num_tokens are freed (or nothing evictable is left). Returns freed slots.
        A parent whose last child is evicted becomes a leaf and joins the heap."""
        heap = [(n.last_access, id(n), n) for n in self._leaves() if n.ref == 0]
        heapq.heapify(heap)
        freed: list = []
        while heap and len(freed) < num_tokens:
            _, _, leaf = heapq.heappop(heap)
            if leaf.ref > 0 or leaf.children:
                continue
            freed.extend(leaf.value)
            parent = leaf.parent
            del parent.children[leaf.key[0]]
            self.size -= len(leaf.key)
            if parent is not self.root and not parent.children and parent.ref == 0:
                heapq.heappush(heap, (parent.last_access, id(parent), parent))
        return freed

    def evictable(self) -> int:
        return self.size - self.locked


class NaiveTrie:
    """Reference for property tests: a set of inserted sequences; match = longest common prefix with any of them."""

    def __init__(self):
        self.seqs: list[tuple] = []

    def insert(self, tokens) -> None:
        self.seqs.append(tuple(tokens))

    def match_len(self, tokens) -> int:
        best = 0
        for s in self.seqs:
            best = max(best, _common(s, tokens))
        return best
