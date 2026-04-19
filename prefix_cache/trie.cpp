#include <cassert>
#include <iostream>
#include <memory>
#include <string.h>
using namespace std;

struct TrieNode {
  bool is_end;
  int ref_cnt;
  unique_ptr<TrieNode> next[26];

  TrieNode() : is_end(false), ref_cnt(0) {}
};

class PrefixCache {
private:
  unique_ptr<TrieNode> root;
  bool remove(TrieNode *cur, const string &key, int idx) {
    // 递归终止，单次不存在
    if (cur == nullptr)
      return false;
    if (idx == key.size()) {
      if (!cur->is_end)
        return false; // 单词未插入
      cur->is_end = false;
      cur->ref_cnt--;
      return cur->ref_cnt == 0; // 无引用即可删除
    }

    int pos = key[idx] - 'a';
    bool need_del = remove(cur->next[pos].get(), key, idx + 1);
    if (need_del) {
      cur->next[pos].reset(); // NOTE API
    }
    cur->ref_cnt--;
    return cur->ref_cnt == 0;
  }

public:
  PrefixCache() : root(make_unique<TrieNode>()) {};
  ~PrefixCache() = default;

  void add(const string &key) {
    TrieNode *cur = root.get();
    for (char c : key) {
      int pos = c - 'a';
      if (!cur->next[pos]) {
        cur->next[pos] = make_unique<TrieNode>();
      }
    }
  }

  void deleteKey(const string &key) { remove(root.get(), key, 0); }

  bool query(const string &prefix) {
    TrieNode *cur = root.get();
    for (char c : prefix) {
      int pos = c - 'a';
      if (!cur->next[pos])
        return false;
      cur = cur->next[pos].get();
    }
    return true;
  }
};
