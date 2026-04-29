import re, collections

f"""
比如子串”abcde“ 出现频率最多，8次。
那么 ab, bc, cd, de 都会增加8次的频率。
初始词表 {'a', 'b', 'c', 'd', 'e'}
* 加入 'ab', 词表 {'a', 'b', 'c', 'd', 'e', 'ab'}
# 分裂优先匹配最长前缀，如'abcd' 会优先'ab' + 'c' + 'd',而不是 'a' + 'bcd'。尽管后者分裂次数更少。
# 'bc'将不会加入词表，因为下一步优先匹配最长前缀'ab'，计算'abc'的频率，也是8次。
* 加入 'abc', 词表 {'a', 'b', 'c', 'd', 'e', 'ab', 'abc'}
* 加入 'abcd', 词表 {'a', 'b', 'c', 'd', 'e', 'ab', 'abc', 'abcd'}  
* 加入 'abcde', 词表 {'a', 'b', 'c', 'd', 'e', 'ab', 'abc', 'abcd', 'abcde'}
"""

def get_stats(vocab):
    """
    统计词元对频率
    """
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # python3.7后 字典有序。元组key在这里省略了括号。
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    """
    合并词元对
    pair,形如：{"a", "b"}
    v_in,形如：{"a b": 1, "c d": 2}
    """
    v_out = {}
    # re.escape() 就是给字符串里的「正则特殊符号」自动加反斜杠 \，让它们变成普通文本，不被正则误解。
    bigram = re.escape(" ".join(pair)) # 形如"a b"
    # ?<! 不能是 
    # \S 非空白（字母数字符号）
    # ?! 后面不能是
    # (?<!\S)  确保 【左边是空格/开头】
    # (?!\S)   确保 【右边是空格/结尾】
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        # 替换。
        # 如 ‘a b c’ -> ‘ab c’, 'x a b y' -> 'x ab y'
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

# 准备语料库，每个词末尾加上</w> 表示结束，并切分字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4 # 设置合并次数

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # key=pairs.get 表示根据 pairs 中的值（频率）来排序，而不是键名
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(f"第{i+1}次合并: {best} -> {''.join(best)}")
    print(f"新词表（部分）: {list(vocab.keys())}")
    print("-" * 60)