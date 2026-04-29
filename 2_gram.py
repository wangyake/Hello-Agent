import collections  

corpus = 'datawhale agent learns datawhale agent works'
tokens = corpus.split()
total_tokens = len(tokens)

# P(datawhale)
count_datawhale = tokens.count('datawhale')
p_datawhale = count_datawhale / total_tokens
print(p_datawhale)

# p(agent|datawhale)
# 计算 bigrams
bigrams = zip(tokens,tokens[1:])
bigrams_count = collections.Counter(bigrams)
count_datawhale_agent = bigrams_count[('datawhale','agent')]

p_agent_given_datawhale = count_datawhale_agent / count_datawhale
print(f'P(agent|datawhale) = {count_datawhale_agent}/{count_datawhale} = {p_agent_given_datawhale:.4f}')


# p(learns|agent)
count_agent_learns = bigrams_count[('agent','learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent
print(f'P(learns|agent) = {count_agent_learns}/{count_agent} = {p_learns_given_agent:.4f}')

# 概率连乘
p_sentence= p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(f'P(datawhale agent learns datawhale agent works) = {p_sentence:.4f}')