import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

""" These values are updated with latest values. """

# plot 2
p = [0.6, 0.7, 0.8, 0.9, 1.0]

num_tokens_testbed = [3.324324324, 3.253203485, 3.242292994, 3.235626283, 3.18834309]

num_tokens_testbed_dynamic = [4.412883845, 4.437121722, 4.24958841, 4.070137157, 3.745985401]


# plot scatter plot and line plot using seaborn
plt.figure(figsize=(5, 4))
sns.scatterplot(x=p, y=num_tokens_testbed, label='Sequoia')
sns.lineplot(x=p, y=num_tokens_testbed)
sns.scatterplot(x=p, y=num_tokens_testbed_dynamic, label='Ours')
sns.lineplot(x=p, y=num_tokens_testbed_dynamic)

# plt.ylim(3, 4.5)
plt.xticks(np.arange(0.6, 1.05, step=0.1)) 
plt.yticks(np.arange(3, 4.6, step=0.25)) 

plt.xlabel('top_p')
plt.ylabel('Avg Num Tokens per decoding step')
plt.title('Avg Num Tokens per decoding step vs top_p')
plt.legend()
plt.savefig('top_p.png')