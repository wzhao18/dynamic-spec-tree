import seaborn as sns
import matplotlib.pyplot as plt


# plot 1
tree_sizes = [16, 32, 64, 128, 256]

num_tokens_testbed = [2.665326846, 3.042690198, 3.320994, 3.562375107, 3.905272838]

num_tokens_testbed_dynamic = [3.172736, 3.46374, 3.8729136, 3.938185, 3.9424]


# plot scatter plot and line plot using seaborn
plt.figure(figsize=(5, 4))
sns.scatterplot(x=tree_sizes, y=num_tokens_testbed, label='Sequoia')
sns.lineplot(x=tree_sizes, y=num_tokens_testbed)
sns.scatterplot(x=tree_sizes, y=num_tokens_testbed_dynamic, label='Ours')
sns.lineplot(x=tree_sizes, y=num_tokens_testbed_dynamic)

# start the y axis at 3
plt.ylim(2.5, 4.5)

plt.xlabel('Tree Size')
plt.ylabel('Avg Num Tokens per decoding step')
plt.title('Avg Num Tokens per decoding step vs Tree Size')
plt.legend()
plt.savefig('tree_sizes.png')