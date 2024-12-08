import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# plot 1
tree_sizes = [4, 8, 16, 32, 64, 128, 256]

num_tokens_baseline = [2.04230423, 2.243820324, 2.284203868, np.NaN,np.NaN,np.NaN,np.NaN]
num_tokens_specinfer = [np.NaN, np.NaN, np.NaN, 2.556339468, 2.663418767, np.NaN,np.NaN]
num_tokens_testbed = [np.NaN, np.NaN, 2.665326846, 3.042690198, 3.320994, 3.562375107, 3.905272838]

num_tokens_testbed_dynamic = [np.NaN, np.NaN, 3.345717234, 3.702733485, 4.0701371571, 4.3726666666, 4.80154639175]


# plot scatter plot and line plot using seaborn
plt.figure(figsize=(5, 4))
sns.scatterplot(x=tree_sizes, y=num_tokens_baseline, label='Original Spec', color = sns.color_palette()[2])
sns.lineplot(x=tree_sizes, y=num_tokens_baseline, color = sns.color_palette()[2])
sns.scatterplot(x=tree_sizes, y=num_tokens_specinfer, label='SpecInfer', color = sns.color_palette()[3])
sns.lineplot(x=tree_sizes, y=num_tokens_specinfer, color = sns.color_palette()[3])
sns.scatterplot(x=tree_sizes, y=num_tokens_testbed, label='Sequoia', color = sns.color_palette()[0])
sns.lineplot(x=tree_sizes, y=num_tokens_testbed, color = sns.color_palette()[0])
sns.scatterplot(x=tree_sizes, y=num_tokens_testbed_dynamic, label='Ours', color = sns.color_palette()[1])
sns.lineplot(x=tree_sizes, y=num_tokens_testbed_dynamic, color = sns.color_palette()[1])

# start the y axis at 3
# plt.ylim(2.5, 5.0)
plt.xticks(tree_sizes[1:])

plt.xlabel('Tree Size')
plt.ylabel('Avg Num Tokens per decoding step')
plt.title('Avg Num Tokens per decoding step vs Tree Size')
plt.legend()
plt.savefig('tree_sizes.png')