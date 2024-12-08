import seaborn as sns
import matplotlib.pyplot as plt



### TODO: Need to update these values with Zhao Wei's results. 

# plot 2
draft_temps = [0.1, 0.3, 0.5, 0.7, 0.9]

num_tokens_testbed_dynamic = [4.508, 4.328, 4.179, 4.085, 4.154]

num_tokens_testbed = [3.157, 3.537, 3.305, 3.240, 3.021]


# plot scatter plot and line plot using seaborn
plt.figure(figsize=(5, 4))
sns.scatterplot(x=draft_temps, y=num_tokens_testbed, label='Sequoia')
sns.lineplot(x=draft_temps, y=num_tokens_testbed)
sns.scatterplot(x=draft_temps, y=num_tokens_testbed_dynamic, label='Ours')
sns.lineplot(x=draft_temps, y=num_tokens_testbed_dynamic)

# plt.ylim(3, 5)
plt.xticks(draft_temps)

plt.xlabel('Target Model Temp.')
plt.ylabel('Avg Num Tokens per decoding step')
plt.title('Avg Num Tokens per decoding step vs target model temp.')
plt.legend()
plt.savefig('target_temp.png')