import seaborn as sns
import matplotlib.pyplot as plt



### TODO: Need to update these values with Zhao Wei's results. 

# plot 2
draft_temps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

num_tokens_testbed = []

num_tokens_testbed_dynamic = []


# plot scatter plot and line plot using seaborn
plt.figure(figsize=(5, 4))
sns.scatterplot(x=draft_temps, y=num_tokens_testbed, label='Sequoia')
sns.lineplot(x=draft_temps, y=num_tokens_testbed)
sns.scatterplot(x=draft_temps, y=num_tokens_testbed_dynamic, label='Ours')
sns.lineplot(x=draft_temps, y=num_tokens_testbed_dynamic)

plt.ylim(2, 4.5)

plt.xlabel('Draft Model Temp.')
plt.ylabel('Avg Num Tokens per decoding step')
plt.title('Avg Num Tokens per decoding step vs draft model temp.')
plt.legend()
plt.savefig('draft_temp.png')