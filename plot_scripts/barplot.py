import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# tree size 64
# temp 0.6
# top_p 0.9 

 
gpu_testbed = 2.05159
gpu_testbed_dynamic = 2.1020558
cpu_testbed = 2    # TODO: replace these values.
cpu_testbed_dynamic = 3     # TODO: replace these values.

data = {
    'Type': ['GPU', 'GPU', 'CPU', 'CPU'],
    'Legend': ['Sequoia', 'Ours', 'Sequoia', 'Ours'],
    'Speedup': [gpu_testbed, gpu_testbed_dynamic, cpu_testbed, cpu_testbed_dynamic]
}

df = pd.DataFrame(data)
plt.figure(figsize=(5, 4))
sns.barplot(x='Type', y='Speedup', hue='Legend', data=df)

plt.xlabel('Type')
plt.ylabel('Latency Speedup')
plt.title('Speedup Comparison')

plt.savefig('barplot.png')