import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# tree size 64
# temp 0.6
# top_p 0.9 

# gpu_specinfer = 1.799497172
gpu_testbed = 2.05159
gpu_testbed_dynamic = 2.1020558
cpu_testbed = 3.936060401    # TODO: replace these values.
cpu_testbed_dynamic = 4.472520107     # TODO: replace these values.

# gpu_testbed_dynamic /= gpu_testbed
# cpu_testbed_dynamic /= cpu_testbed
# gpu_testbed /= gpu_testbed
# cpu_testbed /= cpu_testbed

data = {
    'Type': ['In GPU', 'In GPU', 'Offloading', 'Offloading'],
    'Legend': ['Sequoia', 'Ours', 'Sequoia', 'Ours'],
    'Speedup': [gpu_testbed, gpu_testbed_dynamic, cpu_testbed, cpu_testbed_dynamic]
}

df = pd.DataFrame(data)
ax = plt.figure(figsize=(5, 4))
sns.barplot(x='Type', y='Speedup', hue='Legend', data=df)

# plt.ylim(2, 5)
plt.gca().legend().set_title('')

plt.xlabel('Type')
plt.ylabel('Latency Speedup')
plt.title('Speedup Comparison')

plt.savefig('barplot.png')