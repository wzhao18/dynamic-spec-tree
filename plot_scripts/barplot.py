import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# tree size 64
# temp 0.6
# top_p 0.9 

# TODO: replace these values. 
gpu_testbed = 2
gpu_testbed_dynamic = 2
cpu_testbed = 2
cpu_testbed_dynamic = 3

# Data for plotting
data = {
    'Type': ['GPU', 'GPU', 'CPU', 'CPU'],
    'Legend': ['Sequoia', 'Ours', 'Sequoia', 'Ours'],
    'Speedup': [gpu_testbed, gpu_testbed_dynamic, cpu_testbed, cpu_testbed_dynamic]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plot a bar plot with four bars using seaborn
plt.figure(figsize=(5, 4))
sns.barplot(x='Type', y='Speedup', hue='Legend', data=df)

# Set labels and title
plt.xlabel('Type')
plt.ylabel('Speedup')
plt.title('Speedup Comparison')

# Show plot
plt.savefig('barplot.png')