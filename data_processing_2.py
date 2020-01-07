import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

window_length = 1
data = pd.read_csv("./data/final_output.csv", sep='\t', encoding='utf-8')

final_dataset = np.zeros((data.shape[0] - window_length ,window_length*data.shape[1] + 1))
final_dataset = np.zeros((data.shape[0] - window_length ,window_length*(data.shape[1] - 0) + 1))

all_closing = []
counter = 0
# final_dataset = np.zeros((data.shape[0] - window_length , 6))
for index, row in data.iterrows():
    if index <  data.shape[0] - window_length - 1:

        datapoint = data[index: index + window_length].as_matrix().reshape(-1,1)
        # datapoint = data.iloc[index: index + window_length, :3].as_matrix().reshape(-1,1)
        # datapoint = np.array([data.loc[index]]).reshape(-1,1)

        label = np.array([data.loc[index + window_length].iloc[0]]).reshape(-1,1)
        # label = np.array([data.loc[index + 1].iloc[0]]).reshape(-1,1)

        datapoint_label = np.concatenate((datapoint, label))
        final_dataset[index] = datapoint_label.squeeze()

        all_closing.append(row[0])
        counter += 1

x = range(counter)
y1 = all_closing
y2 = np.linspace(all_closing[0], all_closing[-1], counter)
plt.plot(x, y1, "-b", label="data")
plt.plot(x, y2, "--r", label="linear")
plt.title("Closing Prices in Dataset")
plt.legend(loc="upper left")
plt.show()
df = pd.DataFrame(final_dataset)

df.to_csv("./data/final_dataset.csv", sep='\t', encoding='utf-8', index=False, header = False)
print("DONE!")