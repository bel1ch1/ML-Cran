import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('CSV/right_data.csv', header=None)

# Extract the sub-arrays
sub_arrays = df[0].apply(eval)

new_df = pd.DataFrame([array for sub_array in sub_arrays for array in sub_array])

# Assign the new columns to the original DataFrame
df[['X', 'Y', 'spd']] = new_df


############# Итерации положений X, Y
# # Create plots for each time series
# fig, ax = plt.subplots()

# for i in range(20):
#     ax.plot(df['X'].iloc[i*3:(i+1)*3], df['Y'].iloc[i*3:(i+1)*3], label=f'Time Series {i+1}')

# ax.set_title('Time Series Plots')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.legend()

################### Зависимости по скорости
# # Create plots for X vs spd and Y vs spd
# fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# # Plot X vs spd
# ax[0].scatter(df['X'], df['spd'])
# ax[0].set_title('X vs Speed')
# ax[0].set_xlabel('X')
# ax[0].set_ylabel('Speed')


# # Plot Y vs spd
# ax[1].scatter(df['Y'], df['spd'])
# ax[1].set_title('Y vs Speed')
# ax[1].set_xlabel('Y')
# ax[1].set_ylabel('Speed')

# plt.tight_layout()

############### Итерации положения X
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(20):
    ax.plot(range(3), df['X'].iloc[i*3:(i+1)*3], label=f'Iteration {i+1}')

ax.set_title('Change in X Coordinates over Each Iteration')
ax.set_xlabel('Iteration')
ax.set_ylabel('X Coordinate')
ax.legend()

plt.show()
