import csv
import ast
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('right_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    coordinates = []
    for row in reader:
        coordinates.extend([ast.literal_eval(coord) for coord in row])

fig = plt.figure(figsize=(12, 6))

# 3D scatter plot
ax1 = fig.add_subplot(131, projection='3d')
for coord in coordinates:
    ax1.scatter(coord[0], coord[1], coord[2])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# X vs Speed plot
ax2 = fig.add_subplot(132)
xs = [coord[0] for coord in coordinates]
speeds = [coord[2] for coord in coordinates]
ax2.scatter(xs, speeds)
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

# Y vs Speed plot
ax3 = fig.add_subplot(133)
ys = [coord[1] for coord in coordinates]
ax3.scatter(ys, speeds)
ax3.set_xlabel('Y')
ax3.set_ylabel('Z')

plt.show()
