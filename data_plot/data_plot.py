import pandas as pd
import matplotlib.pyplot as plt

input_file = pd.read_csv("./result.csv", encoding="ms932", sep=",")

x = input_file[["size"]]

naive = input_file[["naive"]]

cuda = input_file[["cuda"]]

plt.figure(dpi=500)

# plt.plot(x, naive, label="naive")
plt.plot(x, cuda, label="cuda")

plt.xlabel("matrix size")
plt.ylabel("elapsed time[micro sec]")
plt.legend()
plt.grid()

plt.savefig("elapsed_time_cuda.png")
plt.show()
