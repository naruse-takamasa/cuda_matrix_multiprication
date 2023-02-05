import pandas as pd
import matplotlib.pyplot as plt

input_file = pd.read_csv("./result.csv", encoding="ms932", sep=",")

x = input_file[["size"]]

# naive = input_file[["naive"]]
cuda = input_file[["cuda"]]
cuda_shared = input_file[["cuda-shared"]]

# def naive_cuda():
#     plt.figure(dpi=500)

#     plt.plot(x, naive, label="naive")
#     plt.plot(x, cuda, label="cuda")
#     # plt.plot(x, cuda_shared, label="cuda(shared)")

#     plt.xlabel("matrix size")
#     plt.ylabel("elapsed time[micro sec]")
#     plt.legend()
#     plt.grid()

#     plt.savefig("./elapsed_time_naive_cuda.png")
#     plt.show()

plt.figure(dpi=500)

# plt.plot(x, naive, label="naive")
plt.plot(x, cuda, label="cuda")
plt.plot(x, cuda_shared, label="cuda(shared)")

plt.xlabel("matrix size")
plt.ylabel("elapsed time[micro sec]")
plt.legend()
plt.grid()

plt.savefig("./elapsed_time_cuda_shared.png")
plt.show()

# naive_cuda

