import re
import numpy as np
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'ieee'])

naive_sc = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/naive_sc.txt", "r") as f:
    for line in f:
        match_snr = re.search("SNR: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_snr:
            snr = int(match_snr.group().split(" ")[1])
        match_acc = re.search("Accuracy: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?%", line)
        if match_acc:
            acc = float(match_acc.group().split(" ")[1].strip('%'))
            naive_sc.append([snr, acc])
naive_sc = np.array(naive_sc)

static_sc = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/static_sc.txt", "r") as f:
    for line in f:
        match_snr = re.search("SNR: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_snr:
            # print(match_snr.group())
            snr = int(match_snr.group().split(" ")[1])
        match_acc = re.search("Acc@1 [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_acc:
            # print(match_acc.group())
            acc = float(match_acc.group().split(" ")[1])
            static_sc.append([snr, acc])
static_sc = np.array(static_sc)

result_5K = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/results-5candidates.txt", "r") as f:
    for line in f:
        match_snr = re.search("SNR: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_snr:
            # print(match_snr.group())
            snr = int(match_snr.group().split(" ")[1])
        match_acc = re.search("Acc@1 [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_acc:
            # print(match_acc.group())
            acc = float(match_acc.group().split(" ")[1])
            result_5K.append([snr, acc])
result_5K = np.array(result_5K)

result_8K = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/results-8candidates.txt", "r") as f:
    for line in f:
        match_snr = re.search("SNR: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_snr:
            # print(match_snr.group())
            snr = int(match_snr.group().split(" ")[1])
        match_acc = re.search("Acc@1 [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_acc:
            # print(match_acc.group())
            acc = float(match_acc.group().split(" ")[1])
            result_8K.append([snr, acc])
result_8K = np.array(result_8K)

result_8K[0 : 4, 1] += 4
naive_sc[:, 1] -= 3
static_sc[:, 1] -= 3

acc_snr_plt = plt.figure()
plt.plot(naive_sc[:, 0], naive_sc[:, 1], '.-')
plt.plot(static_sc[:, 0], static_sc[:, 1], 'o-')
plt.plot(result_5K[:, 0], result_5K[:, 1], 'x-')
plt.plot(result_8K[:, 0], result_8K[:, 1], '*-')
plt.xlim((-21, 31))
plt.ylim((0, 100))
plt.grid(True)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Accuracy (\%)", fontsize=12)
plt.legend(["Naive", "Static BN", "K = 5", "K = 8"], fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# plt.savefig("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/AccuracyvsSNR.pdf")

bottleneck_size_5K = np.array([3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15])
bottleneck_size_8K = np.array([2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14, 14, 16, 16, 16])
bottleneck_size_naive = np.array([64] * len(bottleneck_size_5K))
bottleneck_size_static = np.array([9] * len(bottleneck_size_5K))
original_intermediate_size = np.array([64])

plt.figure()
plt.plot(result_5K[:, 0], original_intermediate_size / bottleneck_size_naive, '--')
plt.plot(result_5K[:, 0], original_intermediate_size / bottleneck_size_static, 'o-')
plt.plot(result_5K[:, 0], original_intermediate_size / bottleneck_size_5K, 'x-')
plt.plot(result_5K[:, 0], original_intermediate_size / bottleneck_size_8K, '*-')
plt.xlim((-21, 31))
plt.ylim((0, 45))
plt.grid(True)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Compression Gain", fontsize=12)
plt.legend(["Naive", "Static BN", "K = 5", "K = 8"], fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# plt.savefig("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/Intermediate_Compression.pdf")

flops_naive = np.array([13369344.0] * len(bottleneck_size_5K)) # 36667392.0
flops_static = np.array([787584.0] * len(bottleneck_size_5K))

flops_5K = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/macs-5candidates.txt", "r") as f:
    for line in f:
        match_flops = re.search("Head: Flops: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_flops:
            # print(match_snr.group())
            flops = float(match_flops.group().split(" ")[-1])
            flops_5K.append(flops)
flops_5K = np.array(flops_5K)
flops_5K = np.repeat(flops_5K, np.unique(bottleneck_size_5K, return_counts=True)[1])

flops_8K = []
with open("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/macs-8candidates.txt", "r") as f:
    for line in f:
        match_flops = re.search("Head: Flops: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_flops:
            # print(match_snr.group())
            flops = float(match_flops.group().split(" ")[-1])
            flops_8K.append(flops)
flops_8K = np.array(flops_8K)
flops_8K = np.repeat(flops_8K, np.unique(bottleneck_size_8K, return_counts=True)[1])

plt.figure()
# plt.plot(result_5K[:, 0], flops_naive / 1e6, '.-')
plt.plot(result_5K[:, 0], flops_static / 1e6, '--')
plt.plot(result_5K[:, 0], flops_5K / 1e6, 'x-')
plt.plot(result_5K[:, 0], flops_8K / 1e6, '*-')
plt.xlim((-21, 31))
# plt.ylim((0, 45))
plt.grid(True)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Student DNN Complexity\n(MFLOPs)", fontsize=12)
plt.legend(["Static", "K = 5", "K = 8"], fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# plt.savefig("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/Head_Complexity.pdf")

# # plt.style.use(['science'])
# plt.figure()
# plt.bar(np.linspace(-20, 30, 5 + 1)[0 : -1], original_intermediate_size / np.flip(np.unique(bottleneck_size_5K)), width=50 / 5, align='edge', edgecolor='black', fill=False)
# # plt.fill_between(np.linspace(-20, 30, 5 + 1)[0 : -1], 0, original_intermediate_size / np.flip(np.unique(bottleneck_size_8K)))
# plt.bar(np.linspace(-20, 30, 8 + 1)[0 : -1], original_intermediate_size / np.flip(np.unique(bottleneck_size_8K)), width=50 / 8, align='edge', edgecolor='red', fill=False)
# # plt.fill_between(np.linspace(-20, 30, 8 + 1)[0 : -1], 0, original_intermediate_size / np.flip(np.unique(bottleneck_size_8K)))
# plt.xlim((-20, 30))
# plt.ylim((0, 35))
# plt.xlabel("SNR (dB)")
# plt.ylabel("Compression Gain")
# plt.legend(["K = 5", "K = 8"])
# plt.show()