import numpy as np

reeig = np.array([76.0, 69.27, 60.33, 82.0, 63.30, 80.82, 72.62, 85.34])
coshP = np.array([76.69, 68.58, 62.64, 78.25, 66.14, 78.93, 70.85, 84.79])
expT = np.array([76.97, 68.75, 64.44, 82.49, 69.99, 81.39, 74.11, 86.07])

reeig_2 = np.array([63.91, 68.67])
coshP_2 = np.array([63.70, 66.84])
expT_2 = np.array([67.43, 72.64])

reeig_3 = np.array([1])
coshP_3 = np.array([1])
expT_3 = np.array([1])

print(50*'=')
print('ModelSPDnet with 1 block SPDNet')
print(f"Mean of reeig : {reeig.mean()} \n Mean of coshP : {coshP.mean()} \n Mean of expT : {expT.mean()}")


print(50*'=')
print('ModelSPDnet with 2 blocks SPDNet')
print(f"Mean of reeig : {reeig_2.mean()} \n Mean of coshP : {coshP_2.mean()} \n Mean of expT : {expT_2.mean()}")


print(50*'=')
print('ModelSPDnet with 3 blocks SPDNet')
print(f"Mean of reeig : {reeig_3.mean()} \n Mean of coshP : {coshP_3.mean()} \n Mean of expT : {expT_3.mean()}")