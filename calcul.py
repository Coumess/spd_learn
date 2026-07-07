import numpy as np

reeig = np.array([76.0, 69.27, 60.33, 82.0, 63.30, 80.82, 72.62, 85.34])
coshP = np.array([76.69, 68.58, 62.64, 78.25, 66.14, 78.93, 70.85, 84.79])
expT = np.array([76.97, 68.75, 64.44, 82.49, 69.99, 81.39, 74.11, 86.07])

print('ModelSPDnet with 1 block SPDNet')
print(f"Mean of reeig : {reeig.mean()} \n Mean of coshP : {coshP.mean()} \n Mean of expT : {expT.mean()}")

print(50*'=')
print('ModelSPDnet with 2 blocks SPDNet')

print(50*'=')
print('ModelSPDnet with 3 blocks SPDNet')