# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
file_trans_49_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_trans/49/azotrans_00049_SF.dat")
file_trans_53_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_trans/53/azotrans_00053_SF.dat")
file_trans_57_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_trans/57/azotrans_00057_SF.dat")
file_trans_61_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_trans/61/azotrans_00061_SF.dat")
file_cis_77_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis/77/azocis_00077_SF.dat")
file_cis_81_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis/81/azocis_00081_SF.dat")
file_cis_85_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis/85/azocis_00085_SF.dat")
file_cis_89_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis/89/azocis_00089_SF.dat")
file_cis_93_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis/93/azocis_00093_SF.dat")
file_cis_2_105_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis_2/105/azocis02_00105_SF.dat")
file_cis_2_109_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis_2/109/azocis02_00109_SF.dat")
file_cis_3_114_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis_3/114/azocis03_00114_SF.dat")
file_cis_3_118_SF = Path("/Users/jack/Documents/lipid/GIXOS/azo_cis_3/118/azocis03_00118_SF.dat")


# %%
data_trans_49_SF = np.loadtxt(file_trans_49_SF, skiprows=29)
data_trans_53_SF = np.loadtxt(file_trans_53_SF, skiprows=29)
data_trans_57_SF = np.loadtxt(file_trans_57_SF, skiprows=29)
data_trans_61_SF = np.loadtxt(file_trans_61_SF, skiprows=29)
data_cis_77_SF = np.loadtxt(file_cis_77_SF, skiprows=29)
data_cis_81_SF = np.loadtxt(file_cis_81_SF, skiprows=29)
data_cis_85_SF = np.loadtxt(file_cis_85_SF, skiprows=29)
data_cis_89_SF = np.loadtxt(file_cis_89_SF, skiprows=29)
data_cis_93_SF = np.loadtxt(file_cis_93_SF, skiprows=29)
data_cis_2_105_SF = np.loadtxt(file_cis_2_105_SF, skiprows=29)
data_cis_2_109_SF = np.loadtxt(file_cis_2_109_SF, skiprows=29)
data_cis_3_114_SF = np.loadtxt(file_cis_3_114_SF, skiprows=29)
data_cis_3_118_SF = np.loadtxt(file_cis_3_118_SF, skiprows=29)


# %%
# plt.plot(data_trans_49_SF[:, 0], data_trans_49_SF[:, 1], color="blue", label="trans_49_0.5mN/m")
plt.plot(data_trans_53_SF[:, 0], data_trans_53_SF[:, 1], color="blue", label="trans_53_10mN/m")
# plt.plot(data_trans_57_SF[:, 0], data_trans_57_SF[:, 1], color="blue", label="trans_57_20mN/m")
# plt.plot(data_trans_61_SF[:, 0], data_trans_61_SF[:, 1], color="blue", label="trans_61_30mN/m")
# plt.plot(data_cis_77_SF[:, 0], data_cis_77_SF[:, 1], color="red", label="cis_77_5mN/m")
plt.plot(data_cis_81_SF[:, 0], data_cis_81_SF[:, 1], color="red", label="cis_81_10mN/m")
# plt.plot(data_cis_85_SF[:, 0], data_cis_85_SF[:, 1], color="red", label="cis_85_20mN/m")
# plt.plot(data_cis_89_SF[:, 0], data_cis_89_SF[:, 1], color="red", label="cis_89_30mN/m")
# plt.plot(data_cis_93_SF[:, 0], data_cis_93_SF[:, 1], color="red", label="cis_93_30mN/m")
# plt.plot(data_cis_2_105_SF[:, 0], data_cis_2_105_SF[:, 1], color="green", label="cis_2_105_3.3mN/m")
# plt.plot(data_cis_2_109_SF[:, 0], data_cis_2_109_SF[:, 1], color="green", label="cis_2_109_30mN/m")
# plt.plot(data_cis_3_114_SF[:, 0], data_cis_3_114_SF[:, 1], color="orange", label="cis_3_114_0.1mN/m")
# plt.plot(data_cis_3_118_SF[:, 0], data_cis_3_118_SF[:, 1], color="orange", label="cis_3_118_30mN/m")
plt.legend()
plt.yscale("log")
plt.ylabel("Intensity (arb.u.)")
plt.xlabel("q_z (A$^{-1}$)")
plt.title("GIXOS")
plt.savefig("gixos.png", dpi=100)
plt.show()


# %%
# plt.plot(data_trans_49_SF[:, 0], data_trans_49_SF[:, 1], color="blue", label="trans_49_0.5mN/m")
# plt.plot(data_trans_53_SF[:, 0], data_trans_53_SF[:, 1], color="blue", label="trans_53_10mN/m")
plt.plot(data_trans_57_SF[:, 0], data_trans_57_SF[:, 1], color="blue", label="trans_57_20mN/m")
# plt.plot(data_trans_61_SF[:, 0], data_trans_61_SF[:, 1], color="blue", label="trans_61_30mN/m")
# plt.plot(data_cis_77_SF[:, 0], data_cis_77_SF[:, 1], color="red", label="cis_77_5mN/m")
# plt.plot(data_cis_81_SF[:, 0], data_cis_81_SF[:, 1], color="red", label="cis_81_10mN/m")
plt.plot(data_cis_85_SF[:, 0], data_cis_85_SF[:, 1], color="red", label="cis_85_20mN/m")
# plt.plot(data_cis_89_SF[:, 0], data_cis_89_SF[:, 1], color="red", label="cis_89_30mN/m")
# plt.plot(data_cis_93_SF[:, 0], data_cis_93_SF[:, 1], color="red", label="cis_93_30mN/m")
# plt.plot(data_cis_2_105_SF[:, 0], data_cis_2_105_SF[:, 1], color="green", label="cis_2_105_3.3mN/m")
# plt.plot(data_cis_2_109_SF[:, 0], data_cis_2_109_SF[:, 1], color="green", label="cis_2_109_30mN/m")
# plt.plot(data_cis_3_114_SF[:, 0], data_cis_3_114_SF[:, 1], color="orange", label="cis_3_114_0.1mN/m")
# plt.plot(data_cis_3_118_SF[:, 0], data_cis_3_118_SF[:, 1], color="orange", label="cis_3_118_30mN/m")
plt.legend()
plt.yscale("log")
plt.ylabel("Intensity (arb.u.)")
plt.xlabel("q_z (A$^{-1}$)")
plt.title("GIXOS")
plt.savefig("gixos.png", dpi=100)
plt.show()

# %%



