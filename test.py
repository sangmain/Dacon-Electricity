import numpy as np

data = [10, 20, 30, 40, 50, 60,]

comp =[10,20,30]

print(np.where(comp == data))


# aa = np.isin(data, comp)
# print(aa)
# found = data[aa]
# print(comp)
# print(found)
# if len(comp) == len(found):
#     is_same = np.equal(comp, found).any()
#     print(is_same)
