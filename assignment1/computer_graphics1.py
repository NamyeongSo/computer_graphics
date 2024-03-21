import numpy as np
list = []
for i in range(2, 27):
    list.append(i)

d1_array = np.array(list)
print(d1_array)

d2_array = np.reshape(d1_array, (5,5))
print(d2_array)

d2_array[1:-1, 1:-1] = 0
print(d2_array)

m_square = d2_array@d2_array
print(m_square)

v = np.power(m_square[0][:],2)
sum = np.sum(v)
print(np.sqrt(sum))