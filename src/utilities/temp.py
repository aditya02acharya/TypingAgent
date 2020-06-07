import numpy
from itertools import product

layout = [['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '-'],
          ['-', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '-'],
          ['-', '-', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '<'],
          ['-', '-', '-', ' ', ' ', ' ', ' ', ' ', '>', '>', '>']]

# save array
# numpy.save('../layouts/english_layout.npy', layout)

# load array
data = numpy.load('../../layouts/finnish_layout.npy')
# print the array
print(data)
# foo = numpy.where(data == 'm')
# d = [numpy.where(data == 'm')[0][0], numpy.where(data == 'm')[1][0]]
# print(numpy.hstack(foo))
