# IPython log file

get_ipython().magic('logstart')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
print(arr2.ndim)
print(arr1.ndim)
arr = np.array([[1., 2., 3.], [4.,5.,6.]])
print(arr*arr)
print(arr-arr)
arr = np.arange(10)
print(arr[5:8])
arr[5:8] = 12
print(arr)
arr_slice = arr[5:8]
arr_slice[:] = 64
print(arr)
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d)
print(arr2d[0][2])
print(arr2d[0][0])
print(arr2d[:2, 1:])
names = np.array(['Asterix', 'Obelix', 'Idefix', 'Asterix', 'Idefix', 'Obelix', 'Obelix'])
names
data = np.random.randn(7,4)
data
names == 'Asterix'
data[names == 'Asterix']
data [ names == 'Asterix', 2: ]
arr = np.arange(15).reshape((3,5))
arr
arr.T
np.dot(arr.T,arr)
arr = np.arang(10)
arr = np.arange(10)
np.save('some_array', arr)
arr = np.loadtxt('servicenumbers.txt', delimiter=',')
arr
arr2 = np.random.randn(8)
np.savetxt('CelticWarr.txt', arr2)
a = np.array([[1,2,3], [4,5,6]])
np.cumsum(a)
np.cumsum(a,axis=0)
np.cumsum(a,axis=1)
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2['c']
np.exp(obj2)
data = {'Celt': ['Asterix', 'Asterix', 'Asterix', 'Obelix', 'Obelix'], 'age' :
[18, 19, 20, 19, 20 ], 'numberofromans': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
frame
frame2 = pd.DataFrame(data, columns=['numberofromans','Celt'], index=['one',
'two', 'three', 'four', 'five'])
frame2
frame2.to_csv('out_frame2.csv')
frame2['thick'] = frame2.Celt == 'Obelix'
frame2
del frame2['thick']
frame2
A = pd.Series([7.3, -2.5, 3.4, 1.5], index = ['a', 'c', 'd', 'e'])
B = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index = ['a', 'c', 'e', 'f', 'g'])
A-B
frame = pd.DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'],columns=['d', 'a', 'b', 'c'])
frame
frame.sort_index()
frame.sort_index(axis=0)
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending)
frame.sort_index(axis=1, ascending = False)
frame = pd.DataFrame({'b': [4, 7, -3], 'a': [0, 1, 0]})
frame.sort_values(by='b')
get_ipython().magic('cd Names')
names1880 = pd.read_csv('Names/yob1880.txt', names=['name', 'sex', 'births'])
names1880 = pd.read_csv('yob1880.txt', names=['name', 'sex', 'births'])
print(names1880)
names1880.groupby('sex')['births'].sum()
get_ipython().magic('cd ..')
xls_file = pd.ExcelFile('data.xls')
table = xls_file.parse('Sheet1')
table
xlsx_file = read_excel('data_xlsx.xlsx')
xlsx_file = pd.read_excel('data_xlsx.xlsx')
xlsx_file
xls_file = pd.ExcelFile('data.xls')
table = xls_file.parse('Sheet1')
table
xlsx_file = pd.read_excel('data_xlsx.xlsx')
xlsx_file
somethingelse = pd.read_excel('data.xls')
somethingelse
plt.plot(np.random.randn(30), linestyle='--', color='g')
plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='solid', marker='*')
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1 = fig.add_subplot(2,2,2)
ax1 = fig.add_subplot(2,2,3)
fig= plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'],rotation=30, fontsize='small')
ax.set_title('One of my first matplotlib plots')
ax.set_xlabel('Stages')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')
plt.savefig('myfavouritelastpic.png', dpi=400)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
data.plot(kind='barh', color='r', alpha=0.5)
data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnopq'))
tb=pd.read_csv('out_total_births.csv')
tb
tb.plot(title='Total births by sex and year', x='year',y=['F','M'])
quit()
