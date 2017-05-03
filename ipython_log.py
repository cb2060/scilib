# IPython log file

get_ipython().magic('logstart')
import pandas as pd
import numpy as np
import matpplotlib.pyplot as plt
import matplotlib.pyplot as plt
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
data2 = [[1,2,3,4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print(arr2)
data2
print(arr2.ndim)
print(arr2.shape)
print(np.zeros(10))
print(np.zeros((3,6)))
print(np.arange(15))
print(range(15))
print(arange(15))
print(np.empty((2,3,2)))
print(np.eye(3))
arr = np.array([[1., 2., 3.], [4.,5.,6.]])
arr
print(arr*arr)
print(arr-arr)
print(arr**0.5)
arr = np.arange(10)
print(arr[5])
print(arr[5:8])
arr[5:8] = 12
print(arr)
arr_slice = arr[5:8]
arr_slice[:] = 64
print(arr)
try = arange(10)
try = range(10)
print(range(10))
print(arange(10))
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
print(arr2d[0][2])
print(arr2d[:2,1:])
names = np.array(['Asterix', 'Obelix', 'Idefix', 'Asterix', 'Idefix', 'Obelix', 'Obelix'])
names
data = np.random.randn(7,4)
data
random.randn(7,4)
randn(7,4)
np.randn(7,4)
names == 'Asterix'
names1 = array(['Asterix', 'Obelix', 'Idefix', 'Asterix', 'Idefix', 'Obelix', 'Obelix'])
names1 == 'Asterix'
data[names == 'Asterix']
data[names1 == 'Asterix']
data1 = random.randn(7,4)
data1[names1 == 'Asterix']
data1
data[names == 'Asterix', 2:]
names != 'Asterix'
data[-(names == 'Asterix')]
data[~(names == 'Asterix')]
mask = (names == 'Asterix') | (names == 'Idefix')
mask
data[names != 'Asterix'] = 7 
data
arr = np.arange(15).reshape((3,5))
arr
np.arange(15)
arr.T
np.dot(arr.T, arr)
np.dot(transpose(arr), arr)
arr = np.arange(10)
arr
np.sqrt(arr)
arr2 = np.random.randn(10)
arr2
arr3 = np.random.randn(10)
arr3
np.maximum(arr2, arr3)
np.minimum(arr2, arr3)
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(cond, xarr, yarr)
result
bools = np.array([False, False, True, False])
np.any(bools)
np.all(bools)
arr = np.random.randn(8)
arr
np.sort(arr)
names = np.array(['Asterix', 'Kanalltix', 'Kaningentix', 'Asterix', 'Kaningentix'])
np.unique(names)
arr = np.arange(10)
np.save('some_array', arr)
np.load('some_array.npy')
arr = np.loadtxt('servicenumbers.txt', delimiter=',')
arr
try = np.loadtxt('CeltixWarr.txt')
try = np.loadtxt('CeltixWarr.txt', delimiter = ',')
someth = np.loadtxt('CeltixWarr.txt', delimiter = ',')
someth = np.loadtxt('CelticWarr.txt', delimiter = ',')
someth
arr2 = np.random.randn(8)
np.savetxt('CelticWarr.txt', arr2)
X = np.random.randn(2,2)
mat = X.T.dot(X)
np.linalg.inv(mat)
np.linalg.inv(mat).dot(mat)
mat
a= np.array([[1,2,3], [4,5,6]])
np.cumsum(a)
np.cumsum(a,axis=0)
np.cumsum(a,axis=1)
a
obj = pd.Series([4, 7, -5, 3])
obj
obj.values
obj.index
for j in obj.index: print j
for j in obj.index: print (j)
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2['a']
obj2['d']
obj2['d'] = 6
obj2
obj2['e'] = 6
obj2
np.exp(obj2)
data = }'Celt': ['Asterix', 'Aseterix', 'Aseterix', 'Obelix', 'Obelix'], 'age': [18, 19, 20, 19, 20 ], 'numberofromans': [1.5, 1.7, 3.6, 2.4, 2.9]}
data = {'Celt': ['Asterix', 'Aseterix', 'Aseterix', 'Obelix', 'Obelix'], 'age': [18, 19, 20, 19, 20 ], 'numberofromans': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
frame
frame2 = pd.DataFrame(data, columns=['numberofromans','Celt'], index=['one', 'two', 'three', 'four', 'five'])
frame2
frame2.to_csv('out_frame2.csv')
data = {'Celt': ['Asterix', 'Asterix', 'Asterix', 'Obelix', 'Obelix'], 'age': [18, 19, 20, 19, 20 ], 'numberofromans': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
frame2 = pd.DataFrame(data, columns=['numberofromans','Celt'], index=['one', 'two', 'three', 'four', 'five'])
frame2.to_csv('out_frame2.csv')
{color : [brown, yellow, green, brown, yellow]}
{'color' : ['brown', 'yellow', 'green', 'brown', 'yellow']}
frame3 = pd.DataFrame(data, columns=['numberofromans','Celt','color'], index=['one', 'two', 'three', 'four', 'five'])
frame3
frame2['Celt']
frame2.Celt
frame2.numberofromans
frame2.ix['three']
frame2['thick'] = frame2.Celt == 'Obelix'
frame2
frame2['color'] = ['brown', 'yellow', 'green', 'brown', 'yellow']
frame2
del frame2['thick']
frame2
punch = { 'Asterix': {18: 1.5, 19: 1.7, 20: 3.6}, 'Obelix': {19: 2.4, 20: 2.9}}
isinstance(punch,dict}
isinstance(punch,dict)
frame = pd.DataFrame(punch)
frame
punch
frame.T
data
isinstance(data,dict)
A = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
B = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index = ['a', 'c', 'e', 'f', 'g'])
A+B
df1 = pd.DataFrame(np.arange(9.).reshape((3,3)), columns=list('bcd'),index=['Asterix', 'Obelix', 'Kanalltix'])
df2 = pd.DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'), index=['Miraculix', 'Asterix', 'Obelix', 'Kaningentix'])
df1
df2
df1 + df2
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
obj2 = pd.SEries([4, 7, -3, 2], index=['d', 'a', 'b', 'c'])
obj2 = pd.Series([4, 7, -3, 2], index=['d', 'a', 'b', 'c'])
obj2.sort_values()
frame = pd.DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame.sort_index()
frame.sort_index(axis=0)
frame.sort_index(axis=1)
frame.sort_index(axis=1, ascending=False)
frame = pd.DataFrame({'b': [ 4, 7, -3], 'a': [0, 1, 0]})
frame.sort_values(by='b')
head 5 names/yob1880.txt
head -5 names/yob1880.txt
head -5 Names/yob1880.txt
get_ipython().magic('cd Names')
head -5 yob1880.txt
head yob1880.txt
sed -n '10,30p' yob1880.txt
get_ipython().magic('cd ..')
names1880 = pd.read_csv('Names/yob1880.txt', names=['name', 'sex', 'births'])
print(names1880)
names1880.groupby('sex')['births'].sum()
xls_file=pd.ExcelFile('data.xls')
table = xls_file.parse('sheet1')
table = xls_file.parse('Sheet1')
table
plt.plot(np.random.randn(30), linestyle='--', color='g')
plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='solid', marker='*')
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
fig
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation= 30, fontsize='small')
ax.set_title('One of my first matplotlib plots')
ax.set_xlabel('Stages')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation= 30, fontsize='small')
ax.set_title('One of my first matplotlib plots')
ax.set_xlabel('Stages')
fig = plt.figure(); ax = fig.add_subplot(1, 1, 1)
ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
ax.legend(loc='best')
plt.savefig('try.png', dpi=400)
plt.savefig('myfavouritelastpic.png', dpi=400)
s= pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
df = DataFrame(np.random.randn(10,4).cumsum(0), columns = ['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df = np.DataFrame(np.random.randn(10,4).cumsum(0), columns = ['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df = pd.DataFrame(np.random.randn(10,4).cumsum(0), columns = ['A', 'B', 'C', 'D'], index=np.arange(0, 100, 10))
df.plot()
np.random.randn(10,4)
A=np.random.randn(10,4)
A
A.cumsum()
A.cumsum(0)
data = pd.Series(np.random.rand(16), index=list('abdefghijklmnop'))
data = pd.Series(np.random.rand(16), index=list('abdefghijklmnopq'))
data.plot(kind='barh', color='r', alpha=0.5)
data.plot(kind='barh', color='r', alpha=0.5)
data = pd.Series(np.random.rand(16), index=list('abdefghijklmnopq'))
data.plot(kind='barh', color='r', alpha=0.5)
df = pd.DataFrame(np.random.rand(6,4), index= ['one', 'two', 'three', 'four' ])
df = pd.DataFrame(np.random.rand(6,4), index= ['one', 'two', 'three', 'four', 'five', 'six'], columns=pd.Index(['A', 'B', 'C', 'D'], name='Genus'))
df
df.plot(kind='bar')
tb = pd.read_csv('out_total_births.csv')
tb
tb.plot(title= 'Total births by sex and year', x='year', y=['F', 'M'])
get_ipython().magic('pip install BeautifulSoup4')
