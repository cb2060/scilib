<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
# External libraries

## 

BB1000 Programming in Python
KTH

---

layout: false

## Essential Python libraries

- NumPy: 'Numerical python' - package for scientific computing in Python. Ment primarily to sort, reshape, and index array types...
- SciPy: functions like numerical integration, linear algebra routinges,... Numpy and SciPy are often used together; the difference is not sharp; newer functionalities might reside in SciPy. 
- pandas: data structures and functions to work with structured data. The main object in pandas is the `DataFrame`, which is a two-dimentional tabular.
- matplotlib: producing plots; the basic functions handled in this course are all in the matplotlib.pyplot module.

```
>>> import pandas as pd
>>> import numpy as np
>>> import scipy as sp
>>> import matplotlib.pyplot as plt

```
---

## Pandas - Baby names 1880-2015

On http://www.ssa.gov/oact/babynames/limits.html the total number of births for each gender/name combination is given as a raw archive. 

```
Mary,F,7065
Anna,F,2604
Emma,F,2003
Elizabeth,F,1939
Minnie,F,1746

```

Since this is a comma-separated form, use is made of `pandas.read_csv` to load the data

```
import pandas as pd
names1880 = pd.read_csv('names/yob1880.txt', names=['name', 'sex', 'births'])

```

---

The data are printed as

```
>>> print(names1880)
         name sex  births
0        Mary   F    7065
1        Anna   F    2604
2        Emma   F    2003
3        Elizabeth   F    1939
4        Minnie   F    1746
...
1996     Worthy   M       5
1997     Wright   M       5
1998       York   M       5
1999  Zachariah   M       5

[2000 rows x 3 columns]

```

To get an overview over all births, we can use the sum of the births by sex:

```
>>> names1880.groupby('sex')['births'].sum()
sex
F     90992
M    110490
Name: births, dtype: int64

```


---

## Pandas - Excel

On the internet, Kaningentix finds an excel sheet containing all herbs, grasses and vegetables which can be found in the forest. The list contains not only the names and the subsequent characterizations, but also where these are found and the time of the medicinal effect.

It is advisable to use pandas, making use of the ExcelFile class.


```
>>> import pandas as pd
>>> xls_file = pd.ExcelFile('data.xls') 

```

Data stored in a sheet can then be read into DataFrame using parse:

```
>>> table = xls_file.parse('sheet1')

```

---

## NumPy

One of the key fatures of NumPy is its N-dimensional array object: `ndarray`. They enable to perform mathematical operations on blocks of data.

```
>>> import numpy as np
>>> data1 = [6, 7.5, 8, 0, 1]
>>> arr1 = np.array(data1)

>>> print(arr1)
[ 6.   7.5  8.   0.   1. ]

>>> data2 = [[1, 2, 3, 4], [ 5, 6, 7, 8]]
>>> arr2 = np.array(data2)

>>> print(arr2)
[[1 2 3 4]
 [5 6 7 8]] 

>>> print(arr2.ndim)
2
>>> print(arr2.shape)
(2, 4)

```

---

## NumPy - Default arrays

```
>>> print(np.zeros(10))
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

>>> print(np.zeros((3,6)))
[[ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.]]  

>>> print(np.arange(15))
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

```

Remark that for higher dimensional arrays, we have used tuples.

---

## NumPy - Default arrays

```
>>> print(np.empty((2,3,2)))
[[[  2.35558336e-310   2.02731498e-316]
  [  2.35558575e-310   2.35558575e-310]
  [  2.35558575e-310   2.35558575e-310]]

 [[  2.35558575e-310   2.35558575e-310]
  [  2.35558575e-310   2.35558575e-310]
  [  2.35558575e-310   2.42092166e-322]]]

>>> print(np.eye(3))
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]

```

`empty` creates an array without initializing its values to any particular value. It does the ideal recipe to return garbage...

---

## NumPy - Operations between arrays and scalars

```
>>> arr = np.array([[1., 2., 3.], [4., 5., 6.]])
>>> print(arr*arr)
[[  1.   4.   9.]
 [ 16.  25.  36.]]

>>> print(arr-arr)
[[ 0.  0.  0.]
 [ 0.  0.  0.]]
 
>>> print(1/arr)
[[ 1.          0.5         0.33333333]
 [ 0.25        0.2         0.16666667]]
 
>>> print(arr**0.5)
[[ 1.          1.41421356  1.73205081]
 [ 2.          2.23606798  2.44948974]]
 
```
---

## NumPy - Basic indexing and slicing

```
>>> arr= np.arange(10)
>>> print(arr[5])
5
>>> print(arr[5:8])
[5 6 7]
>>> arr[5:8] = 12
>>> print(arr)
[ 0  1  2  3  4 12 12 12  8  9]

>>> arr_slice = arr[5:8]
>>> arr_slice[:] = 64
>>> print(arr)
[ 0  1  2  3  4 64 64 64  8  9]

```

---

## NumPy - Basic indexing and slicing

```
>>> arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> print(arr2d[2])
[7 8 9]
>>> print(arr2d[0][2])
3

```

```
>>> print(arr2d[:2, 1:])
[[2 3]
 [5 6]]

```

Entries up till (but not including) the second row are kept, as well as the column starting from (and including) the first one.

---

## NumPy - Boolean indexing

```
>>> names = np.array(['Asterix', 'Obelix', 'Idefix', 'Asterix', 'Idefix', 'Obelix', 'Obelix'])
>>> names
array(['Asterix', 'Obelix', 'Idefix', 'Asterix', 'Idefix', 'Obelix',
       'Obelix'],
      dtype='|S7')	     

```

```
>>> data = np.random.randn(7,4)
>>> data
array([[ 0.02062421, -0.1369847 ,  0.90160195,  0.75181516],
       [-1.1268401 , -0.41237719, -0.21513891,  0.2190537 ],
       [-0.00535594,  0.15848914, -0.99522448,  0.93785222],
       [ 0.84553696, -1.7851311 ,  0.74135975,  0.36109035],
       [ 1.22254501, -0.68403217,  0.39343747,  1.59037781],
       [ 0.02684093, -0.62523998,  0.06727077, -1.3981326 ],
       [ 0.70864672, -1.46741426, -1.69648987, -0.47846134]])
>>> names == 'Asterix'
array([ True, False, False,  True, False, False, False], dtype=bool)

```

---

## NumPy - Boolean indexing

Those rows in data indexed with 'True' can be selected:

```
>>> data[names == 'Asterix']
array([[ 0.02062421, -0.1369847 ,  0.90160195,  0.75181516],
       [ 0.84553696, -1.7851311 ,  0.74135975,  0.36109035]])
       
```

And also slicing is possible:

```
>>> data[names == 'Asterix', 2:]
array([[ 0.90160195,  0.75181516],
       [ 0.74135975,  0.36109035]])
       
```

For negation `!=` can be used as wel as `-`

```
>>> names != 'Asterix'
array([False,  True,  True, False,  True,  True,  True], dtype=bool)
>>> data[-(names == 'Asterix')]
array([[-1.1268401 , -0.41237719, -0.21513891,  0.2190537 ],
       [-0.00535594,  0.15848914, -0.99522448,  0.93785222],
       [ 1.22254501, -0.68403217,  0.39343747,  1.59037781],
       [ 0.02684093, -0.62523998,  0.06727077, -1.3981326 ],
       [ 0.70864672, -1.46741426, -1.69648987, -0.47846134]])

```

---

## NumPy - Boolean indexing

To select two of the three names to combine multiple boolean conditions, use boolean arithmetic operators like `&` (and) and `|` (or):

```
>>> mask = (names == 'Asterix') | (names == 'Idefix')
>>> mask
array([ True, False,  True,  True,  True, False, False], dtype=bool)

```

In this way, it is possible to set data to whole rows:

```
>>> data[names != 'Asterix'] = 7
>>> data
array([[ 0.02062421, -0.1369847 ,  0.90160195,  0.75181516],
       [ 7.        ,  7.        ,  7.        ,  7.        ],
       [ 7.        ,  7.        ,  7.        ,  7.        ],
       [ 0.84553696, -1.7851311 ,  0.74135975,  0.36109035],
       [ 7.        ,  7.        ,  7.        ,  7.        ],
       [ 7.        ,  7.        ,  7.        ,  7.        ],
       [ 7.        ,  7.        ,  7.        ,  7.        ]])
					  
```

---

## NumPy - Transposing arrays

Arrays have the `transpose` method and also the special `T` attribute:

```
>>> arr = np.arange(15).reshape((3,5))
>>> arr
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> arr.T
array([[ 0,  5, 10],
       [ 1,  6, 11],
       [ 2,  7, 12],
       [ 3,  8, 13],
       [ 4,  9, 14]])			    

```

The method is very interesting in linear algebra, as the inner matrix product 'X^TX' can be easily calculated:

```
>>> np.dot(arr.T, arr)
array([[125, 140, 155, 170, 185],
       [140, 158, 176, 194, 212],
       [155, 176, 197, 218, 239],
       [170, 194, 218, 242, 266],
       [185, 212, 239, 266, 293]])

```

---

## NumPy - Unary and binary universal functions

A universal function ('ufunction') is a function that performs elementwise operations on data in ndarrays. A unary one only focusses upon one array, while binary ones require 2 arrays.

Examples of unary ufunctions are `sqrt`, `exp`, `abs`, `log`, `sign`, `floor` (largest integer less than or equal to the element), `ceil` (analogon for `floor` but then higher or equal to the element), `cos`,...

Examples of binary ufunctions are `add`, `subtract`, `multiply`, `divide`, `power`, `max`, `min`, `mod` (remainder of division), `greater`, `less`, `less_equal`, ...


---

## NumPy - Unary and binary universal functions

A few examples...	

```
>>> arr = np.arange(10)
>>> np.sqrt(arr)
array([ 0.        ,  1.        ,  1.41421356,  1.73205081,  2.        ,
        2.23606798,  2.44948974,  2.64575131,  2.82842712,  3.        ])

```

```
>>> arr2 = np.random.randn(10)
>>> arr2
array([-1.32672421, -2.02196629, -1.87814963,  1.3586335 ,  0.66869694,
        1.64577817,  0.01575116,  0.09529667, -0.32427566,  0.73408638])
>>> arr3 = np.random.randn(10)
>>> arr3
array([-0.09059814, -0.05915682,  1.39919745, -0.96167955, -2.70897768,
       -1.44743637,  0.47766619, -0.18136026,  0.87246909, -0.43929249])
>>> np.maximum(arr2, arr3)
array([-0.09059814, -0.05915682,  1.39919745,  1.3586335 ,  0.66869694,
        1.64577817,  0.47766619,  0.09529667,  0.87246909,  0.73408638])
	
```

---

## NumPy - Conditions and arrays

The `numpy.where(condition,firstargument,secondargument)` function reduces the expression `x if condition else y` for arrays: if the `condition` is true, then the `firstargument` is executed, else the `secondargument` is done.

```
>>> xarr = np.array([1.1, 1.2, 1.3 , 1.4, 1.5])
>>> yarr = np.array([2.1, 2.2, 2.3 , 2.4, 2.5])
>>> cond = np.array([True, False, True, True, False])
>>> result = np.where(cond, xarr, yarr)
>>> result
array([ 1.1,  2.2,  1.3,  1.4,  2.5])

``` 

For boolean arrays, `any` tests whether one or more values in an array is `True`, while `all` checks if every value is `True`.

```
>>> bools = np.array([False, False, True, False])
>>> np.any(bools)
True
>>> np.all(bools)
False

```

---

## NumPy - Sorting and Unique

```
>>> arr= np.random.randn(8)
>>> arr
array([ 0.71176752,  0.24762018, -0.61990769,  0.77071301,  0.67810754,
        1.92071058,  1.01916251,  1.06109087])
>>> np.sort(arr)
array([-0.61990769,  0.24762018,  0.67810754,  0.71176752,  0.77071301,
        1.01916251,  1.06109087,  1.92071058])

```

```
>>> names = np.array(['Asterix', 'Kanalltix', 'Kaningentix', 'Asterix', 'Kaningentix', 'Asterix', 'Kaningentix'])
>>> np.unique(names)
array(['Asterix', 'Kanalltix', 'Kaningentix'],
      dtype='|S11')

```

---

## NumPy - Storing array, saving and loading

`np.save` and `np.load` allow to save and load data on disk. It will be in raw binary format and have file extension `.npy`.   

```
>>> arr = np.arange(10)
>>> np.save('some_array', arr)
>>> np.load('some_array.npy')
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

```

Heroix finds a paper with the service numbers of the roman soldiers which attacked his village yesterday. How can he manipulate this using arrays?

```
125481,568937,428937,585667,889375,442568,558937,557934,554723,258649,34582
>>> arr = np.loadtxt('servicenumbers.txt', delimiter=',')
>>> arr
array([ 125481.,  568937.,  428937.,  585667.,  889375.,  442568.,
        558937.,  557934.,  554723.,  258649.,   34582.])
	
```

When Heroix wants to write the service numbers of his Celtic warriors on a file, he uses `np.savetxt`:

```
>>> arr2=np.random.randn(8)
>>> np.savetxt('CelticWarr.txt',arr2)

```

---

## NumPy - Linear algebra

Attention has to be paid at `*` which is an element-wsie product instead of a matrix dot product. The function `dot` is used in numpy (see 'Transposing arrays').

To do calculations on matrices, `numpy.linalg` has a standard set of functions, like `diag` (return the diagonal elements of a square matrix), `trace`, `det` (matrix determinant), `eig` (eigenvalues and eigenvectors of a square matrix), `inv` (inverse),...

```
>>> X = np.random.randn(2,2)
>>> mat = X.T.dot(X)
>>> np.linalg.inv(mat)
array([[  1.70182387,   3.91458018],
       [  3.91458018,  10.22597796]])
       
```



---




