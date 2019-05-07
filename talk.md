<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
# External libraries

## 

BB1000 Programming in Python
KTH

---

layout: false



--
## What can you do with Python libraries

*This year’s Nobel Prize in economics was awarded to a Python convert*
<img height=200 src="https://cms.qz.com/wp-content/uploads/2018/10/Paul-Romer-Jupyter.jpg?quality=75&strip=all&w=1600&h=901">


https://qz.com/1417145/economics-nobel-laureate-paul-romer-is-a-python-programming-convert/


*Instead of using Mathematica, Romer discovered that he could use a Jupyter
notebook for sharing his research. <mark>Jupyter notebooks</mark> are web
applications that allow programmers and researchers to share documents that
include code, charts, equations, and data. Jupyter notebooks allow for code
written in dozens of programming languages. For his research, Romer used
<mark>Python—the most popular language for data science and statistics.</mark>*

---
## What can you do with Python libraries


*Take a picture of a black hole*

<img height=200 src="https://static.projects.iq.harvard.edu/files/styles/os_files_xlarge/public/eht/files/20190410-78m-800x466.png?m=1554877319&itok=fLnjP-iS">


https://doi.org/10.3847/2041-8213/ab0e85



*Software: DiFX (Deller et al. 2011), CALC, PolConvert (Martí-Vidal et al.  2016), HOPS (Whitney et al. 2004), CASA (McMullin et al. 2007), AIPS (Greisen 2003), ParselTongue (Kettenis et al. 2006), GNU Parallel (Tange 2011), GILDAS, eht-imaging (Chael et al. 2016, 2018), <mark>Numpy</mark> (van der Walt et al.  2011), <mark>Scipy</mark> (Jones et al. 2001), <mark>Pandas</mark> (McKinney 2010), Astropy (The Astropy Collaboration et al. 2013, 2018), <mark>Jupyter</mark> (Kluyver et al. 2016), <mark>Matplotlib</mark> (Hunter 2007).*


---

## What is a library?

* A file
* A directory
* Builtin
* Standard library (requires import)
* External libraries (requires install)

---

## Builtins

~~~
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
~~~
~~~
>>> __builtins__
<module 'builtins' (built-in)>
~~~
~~~
>>> dir(__builtins__)
[
...
 'print',
 'property',
 'quit',
 'range',
 'repr',
 'reversed',
 'round',
 'set',
 'setattr',
 'slice',
 'sorted',
 'staticmethod',
 'str',
 'sum',
 'super',
 'tuple',
 'type',
 'vars',
 'zip']
~~~

---

## Standard library

Included in all distributions, but requires an import statement for access

~~~
>>> import math
>>> print(math.pi)
3.141592653589793
~~~

https://docs.python.org/3/library/

<img height="200" src="https://m.media-amazon.com/images/I/517+o9DVBqL._AC_UL436_.jpg">
<img height="200" src="https://m.media-amazon.com/images/I/718xn9vrebL._AC_UL436_.jpg">
<img height="200" src="https://m.media-amazon.com/images/I/513mJHxeseL._AC_UL436_.jpg">
---

## External Python libraries

- NumPy: 'Numerical python', linear algebra, https://www.numpy.org/

- Pandas: High-level library for tabular data, https://pandas.pydata.org/

- Matplotlib: fundamental plotting module, https://matplotlib.org/


<!--
If ModuleNotFoundError, install first

import sys
!conda install --yes --prefix {sys.prefix} numpy (or pandas or matplotlib)

import pandas (or numpy or matplotlib.pyplot) as pd (or np or plt)

-->


---

## Are you a math genius?

<img height=300 src="mathgenius.jpg" >

* First three rows a linear system of equations
* Identify the coefficient matrix and right-hand side
* Last line represents an expression of the solutions

$$
\begin{pmatrix}
3 &0 &0 \\\
1 &8 &0 \\\
4 &0 &-2
\end{pmatrix}
\begin{pmatrix}
a\\\ b \\\ c
\end{pmatrix}
=
\begin{pmatrix}
30\\\ 18 \\\ 2
\end{pmatrix}
$$
$$
a + 3b + c = ?
$$


---

### Linear Algebra in Python: NumPy

* Libraries provided by ``numpy`` provide computational speeds close to compiled languages
* Generally written in C
* From a user perspective they are imported as any python module
* http://www.numpy.org

---

### Creating arrays

* one- and two-dimensional

```
>>> import numpy
>>> a = numpy.zeros(3)
>>> print(type(a), a)
<class 'numpy.ndarray'> [ 0.  0.  0.]
>>> b = numpy.zeros((3, 3))
>>> print(b)
[[ 0.  0.  0.]
 [ 0.  0.  0.]
 [ 0.  0.  0.]]

```

---

### Copying arrays

```
    >>> x = numpy.zeros(2)
    >>> y = x
    >>> x[0] = 1; print(x, y)
    [ 1.  0.] [ 1.  0.]

```
    
Note that assignment (like lists) here is by reference

```
    >>> print(x is y)
    True

```

Numpy array copy method

```
    >>> y = x.copy()
    >>> print(x is y)
    False

```

---

### Filling arrays

``linspace`` returns an array with sequence data

```
    >>> print(numpy.linspace(0,1,6))
    [ 0.   0.2  0.4  0.6  0.8  1. ]

```

``arange`` is a similar function
::
    >>> print(numpy.arange(0, 1, 0.2))
    [ 0.   0.2  0.4  0.6  0.8]

---

### Arrays from list objects

```
    >>> la=[1.,2.,3.]
    >>> a=numpy.array(la)
    >>> print(a)
    [ 1.  2.  3.]

```
```
    >>> lb=[4., 5., 6.]
    >>> ab=numpy.array([la,lb])
    >>> print(ab)
    [[ 1.  2.  3.]
     [ 4.  5.  6.]]

```

---

### Arrays from file data:

* Using `numpy.loadtxt`


```
    #a.dat
    1 2 3
    4 5 6
```

<!--
>>> with open('a.dat', 'w') as adat:
...     n = adat.write('1 2 3\n4 5 6\n')

-->

```
    >>> a = numpy.loadtxt('a.dat')
    >>> print(a)
    [[ 1.  2.  3.]
     [ 4.  5.  6.]]

```
If you have a text file with only numerical data
arranged as a matrix: all rows have the same number of elements

---

### Reshaping

by changing the shape attribute

```
>>> print(ab.shape)
(2, 3)
>>> ab.shape = (6,)
>>> print(ab)
[ 1.  2.  3.  4.  5.  6.]

```

with the reshape method

```
    >>> ba = ab.reshape((3, 2))
    >>> print(ba)
    [[ 1.  2.]
     [ 3.  4.]
     [ 5.  6.]]

```
---

### Views of same data

* ab and ba are different objects but represent different views  of the same data

```
>>> ab[0] = 0
>>> print(ab)
[ 0.  2.  3.  4.  5.  6.]
>>> print(ba)
[[ 0.  2.]
 [ 3.  4.]
 [ 5.  6.]]

```
---

### Array indexing

like lists

* ``a[2:4]`` is an array slice with elements ``a[2]`` and ``a[3]``
* ``a[n:m]`` has size ``m-n``
* ``a[-1]`` is the last element of ``a``
* ``a[:]`` are all elements of ``a``

---

### Looping over elements

```
>>> r, c = ba.shape
>>> for i in range(r):
...    line = ""
...    for j in range(c):
...        line += "%10.3f" % ba[i, j]
...    print(line)
     0.000     4.000
     2.000     5.000
     3.000     6.000

```

or

```
>>> for row in ba:
...     print("".join("%10.3f" % el for el in row))
     0.000     4.000
     2.000     5.000
     3.000     6.000

```

more *Pythonic*

---

* The `ravel` methods returns a one-dim array

```
    >>> for e in ba.ravel():
    ...    print(e)
    0.0
    4.0
    2.0
    5.0
    3.0
    6.0

```
    
---

* ``ndenumerate`` returns indices as well

```
    >>> for ind, val in numpy.ndenumerate(ba):
    ...    print(ind, val )
    (0, 0) 0.0
    (0, 1) 4.0
    (1, 0) 2.0
    (1, 1) 5.0
    (2, 0) 3.0
    (2, 1) 6.0

```

---


### Matrix operations

* explicit looping

```
    >>> import numpy, time
    >>> n=256
    >>> a=numpy.ones((n,n))
    >>> b=numpy.ones((n,n))
    >>> c=numpy.zeros((n,n))
    >>> t1=time.clock()
    >>> for i in range(n):
    ...    for j in range(n):
    ...        for k in range(n):
    ...            c[i,j]+=a[i,k]*b[k,j]
    >>> t2=time.clock()
    >>> print("Loop timing",t2-t1 )
    Loop timing ...

```

---

* using `numpy.dot`

```
    >>> import numpy, time
    >>> n=256
    >>> a=numpy.ones((n,n))
    >>> b=numpy.ones((n,n))
    >>> t1=time.clock()
    >>> c=numpy.dot(a,b)
    >>> t2=time.clock()
    >>> print("dot timing",t2-t1)
    dot ...

```

---

### The Fortran version

```fortran
    INTEGER, PARAMETER :: N = 256
    REAL*8, DIMENSION(N,N) :: A, B, C
    ! Timing
    INTEGER :: T1, T2, RATE
    ! Initialize
    A = 1.0
    B = 1.0
    !
    CALL SYSTEM_CLOCK(COUNT_RATE=RATE)
    CALL SYSTEM_CLOCK(COUNT=T1)
    C = MATMUL(A, B)
    CALL SYSTEM_CLOCK(COUNT=T2)
    PRINT '(A, F6.2)', 'MATMUL timing',  DBLE(T2-T1)/RATE
    END
```
---

### Conclusion

* Provided that numpy has been install properly (difficult) and linked with optimized libraries, basic linear algebra work as fast in python as in Fortran (or C/C++)

---

### More vector operations

* Scalar multiplication ``a*2`` 
* Scalar addition ``a + 2``
* Power (elementwise) ``a**2``

Note that for objects of ``ndarray`` type, multiplication means elementwise multplication and not matrix multiplication

---

### Vectorized elementary functions

```
    >>> v = numpy.arange(0, 1, .2)
    >>> print(v)
    [ 0.   0.2  0.4  0.6  0.8]

```
--
```
    >>> print(numpy.cos(v))
    [ 1.          0.98006658  0.92106099  0.82533561  0.69670671]

```
--
```
    >>> print(numpy.sqrt(v))
    [ 0.          0.4472136   0.63245553  0.77459667  0.89442719]

```
--
```
    >>> print(numpy.log(v)) #doctest: +SKIP
    ./linalg.py:98: RuntimeWarning: divide by zero encountered in log
      print(numpy.log(v))
    [       -inf -1.60943791 -0.91629073 -0.51082562 -0.22314355]

```

---

### More linear algebra

* Solve a linear  system of equations
$$Ax = b$$

--


```
    x = numpy.linalg.solve(A, b)

```

--

* Determinant of a matrix

$$det(A)$$

--


```
    x = numpy.linalg.det(A)

```

---


* Inverse of a matrix 
$$A^{-1}$$

--

```
    x = numpy.linalg.inverse(A)

```

--

*  Eigenvalues  of a matrix

$$Ax = x\lambda$$

--

```
    x, l = numpy.linalg.eig(A)

```

---

### Refernces

* http://www.numpy.org
* http://www.scipy-lectures.org/intro/numpy/index.html
* Videos: https://pyvideo.org/search.html?q=numpy
