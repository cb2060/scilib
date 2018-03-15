"""
This test file exemplifies use of the mock library.
The purpose is to test the logic only in your own function, not
on what external libraries return.

Try to understand what it does.
You may use this as a guide for completing LAB1112
"""
import pytest

import unittest.mock as mock
import pandas as pd
import pandas.util.testing as pdt
#import lab
#from 1_name_extlib import *
#import 1_name_extlib
#from 2_average_extlib import *
#from 3_protein_other_extlib import *
#from 4_shift_pdb_extlib import *

def test_1():
    """Test exercise 1
    Assume that the functions returns a pandas.Series object which may be
    forwarded to its own plotting method, e.g. with

    >>> lab.ex_1('data.csv').plot()
    """

    df = pd.DataFrame(
        {'name': ['Mary', 'Anna', 'Emma', 'Elizabeth'],
          'prop': [0.0776441885001, 0.0286179004748, 0.0220129242131, 0.0213095656761],
          'year': [1880, 1880, 1881, 1882]
        }
    )
    
    with mock.patch('pandas.read_csv') as fake_read_csv:
        fake_read_csv.return_value = df

#        births = pd.read_csv('out_donald.csv')
        births = pd.read_csv('out_donald.csv')

    fake_read_csv.assert_called_once_with('out_donald.csv')

    year = pd.Series([1880, 1881, 1882], name='year')
#    expected_births = pd.Series([9669, 2003, 1939], index=year, name='births')
    expected_births = pd.DataFrame({'prop': [0.1062620889749, 0.0220129242131, 0.0213095656761], 'year': [1880, 1881, 1882]})
#    expected_births = pd.Series([0.1062620889749, 0.0220129242131, 0.0213095656761], index= [1880, 1881, 1882])
    
    pdt.assert_frame_equal(births, expected_births)

def test_2():
    """
    Emulate terminal input and printed output to the screen.
    """
    with mock.patch('lab.input') as fake_input:
        fake_input.side_effect = ["0", "1", "1", "2", "quit"]

        with mock.patch('lab.print') as fake_print:
            lab.ex_2()
#             2_average_extlib()
            
    fake_print.assert_called_once_with('1.000 0.707')

def test_3():
    """
    Veryfy that input data is converted and written to output file
    """
    inp = """
...
ATOM      1  N   ASP A   1     -52.682  -1.842  -9.027  1.00 50.37           N  
ATOM      2  CA  ASP A   1     -51.951  -0.605  -9.262  1.00 42.07           C  
ATOM      3  O   ASP A   1     -50.463  -0.871  -9.218  1.00 37.13           O  
...
""".split('\n')

    with mock.patch('lab.open', mock.mock_open()) as fake_open:

        lab.ex_3(inp)

    fake_open.assert_called_once_with('HETATOMS_extra.out', 'w')
    write_calls = [
        mock.call("N  -52.682  -1.842  -9.027 False"),
        mock.call("C  -51.951  -0.605  -9.262 False"),
        mock.call("O  -50.463  -0.871  -9.218 True")
    ]
    fake_open().write.assert_has_calls(write_calls, any_order=True) 

def test_4():
    """
    Verify that relevant parts of input data is shifted and written to file
    """
    inp = """
...
HETATM  809  P   PO4 A 106     -49.810  -2.507 -17.593  0.94 64.95           P  
HETATM  810  O1  PO4 A 106     -48.495  -2.289 -18.306  1.00 55.57           O  
HETATM  813 ZN    ZN A 108     -49.810  28.758  -7.406  0.67 58.21          ZN  
HETATM  811  O2  PO4 A 106     -50.298  -3.632 -18.476  1.00 56.57           O  
...
"""
    with mock.patch('lab.open', mock.mock_open(read_data=inp)) as fake_open:

        lab.ex_4('dum.pdb')

    open_calls = [
        mock.call('dum.pdb'),
        mock.call('dum_shifted_negatively.xyz', 'w')
    ]
    fake_open.assert_has_calls(open_calls, any_order=True)
    fake_open().write.assert_called_once_with('ZN -52.810  24.758  -9.406\n')
