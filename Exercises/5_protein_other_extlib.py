from sys import argv
import pandas as pd
inp = open(argv[1],'r')
out_atom = open('ATOM.out','w')
out_hetatom = open('HETATOM.out','w')

lines = inp.readlines()
print ('name, xcoord, ycoord, zcoord', file=out_atom)
print ('name, xcoord, ycoord, zcoord', file=out_hetatom)

for line in lines:
   if line.startswith('ATOM'):
      l = line.split()
      coord = (l[11], float(l[6]), float(l[7]), float(l[8]))
      print ("%5s, %8.3f, %8.3f, %8.3f"%coord, file=out_atom)
	
   if line.startswith('HETATM'):
      m = line.split()
      coord2 = (m[11], float(m[6]), float(m[7]), float(m[8]))
      print ("%5s, %8.3f, %8.3f, %8.3f"%coord2, file=out_hetatom)

out_atom.close()
out_hetatom.close()

out_hetatom_extra = open('HETATOM_extra.out','w')

data=pd.read_csv('HETATOM.out')

data['Oxygen'] = data.name == '    O'

print(data, file=out_hetatom_extra)

out_hetatom_extra.close()
