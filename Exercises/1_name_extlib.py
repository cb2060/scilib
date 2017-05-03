import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('out_top1000.csv')

frame = pd.DataFrame(data)

#if frame.name == 'Donald':
#    print('Donald')
#    print(frame.year)

tot_year=frame.groupby('year')['births'].sum()

tot_year.plot(title='tot per year', x='year', y='prop')

plt.savefig('tot_year.png', dpi=400)
    
inp = open('out_top1000.csv', 'r')

out_donald = open('out_donald.csv', 'w') 

out_george = open('out_george.csv', 'w')

lines= inp.readlines()

print('year,prop', file = out_donald)

print('year,prop', file = out_george)

for line in lines:
    if line.split(',')[1] == 'Donald':
#        val = True
#        iets = line.split(',')[3]
#        out_donald.write("%s,%s" % (str(line.split(',')[4]),str(iets)))
#        if float(line.split(',')[5]) < 0.0001:
#            val == False
#        if val:    
        out_donald.write("%s,%s" % (str(line.split(',')[4]),str(line.split(',')[5])))
    if line.split(',')[1] == 'George':
        out_george.write("%s,%s" % (str(line.split(',')[4]),str(line.split(',')[5])))

out_donald.close()

out_george.close()
        
data_donald = pd.read_csv('out_donald.csv')

data_george = pd.read_csv('out_george.csv')

frame_donald = pd.DataFrame(data_donald)

frame_george = pd.DataFrame(data_george)

frame_donald.plot(title='Donald', x='year', y='prop')

plt.savefig('Donald.png', dpi=400)

frame_george.plot(title='George', x='year', y='prop')

plt.savefig('George.png', dpi=400)



