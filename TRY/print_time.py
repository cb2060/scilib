#from datetime import datetime,timedelta 
#from datetime import timedelta,datetime
import datetime
#from datetime import today
#import time
#from datetime import now
#import date
import sys
d1=datetime.timedelta(hours=5)
d2=datetime.timedelta(seconds=2)
d=d1-d2
print(d.days,d.seconds,d.microseconds)
#d3=datetime.today()
#d3=datetime.utcnow()
d3=datetime.date(2015,12,31)
d4=datetime.date(2017,12,31)
t=d4-d3
print(t.days,t.seconds,t.microseconds)
#u=datetime.date(2000,12,05)
#u2=datetime.date(2001,12,05)
#sys.stdout.write("%s" % u.year)
#sys.stdout.write(u.day)
#print(datetime.timedelta(microseconds=u-u2))

