import datetime as dt

t = dt.datetime.now()

while True:
  delta=dt.datetime.now()-t
  if delta.seconds >= 60:
     print("1 Min")
     t = dt.datetime.now()