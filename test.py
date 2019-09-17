import datetime as dt
start = dt.datetime.now()
t = dt.datetime.now()
i = 0
while True:
	delta=dt.datetime.now()-t
	if delta.seconds >= 60:
		i+=1
		print(i)
		t = dt.datetime.now()