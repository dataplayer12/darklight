import time

class Timer(object):
	def __init__(self):
		self.stime=time.time()
		self.etime=self.stime
		self.spause=[]
		self.epause=[]

	def pause(self):
		self.spause.append(time.time())

	def unpause(self):
		self.epause.append(time.time())

	def reset(self):
		self.stime=time.time()
		self.etime=self.stime
		self.spause=[]
		self.epause=[]

	def tick(self, full=False):
		self.etime=time.time()
		if full:
			ptime=0
		else:
			ptime=sum([e-s for (s,e) in zip(self.spause, self.epause)])
			
		duration=(self.etime-self.stime)-ptime
		return duration