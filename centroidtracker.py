# import the necessary packages
import math
from collections import OrderedDict
import numpy as np
from collections import deque

UNDEFINED = 'UNDEFINED'
RIGHT = 'RIGHT'
LEFT = 'LEFT'
UP = 'UP'
BOTTOM = 'BOTTOM'


class Centroid():
	def __init__(self, x, y, centroidID, buffer_size=70):
		self.x = x
		self.y = y
		self.centroidID = centroidID
		self.detect = True
		self.orientation = [UNDEFINED, UNDEFINED] 
		self.buffer = deque(maxlen=buffer_size)
		self.bounce = False

	def update(self, x, y):
		if(x > self.x):
			self.orientation[0] = RIGHT
		elif(x < self.x):
			self.orientation[0] = LEFT
		
		if(y > self.y):
			self.orientation[1] = BOTTOM
		elif(y < self.y):
			self.orientation[1] = UP
		
		self.buffer.appendleft((self.orientation[0], self.orientation[1]))

		if len(self.buffer) >= 2:
			A = self.buffer[1]
			B = self.buffer[0]
			if A[1] is BOTTOM and B [1] is UP and not self.bounce:
				self.bounce = True
				print("SERÁ QUE QUICOU??")

		self.x = x
		self.y = y
		self.detect = True


class CentroidTracker():
	def __init__(self, maxDisappeared=150):
		self.nextObjectID = 0
		self.centroids = dict()
		self.maxDisappeared = maxDisappeared

	def register(self, x, y):
		centroid = Centroid(x, y, self.nextObjectID)
		self.centroids[self.nextObjectID] = centroid
		self.nextObjectID += 1

	def unregister(self, objectID):
		del self.centroids[objectID]

	def update(self, x, y):
		update = False
		for key in self.centroids.keys():
			if(math.hypot(x - self.centroids[key].x, y - self.centroids[key].y) < self.maxDisappeared):
				self.centroids[key].update(x, y)
				update = True
				break

		if update == False:
			self.register(x, y)

		self.removeAllUndetected()
		self.setAllUndetected()

	def setAllUndetected(self):
		for key in self.centroids.keys():
			self.centroids[key].detect = False

	def removeAllUndetected(self):
		for key in self.centroids.copy().keys():
			if(self.centroids[key].detect is not True):
				self.unregister(key)
