import numpy as np
import cv2 
import time
import math
import HVal


#Case 1: No diagonal movement allowed
#Case 2: Diagonal movement allowed

case = 2
algo = 4

# DIJSKTRA    --> 1
# A*EUC       --> 2 
# A*2         --> 3 Heuristic = Euclidean distance of current cell + Euclidean distance of parent cell upto the target cell
# A*3         --> 4 Heuristic = 0.55*(l/L) + 0.45*(theta/alpha)  
#								Where, l = euclidean distance of current cell to target
#									   L = euclidean distance of start cell to target cell.
#									   theta = Angle made by line joining current and target cell with horizontal
#									   alpha = Angle made by line joining start and target cell with horizontal
# A*Manhattan --> 5	
# A*Diagonal  --> 6

z=1  #f = g + zh

cv2.namedWindow("unsolved", cv2.WINDOW_NORMAL)
maze = cv2.imread("maze.png", 1)
cv2.imshow("unsolved", maze)

cv2.namedWindow("solved", cv2.WINDOW_NORMAL)

h,w,c = maze.shape
arr = np.zeros((100,100)) #creating the array which represents the maze with zeroes(non-navigable) and ones(navigable)

for i in range(100):
	for j in range(100):
		if maze[i*10][j*10][0]==0: 		#detects black cells
			arr[i][j] = 1	
		elif maze[i*10][j*10][0]==113:	#detecs the starting cell
			arr[i][j] = 1
			start = (i,j)
		elif maze[i*10][j*10][0]==60:	#detects the target cell
			arr[i][j] = 1
			end = (i,j)
		else:
			arr[i][j]=0					#white non navigable cells

class cell:    #Class which stores details of every cell
	def setparents(self,pi,pj): #assigns parent cell coordinates to the current cell
		self.pi = pi
		self.pj = pj
	def setfgh(self,f,g,h):  	#assigns f,g,h values of the current cell
		self.f = f
		self.g = g
		self.h = h
	def __init__(self, i, j):	#constructor, assigns the cell coordinates
		self.i = i
		self.j = j

def isValid(i,j):	#returns true if a coordinate (i,j) is a valid one
	return i>=0 and i<100 and j>=0 and j<100

def isUnblock(i,j):	#returns true if a cell (i,j) is navigable
	return arr[i][j]==1.0

def isEnd(i,j):		#returns true if (i,j) is the target cell
	return (i,j) == end

def colorOrange(x,y):	#colors a cell orange
	x = x*10
	y = y*10
	for p in range(10):
		for q in range(10):
			maze[x+p][y+q][0]=52
			maze[x+p][y+q][1]=149
			maze[x+p][y+q][2]=235

def colorGreen(x,y):	#colors a cell green
	x = x*10
	y = y*10
	for p in range(10):
		for q in range(10):
			maze[x+p][y+q][0]=52
			maze[x+p][y+q][1]=235
			maze[x+p][y+q][2]=79
	
def colorBlue(x,y):	#colors a cell blue
	x = x*10
	y = y*10
	for p in range(10):
		for q in range(10):
			maze[x+p][y+q][0]=235
			maze[x+p][y+q][1]=88
			maze[x+p][y+q][2]=52

closedList = np.full((100,100), False) #has a true value if the cell is explored, else true. Initially all cells are unexplored.

cellDet = []

for m in range(100):
	cellDet.append([])
	for n in range(100):
		cellDet[m].append(cell(m,n))   #Stores cell class objects in a 2D list


	
for m in range(100):   
	for n in range(100):
		cellDet[m][n].setparents(-1, -1)	#initialises the parents of all cells as (-1,-1)
		cellDet[m][n].setfgh(float('inf'), float('inf'), float('inf')) #initialises the f,g,h values of all cells as infinite


def hValue(i,j):	#returns h-value
	if algo==1:
		return HVal.dijkstra(i,j,end)
	elif algo==2:
		return HVal.euc(i,j,end)
	elif algo==3:
		return HVal.euc(i,j,end)+HVal.euc(cellDet[i][j].pi, cellDet[i][j].pi, end)
	elif algo==4:
		return HVal.Astar3(i,j,start,end)
	elif algo==5:
		return HVal.manh(i,j,end)
	elif algo==6:
		return HVal.diag(i,j,end)


i=start[0]
j=start[1]

cellDet[start[0]][start[1]].setparents(start[0], start[1])
cellDet[start[0]][start[1]].setfgh(0, 0, 0)
openList = [] #stores cell coordinates and their h values which are yet to be explored
openList.append((0,(i,j))) #appends the starting cell to the open list
found = False


def inOpenList(i,j): #returns true value if the cell (i,j) is in the open list
	for x in range(len(openList)):
		if openList[x][1]==(i,j):
			return True
	return False

def posOpen(i,j):  #returns position of a cell (i,j) in the open list
	for x in range(len(openList)):
		if openList[x][1]==(i,j):
			return x

begin = time.time()



q=0 #no of cells explored

while openList:
	#print("1")
	min = 0 #keeps track of the position of the cell with minimum h value in the open list
	for k in range(len(openList)): 
		if openList[k][0]<openList[min][0]:
			min = k

	p = openList[min]
	openList.remove(openList[min])
	i,j = p[1]
	closedList[i][j] = True
	colorOrange(i,j) 
	q = q+1

	#NORTH (i,j) -> (i-1,j)
	if isValid(i-1,j):
		if isEnd(i-1,j):
			cellDet[i-1][j].setparents(i,j)
			print("DESTINATION REACHED")
			found = True
			print(f"Cost = {cellDet[i][j].g+1.0}")
			break
		elif closedList[i-1][j] == False and isUnblock(i-1,j) == True:
			gnew = cellDet[i][j].g + 1.0
			hnew = hValue(i-1,j)
			fnew = gnew + (z*hnew)
			if not inOpenList(i-1,j):
				openList.append((fnew,(i-1,j)))
				cellDet[i-1][j].setfgh(fnew, gnew, hnew)
				cellDet[i-1][j].setparents(i,j)
			else:
				if gnew < cellDet[i-1][j].g:
					cellDet[i-1][j].setparents(i,j)
					cellDet[i-1][j].setfgh(fnew, gnew, hnew)
					pos = posOpen(i-1,j)
					openList[pos]  = (fnew,(i-1,j))


				
	#SOUTH (i,j) -> (i+1,j)
	if isValid(i+1,j):
		if isEnd(i+1,j):
			cellDet[i+1][j].setparents(i,j)
			print("DESTINATION REACHED")
			found = True
			print(f"Cost = {cellDet[i][j].g+1.0}")
			break
		elif closedList[i+1][j] == False and isUnblock(i+1,j) == True:
			gnew = cellDet[i][j].g + 1.0
			hnew = hValue(i+1,j)
			fnew = gnew + (z*hnew)
			if not inOpenList(i+1,j):
				openList.append((fnew,(i+1,j)))
				cellDet[i+1][j].setfgh(fnew, gnew, hnew)
				cellDet[i+1][j].setparents(i,j)
			else:
				if gnew < cellDet[i+1][j].g:
					cellDet[i+1][j].setparents(i,j)
					cellDet[i+1][j].setfgh(fnew, gnew, hnew)
					pos = posOpen(i+1,j)
					openList[pos]= (fnew,(i+1,j))
				
				
	#EAST (i,j) -> (i,j+1)
	if isValid(i,j+1):
		if isEnd(i,j+1):
			cellDet[i][j+1].setparents(i,j)
			print("DESTINATION REACHED")
			found = True
			print(f"Cost = {cellDet[i][j].g+1.0}")
			break
		elif closedList[i][j+1] == False and isUnblock(i,j+1) == True:
			gnew = cellDet[i][j].g + 1.0
			hnew = hValue(i,j+1)
			fnew = gnew + (z*hnew)
			if not inOpenList(i,j+1):
				openList.append((fnew,(i,j+1)))
				cellDet[i][j+1].setfgh(fnew, gnew, hnew)
				cellDet[i][j+1].setparents(i,j)
			else:
				if gnew < cellDet[i][j+1].g:
					cellDet[i][j+1].setparents(i,j)
					cellDet[i][j+1].setfgh(fnew, gnew, hnew)
					pos = posOpen(i,j+1)
					openList[pos]= (fnew,(i,j+1))
				
	#WEST (i,j) -> (i,j-1)
	if isValid(i,j-1):
		if isEnd(i,j-1):
			cellDet[i][j-1].setparents(i,j)
			print("DESTINATION REACHED")
			found = True
			print(f"Cost = {cellDet[i][j].g+1.0}")
			break
		elif closedList[i][j-1] == False and isUnblock(i,j-1) == True:
			gnew = cellDet[i][j].g + 1.0
			hnew = hValue(i,j-1)
			fnew = gnew + (z*hnew)

			if not inOpenList(i,j-1):
				openList.append((fnew,(i,j-1)))
				cellDet[i][j-1].setfgh(fnew, gnew, hnew)
				cellDet[i][j-1].setparents(i,j)
			else:
				if gnew < cellDet[i][j-1].g:
					cellDet[i][j-1].setparents(i,j)
					cellDet[i][j-1].setfgh(fnew, gnew, hnew)
					pos = posOpen(i,j-1)
					openList[pos] = (fnew,(i,j-1))
	if case == 2:
		
		#NE (i,j) -> (i-1,j+1)
		if isValid(i-1,j+1):
			if isEnd(i-1,j+1):
				cellDet[i-1][j+1].setparents(i,j)
				print("DESTINATION REACHED")
				found = True
				print(f"Cost = {cellDet[i][j].g+1.414}")
				break
			elif closedList[i-1][j+1] == False and isUnblock(i-1,j+1) == True:
				gnew = cellDet[i][j].g + 1.414
				hnew = hValue(i-1,j+1)
				fnew = gnew + (z*hnew)

				if not inOpenList(i-1,j+1):
					openList.append((fnew,(i-1,j+1)))
					cellDet[i-1][j+1].setfgh(fnew, gnew, hnew)
					cellDet[i-1][j+1].setparents(i,j)
				else:
					if gnew < cellDet[i-1][j+1].g:
						cellDet[i-1][j+1].setparents(i,j)
						cellDet[i-1][j+1].setfgh(fnew, gnew, hnew)
						pos = posOpen(i-1,j+1)
						openList[pos] = (fnew,(i-1,j+1))
					
		#NW (i,j) -> (i-1,j-1)
		if isValid(i-1,j-1):
			if isEnd(i-1,j-1):
				cellDet[i-1][j-1].setparents(i,j)
				print("DESTINATION REACHED")
				found = True
				print(f"Cost = {cellDet[i][j].g+1.414}")
				break
			elif closedList[i-1][j-1] == False and isUnblock(i-1,j-1) == True:
				gnew = cellDet[i][j].g + 1.414
				hnew = hValue(i-1,j-1)
				fnew = gnew + (z*hnew)

				if not inOpenList(i-1,j-1):
					openList.append((fnew,(i-1,j-1)))
					cellDet[i-1][j-1].setfgh(fnew, gnew, hnew)
					cellDet[i-1][j-1].setparents(i,j)
				else:
					if gnew < cellDet[i-1][j-1].g:
						cellDet[i-1][j-1].setparents(i,j)
						cellDet[i-1][j-1].setfgh(fnew, gnew, hnew)
						pos = posOpen(i-1,j-1)
						openList[pos] = (fnew,(i-1,j-1))
		
		#SE (i,j) -> (i+1,j+1)
		if isValid(i+1,j+1):
			if isEnd(i+1,j+1):
				cellDet[i+1][j+1].setparents(i,j)
				print("DESTINATION REACHED")
				found = True
				print(f"Cost = {cellDet[i][j].g+1.414}")
				break
			elif closedList[i+1][j+1] == False and isUnblock(i+1,j+1) == True:
				gnew = cellDet[i][j].g + 1.414
				hnew = hValue(i+1,j+1)
				fnew = gnew + (z*hnew)

				if not inOpenList(i+1,j+1):
					openList.append((fnew,(i+1,j+1)))
					cellDet[i+1][j+1].setfgh(fnew, gnew, hnew)
					cellDet[i+1][j+1].setparents(i,j)
				else:
					if gnew < cellDet[i+1][j+1].g:
						cellDet[i+1][j+1].setparents(i,j)
						cellDet[i+1][j+1].setfgh(fnew, gnew, hnew)
						pos = posOpen(i+1,j+1)
						openList[pos] = (fnew,(i+1,j+1))
		
		#SW (i,j) -> (i+1,j-1)
		if isValid(i+1,j-1):
			if isEnd(i+1,j-1):
				cellDet[i+1][j-1].setparents(i,j)
				print("DESTINATION REACHED")
				found = True
				print(f"Cost = {cellDet[i][j].g+1.414}")
				break
			elif closedList[i+1][j-1] == False and isUnblock(i+1,j-1) == True:
				gnew = cellDet[i][j].g + 1.414
				hnew = hValue(i+1,j-1)
				fnew = gnew + (z*hnew)

				if not inOpenList(i+1,j-1):
					openList.append((fnew,(i+1,j-1)))
					cellDet[i+1][j-1].setfgh(fnew, gnew, hnew)
					cellDet[i+1][j-1].setparents(i,j)
				else:
					if gnew < cellDet[i+1][j-1].g:
						cellDet[i+1][j-1].setparents(i,j)
						cellDet[i+1][j-1].setfgh(fnew, gnew, hnew)
						pos = posOpen(i+1,j-1)
						openList[pos] = (fnew,(i+1,j-1))
	

p = 0 #length of the path				

if found:
	finish = time.time()
	print(f"Execution Time: {finish-begin}")
	x = cellDet[end[0]][end[1]].pi
	y = cellDet[end[0]][end[1]].pj
	while x!=start[0] or y!=start[1]:
		colorGreen(x,y)
		q = q+1
		p = p+1
		m=x
		n=y
		x = cellDet[m][n].pi
		y = cellDet[m][n].pj
		

	for i in range(len(openList)):
		colorBlue(openList[i][1][0], openList[i][1][1])

	print(f"\nNo of cells explored: {q}\nPath Length: {p}\nEfficiency: {p/q}")
	cv2.imshow("solved", maze)


cv2.waitKey(0)
cv2.destroyAllWindows()












