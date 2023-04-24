from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import math, struct, cv2

# Set dataframe filepath
filepath = open("./datasets/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# Dataframe
#print(df)
#print(len(df))

# The first column name
#print(df.columns[0])

# The last column name
#print(df.columns[78])
def dataframeFloat2RGBHex(dataframe):
	newDataframe = []

	for df in dataframe:
		newDf = [ hex(struct.unpack('<I', struct.pack('<f', value))[0])[2:].zfill(8) for value in df ]
		newDataframe.append(newDf)

	return newDataframe

class Dataframe2RGB:
	def __init__(self, filepath):
		self.benign = []
		self.ddos   = []

		self.benignRGB = []
		self.ddosRGB   = []

		# Get Dataframe from csv file
		self.dataframe = pd.read_csv(filepath)

		# Init Dataframe
		self.init()

	def init(self):

		# Delete raw which has Nan or Infinity
		self.dataframe = self.dataframe.replace([np.inf, -np.inf], np.nan)
		self.dataframe = self.dataframe.dropna()

		# Set Label column name
		columnName = ' Label'

		# Init benign/malicious dataframe
		benignDataframe = self.dataframe.loc[ self.dataframe[columnName] == 'BENIGN' ]
		ddosDataframe   = self.dataframe.loc[ self.dataframe[columnName] == 'DDoS'   ]

		# Drop Label column
		self.benign = benignDataframe.drop(columns=[columnName])
		self.ddos   = ddosDataframe  .drop(columns=[columnName])

		#print(self.benign)
		#print(self.ddos)

		# Set rgbRange
		rgbRange =  256 * 256 * 256

		# Rescale benign/ddos frame
		s = MinMaxScaler(feature_range=(0, rgbRange))

		s.fit(self.benign)
		self.benignRGB = dataframeFloat2RGBHex(s.transform(self.benign))

		s.fit(self.ddos)
		self.ddosRGB = dataframeFloat2RGBHex(s.transform(self.ddos))

# Generate Dataframe2RGB
df = Dataframe2RGB(filepath)

#print(df.benignRGB[0])

columnLength = len(df.benign.columns)

width  = int(math.sqrt(columnLength))
height = int( columnLength / width ) + 1

#print('len:', len(df.benign.columns)) 
#print('w: %d, h: %d' % (width, height) )

scale  = 400
img = np.zeros( (height*scale, width*scale, 3), np.uint8 )

thickness = 1
columnFontScale = 1/2
font = cv2.FONT_HERSHEY_SIMPLEX

indexNum = 96

for ci in range(columnLength):
	rgb = df.benignRGB[indexNum][ci]	# [record index] [column index]
	x, y = int(ci % width), int(ci / width)

#	print(x,y)
#	print(int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))

	# Draw rects
	cv2.rectangle(	img,
					(scale*x, scale*y),
					(scale*(x+1), scale*(y+1)),
					(int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16)),
					cv2.FILLED	)
	# Get value text and textSize
	text = str(df.benign.values[indexNum][ci])
	textsize = cv2.getTextSize(text, font, 1, thickness)[0]

	# Put info text
	cv2.putText(img, text, ( int(x*scale + (scale - textsize[0])/2) , int(scale*(y+0.5)) ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness)
	cv2.putText(img, df.benign.columns[ci], ( scale*x ,int(scale*(y+0.6)) ), font, columnFontScale, (255,255,255), thickness)

#cv2.imshow('test', img)

cv2.imwrite("output.jpg", img)
