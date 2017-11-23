import scipy
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from scipy.signal import hann
from scipy.fftpack import rfft
from math import log10
import math as math
import numpy as np
import peak_detect as dp

blockSize = 512
SCMax = 0;


def fftMag(array):
	window = hann(len(array))
	X = rfft(array*window)
	#print len(X)
	# X holds the complex-valued samples of the spectrum of the input data
	# with X[0] real DC component
	# with X[-1] real Nyquist component
	# in-between the above there are alternating real and imaginary parts of the complex spectrum

	# magnitude spectrum:
	# if a,b is a point in the complex plane,
	# then the magnitude is the vector from the origin to this point
	# sqrt( a*a + b*b )

	mags = []
	for k in range(len(X)/2-1):
	    if k == 0:
	        mags.append(X[0])
	    else:
	        k = k * 2 - 1
	        mags.append((scipy.sqrt(X[k]*X[k] + X[k+1]*X[k+1])))
	    
	mags.append(X[-1])

	mags = 20 * scipy.log10(mags)
	# normalise to 0 dB max
	#mags -= max(mags)
	return mags


def magListFile(audioFile):
	magTempList = []
	for i in range(0, len(audioFile)-blockSize, blockSize):
		fftMags = fftMag(audioFile[i:i+blockSize]);
		magTempList.append([num.real for num in fftMags])

	
	return magTempList


#2
def maxPower(l):
	return max([int(el)*int(el) for el in l])

def maxPowerFile(audioFile):
	#get power of each block
	powerTempList = [];
	for i in range(0, len(audioFile)-blockSize, blockSize):
		powerTempList.append(maxPower(audioFile[i:i+blockSize]))

	#power of remainder
	powerTempList.append(maxPower(audioFile[len(audioFile)-(len(audioFile)%blockSize):len(audioFile)]))
	return powerTempList



#3
def rms(extracted):
	rmq = 0.0;
	temp = 0;
	for el in extracted:
		temp += (int(el)*int(el));
	rmq = math.sqrt(temp/blockSize);
	return rmq;

def rmsFile(audioFile):
	#used to call rms for an audio file 
	rmsTempList = []
	for i in range(0, len(audioFile)-blockSize, blockSize):
		rmsTempList.append(rms(audioFile[i:i+blockSize]))

	#power of remainder
	rmsTempList.append(rms(audioFile[len(audio)-(len(audioFile)%blockSize):len(audioFile)]))
	return rmsTempList

#4 
def spectralCentroid(mag):
	tempTop = 0;
	tempBottom = 0;
	sc = 0.0
	for k in range(0,len(mag)/2):
		tempTop += k * (mag[k] * mag[k])
		tempBottom += mag[k] * mag[k]
	if(tempBottom <= 0):
		tempBottom = 1
	sc = tempTop/tempBottom

	
	return sc



def plotData(figure, data, title, ylabel = "", xlabel = ""):
	plt.figure(figure)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.title(title)
	plt.grid(True)
	plt.plot(data)

#6
#max sliding window 
def maxSlidingWindow(pList):
	#list for max power per window 
	maxList = []
	for i in range(1, len(pList)):
		maxList.append(pow(pList[i],2) - pow(pList[i-1],2));
	return maxList; 

#7
#first order difference for high frequency 
def firstOrderDif(mag):
	firstOrder = 0.0;
	for k in range(1, len(mag)):
		firstOrder += mag[k] * k; 
	return firstOrder;

#8
#bandwise difference 
def bandwiseDifference(mag, magMinOne): 
	freqDomain = 0.0;
	for k in range(1,len(mag)):
		if((pow(mag[k],2) - pow(magMinOne[k],2)) < 0):
			continue; 
		else:
			freqDomain += pow(mag[k],2) - pow(magMinOne[k]-1,2)
	return freqDomain;

def onsetTimes(numSeconds, peakList,originList,totalSamples):
	temp = []
	global blockSize
	for i in range(len(peakList)):
		temp.append((float(peakList[i] * blockSize) / float(totalSamples)) * numSeconds)
	return temp



# read audio samples
input_data = read("flute.wav")
audio = input_data[1]
fluteSeconds = len(input_data[1])/input_data[0]
input_data2 = read("voice.wav")
audioVoice = input_data2[1]
voiceSeconds = len(input_data2[1])/input_data2[0]
input_data3 = read("drumrec16m.wav")
audioDrum = input_data3[1]
drumSeconds = len(input_data3[1])/input_data3[0]
input_data4 = read("Sine.wav")
audioVocal = input_data4[1]
vocalSeconds = len(input_data4[1])/input_data4[0]
input_data5 = read("guitar.wav")
audioGuitar = input_data5[1]
input_data6 = read("sine.wav")
audioSine = input_data6[1]
sineSeconds = len(input_data6[1])/input_data6[0]

#sampling rate = input data 0
#in seconds input data 1 / inputdata 0 


# ************************* FLUTE **************************

#list of powers
powerListFlute = [];
#list of RMS
rmsListFlute = [];
## generate list of fft's (spectrogram)
magListFlute = [];
#max sliding window list
MSWFlute = [];
#First order difference list 
FODFlute= [];
#spectral centroid flute list 
scListFlute = [];
#bandwise difference for flute 
bdListFlute = []; 
#onset times 
onsetDrum1 = []
onsetDrum2 = []
onsetDrum3 = []



#get power of each block
# for i in range(0, len(audio)-blockSize, blockSize):
# 	powerList.append(maxPower(audio[i:i+blockSize]))

# #power of remainder
# powerList.append(maxPower(audio[len(audio)-(len(audio)%blockSize):len(audio)]))

powerListFlute = maxPowerFile(audio)
rmsListFlute = rmsFile(audio)
magListFlute = magListFile(audio)
MSWFlute = maxSlidingWindow(powerListFlute);

for i in range(len(magListFlute)):
	FODFlute.append(firstOrderDif(magListFlute[i]))
	scListFlute.append(spectralCentroid(magListFlute[i]))
	
for i in range(1,len(magListFlute)):
	bdListFlute.append(bandwiseDifference(magListFlute[i],magListFlute[i-1]))


#PLOT FLUTE 

plt.subplot(5,2,1)
plotData(1, powerListFlute, "Flute Max Power", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot RMS for Flute 
plt.subplot(5,2,2)
plotData(1, rmsListFlute, "Flute RMS", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot Spectral Centroid 
plt.subplot(5,2,3)
plotData(1, scListFlute, "Flute Spectral Centroid")

#plot max sliding window 
plt.subplot(5,2,4)
plotData(1,MSWFlute, "Max Sliding Window");

#plot first order difference
plt.subplot(5,2,5)
plotData(1,FODFlute, "First Order Difference");

#plot bandwise difference 
plt.subplot(5,2,6)
plotData(1,bdListFlute, "Bandwise difference");

#plot peak detection 
plt.subplot(5,2,7)
plt.figure(1)
peakFlute1 = dp.detect_peaks(MSWFlute, show = True, ax = plt)
onsetFlute1 = onsetTimes(fluteSeconds, peakFlute1,MSWFlute,len(audio))

plt.subplot(5,2,8)
plt.figure(1)
peakFlute2 = dp.detect_peaks(FODFlute, show = True, ax = plt)
onsetFlute2 = onsetTimes(fluteSeconds, peakFlute2,FODFlute,len(audio))

plt.subplot(5,2,9)
plt.figure(1)
peakFlute3 = dp.detect_peaks(bdListFlute, show = True, ax = plt)
onsetFlute3 = onsetTimes(fluteSeconds, peakFlute3,bdListFlute,len(audio))

plt.show

# ********************** DRUM ***************************
#list of powers
powerListDrum = [];
#list of RMS
rmsListDrum = [];
## generate list of fft's (spectrogram)
magListDrum = [];
#max sliding window list
MSWDrum = [];
#First order difference list 
FODDrum= [];
#spectral centroid Drum list 
scListDrum = [];
#bandwise difference for Drum 
bdListDrum = []; 
#onset times 
onsetDrum1 = []
onsetDrum2 = []
onsetDrum3 = []

powerListDrum = maxPowerFile(audioDrum)
rmsListDrum = rmsFile(audioDrum)
magListDrum = magListFile(audioDrum)
MSWDrum = maxSlidingWindow(powerListDrum);

for i in range(len(magListDrum)):
	FODDrum.append(firstOrderDif(magListDrum[i]))
	scListDrum.append(spectralCentroid(magListDrum[i]))
	
for i in range(1,len(magListDrum)):
	bdListDrum.append(bandwiseDifference(magListDrum[i],magListDrum[i-1]))



#PLOT Drum

#plot maxPower for drum
plt.figure(2) 
plt.subplot(5,2,1)
plotData(2, powerListDrum, "Drum Max Power", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot RMS for drum
plt.subplot(5,2,2)
plotData(2, rmsListDrum, "Drum RMS", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot Spectral Centroid 
plt.subplot(5,2,3)
plotData(2, scListDrum, "Drum Spectral Centroid")

#plot max sliding window 
plt.subplot(5,2,4)
plotData(2,MSWDrum,"Max Sliding Window");

#plot first order difference
plt.subplot(5,2,5)
plotData(2,FODDrum, "First Order Difference");

#plot bandwise difference 
plt.subplot(5,2,6)
plotData(2,bdListDrum, "Bandwise difference");

#plot peak detection 
plt.subplot(5,2,7)
plt.figure(2)
peakDrum1 = dp.detect_peaks(MSWDrum, show = True, ax = plt)
onsetDrum1 = onsetTimes(drumSeconds, peakDrum1,MSWDrum,len(audioDrum))

plt.subplot(5,2,8)
plt.figure(2)
peakDrum2 = dp.detect_peaks(FODDrum, show = True, ax = plt)
onsetDrum2 = onsetTimes(drumSeconds, peakDrum2,FODDrum,len(audioDrum))


plt.subplot(5,2,9)
plt.figure(2)
peakDrum3 = dp.detect_peaks(bdListDrum, show = True, ax = plt)
onsetDrum3 = onsetTimes(drumSeconds, peakDrum3,bdListDrum,len(audioDrum))

plt.show()


# ********************** Voice ***************************
#list of powers
powerListVoice = [];
#list of RMS
rmsListVoice = [];
## generate list of fft's (spectrogram)
magListVoice = [];
#max sliding window list
MSWVoice = [];
#First order difference list 
FODVoice= [];
#spectral centroid Voice list 
scListVoice = [];
#bandwise difference for Voice
bdListVoice = []; 
#onsetTimes
onsetVoice1 = []
onsetVoice2 = []
onsetVoice3 = []


powerListVoice = maxPowerFile(audioVoice)
rmsListVoice = rmsFile(audioVoice)
magListVoice = magListFile(audioVoice)
MSWVoice = maxSlidingWindow(powerListVoice);

for i in range(len(magListVoice)):
	FODVoice.append(firstOrderDif(magListVoice[i]))
	scListVoice.append(spectralCentroid(magListVoice[i]))
	
for i in range(1,len(magListVoice)):
	bdListVoice.append(bandwiseDifference(magListVoice[i],magListVoice[i-1]))



#PLOT Voice

#plot maxPower for Voice
plt.figure(3) 
plt.subplot(5,2,1)
plotData(3, powerListVoice, "Voice Max Power", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot RMS for Voice 
plt.subplot(5,2,2)
plotData(3, rmsListVoice, "Voice RMS", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot Spectral Centroid 
plt.subplot(5,2,3)
plotData(3, scListVoice,  "Voice Spectral Centroid")

#plot max sliding window 
plt.subplot(5,2,4)
plotData(3,MSWVoice, "Max Sliding Window");

#plot first order difference
plt.subplot(5,2,5)
plotData(3,FODVoice, "First Order Difference");

#plot bandwise difference 
plt.subplot(5,2,6)
plotData(3,bdListVoice, "Bandwise difference");

#plot peak detection 
plt.subplot(5,2,7)
plt.figure(3)
peakVoice1 = dp.detect_peaks(MSWVoice, show = True, ax = plt)
onsetVoice1 = onsetTimes(voiceSeconds, peakVoice1,MSWVoice,len(audioVoice))

plt.subplot(5,2,8)
plt.figure(3)
peakVoice2 = dp.detect_peaks(FODVoice, show = True, ax = plt)
onsetVoice2 = onsetTimes(voiceSeconds,  peakVoice2,FODVoice,len(audioVoice))

plt.subplot(5,2,9)
plt.figure(3)
peakVoice3 = dp.detect_peaks(bdListVoice, show = True, ax = plt)
onsetVoice3 = onsetTimes(voiceSeconds, peakVoice3,bdListVoice,len(audioVoice))



plt.show()

# ********************** Vocal ***************************
#list of powers
powerListVocal = [];
#list of RMS
rmsListVocal = [];
## generate list of fft's (spectrogram)
magListVocal = [];
#max sliding window list
MSWVocal = [];
#First order difference list 
FODVocal= [];
#spectral centroidVocal list 
scListVocal = [];
#bandwise difference forVocal 
bdListVocal = []; 
#onset Times
onsetVocal1 = []
onsetVocal2 = []
onsetVocal3 = []


powerListVocal = maxPowerFile(audioVocal)
rmsListVocal = rmsFile(audioVocal)
magListVocal = magListFile(audioVocal)
MSWVocal = maxSlidingWindow(powerListVocal);

for i in range(len(magListVocal)):
	FODVocal.append(firstOrderDif(magListVocal[i]))
	scListVocal.append(spectralCentroid(magListVocal[i]))
	
for i in range(1,len(magListVocal)):
	bdListVocal.append(bandwiseDifference(magListVocal[i],magListVocal[i-1]))



#PLOT Vocal

#plot maxPower for Vocal
plt.figure(4) 
plt.subplot(5,2,1)
plotData(4, powerListVocal, "Vocal Max Power", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot RMS forVocal
plt.subplot(5,2,2)
plotData(4, rmsListVocal, "Vocal RMS", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot Spectral Centroid 
plt.subplot(5,2,3)
plotData(4, scListVocal,  "Vocal Spectral Centroid")

#plot max sliding window 
plt.subplot(5,2,4)
plotData(4,MSWVocal, "Max Sliding Window");

#plot first order difference
plt.subplot(5,2,5)
plotData(4,FODVocal, "First Order Difference");

#plot bandwise difference 
plt.subplot(5,2,6)
plotData(4,bdListVocal, "Bandwise difference");

#plot peak detection 
plt.subplot(5,2,7)
plt.figure(4)
peakVocal1 = dp.detect_peaks(MSWVocal, show = True, ax = plt)
onsetVocal1 = onsetTimes(vocalSeconds,  peakVocal1,MSWVocal,len(audioVocal))

plt.subplot(5,2,8)
plt.figure(4)
peakVocal2 = dp.detect_peaks(FODVocal, show = True, ax = plt)
onsetVocal2 = onsetTimes(vocalSeconds,  peakVocal2,FODVocal,len(audioVocal))

plt.subplot(5,2,9)
plt.figure(4)
peakVocal3 = dp.detect_peaks(bdListVocal, show = True, ax = plt)
onsetVocal3 = onsetTimes(vocalSeconds,  peakVocal3,bdListVocal,len(audioVocal))



plt.show()



# ********************** Sine ***************************
#list of powers
powerListSine = [];
#list of RMS
rmsListSine = [];
## generate list of fft's (spectrogram)
magListSine = [];
#max sliding window list
MSWSine = [];
#First order difference list 
FODSine= [];
#spectral centroid Sine list 
scListSine = [];
#bandwise difference for Sine
bdListSine = []; 
#onset Times
onsetSine1 = []
onsetSine2 = []
onsetSine3 = []

powerListSine = maxPowerFile(audioSine)
rmsListSine = rmsFile(audioSine)
magListSine = magListFile(audioSine)
MSWSine = maxSlidingWindow(powerListSine);

for i in range(len(magListSine)):
	FODSine.append(firstOrderDif(magListSine[i]))
	scListSine.append(spectralCentroid(magListSine[i]))
	
for i in range(1,len(magListSine)):
	bdListSine.append(bandwiseDifference(magListSine[i],magListSine[i-1]))



#PLOT Sine

#plot maxPower for Sine
plt.figure(5) 
plt.subplot(5,2,1)
plotData(5, powerListSine, "Sine Max Power", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot RMS for Flute 
plt.subplot(5,2,2)
plotData(5, rmsListSine, "Sine RMS", "Amplitude", "Time (bins of "+str(blockSize)+")")
#plot Spectral Centroid 
plt.subplot(5,2,3)
plotData(5, scListSine,  "Sine Spectral Centroid")

#plot max sliding window 
plt.subplot(5,2,4)
plotData(5,MSWSine, "Max Sliding Window");

#plot first order difference
plt.subplot(5,2,5)
plotData(5,FODSine, "First Order Difference");

#plot bandwise difference 
plt.subplot(5,2,6)
plotData(5,bdListSine, "Bandwise difference");

#plot peak detection 
plt.subplot(5,2,7)
plt.figure(5)
peakSine1 = dp.detect_peaks(MSWSine, show = True, ax = plt)
onsetSine1 = onsetTimes(sineSeconds,  peakSine1,MSWSine,len(audioSine))

plt.subplot(5,2,8)
plt.figure(5)
peakSine2 = dp.detect_peaks(FODSine, show = True, ax = plt)
onsetSine2 = onsetTimes(sineSeconds,  peakSine2,FODSine,len(audioSine))

plt.subplot(5,2,9)
plt.figure(5)
peakSine3 = dp.detect_peaks(bdListSine, show = True, ax = plt)
onsetSine3 = onsetTimes(sineSeconds,  peakSine3,bdListSine,len(audioSine))




plt.show()