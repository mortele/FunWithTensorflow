import os
import numpy 		as np

nR 		= 100#int(1e5)
r12 	= np.linspace(1.0,10.0, nR)

def writeToInputFile(r) :
	with open("input", "w") as inputFile :
		inputFile.write("! DFT-Energy+ ENGRAD\n")
		#inputFile.write("! UHF ENGRAD\n")
		inputFile.write("! BOHRS\n")
		inputFile.write("* xyz 0 1\n")
		inputFile.write("  O 0 0 0\n")
		inputFile.write("  O 0 0 %f\n" % (r))
		inputFile.write("*")

def readFromOutputFile(inFileName) :
	E  = 0
	r1 = [0, 0, 0]
	r2 = [0, 0, 0]
	g1 = [0, 0, 0]
	g2 = [0, 0, 0]
	with open(inFileName, "r") as inFile :
		lines 	= inFile.readlines()
		E 		= float(lines[7].split('\n')[0])

		g1[0] 	= float(lines[11].split('\n')[0])
		g1[1] 	= float(lines[12].split('\n')[0])
		g1[2] 	= float(lines[13].split('\n')[0])

		g2[0] 	= float(lines[14].split('\n')[0])
		g2[1] 	= float(lines[15].split('\n')[0])
		g2[2] 	= float(lines[16].split('\n')[0])

		r 		= lines[20].split('\n')[0].split()
		r1[0] 	= float(r[1])
		r1[1] 	= float(r[2])
		r1[2] 	= float(r[3])

		r 		= lines[21].split('\n')[0].split()
		r1[0] 	= float(r[1])
		r1[1] 	= float(r[2])
		r1[2] 	= float(r[3])
	return E,r1,r2,g1,g2


for i in range(len(r12)) :
	print("Progress: %d of %d runs complete. r12=%f" % (i,nR,r12[i]))
	writeToInputFile(r12[i])
	if os.system("/Users/morten/Library/Orca/orca input > output") != 0 :
		print("Orca run aborted. Exiting.")
		exit()

	E,r1,r2,g1,g2 = readFromOutputFile("input.engrad")

	fileFlag = None
	if i==0 :
		fileFlag = "w"
	else :
		fileFlag = "a"
	with open("qdata.dat", fileFlag) as outFile :
		outFile.write("%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f\n" % \
				(E, \
				r1[0], r1[1], r1[2], \
				r2[0], r2[1], r2[2], \
				g1[0], g1[1], g1[2], \
				g2[0], g2[1], g2[2]))
