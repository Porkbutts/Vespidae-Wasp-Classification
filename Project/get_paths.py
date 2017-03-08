import os, sys

# script to generate absolute paths (and truth labels for training)
#
# usage: python get_paths.py <mode={0,1}> <dir>
#
# mode = 0 for training
# <dir> contains subdirectories of images for each species
# each line in the output is of the format:
# "FILE_PATH@LABEL"
#
# mode = 1 for testing
# <dir> contains the images
# each line in the output is of the format:
# "FILE_PATH"
#
# FILE_PATH is the absolute file path to the image,
# @ is a delimiter,
# LABEL is an integer corresponding to the class label

def main(argv):
	if len(argv) > 2:
		sys.exit("Usage: python get_paths.py <mode> <dir>")
	
	mode = int(argv[0])
	cd = argv[1]
	index = 0
	numImages = 0
	
	f = open('filepaths_and_labels.txt', 'w')
	f.write("                            \n") # replace with number of lines afterward
	
	if mode == 0:
		for dir in os.listdir(cd):
			dirPath = os.path.join(cd, dir)
			if os.path.isdir(dirPath):
				for file in os.listdir(dirPath):
					filePath = os.path.join(dirPath, file)
					f.write(filePath + "@" + repr(index) + "\n")
					numImages += 1
			index += 1
	elif mode == 1:
		for file in os.listdir(cd):
			filePath = os.path.join(cd, file)
			if os.path.isfile(filePath):
				f.write(filePath + "\n")
				numImages += 1
	else:
		sys.exit("<mode> = 0 for training, or 1 for testing")
	
	# seek to first line and write number of lines
	f.seek(0);
	f.write(repr(numImages));
	f.close()
					
if __name__ == "__main__":
	main(sys.argv[1:])