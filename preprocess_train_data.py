import sys
import random

# segment data of file1, add the segmented data to file0
def add_train_data(file0, file1, label):
	outfile = open(file0, "a")
	infile = open(file1, "r")

	lines = infile.readlines()
	for line in lines:
		waves = line.split('\t')
		wave0 = waves[0].split(',')

		# omit the extremely quite data
		if abs(float(wave0[0])) < 0.1:
			continue

		outfile.write(label)
		outfile.write(':') 
		outfile.write(line)


def shuffle_train_data(file):
	infile = open(file, "r")
	lines = infile.readlines()
	infile.close()

	outfile = open(file, "w")
	random.shuffle(lines)
	for line in lines:
		outfile.write(line)




if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "usage: merge outfile infile label\n"
		print "usage: shuffle outfile"

	else:
		if sys.argv[1] == "merge":
			add_train_data(sys.argv[2], sys.argv[3], sys.argv[4])

		elif sys.argv[1] == "shuffle":
			shuffle_train_data(sys.argv[2])

		else:
			print "wrong parameter"