CC = g++ -Wall
CFLAGS = `pkg-config --cflags --libs opencv` -fopenmp


all: haar haar_serial CDF_9_7_seriale CDF_9_7_parallela

haar: 
	$(CC) Haar_parallela.cpp -o haar_parallela $(CFLAGS)
	
haar_serial:
	$(CC) Haar_seriale.cpp -o haar_seriale $(CFLAGS)

CDF_9_7_seriale: 
	$(CC) CDF_9_7_seriale.cpp -o cdf_9_7_seriale $(CFLAGS)

CDF_9_7_parallela: 
	$(CC) CDF_9_7_parallela.cpp -o cdf_9_7_parallela $(CFLAGS)

clean:
	rm -f haar* cdf* *.o *~