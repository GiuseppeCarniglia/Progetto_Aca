CC = g++ -Wall
CFLAGS = `pkg-config --cflags --libs opencv` -fopenmp


all: haar haar_serial

haar: 
	$(CC) Haar.cpp -o haar $(CFLAGS)
	
haar_serial:
	$(CC) Haar_serial.cpp -o haar_serial $(CFLAGS)

clean:
	rm -f haar* *.o *~
