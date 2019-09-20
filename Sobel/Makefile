CC=gcc
LDFLAGS=-shared
EXEC=sobel.so

all : $(EXEC)

$(EXEC) : *.c
	$(CC) $(LDFLAGS) -o $@ -fPIC $^ 

mrproper : 
	rm -rf $(EXEC)