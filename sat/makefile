OUTPUT = sat.out
CFLAGS = -g
LDFLAGS = -lm

CC = nvcc
CFORMAT = cu
MAIN = main.o
TEST = test.o
OBJECTS = sat.o loader.o dependence.o
MAIN_OBJECTS = $(MAIN) $(OBJECTS)
TEST_OBJECTS = $(TEST) $(OBJECTS)

# targets
sat: $(MAIN_OBJECTS)
	$(CC) -o $(OUTPUT) $(CFLAGS) $(MAIN_OBJECTS) $(LDFLAGS)

test: $(TEST_OBJECTS)
	$(CC) -o $(OUTPUT) $(CFLAGS) $(TEST_OBJECTS) $(LDFLAGS)

# objects
main.o: config.h loader.h sat.h main.$(CFORMAT)
	$(CC) -o main.o $(CFLAGS) -c main.$(CFORMAT)

test.o: config.h loader.h sat.h test.$(CFORMAT)
	$(CC) -o test.o $(CFLAGS) -c test.$(CFORMAT)

sat.o: config.h sat.h
	$(CC) -o sat.o $(CFLAGS) -c sat.$(CFORMAT)

loader.o: config.h loader.h sat.h
	$(CC) -o loader.o $(CFLAGS) -c loader.$(CFORMAT)

dependence.o: dependence.h
	$(CC) -o dependence.o $(CFLAGS) -c dependence.$(CFORMAT)

.PHONY: clean
clean:
	-rm $(OUTPUT) $(OBJECTS)
	-rm $(MAIN)
	-rm $(TEST)