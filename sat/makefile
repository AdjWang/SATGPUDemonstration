OUTPUT = sat.out
CFLAGS = -g -Wall
LDFLAGS = -lm

CC = cc
MAIN = main.o
TEST = test.o
OBJECTS = sat.o loader.o
MAIN_OBJECTS = $(MAIN) $(OBJECTS)
TEST_OBJECTS = $(TEST) $(OBJECTS)

# targets
sat: $(MAIN_OBJECTS)
	$(CC) -o $(OUTPUT) $(CFLAGS) $(MAIN_OBJECTS) $(LDFLAGS)

test: $(TEST_OBJECTS) test_input.txt
	$(CC) -o $(OUTPUT) $(CFLAGS) $(TEST_OBJECTS) $(LDFLAGS)\
	&& ./$(OUTPUT)

# objects
main.o: config.h loader.h sat.h main.c
	$(CC) -o main.o $(CFLAGS) -c main.c

test.o: config.h loader.h sat.h test.c
	$(CC) -o test.o $(CFLAGS) -c test.c

sat.o: config.h sat.h
	$(CC) -o sat.o $(CFLAGS) -c sat.c

loader.o: config.h loader.h sat.h
	$(CC) -o loader.o $(CFLAGS) -c loader.c

.PHONY: clean
clean:
	-rm $(OUTPUT) $(MAIN_OBJECTS)