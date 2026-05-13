CXX = g++
CXXFLAGS = $(shell root-config --cflags)
LIBS = $(shell root-config --libs)

missasso: main.cpp helper.cpp helper.h
	clear
	$(CXX) $(CXXFLAGS) -o main main.cpp helper.cpp root_helper.cpp $(LIBS)

clean:
	rm -f main