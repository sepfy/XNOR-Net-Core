OUTDIR = objs
SRCDIR = src
SAMPLE = samples

SRC = $(wildcard $(SRCDIR)/*.cc)
OBJ = $(addsuffix .o, $(basename $(patsubst $(SRCDIR)/%,$(OUTDIR)/%,$(SRC))))

CXXFLAGS = -O2 -std=c++11
INCLUDE = -I ./include/
LIBS = -lm -fopenmp
LIB = libxnnc.a

all: mnist lib 

mnist: lib
	$(CXX) $(CXXFLAGS) -Wno-unused-result $(INCLUDE) $(LIBS) $(SAMPLE)/mnist_train.cc $(LIB) -o $(SAMPLE)/mnist_train
	$(CXX) $(CXXFLAGS) -Wno-unused-result $(INCLUDE) $(LIBS) $(SAMPLE)/mnist_deploy.cc $(LIB) -o $(SAMPLE)/mnist_deploy

lib: $(OBJ)
	$(AR) rvs $(LIB) $^
	chmod 777 $(LIB)

$(OUTDIR)/%.o: $(SRCDIR)/%.cc $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -c $< -o $@ 

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(SAMPLE)/mnist_deploy $(SAMPLE)/mnist_train $(OUTDIR) libxnnc.a

.PHONY: clean $(OUTDIR)

