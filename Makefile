OUTDIR = objs
SRCDIR = src
SAMPLE = samples

SRC = $(wildcard $(SRCDIR)/*.cc)
OBJS = $(addsuffix .o, $(basename $(patsubst $(SRCDIR)/%,$(OUTDIR)/%,$(SRC))))

CXXFLAGS = -O2 -std=c++11 -Wno-unused-result
INCLUDE = -I ./include/
LIBS = -lm -fopenmp
LIB = libxnnc.a

OPENCV = `pkg-config opencv --cflags --libs`

all: $(OUTDIR) $(LIB) samples

samples: mnist cifar lenet

test: $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) unittest/conn_test.cc $(LIB) -o unittest/conn_test
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) unittest/conv_test.cc $(LIB) -o unittest/conv_test
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) unittest/bn_test.cc $(LIB) -o unittest/bn_test

vgg: $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) $(SAMPLE)/vgg.cc $(LIB) $(OPENCV) -o $(SAMPLE)/$@

mnist: $(SAMPLE)/mnist.cc $(LIB) 
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) $^ -o $(SAMPLE)/$@

cifar: $(SAMPLE)/cifar.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) $^ $(OPENCV) -o $(SAMPLE)/$@

lenet: $(SAMPLE)/lenet.cc $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) $^ $(OPENCV) -o $(SAMPLE)/$@

$(LIB): $(OBJS) 
	$(AR) rcs $@ $(OBJS)

$(OUTDIR)/%.o: $(SRCDIR)/%.cc 
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -c $< -o $@ 

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf $(SAMPLE)/mnist $(SAMPLE)/cifar $(SAMPLE)/lenet $(OUTDIR) libxnnc.a

.PHONY: clean $(OUTDIR)

