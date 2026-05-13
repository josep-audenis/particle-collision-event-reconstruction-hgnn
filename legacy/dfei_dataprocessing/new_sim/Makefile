# Compiler and flags
CXX = g++
CXXFLAGS = -g -std=c++11 -pedantic -W -Wall -Wshadow -fPIC

# Base paths
CVMFS_VIEW = /cvmfs/sft.cern.ch/lcg/views/LCG_107a/x86_64-el9-gcc13-opt
LOCAL_BASE = /afs/cern.ch/work/y/yukaiz/public/sim_packages_test

# Package paths
PYTHIA8 = $(LOCAL_BASE)
PYTHIA8DATA = $(PYTHIA8)/share/Pythia8/xmldoc
HEPMC = $(LOCAL_BASE)
EVTGEN = $(LOCAL_BASE)

# VDT and TBB paths
VDT_PATH = /cvmfs/sft.cern.ch/lcg/releases/vdt/0.4.4-48c41/x86_64-el9-gcc13-opt
TBB_PATH = /cvmfs/sft.cern.ch/lcg/releases/tbb/2021.10.0-2a247/x86_64-el9-gcc13-opt

# Include paths
INCLUDES = -I$(PYTHIA8)/include \
           -I$(EVTGEN)/include \
           -I$(HEPMC)/include

# Library paths and linking
LDFLAGS = -L$(EVTGEN)/lib64 -Wl,-rpath,$(EVTGEN)/lib64 -lEvtGen -lEvtGenExternal \
          -L$(HEPMC)/lib64 -Wl,-rpath,$(HEPMC)/lib64 -lHepMC3 \
          -L$(PYTHIA8)/lib -Wl,-rpath,$(PYTHIA8)/lib -lpythia8 \
          -L/lib64 -lssl -lcrypto \
          -L$(VDT_PATH)/lib -Wl,-rpath,$(VDT_PATH)/lib \
          $(shell root-config --cflags --glibs)

# Source files
SOURCES = gen.cc helpers.cc frag_helper.cc reco_helper.cc

# Target executable
TARGET = gen

# Default target
all: setup $(TARGET)

# Setup environment (mimics the source commands)
.PHONY: setup
setup:
	@echo "Note: Source the CVMFS setup before running make:"
	@echo "  source $(CVMFS_VIEW)/setup.sh"
	@echo "Set environment variables:"
	@echo "  export PYTHIA8=$(PYTHIA8)"
	@echo "  export PYTHIA8DATA=$(PYTHIA8DATA)"
	@echo "  export HEPMC=$(HEPMC)"
	@echo "  export EVTGEN=$(EVTGEN)"
	@echo "  export LD_LIBRARY_PATH=$(VDT_PATH)/lib:$(TBB_PATH)/lib64:$(PYTHIA8)/lib:$(EVTGEN)/lib64:$(HEPMC)/lib64:\$$LD_LIBRARY_PATH"

# Build target
$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(INCLUDES) $(LDFLAGS)

# Archive decfiles directory
.PHONY: archive
archive:
	tar -czf decfiles.tar.gz decfiles

# Clean target
.PHONY: clean
clean:
	rm -f $(TARGET) decfiles.tar.gz

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all (default) - Build the gen executable"
	@echo "  archive       - Create decfiles.tar.gz from decfiles directory"
	@echo "  clean         - Remove the gen executable and decfiles.tar.gz"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Before building, ensure you source the CVMFS setup:"
	@echo "  source $(CVMFS_VIEW)/setup.sh"
