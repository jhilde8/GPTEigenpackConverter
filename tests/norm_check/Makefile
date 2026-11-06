HADRONS_CONFIG ?= hadrons-config

CXX      := $(shell $(HADRONS_CONFIG) --cxx)
CXXFLAGS := $(shell $(HADRONS_CONFIG) --cxxflags)
LDFLAGS  := $(shell $(HADRONS_CONFIG) --ldflags)
LIBS     := $(shell $(HADRONS_CONFIG) --libs)

Test_norm_check: Test_norm_check.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f Test_norm_check

.PHONY: clean
