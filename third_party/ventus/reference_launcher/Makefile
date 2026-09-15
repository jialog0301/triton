VENTUS_INSTALL_PREFIX ?= /home/weijiale/Code/cuda2rvv/ventus-env/install
CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror

TARGET := ventus_spike_smoke
SRC := ventus_spike_smoke.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -I$(VENTUS_INSTALL_PREFIX)/include $< \
		-L$(VENTUS_INSTALL_PREFIX)/lib -Wl,-rpath,$(VENTUS_INSTALL_PREFIX)/lib \
		-lspike_driver -o $@

clean:
	rm -f $(TARGET)
