# Build the GTK4 Ollama client.
#
#   sudo apt install build-essential pkg-config libgtk-4-dev \
#                    libjson-glib-dev libcurl4-openssl-dev
#   make
#   ./ollama-gui

CC      ?= gcc
PKGS     = gtk4 json-glib-1.0 libcurl
CFLAGS  += -O2 -Wall -Wextra -Wno-unused-parameter $(shell pkg-config --cflags $(PKGS))
LDLIBS  += $(shell pkg-config --libs $(PKGS))

TARGET   = ollama-gui
SRC      = ollama_gui.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

deps:
	sudo apt install build-essential pkg-config libgtk-4-dev \
	                 libjson-glib-dev libcurl4-openssl-dev

clean:
	rm -f $(TARGET)

.PHONY: all deps clean
