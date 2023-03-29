# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

CXX ?= g++
OUT ?= .
OUTNAME ?= libpoptorch_dynamic_sparse_op.so
CXXFLAGS = -Wall -Wextra -Werror -std=c++17 -O2 -g -fPIC -DONNX_NAMESPACE=onnx

$(OUT)/libpoptorch_dynamic_sparse_op.so: libpoptorch_dynamic_sparse_op.cpp
	mkdir -p $(OUT) && $(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined -lpoplar -lpopart -lpopops -lpopsparse -lpoputil

all: $(OUT)/libpoptorch_dynamic_sparse_op.so
