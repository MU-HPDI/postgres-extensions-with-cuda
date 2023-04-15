include .env

# Use variables for directories and files
SRCFILE = main
PG_SERVER_DIR = /usr/include/postgresql/14/server
PG_BINDIR = /tmp
BUILD_DIR = build

cuda_wrappers.o: cuda_funcs/cuda_wrappers.cu
	nvcc -Xcompiler -fPIC -c $< -o $(BUILD_DIR)/$@

host_wrapper.o: cpu_funcs/host_wrapper.cpp
	g++ -fPIC -c $< -o $(BUILD_DIR)/$@

cuda.o: cuda_funcs/main.cu
	nvcc -Xcompiler -fPIC -c $< -o $(BUILD_DIR)/$@

main.o: main.cpp
	nvcc -Xcompiler "-fPIC -w" -c -I $(PG_SERVER_DIR) $< -o $(BUILD_DIR)/$@
	

target: main.o cuda_wrappers.o cuda.o host_wrapper.o
	nvcc -Xcompiler "-shared -rdynamic" -o $(BUILD_DIR)/$(SRCFILE).so $(BUILD_DIR)/main.o $(BUILD_DIR)/cuda_wrappers.o $(BUILD_DIR)/cuda.o $(BUILD_DIR)/host_wrapper.o
	mv $(BUILD_DIR)/$(SRCFILE).so $(PG_BINDIR)/$(SRCFILE).so
	psql postgresql://$(PGUSER):$(PGPASSWORD)@$(PGHOST):$(PGPORT)/$(PGDATABASE) -f scripts/script.sql

# Use phony targets for non-file targets, such as "clean" and "all"
.PHONY: all insert clean

# The default target is "all"
all: clean target

# Use variables for insert arguments
NUM_RECORDS = 10
ARRAY_LENGTH = 1024

SHELL := /bin/bash

insert:
	python3 -m venv env && source env/bin/activate && pip install -r requirements.txt
	source env/bin/activate && python3 scripts/load_bed_data.py
	source env/bin/activate && python3 scripts/dummy_data.py --num_records $(NUM_RECORDS) --array_length $(ARRAY_LENGTH) 

plot:
	python3 -m venv env && source env/bin/activate && pip install -r requirements.txt
	source env/bin/activate && python3 scripts/plots.py

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.so
