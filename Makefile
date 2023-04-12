include .env

# Use variables for directories and files
SRCFILE = main
PG_SERVER_DIR = /usr/include/postgresql/14/server
PG_BINDIR = /tmp

# Use automatic variables to avoid repetition and simplify the Makefile
cuda_kernel.o: cuda_funcs/cuda_kernel.cu
	nvcc -Xcompiler -fPIC -c $< -o $@

cuda_wrappers.o: cuda_funcs/cuda_wrappers.cu
	nvcc -Xcompiler -fPIC -c $< -o $@

cuda.o: cuda_funcs/main.cu
	nvcc -Xcompiler -fPIC -c $< -o $@

main.o: main.cpp
	nvcc -Xcompiler "-fPIC -w" -c -I $(PG_SERVER_DIR) $< -o $@

target: main.o cuda_wrappers.o cuda_kernel.o cuda.o
	nvcc -Xcompiler "-shared -rdynamic" -o $(SRCFILE).so $^
	mv $(SRCFILE).so $(PG_BINDIR)/$(SRCFILE).so
	psql postgresql://$(PGUSER):$(PGPASSWORD)@$(PGHOST):$(PGPORT)/$(PGDATABASE) -f script.sql

docker: main.o cuda_wrappers.o cuda_kernel.o cuda.o
	nvcc -Xcompiler "-shared -rdynamic" -o $(SRCFILE).so $^
	mv $(SRCFILE).so $(PG_BINDIR)/$(SRCFILE).so

# Use phony targets for non-file targets, such as "clean" and "all"
.PHONY: all insert clean

# The default target is "all"
all: clean target

# Use variables for insert arguments
NUM_RECORDS = 10
ARRAY_LENGTH = 1024

insert:
	env/bin/python3 dummy_data.py --num_records $(NUM_RECORDS) --array_length $(ARRAY_LENGTH)

clean:
	rm -f *.o *.so
