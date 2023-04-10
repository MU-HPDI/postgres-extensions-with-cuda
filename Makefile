include .env

SRCFILE = main
PG_SERVER_DIR = /usr/include/postgresql/14/server
PG_BINDIR = /tmp

cuda_kernel.o:
	nvcc -Xcompiler -fPIC -c cuda_funcs/cuda_kernel.cu -o cuda_kernel.o

cuda_wrappers.o: cuda_kernel.o
	nvcc -Xcompiler -fPIC -c cuda_funcs/cuda_wrappers.cu -o cuda_wrappers.o

cuda: cuda_wrappers.o
	nvcc -Xcompiler -fPIC -c cuda_funcs/main.cu -o cuda.o 
	nvcc cuda.o cuda_wrappers.o cuda_kernel.o -o cuda.out

main.o: cuda_wrappers.o
	nvcc -Xcompiler "-fPIC -w" -c -I $(PG_SERVER_DIR) main.cpp -o main.o

target: main.o 
	nvcc -Xcompiler "-shared -rdynamic" -o $(SRCFILE).so main.o cuda_wrappers.o cuda_kernel.o
	mv $(SRCFILE).so $(PG_BINDIR)/$(SRCFILE).so
	psql postgresql://$(PGUSER):$(PGPASSWORD)@$(PGHOST):$(PGPORT)/$(PGDATABASE) -f $ script.sql

all: clean target

insert:
	env/bin/python3  dummy_data.py --num_records 10 --array_length 1024

clean:
	rm -f *.o *.so