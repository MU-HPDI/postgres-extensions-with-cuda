SRCFILE = main
PG_SERVER_DIR = /usr/include/postgresql/14/server
PG_BINDIR = /tmp

taget:
	nvcc -Xcompiler -fPIC -c cuda_wrappers.cu -o cuda_wrappers.o
	nvcc -Xcompiler "-fPIC -w" -c -I $(PG_SERVER_DIR) cuda_kernel.cu -o cuda_kernel.o
	nvcc -Xcompiler "-fPIC -w" -c -I $(PG_SERVER_DIR) main.cpp -o main.o
	nvcc -Xcompiler "-shared -rdynamic" -o $(SRCFILE).so main.o cuda_wrappers.o cuda_kernel.o

	mv $(SRCFILE).so $(PG_BINDIR)/$(SRCFILE).so
	# export PGPASSWORD="Jamal.s" && psql -d gpu_dbl -f script.sql

all: clean taget

clean:
	rm -f *.o *.so