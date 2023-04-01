DROP FUNCTION vector_addition_cuda;
CREATE OR REPLACE FUNCTION vector_addition_cuda(integer [], integer[])
        RETURNS integer []
     AS '/tmp/main.so', 'vector_addition_cuda'
     LANGUAGE C STRICT;


SELECT vector_addition_cuda(ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);