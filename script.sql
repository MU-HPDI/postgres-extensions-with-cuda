DROP FUNCTION IF EXISTS vector_addition_cuda(integer [], integer[]);
CREATE OR REPLACE FUNCTION vector_addition_cuda(integer [], integer[])
        RETURNS integer []
     AS '/tmp/main.so', 'vector_addition_cuda'
     LANGUAGE C STRICT;


SELECT vector_addition_cuda(ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

DROP FUNCTION IF EXISTS max_reduction_cuda(start_time timestamp without time zone, end_time timestamp without time zone);
DROP TYPE IF EXISTS __retcomposite_max_reduction;
CREATE TYPE __retcomposite_max_reduction AS (
        tstamp timestamp without time zone, 
        value smallint
);

CREATE OR REPLACE FUNCTION max_reduction_cuda(start_time timestamp without time zone, end_time timestamp without time zone)
        RETURNS SETOF __retcomposite_max_reduction
     AS '/tmp/main.so', 'max_reduction_cuda'
     LANGUAGE C STRICT;

SELECT * FROM max_reduction_cuda('2023-04-08 22:00:00'::timestamp, '2023-04-09 22:00:00'::timestamp);
