-- Vector Addition
DROP FUNCTION IF EXISTS vector_addition_cuda(integer [], integer[]);
CREATE OR REPLACE FUNCTION vector_addition_cuda(integer [], integer[])
        RETURNS integer []
     AS '/tmp/main.so', 'vector_addition_cuda'
     LANGUAGE C STRICT;


SELECT vector_addition_cuda(ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ARRAY[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

-- Reduction
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


-- Heart Rate 
DROP FUNCTION IF EXISTS heart_rate_estimation_cuda(character varying, timestamp without time zone, timestamp without time zone);
DROP TYPE IF EXISTS __heart_rate_type;
CREATE TYPE __heart_rate_type AS (
        tstmp timestamp without time zone, 
        heart_rate real
);

CREATE OR REPLACE FUNCTION heart_rate_estimation_cuda(character varying, timestamp without time zone, timestamp without time zone)
        RETURNS SETOF __heart_rate_type
     AS '/tmp/main.so', 'heart_rate_estimation_cuda'
     LANGUAGE C STRICT
;


SELECT * FROM heart_rate_estimation_cuda('bed_data', '2022-06-20 00:00:00'::timestamp, '2022-06-20 01:00:00'::timestamp);
