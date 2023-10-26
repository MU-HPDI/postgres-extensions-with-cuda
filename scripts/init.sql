CREATE OR REPLACE FUNCTION vector_addition_cuda(integer [], integer[])
        RETURNS integer []
     AS '/tmp/main.so', 'vector_addition_cuda'
     LANGUAGE C STRICT;

CREATE TYPE __retcomposite_max_reduction AS (
        tstamp timestamp without time zone, 
        value smallint
);

CREATE OR REPLACE FUNCTION max_reduction_cuda(start_time timestamp without time zone, end_time timestamp without time zone)
        RETURNS SETOF __retcomposite_max_reduction
     AS '/tmp/main.so', 'max_reduction_cuda'
     LANGUAGE C STRICT;


CREATE TYPE __heart_rate_comp_type AS (
        tstmp timestamp without time zone, 
        heart_rate real
);

CREATE OR REPLACE FUNCTION heart_rate_estimation(character varying, timestamp without time zone, timestamp without time zone, integer, character varying, character varying)
        RETURNS SETOF __heart_rate_comp_type
     AS '/tmp/main.so', 'heart_rate_estimation'
     LANGUAGE C STRICT
;