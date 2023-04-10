#ifndef SPI_HELPERS_H /* Include guard */
#define SPI_HELPERS_H

extern "C"
{
#include "datatype/timestamp.h"
#include "executor/spi.h"
#include "utils/array.h"
#include "postgres.h"
#include <utils/lsyscache.h> // get_typlenbyvalalign

#define DatumGetTimestamp(X) ((Timestamp)DatumGetInt64(X))
#define TimestampGetDatum(X) Int64GetDatum(X)
#define PG_GETARG_TIMESTAMP(n) DatumGetTimestamp(PG_GETARG_DATUM(n))
}

#include "cuda_funcs/cuda_wrappers.hpp"
#include <vector>

struct AggregatedResult
{
    Timestamp tstamp;
    int16 value;
} typedef AggregatedResult;

/**
 * @brief Builds a SQL query string using the provided start and end times.
 * @param start_time A C string representing the start time of the query range.
 * @param end_time A C string representing the end time of the query range.
 * @return A pointer to a C string containing the generated SQL query.
 */
const char *build_query(
    const char *start_time,
    const char *end_time)
{

    char *query;
    asprintf(&query, "SELECT * FROM example WHERE tstmp BETWEEN '%s'::TIMESTAMP AND '%s'::TIMESTAMP", start_time, end_time);

    char *pq_query = (char *)palloc(strlen(query) + 1);
    strcpy(pq_query, query);
    pq_query[strlen(pq_query)] = '\0';
    return pq_query;
}

/**
 * Converts a PostgreSQL Datum value to a C string representation of a timestamp.
 * @param d A PostgreSQL Datum value to be converted.
 * @return A pointer to a C string containing the timestamp value as a string.
 */
const char *datum_to_c_string(Datum d)
{
    return DatumGetCString(DirectFunctionCall1(timestamp_out, d));
}

/**
 * Prepares a read-only cursor with the given SQL command.
 * @param command A C string containing the SQL command to be executed by the cursor.
 * @return A Portal object representing the prepared cursor.
 */
Portal prepare_signature_cursor(const char *command)
{

    // variables
    const char *cursor_name = NULL;               // Postgres will self-assign a name
    const int arg_count = 0;                      // No arguments will be passed
    Oid *arg_types = NULL;                        // n/a
    Datum *arg_values = NULL;                     // n/a
    const char *null_args = NULL;                 // n/a
    bool read_only = true;                        // read_only allows for optimization
    const int cursor_opts = CURSOR_OPT_NO_SCROLL; // default cursor options

    return SPI_cursor_open_with_args(
        cursor_name,
        command,
        arg_count,
        arg_types,
        arg_values,
        null_args,
        read_only,
        cursor_opts);
}

/**
 * @brief converts a int array to a PostgreSQL Datum value.
 * @param d A PostgreSQL Datum value to be converted.
 * @param nelemsp A pointer to an integer that will be set to the number of elements in the array.
 * @return A pointer to an array of Datum values.
 */
Datum *get_int16_array_from_datum(Datum d, int *nelemsp)
{
    ArrayType *array = DatumGetArrayTypeP(d);

    Oid elmtype = ARR_ELEMTYPE(array);
    int elmlen = sizeof(int16);
    bool elmbyval = true;
    char elmalign = 's';
    Datum *elemsp;
    bool *nullsp;

    deconstruct_array(array, elmtype, elmlen, elmbyval, elmalign,
                      &elemsp, &nullsp, nelemsp);

    return elemsp;
}

/**
 * @brief Computes the results of the query inside GPU.
 *
 * @param funcctx A FuncCallContext object.
 * @return Datum** A pointer to an array of Datum values.
 */
Datum **compute_results_c_plus_plus(FuncCallContext *funcctx)
{
    unsigned int i;
    bool is_null = true;
    Datum **results;
    int records_processed = 0;
    Datum **aggregated_results;

    if (SPI_processed == 0)
    {
        return NULL;
    }

    results = (Datum **)SPI_palloc(sizeof(Datum *) * SPI_processed);

    std::vector<Timestamp> times_vector;
    std::vector<int16> data_values_vector;
    int cols;

    elog(INFO, "SPI_processed: %d", SPI_processed);

    for (i = 0; i < SPI_processed; i++)
    {
        Datum tstamp = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &is_null);
        Datum array_values = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &is_null);

        int array_size = 0;
        Datum *array_values_int16 = get_int16_array_from_datum(array_values, &array_size);

        for (int j = 0; j < array_size; j++)
        {
            data_values_vector.push_back(DatumGetInt16(array_values_int16[j]));
        }

        times_vector.push_back(DatumGetTimestamp(tstamp));

        records_processed++;

        cols = array_size;
    }

    int number_of_elements = records_processed;

    const short int *data_ptr = (const short int *)data_values_vector.data();
    int data_number_of_elements = data_values_vector.size();

    short int *max_values = (short int *)palloc(sizeof(short int) * number_of_elements);

    cuda_find_max(data_ptr, max_values, number_of_elements, cols);

    aggregated_results = (Datum **)SPI_palloc(sizeof(Datum *) * number_of_elements);

    for (int i = 0; i < number_of_elements; i++)
    {
        AggregatedResult *aggregated_result_row = (AggregatedResult *)SPI_palloc(sizeof(AggregatedResult));

        aggregated_result_row->tstamp = times_vector[i];
        aggregated_result_row->value = max_values[i];
        aggregated_results[i] = (Datum *)aggregated_result_row;

        // elog(INFO, "aggregated_result_row->tstamp: %s aggregated_result_row->value %u \n",
        //      datum_to_c_string(TimestampGetDatum(aggregated_result_row->tstamp)),
        //      aggregated_result_row->value);
    }

    funcctx->max_calls = number_of_elements;

    return aggregated_results;
}

#endif // SPI_HELPERS_H