
extern "C"
{
#include "postgres.h"
#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
    PG_MODULE_MAGIC;
#endif

    PG_FUNCTION_INFO_V1(vector_addition_cuda);
    PG_FUNCTION_INFO_V1(max_reduction_cuda);
#include "utils/geo_decls.h"
#include "funcapi.h"
#include "utils/array.h"

#include "executor/spi.h"
#include "utils/builtins.h"
#include "datatype/timestamp.h"
#include "pgtime.h"
#include "miscadmin.h"
}
#include "spi_helpers.hpp"
#include "cuda_funcs/cuda_wrappers.hpp"

struct FunctionInfo
{
    Portal signature_cursor;
    Datum **results;
    int results_proccesed;
    int results_size;
} typedef FunctionInfo;

Datum vector_addition_cuda(PG_FUNCTION_ARGS)
{

    ArrayType *x = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *y = PG_GETARG_ARRAYTYPE_P(1);

    int n = ArrayGetNItems(ARR_NDIM(x), ARR_DIMS(x));

    int *c = (int *)palloc(n * sizeof(int));

    elog(INFO, "n: %d", n);

    cuda_wrapper_vector_addition((int *)ARR_DATA_PTR(x), (int *)ARR_DATA_PTR(y), c, n);

    for (int i = 0; i < n; i++)
    {
        elog(INFO, "x: %d", ((int *)ARR_DATA_PTR(x))[i]);
        elog(INFO, "y: %d", ((int *)ARR_DATA_PTR(y))[i]);
    }

    Datum *c_datum = (Datum *)palloc(n * sizeof(Datum));

    for (int i = 0; i < n; i++)
    {
        c_datum[i] = Int32GetDatum(c[i]);
    }

    PG_RETURN_ARRAYTYPE_P(construct_array(c_datum, n, INT4OID, sizeof(int), true, 'i'));
}

Datum max_reduction_cuda(PG_FUNCTION_ARGS)
{
    TupleDesc tupleDesc;

    if (get_call_result_type(fcinfo, NULL, &tupleDesc) != TYPEFUNC_COMPOSITE)
    {
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("Function returning record called in context that cannot accept type record")));
    }

    /*
        Variables passed from Postgres

        Ex.
            SELECT * FROM max_reduction_cuda('2020-01-01'::timestamp, '2020-01-02'::timestamp);
    */
    const char *start_time_str = datum_to_c_string(PG_GETARG_DATUM(0));
    const char *end_time_str = datum_to_c_string(PG_GETARG_DATUM(1));
    const char *command = build_query(start_time_str, end_time_str);

    FuncCallContext *funcctx;
    Portal signature_cursor;
    Datum **results;
    bool nulls[3] = {false, false, false};
    bool forward = true;
    const long fetch_count = FETCH_ALL;

    if (SRF_IS_FIRSTCALL())
    {
        elog(INFO, "command: %s", command);
        MemoryContext oldcontext;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        /* One-time setup code appears here: */
        if (SPI_connect() != SPI_OK_CONNECT)
        {
            elog(ERROR, "SPI_connect failed");
        }

        // set function infor for having a global cursor and knowing when the SPI_fecth is done
        FunctionInfo *func_info = (FunctionInfo *)SPI_palloc(sizeof(FunctionInfo));
        func_info->signature_cursor = prepare_signature_cursor(command);
        func_info->results = NULL;
        func_info->results_proccesed = 0;
        func_info->results_size = 0;

        funcctx->user_fctx = func_info;

        funcctx->tuple_desc = tupleDesc;
        /* end one time setup code */

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();

    FunctionInfo *func_info = (FunctionInfo *)funcctx->user_fctx;
    signature_cursor = (Portal)func_info->signature_cursor;

    if (func_info->results_proccesed == func_info->results_size)
    {
        SPI_cursor_fetch(signature_cursor, forward, fetch_count);
        func_info->results = compute_results_c_plus_plus(funcctx);
        func_info->results_proccesed = 0;
        func_info->results_size = funcctx->max_calls;
    }

    if (SPI_processed == 0)
    {
        SPI_finish();
        SRF_RETURN_DONE(funcctx);
    }

    results = func_info->results;
    HeapTuple tuple = heap_form_tuple(funcctx->tuple_desc, results[func_info->results_proccesed], nulls);
    Datum result = HeapTupleGetDatum(tuple);
    ((FunctionInfo *)funcctx->user_fctx)->results_proccesed += 1;

    SRF_RETURN_NEXT(funcctx, result);
}