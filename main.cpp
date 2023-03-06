
extern "C"
{
#include "postgres.h"
#include "fmgr.h"

#ifdef PG_MODULE_MAGIC
    PG_MODULE_MAGIC;
#endif

    PG_FUNCTION_INFO_V1(vector_addition_cuda);

#include "funcapi.h"
#include "utils/array.h"
}

#include "cuda_wrappers.hpp"

Datum vector_addition_cuda(PG_FUNCTION_ARGS)
{

    ArrayType *x = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *y = PG_GETARG_ARRAYTYPE_P(1);

    int n = ArrayGetNItems(ARR_NDIM(x), ARR_DIMS(x));

    int *c = (int *)palloc(n * sizeof(int));

    elog(INFO, "n: %d", n);

    cuda_wrapper_vector_addition((int *)ARR_DATA_PTR(x), (int *)ARR_DATA_PTR(y), c, n);

    // elog(INFO, "result: %d", c[0]);

    Datum *c_datum = (Datum *)palloc(n * sizeof(Datum));

    for (int i = 0; i < n; i++)
    {
        c_datum[i] = Int32GetDatum(c[i]);
    }

    PG_RETURN_ARRAYTYPE_P(construct_array(c_datum, n, INT4OID, sizeof(int), true, 'i'));
}
