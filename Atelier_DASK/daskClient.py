from dask.diagnostics import ProgressBar
import dask.array as da 
from dask.distributed import Client, progress
 
client = Client()
a = da.random.normal(size=(10000, 10000), chunks=(1000, 1000))
res = (a.T + a).mean(axis=0)

#res.compute()
res = res.persist()
progress(res)