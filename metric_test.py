#%%
import pandas as pd
record_path = r"C:\Users\llim\Downloads"
filename = "inflation_india_v2.pkl"

p = Pickler(record_path)

records = p.read_data(filename)
records_obj = records.get("obj", [])
#%%
# test on the speed
import time
from joblib import Parallel, delayed

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

## old parallel
@timeit
def run_single_old():
    return Parallel(n_jobs=1)(
        delayed(get_info)(records_obj) for _ in range(10)
    )

@timeit 
def run_parallel_old():
    return Parallel(n_jobs=-1)(
        delayed(get_info)(records_obj) for _ in range(10)
    )

## new flow on the items
def process_records(records_obj):
    full_records_df = Records.read(records=records_obj)
    return full_records_df.get_scorer(["MAE", "MSE"])

@timeit
def run_single_new():
    return Parallel(n_jobs=1)(
        delayed(process_records)(records_obj) for _ in range(10)
    )

@timeit
def run_parallel_new():
    return Parallel(n_jobs=-1)(
        delayed(process_records)(records_obj) for _ in range(10)
    )


run_single_old()
run_single_new()
run_parallel_old()
run_parallel_new()
#%%
full_records_df = Records.read(records=records_obj)
scorer_df = full_records_df.get_scorer(["MAE", "MSE"])
summary_df = full_records_df.get_summary()
plot_dict = full_records_df.plot()

#%%
scorer = ["MAE", "MSE"]

self = full_records_df
metrics_test = self.pred_true.dropna(subset=["y_true", "y_pred"])
metrics_records = []

func_pairs = {metric: func for metric, func in zip(scorer, get_metrics(scorer))}

for mode_type, d in metrics_test.groupby("modes"):
    metric_df = pd.DataFrame({m: f(d["y_true"], d["y_pred"]) for m, f in func_pairs.items()}, index=[0])

    d["direction"] = self._direction_metrics(d)
    direction_acc = (d["direction"] == 1).sum() / len(d["direction"])
    direction_acc_last = (d.groupby("ref_periods")["direction"].last() == 1).sum() / len(d.groupby("ref_periods")["direction"].last())