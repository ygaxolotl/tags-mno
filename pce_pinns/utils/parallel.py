import os
from tqdm import tqdm
import psutil
import json

os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
import ray

"""
Parallelization
"""
def to_iterator(obj_id):
    # Call this to display tqdm progressbar when using ray parallel processing
    # Source https://github.com/ray-project/ray/issues/5554
    while obj_id:
        done, obj_id = ray.wait(obj_id)
        yield ray.get(done[0])

def print_memory():
    """
    Prints memory stats
    """
    vmem = psutil.virtual_memory()
    print(f'vMem: total:{vmem.total>>30}GB, '\
        f'avail:{vmem.available>>30}GB, '\
        f'used:{vmem.used>>30}GB {vmem.percent}\%, '\
        f'slab:{vmem.slab>>30}GB, '\
        f'cached:{vmem.cached>>30}GB'
    )

def init_preprocessing(fn, parallel=False, verbose=True,
    dir_spill=None):
    """
    Init parallel processing
    Source: https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

    Args:
        fn (fn): Function that's to be parallelized

    Returns:
    """
    if parallel:
        num_cpus = psutil.cpu_count(logical=False)
        if verbose: print('n cpus', num_cpus)
        if not ray.is_initialized():
            if dir_spill is not None:
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem", 
                        "params": {
                            "directory_path": dir_spill # "/nobackup1/lutjens/tmp/spill"
                    }},)
                }
            ray.init(
                _system_config=_system_config,                
                num_cpus=num_cpus, 
                ignore_reinit_error=True)            

    if parallel:
        fn_r = ray.remote(fn).remote
    else:
        fn_r = fn

    fn_tasks = []
    return fn_r, fn_tasks

def get_parallel_fn(model_tasks, verbose=True):
    """
    Waits for parallel model tasks to finish and returns outputs
    
    Args:
        model_outputs (list(tuple)): Outputs of model
    """
    for x in tqdm(to_iterator(model_tasks), total=len(model_tasks), disable=(verbose==False)):
        pass
    model_outputs = ray.get(model_tasks) # [0, 1, 2, 3]
    return model_outputs
