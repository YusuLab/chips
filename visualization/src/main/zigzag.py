import numpy as np
import dionysus as d
import sys

def my_tracer(frame, event, arg = None):
    # extracts frame code
    code = frame.f_code
  
    # extracts calling function name
    func_name = code.co_name
  
    # extracts the line number
    line_no = frame.f_lineno
  
    print(f"A {event} encountered in \
    {func_name}() at line number {line_no} ")
  
    return my_tracer

sys.settrace(my_tracer)
f_lst = np.load("f_lst.npy", allow_pickle=True).tolist()
times = np.load("times.npy", allow_pickle=True).tolist()
f = d.Filtration(f_lst)
zz, dgms, cells = d.zigzag_homology_persistence(f, times)
np.save("dgms.npy", dgms)

