#%%
import pandas as pd
import pathlib


#%%
tables = [
    "Width",
    "Noise",
    "Transition Noise"
]

results = {}
for result_name in tables:
    path = pathlib.Path(fr"S:\PhD\Packages\SELeCT\evaluation\{result_name}_datatable.txt")
    print(path)
    l_num = -1
    for l in path.open():
        l_num += 1
        if l_num < 6 or l_num > 18:
            continue
        print(l)



