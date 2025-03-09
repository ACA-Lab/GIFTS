import pandas as pd
from config import gnns, datasets, keyword

gnnsets_names = []
for gnn in gnns:
    for dataset in datasets:
        gnnsets_names.append(f"{gnn}-{dataset}")

def switch2frame(frame_name):
    if frame_name in ["Ours", "Ours_comp", "Ours_sched", "Ours_val_only"]:
        return "Ours"
    else:
        return frame_name
    
def frame2num(frame_name):
    if frame_name == "PyG":
        return 0
    elif frame_name == "DGL":
        return 1
    elif frame_name == "Ours_comp":
        return 2
    elif frame_name == "Ours_sched":
        return 3
    elif frame_name == "Ours":
        return 4
    elif frame_name == "Ours_val_only":
        return 5
    else:
        raise NotImplementedError
    
def gnnsets2num(gnn, dataset):
    gnnsets = f"{gnn}-{dataset}"
    if gnnsets in gnnsets_names:
        return gnnsets_names.index(gnnsets)
    else:
        print(gnnsets)
        raise NotImplementedError

def proc_contents(contents, keyword=keyword):
    time_res = []
    for content in contents:
        if keyword in content:
            time_res.append(float(content.split(':')[-1].split('s\n')[0].strip()))
    return time_res

def convert_to_excel_mul(data, route='output.xlsx', row_names=True, column_names=True, row_rename=None, start_row=[], index_label=[]):
    writer = pd.ExcelWriter(route, engine='openpyxl')
    for i_ in range(len(data)):
        data_ = data[i_]
        for num_layer, data_res in data_.items():
            df = pd.DataFrame(data_res)
            if row_rename is not None:
                row_rename_=row_rename[num_layer]
                rename_ = {}
                for i in range(len(row_rename_)):
                    rename_[i] = row_rename_[i]
                print(rename_)
                df = df.rename(index=rename_)
            df.to_excel(writer, sheet_name=num_layer, index=row_names, header=column_names, startrow=start_row[i_],index_label=index_label[i_])
    writer.close()