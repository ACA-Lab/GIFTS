import os
import os.path as osp
import utils
from config import framework_names, gnns, datasets, keyword

search_path = osp.join(osp.dirname(osp.realpath(__file__)),'Result')

gnnsets_names = []
for gnn in gnns:
    for dataset in datasets:
        gnnsets_names.append(f"{gnn}-{dataset}")

excel_dir = osp.join(osp.dirname(osp.realpath(__file__)),'Excel')

suffix = []
for i in [32]:
    suffix.append(f"thread_{i}")

for suffix_ in suffix:

    search_path_ =osp.join(search_path, suffix_)
    results = []

    for root, dirs, files in os.walk(search_path_):
        for file in files:
            if file == 'res.out':
                prefix = root.split('/')
                num_layer = prefix[-1]
                dataset = prefix[-2]
                framework = prefix[-3]
                gnn = prefix[-4]
                if dataset in datasets and gnn in gnns:
                    with open(osp.join(root,file), 'r') as f:
                        contents = f.readlines()
                        results.append([num_layer, dataset, framework, gnn, contents])

    if len(results) <1 :
        continue

    summary = {}
    row_name = {}
    for num_layer, dataset, framework, gnn, contents in results:
        if num_layer not in summary:
            summary[num_layer] = {}
        time_res = utils.proc_contents(contents)
        if framework_names[utils.frame2num(framework)] not in summary[num_layer]:
            summary[num_layer][framework_names[utils.frame2num(framework)]] = [0 for _ in range(len(gnnsets_names))]
        row_name[num_layer]=gnnsets_names
        summary[num_layer][framework_names[utils.frame2num(framework)]][utils.gnnsets2num(gnn, dataset)] = sum(time_res)

    if not os.path.exists(excel_dir):
        os.makedirs(excel_dir)

    utils.convert_to_excel_mul(data=[summary], 
                            route=osp.join(excel_dir,f"{suffix_}.xlsx"), row_rename=row_name, start_row=[0], 
                            index_label=["Datasets"])
    # utils.convert_to_excel_mul(data=[summary, summary_conv1, summary_conv2], 
    #                         route='summary{}.xlsx'.format(suffix_), row_rename=row_name, start_row=[0,len(row_name)+1,(len(row_name)+1)*2], 
    #                         index_label=["All","conv1","conv2"])