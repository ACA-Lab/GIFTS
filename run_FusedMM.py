import os
import os.path as osp

program_rt = osp.dirname(osp.realpath(__file__))

datasets = ["ogbn-products", "ogbn-mag", "ogbl-citation2"]

default_K = 256
Out_K = [47, 349, 256]

for num, dataset in enumerate(datasets):
    res_rt = os.path.join(program_rt, "Result_FusedMM", dataset)
    inf_prog_rt = os.path.join(program_rt, "Official_Code", "FusedMM", "bin", "xsOptFusedMMtime_gcn_pt")
    data_rt = os.path.join(program_rt, "Official_Code", "FusedMM", "dataset", f"{dataset}.mtx")
    if not os.path.exists(res_rt):
        os.makedirs(res_rt)
    # Default K
    res1_out_rt = os.path.join(res_rt, "res1.out")
    run_cmd = f"{inf_prog_rt} -input {data_rt} -K {default_K} > {res1_out_rt}"
    print(run_cmd)
    os.system(run_cmd)
    # Output K
    res2_out_rt = os.path.join(res_rt, "res2.out")
    run_cmd2 = f"{inf_prog_rt} -input {data_rt} -K {Out_K[num]} > {res2_out_rt}"
    print(run_cmd2)
    os.system(run_cmd2)
    