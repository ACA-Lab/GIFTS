import os
import os.path as osp
import utils
# You may not need to set this, but something unexpected may happen :(
os.environ['MKL_THREADING_LAYER'] = 'GNU'

program_rt = osp.dirname(osp.realpath(__file__))
Train = False
Inference = True
Train_or_Inference = Inference

Debug = False

gnns = ["gcn", "sage"]
datasets = ["ogbn-products", "ogbn-mag", "ogbl-citation2"]
frameworks = ["DGL", "PyG", "Ours", "Ours_comp", "Ours_sched"]
# num_layers = [3, 4, 5]
num_layers = [3]
# num_threads = [32, 30, 28, 26, 24, 22, 20, 18, 16]
num_threads = [32]

# gnns = ["gcn"]
# datasets = ["ogbn-products"]
# frameworks = ["DGL"]

for num_thread in num_threads:
    for gnn in gnns:
        for dataset in datasets:
            for framework in frameworks:
                for num_layer in num_layers:
                    dataset_rt = os.path.join(program_rt, "dataset")
                    framework_ = utils.switch2frame(framework)
                    misc_rt = os.path.join(program_rt, "MISC", gnn, framework_, dataset, f"num_layer{num_layer}")
                    code_rt = os.path.join(program_rt, "code", framework_, dataset)
                    res_rt = os.path.join(program_rt, "Result", f"thread_{num_thread}", gnn, framework, dataset, f"num_layer{num_layer}")
                    # res_rt = "."
                    res_out_rt = os.path.join(res_rt, "res.out")
                    inf_prog_rt = os.path.join(code_rt, "test_inference_gnn.py")
                    tra_prog_rt = os.path.join(code_rt, "main_gnn.py")
                    checkpoint_rt = os.path.join(misc_rt, "checkpoint_dir")
                    log_rt = os.path.join(misc_rt, "log_dir")

                    if not os.path.exists(checkpoint_rt):
                        os.makedirs(checkpoint_rt)
                    if not os.path.exists(log_rt):
                        os.makedirs(log_rt)
                    if not os.path.exists(res_rt):
                        os.makedirs(res_rt)

                    if framework_ == "Ours":
                        run_cmd_tra = f"python {tra_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} \
                                --log_dir {log_rt} --dataset_name {dataset} --batch_size 1024 --num_layers {num_layer}"
                    else:
                        run_cmd_tra = f"python {tra_prog_rt} --gnn {gnn} \
                        --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} \
                            --log_dir {log_rt} --dataset_name {dataset} --batch_size 1024 \
                            --epochs 10 --num_layers {num_layer}"
                        # run_cmd_tra = f"python {tra_prog_rt}"

                    if framework == "Ours":
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 10 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} --compress_flag --reorder_flag > {res_out_rt}"
                    elif framework == "Ours_comp":
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 10 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} --compress_flag > {res_out_rt}"
                    elif framework == "Ours_sched":
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 10 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} --reorder_flag > {res_out_rt}"
                    elif framework == "Ours_val_only":
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 10 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} --compress_flag --value_only_flag > {res_out_rt}"
                    else:
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 10 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} > {res_out_rt}"
                        
                    if Debug:
                        run_cmd_inf = f"python {inf_prog_rt} --gnn {gnn} \
                            --dataset_route {dataset_rt} --checkpoint_dir {checkpoint_rt} --epochs 1 \
                                --use_cpu_only --dataset_name {dataset} --cpu_threads {num_thread} --num_layers {num_layer} > {res_out_rt}"

                    print(f"gnn: {gnn}, dataset: {dataset}, framework: {framework}, num_layer: {num_layer}, inference: {Train_or_Inference}")

                    # print(run_cmd_inf)

                    if Train_or_Inference == Train:
                        os.system(run_cmd_tra)
                    elif Train_or_Inference == Inference:
                        os.system(run_cmd_inf)
                    else:
                        pass
