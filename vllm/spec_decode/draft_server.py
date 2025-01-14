import multiprocessing as mp
import traceback
import os
import time
from vllm.spec_decode.ngram_worker import NGramWorker
from vllm.spec_decode.multi_step_worker import MultiStepWorker
from vllm.spec_decode.smaller_tp_proposer_worker import SmallerTpProposerWorker
from vllm.sequence import ExecuteModelRequest
class RemoteWorker(object):
    def __init__(self) -> None:
        self.worker_list = {}
        self.all_res = {}

    def setup_env(self, gpu_id, rank, world_size, port, autospec_steps):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["LOCAL_RANK"] = str(gpu_id)
        import torch.distributed as dist
        # Use address of one of the machines
        print("==================================================")
        print(port)
        dist.init_process_group("nccl", init_method='tcp://127.0.0.1:' + str(port),
                        rank=rank, world_size=world_size)
        self.rank = rank
        self.world_size = world_size
        self.autospec_steps = autospec_steps
        print("Worker[" + str(rank) + "] env setup!", flush=True)

    def init_worker(self, method ,model_config):
        print("start init worker")
        self.all_res[method] = []
        if method == "ssm":
            draft_tp = model_config.pop("draft_tp")
            target_tp = model_config.pop("target_tp")
            proposer_worker = MultiStepWorker(**model_config)

            proposer_worker = SmallerTpProposerWorker.maybe_wrap_worker(
                proposer_worker, draft_tp, target_tp)
            proposer_worker.init_device()
            proposer_worker.load_model()
            proposer_worker.cache_config.gpu_memory_utilization = 0.4
            a,b = proposer_worker.determine_num_available_blocks()
            proposer_worker.initialize_cache(a,b)
            proposer_worker.set_include_gpu_probs_tensor()
            proposer_worker.set_should_modify_greedy_probs_inplace()
            self.worker_list[method] = proposer_worker
            print(f"Worker[{method}] init worker!", flush=True)

        elif method == "ngram":
            ngram_prompt_lookup_max = (
                model_config.pop("ngram_prompt_lookup_max"))
            ngram_prompt_lookup_min = (
                model_config.pop("ngram_prompt_lookup_min"))
            self.worker_list[method] = NGramWorker(**model_config)
            self.worker_list[method].set_ngram_window_size(ngram_prompt_lookup_min,
                                                  ngram_prompt_lookup_max)
            self.worker_list[method].init_device()
            print(f"Worker[{method}] init worker! windows size between {ngram_prompt_lookup_min} and {ngram_prompt_lookup_max}", flush=True)
            

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def execute_methods(self, autospec_steps : int, 
                        execute_model_req : ExecuteModelRequest, 
                        seq_with_bonus_token_in_last_step):
        execute_model_req.num_lookahead_slots = 5
        for method, worker in self.worker_list.items():
            t1 = time.time()
            res = worker.get_spec_proposals(execute_model_req, seq_with_bonus_token_in_last_step)
            t2 = time.time()
            self.all_res[method].append([res,t2-t1])
            print(f"Worker[{method}] get result {res}!", flush=True)


def remote_worker_proc(rx, tx):
    remote_worker = RemoteWorker()
    # cmd: [quit]
    # cmd: [setup_env, gpu_id, rank, world_size, port]
    # cmd: [init_worker, model_config, parallel_config, scheduler_config]
    # cmd: [run_cmd, method, args, kwargs]
    cmd = rx.recv()
    port = 0
    
    while cmd[0] != "quit":
        # print(cmd)
        if cmd[0] == "setup_env":
            remote_worker.setup_env(cmd[1], cmd[2], cmd[3], cmd[4], cmd[5])
            port = cmd[4]
        elif cmd[0] == "init_worker":
            remote_worker.init_worker(cmd[1], cmd[2])
        elif cmd[0] == "execute_method":
            try:
                start_time = time.time()
                # if port == 20001:
                print("\nstart:", start_time, flush=True)
                print(cmd[1], cmd[2], cmd[3])
                remote_worker.execute_methods(cmd[1], cmd[2], cmd[3])
                #print("Get result", res)
                end_time = time.time()
                #tx.send((res, end_time - start_time))
                #if port == 20001:
                    # print("\nend: ", end_time, flush=True)

            except:
                traceback.print_exc()
                tx.send("ERROR!")
                
        elif cmd[0] == "get_all_results":
            print(remote_worker.all_res)
            tx.send(remote_worker.all_res)
        cmd = rx.recv()

class PipeWorkerResult(object):
    def __init__(self, worker, res_id):
        self.worker = worker
        self.res_id = res_id
    
    def get(self):
        return self.worker.get_result(self.res_id)

    def available(self):
        return self.worker.has_result(self.res_id)

class RemoteServer(object):
    def __init__(self):
        self.rx, self.rx_child = mp.Pipe(False)
        self.tx_child, self.tx = mp.Pipe(False)
        self.max_id = 0
        self.cur_id = -1
        self.result_map = {}
        self.process = mp.Process(target=remote_worker_proc, args=(self.tx_child, self.rx_child))
        self.process.start()

    def setup_env(self, gpu_id, rank, world_size, port, autospec_steps):
        self.tx.send(["setup_env", gpu_id, rank, world_size, port, autospec_steps])
    
    def init_worker(self, method, model_kwargs):
        self.tx.send(["init_worker", method, model_kwargs])
    
    def run_cmd(self, autospec_step, execute_model_req, seq_with_bonus_token_in_last_step):
        res_id = self.max_id
        self.tx.send(["execute_method", autospec_step, execute_model_req, seq_with_bonus_token_in_last_step])
        self.max_id += 1
        
        
    def get_result(self, res_id):
        assert res_id < self.max_id

        while self.cur_id < res_id:
            result = self.pop_result()
            self.cur_id += 1
            self.result_map[self.cur_id] = result

        result = self.result_map[res_id]
        del self.result_map[res_id]
        
        return result

    def has_result(self, res_id):
        assert res_id < self.max_id

        while self.cur_id < res_id and (self.rx.poll()):
            result = self.pop_result()
            self.cur_id += 1
            self.result_map[self.cur_id] = result

        if self.cur_id < res_id:
            return False

        return True
    
    def pop_result(self):
        return self.rx.recv()

    def stop(self):
        self.tx.send(["quit"])
    
    def get_all_results(self):
        self.tx.send(["get_all_results"])
        return self.rx.recv()
