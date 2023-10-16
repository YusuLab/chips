import os
import time
import subprocess

def is_gpu_available():
    """
    Returns True if any GPU is available based on both memory and utilization, else returns False.
    """
    try:
        # Get GPU utilization from nvidia-smi
        utilization = os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits').read()
        gpu_utils = [int(util) for util in utilization.strip().split('\n')]
        
        # Get GPU memory free
        memory = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits').read()
        gpu_memory_free = [int(mem) for mem in memory.strip().split('\n')]

        # Define thresholds (change as needed)
        UTIL_THRESHOLD = 50  # Consider GPU as available if utilization is less than this
        MEM_THRESHOLD = 9000  # Consider GPU as available if used memory is smaller than this (in MiB)
        
        i = 0
        for util, mem_used in zip(gpu_utils, gpu_memory_free):
            if util < UTIL_THRESHOLD and mem_used < MEM_THRESHOLD:
                return True, i
            i += 1
        return False, -1
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return False, -1

def submit_job(node):
    """
    Your code to submit the job goes here.
    For example purposes, I'm just printing a message.
    """
    subprocess.run(["sh", "train_gnn_hetero_demand.sh", "0", "1", "gat", "hpwl", str(node), "0"])
    print("Job submitted!")

def main():
    # Poll every 30 seconds to check GPU availability
    while True:
        output = is_gpu_available()
        if output[0]:
            submit_job(output[1])
            break
        else:
            print("No GPU available right now. Waiting...")
            time.sleep(30)

if __name__ == '__main__':
    main()
