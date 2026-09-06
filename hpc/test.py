import torch

def check_cuda():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        # CUDA version PyTorch was built with
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Number of GPUs found
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}\n")
        
        # Details for each available GPU
        for i in range(gpu_count):
            print(f"--- GPU {i} ---")
            print(f"Device Name: {torch.cuda.get_device_name(i)}")
            
            # Memory details in GB
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(i) / (1024**3)
            
            print(f"Total Memory: {total_memory:.2f} GB")
            print(f"Allocated Memory: {allocated_memory:.2f} GB")
            print(f"Cached Memory: {cached_memory:.2f} GB\n")
            
        # Quick tensor operation test on GPU
        try:
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            print(f"Tensor Test Success: {x} created on {x.device}")
        except Exception as e:
            print(f"Tensor Test Failed: {e}")
    else:
        print("\nCUDA is not available. Running on CPU only.")

if __name__ == "__main__":
    check_cuda()