import math
from numba import cuda
import numba as nb
import numpy as np
import sys



BLOCK_SIZE = 1024
INDEPENDENT = False
INCLUSIVE = False



def complete_to_power_of_two(array) :
    m = int(np.log2(len(array)) + 1)
    n = 2**m
    new_array = np.zeros(n, dtype=np.int32)
    for i in range(0, n):
        if(i < len(array)):
            new_array[i] = array[i]
        else:
            new_array[i] = 0
    return new_array


#array is 2^m long
#implementation of the scan prefix algorithm
def scanCPU(array) :
    
    len_initial_array = len(array)
    
    array = complete_to_power_of_two(array)

    m = int(np.log2(len(array)))
    
    
    for d in range(0, m):
        for k in range(0, len(array), 2**(d+1)):
            array[k+2**(d+1)-1] += array[k+2**d-1]
            
    array[len(array)-1] = 0
    
    for d in range(m-1, -1, -1):
        for k in range(0, len(array), 2**(d+1)):
            temp = array[k+2**d-1]
            array[k+2**d-1] = array[k+2**(d+1)-1]
            array[k+2**(d+1)-1] += temp
            
    return array[:len_initial_array]

@cuda.jit
def scanKernel(array,sum_array) :
    id = cuda.grid(1)
    local_id = cuda.threadIdx.x
    m  = int(math.ceil(math.log2(BLOCK_SIZE)))

    shared_array = cuda.shared.array(shape=BLOCK_SIZE, dtype=nb.int32)
    cuda.syncthreads()

    if(id < len(array)):
        shared_array[local_id] = array[id]
    cuda.syncthreads()
            
    # print(id,shared_array[local_id])
    
    for d in range(0, m):
        k = BLOCK_SIZE // 2**(d + 1)
        if local_id < k :
            shared_array[local_id* 2**(d + 1)+2**(d+1)-1] += shared_array[local_id* 2**(d + 1)+2**d-1]
        cuda.syncthreads()

    if(local_id == 0):
        sum_array[cuda.blockIdx.x] = shared_array[BLOCK_SIZE-1]
        # print("shared_array = ", shared_array[BLOCK_SIZE-1])
        # print("sum_array_d = ", sum_array[cuda.blockIdx.x])
        shared_array[BLOCK_SIZE-1] = 0
    cuda.syncthreads()
    
    # print(id,shared_array[local_id])
        
    for d in range(m - 1, -1, -1):
        k = BLOCK_SIZE // 2**(d + 1)
        if local_id < k:
            temp = shared_array[local_id* 2**(d + 1)+2**d-1]
            shared_array[local_id* 2**(d + 1)+2**d-1] = shared_array[local_id* 2**(d + 1)+2**(d+1)-1]
            shared_array[local_id* 2**(d + 1)+2**(d+1)-1] += temp    
        cuda.syncthreads()

    if(id < len(array)):
        array[id] = shared_array[local_id]
        
        
def scanGPU(array):
    n = len(array)
    
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    # print("grid = ", grid)
    
    sum_array = np.zeros(grid, dtype=np.int32)
    
    d_array = cuda.to_device(array)
    d_sum_array = cuda.to_device(sum_array)
        
    scanKernel[grid, BLOCK_SIZE](d_array,d_sum_array)
    
    # array = cuda.from_device(array, n, np.int32)
    # sum_array = cuda.from_device(sum_array, grid, np.int32)

    array = d_array.copy_to_host()
    sum_array = d_sum_array.copy_to_host()
    
    # print("sum_array = ", sum_array)
    # print("array = ", array)

    if(grid == 1 or INDEPENDENT):
        return array
    else :
        res = scanGPU(sum_array)
        for i in range(0, len(array)):
            array[i] += res[i // BLOCK_SIZE]
        return array
        # return np.concatenate((array, res))
        # print("res = ", res)
        # for i in range(0, len(array)):
        #     array[i] += res[0]
        # return array
        
        
# tab = np.array([1 for _ in range(1025)], dtype=np.int32)
# tab2 = np.array([1,2,3,4,5,6,7,8,20,55], dtype=np.int32)
# tab3 = np.array([1,2,3,4,5,6,7,8,9,10,11], dtype=np.int32)


def main():
    global BLOCK_SIZE
    global INDEPENDENT
    global INCLUSIVE

    # Check if the user provided the input file.
    if len(sys.argv) < 2:
        # print("Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]")
        sys.exit(1)
    
    # Read the input file and convert it to an array.
    inputFile = sys.argv[1]
    tab = open(inputFile).read().split(',')
    tab  = np.array([int(x) for x in tab ], dtype=np.int32)

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--tb":
            if len(sys.argv) > i + 1:
                bs = int(sys.argv[i + 1])
                # Check if the thread block size is a power of 2.
                if bs > 1024:
                    BLOCK_SIZE = 1024
                if int(math.ceil(math.log2(bs))) != int(math.log2(bs)):
                    # print("The thread block size must be a power of 2.")
                    BLOCK_SIZE = 2 ** int(math.log2(bs))
                    # print("Using " + str(BLOCK_SIZE) + " instead.")
                else:
                    BLOCK_SIZE = bs
                    i += 1
            else:
                # print("Usage: python project-gpu.py <inputFile> [--tb int] [--independent] [--inclusive]")
                sys.exit(1)
        elif sys.argv[i] == "--independent":
            INDEPENDENT = True
        elif sys.argv[i] == "--inclusive":
            INCLUSIVE = True

    scan_tab = scanGPU(tab)

    if INCLUSIVE:
        scan_tab += tab

    print(np.array2string(scan_tab, separator=",",threshold=scan_tab.shape[0]).strip('[]').replace('\n', '').replace(' ',''))

if __name__ == '__main__':
    main()

