
/*
 *  input size N : default 1000
 *  choose input St: =k N default 1.25
 */


/*
 * simple resizable hash table (linear probe/ double)
 * the proble is when we use the resize   ,and how
 * we know the resize condition ... by a global var?
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#define Entry unsigned long long
#define NUM_THREADS 512

//parameters
#define jump_prime 41
#define hp 4294967291  //max int32 prime
#define SLOTEMPTY ((Entry)0xffffffff00000000)
#define LINEAR 1
#define QUADRATIC 2
#define method 3  //differenf kind of probe 3:double hashing
#define kMaxProbes 3000
#define ha 2654435769
#define hb 11

unsigned int N=100000; //St: number of slot  St=1.25N a good trade-off
unsigned int St; //kMaxProbes： max Probe sequence

//hash function
inline __device__ __host__ unsigned  
hash_function(const unsigned key, unsigned size)
{
    //Linear probing  h(k) = g(k) + iteration
    //Quadratic probing  h(k) = g(k) + c0 · iteration + c1 · iteration2
    //Double hashing h(k) = g(k) + jump(k) · iteration

    //p=4,294,967,291 prime
    //f (a, k) = a · k
    //return ((ha*key + hb) %hp )% size;

    return key % size ;
}

inline __device__ __host__ unsigned  
jump_function(const unsigned key, unsigned size)
{
    return (1+key % jump_prime) % size;
}

//insert
__device__ bool 
insert_entry(const unsigned key,
                  const unsigned value,
                  const unsigned table_size,
                  Entry *table)
{
    // Manage the key and its value as a single 64−bit entry.
    Entry entry = ((Entry)key << 32) + value ;

    // Figure out where the item needs to be hashed into.
    unsigned index            = hash_function(key,table_size);
    unsigned double_hash_jump = jump_function(key,table_size) + 1;

    // Keep trying to insert the entry into the hash table
    // until an empty slot is found.
    Entry old_entry ;
    for (unsigned attempt = 1; attempt <= kMaxProbes; ++attempt) 
    {
        // Move the index so that it points somewhere within the table .
        index %= table_size ;

        // Atomically check the slot and insert the key if empty.
        old_entry = atomicCAS((table + index), SLOTEMPTY, entry);

        // If the slot was empty , the item was inserted safely.
        if (old_entry == SLOTEMPTY)
        {
            
            return true;
        }

        // Move the insertion index .
        if (method == LINEAR)
        {
            index += 1;
            
        }
        else if (method == QUADRATIC)
        {
            index += attempt * attempt;
        }
        else
        {
            index += attempt * double_hash_jump;
        }
    }
    return false ;
}//end insert_entry

//init kernel
__global__ void 
initkernrl(Entry *table, unsigned size){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<size){
        table[idx]=SLOTEMPTY;
    }
}

//find


//kernel_creat_table
__global__ void 
creat_table(Entry* table,
            unsigned size,
            unsigned int *key,
            unsigned int *value,
            unsigned int NUM_INSTER)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<NUM_INSTER)
    {
        unsigned KEY=key[idx],VALUE=value[idx];
        //insert two times 
        if(!insert_entry(KEY,idx,size,table))
        {
            if(!insert_entry(KEY,idx,size,table))
            {
                
                //exit(1);
                return ;
            }
        }
    }
}


__device__ bool 
find_key(const unsigned key,
         unsigned &value,
         const unsigned table_size,
         Entry *table)
{
    // Figure out where the item needs to be hashed into.
    unsigned index            = hash_function(key,table_size);
    unsigned double_hash_jump = jump_function(key,table_size) + 1;

    for (unsigned attempt = 1; attempt <= kMaxProbes; ++attempt) 
    {
        index %= table_size ;

        
        if (table[index] == SLOTEMPTY)
        {
            
            return false;
        }
        if ((unsigned)(table[index]>>32) == key)
        {
            value =(unsigned)(table[index]&0xffffffff);
            return true;
        }

        // Move the insertion index .
        if (method == LINEAR)
        {
            index += 1;
            
        }
        else if (method == QUADRATIC)
        {
            index += attempt * attempt;
        }
        else
        {
            index += attempt * double_hash_jump;
        }
    }
    return false ;

}
__global__ void
find_table(Entry* table,
           unsigned table_size,
           unsigned int *key,
           unsigned int *value,
           unsigned int NUM_FIND)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    if(idx<NUM_FIND)
    {
        unsigned KEY=key[idx];
        bool b=find_key(KEY,value[idx],table_size,table);
/*      if(b==true){
           printf("find key:%d v:%d idx:%d\n",KEY,value[idx],idx);
        }else{
           printf("not find key:%d v:%d idx:%d\n",KEY,value[idx],idx);
        }
*/
    }
}

//kernel_creat_table
__global__ void 
reszie(Entry** table,
        unsigned size,
        Entry** newtable;
        int NUM_INSERT)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    Entry *c=(*table);
    *table=*newtable;
    *newtable=c;

    if(idx<NUM_INSTER)
    {
        insert_entry((unsigned)((*newtable)[idx]>>32),
                    (unsigned)((*newtable)[idx]&0xffffffff),
                    size,*table);
    }
}

//main
int main(int argc, char** argv)
{
    int KEYS=10000;
    if (argc >=2 ) 
    {
        N=atoi(argv[1]);
    }
    St=1.25*N;
    if (argc >=3 ) 
    {
        KEYS=atoi(argv[2]);
    }

 
   



    // Allocate hash table
    Entry *h_table,*d_table;
    h_table = (Entry*)malloc( St*sizeof(Entry ));
    cudaMalloc( (void**)&d_table, St*sizeof(Entry)  );
    cudaMemcpy( d_table, h_table, St*sizeof (Entry), cudaMemcpyHostToDevice);
    //init with SlotEmpty
    initkernrl<<<(St+NUM_THREADS-1)/NUM_THREADS,NUM_THREADS>>>(d_table,St);

    //key vlaue  init
    unsigned  int *h_key,*h_value;
    h_key=(unsigned int *)malloc(N*sizeof(unsigned int));
    h_value=(unsigned int *)malloc(N*sizeof(unsigned int));
//int maxrand=0;
    for(int i=0;i<N;++i){
        h_value[i]=i;
        h_key[i]=rand()%KEYS;
//maxrand=(h_key[i]>maxrand?h_key[i]:maxrand);
    
       // if(i<100)
         //   std::cout<<i<<"="<<h_key[i]<<std::endl;
    }
  //printf("maxrand : %d\n",maxrand);
    unsigned  int *d_key,*d_value;
    cudaMalloc( (void**)&d_key, N*sizeof(unsigned int)  );
    cudaMemcpy( d_key, h_key, N*sizeof (unsigned int), cudaMemcpyHostToDevice);
    cudaMalloc( (void**)&d_value, N*sizeof(unsigned int)  );
    cudaMemcpy( d_value, h_value, N*sizeof (unsigned int), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //kernel for creat
    creat_table<<<(N+NUM_THREADS-1)/NUM_THREADS,NUM_THREADS>>>
                                            (d_table,St,d_key,d_value,N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // Print kernel execution time in milliseconds
    printf("creat_kernel time:%lf ms (in milliseconds)\n",time);
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //kernel for creat
    find_table<<<(N+NUM_THREADS-1)/NUM_THREADS,NUM_THREADS>>>
                                            (d_table,St,d_key,d_value,N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("creat_kernel time:%lf ms (in milliseconds)\n",time);


    //copy back
    //Entry *h_out_table;
    //h_out_table = (Entry*)malloc( St*sizeof(Entry ));
    //cudaMemcpy( h_out_table, d_table, St*sizeof (Entry), cudaMemcpyDeviceToHost   );


    return 0;
}

//read in argc=3
//argv[1]=k  St=kN
//argv[2]=N  N=num key
//alloc h_in
//set k in h_in[]
//alloc d_in
//memcpy

//kenrel_creat

//kenrel_find

//alloc memcpy out_d


