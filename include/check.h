#ifndef CHECK_H
#define CHECK_H

#define CUDA_ERR_CHECK(x)                                  \
    do { cudaError_t err = x; if (err != cudaSuccess) {    \
        fprintf(stderr, "CUDA error %d \"%s\" at %s:%d\n", \
            (int)err, cudaGetErrorString(err),             \
            __FILE__, __LINE__);                           \
        abort();                                           \
    }} while (0)

#endif // CHECK_H

