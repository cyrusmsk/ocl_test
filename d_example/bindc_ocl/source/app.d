module app;

import bindbc.opencl;

static MAX_SOURCE_SIZE = 0x100000;

void main()
{
    int i;
    const LIST_SIZE = 1024;
    int[] A = new int[1024];
    int[] B = new int[1024];
    foreach(i; 0..LIST_SIZE) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }
    // Load the kernel source code

}
