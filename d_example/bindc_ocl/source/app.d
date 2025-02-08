module app;

import core.stdc.stdio : printf, fread, fopen, fclose, FILE;
import core.stdc.stdlib : EXIT_FAILURE, exit, malloc, free;
import bindbc.opencl;

static MAX_SOURCE_SIZE = 0x100000;

void main()
{
    auto support = loadOpenCL();

    const int LIST_SIZE = 1024;
    auto A = cast(int*) malloc(int.sizeof * LIST_SIZE);
    auto B = cast(int*) malloc(int.sizeof * LIST_SIZE);
    for (int i = 0; i < LIST_SIZE; i++)
    {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }
    // Load the kernel source code
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("vector_add_kernel.cl", "r");
    if (!fp)
    {
        printf("Failed to load kernel\n");
        exit(1);
    }
    source_str = cast(char*) malloc(MAX_SOURCE_SIZE);
    scope (exit)
        free(source_str);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = null;
    cl_device_id device_id = null;
    cl_uint ret_num_devices, ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
        &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(null, 1, &device_id, null, null, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * int.sizeof, null, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * int.sizeof, null, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        LIST_SIZE * int.sizeof, null, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE * int.sizeof, A, 0, null, null);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        LIST_SIZE * int.sizeof, B, 0, null, null);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        cast(const char**)&source_str, cast(const size_t*)&source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, null, null, null);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, cl_mem.sizeof, cast(void*)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, cl_mem.sizeof, cast(void*)&b_mem_obj);
    ret = clSetKernelArg(kernel, 2, cl_mem.sizeof, cast(void*)&c_mem_obj);

    for (int i = 0; i < 100_000; i++)
    {
        // Execute the OpenCL kernel on the list
        size_t global_item_size = LIST_SIZE; // Process the entire lists
        size_t local_item_size = 64; // Divide work items into groups of 64
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, null,
            &global_item_size, &local_item_size, 0, null, null);
    }

    // Read the memory buffer C on the device to the local variable C
    int* C = cast(int*) malloc(int.sizeof * LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
        LIST_SIZE * int.sizeof, C, 0, null, null);

    // Display the result to the screen
    for (int i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
}
