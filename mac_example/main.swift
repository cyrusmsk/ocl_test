#if canImport(Darwin) // Apple platforms
import OpenCL
#else
// Don't know how to import OpenCL on Linux yet. It should be possible if I
// wrap it in a Clang module and link that to Swift.
#endif

/*
 Sourced from this online tutorial:
 https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/
 
 In the same directory as this script:
 - "vector_add_kernel.cl"
 - source:
 
 __kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
  
     // Get the index of the current element to be processed
     int i = get_global_id(0);
  
     // Do the operation
     C[i] = A[i] + B[i];
 }
 */

let MAX_SOURCE_SIZE = 0x100000

let LIST_SIZE = 1024
let A = UnsafeMutablePointer<Int32>.allocate(capacity: LIST_SIZE)
let B = UnsafeMutablePointer<Int32>.allocate(capacity: LIST_SIZE)
for i in 0..<LIST_SIZE {
  A[i] = Int32(i)
  B[i] = Int32(LIST_SIZE - i)
}

// Load the kernel source code into the array source_str
guard let fp = fopen("vector_add_kernel.cl", "r") else {
  fatalError("Failed to load kernel.")
}

let source_str = UnsafeMutablePointer<CChar>.allocate(capacity: MAX_SOURCE_SIZE)
var source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp)
fclose(fp)

// Get platform and device information
var platform_id: cl_platform_id? = nil
var device_id: cl_device_id? = nil
var ret_num_devices: cl_uint = 0
var ret_num_platforms: cl_uint = 0
var ret: cl_int = clGetPlatformIDs(1, &platform_id, &ret_num_platforms)
ret = clGetDeviceIDs(platform_id, cl_device_type(CL_DEVICE_TYPE_DEFAULT), 1,
  &device_id, &ret_num_devices)

// Create OpenCL context
let context: cl_context = clCreateContext(nil, 1, &device_id, nil, nil, &ret)

// Create a command queue
let command_queue: cl_command_queue = clCreateCommandQueue(context, device_id, 0,
  &ret)

// Create memory buffers on the device for each vector
var a_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)
var b_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)
var c_mem_obj: cl_mem = clCreateBuffer(context, cl_mem_flags(CL_MEM_READ_ONLY),
  LIST_SIZE * MemoryLayout<Int32>.stride, nil, &ret)

// Copy the lists A and B to their respective memory buffers
ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, A, 0, nil, nil)
ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, B, 0, nil, nil)

// Create a program from the kernel source
var source_str_casted = unsafeBitCast(source_str, to: UnsafePointer<CChar>?.self)
let program: cl_program = clCreateProgramWithSource(context, 1,
  &source_str_casted, &source_size, &ret)

// Build the program
ret = clBuildProgram(program, 1, &device_id, nil, nil, nil)

// Create the OpenCL kernel
let kernel: cl_kernel = clCreateKernel(program, "vector_add", &ret)

// Set the arguments of the kernel
let size_of_cl_mem = MemoryLayout<cl_mem>.stride
ret = clSetKernelArg(kernel, 0, size_of_cl_mem, &a_mem_obj)
ret = clSetKernelArg(kernel, 1, size_of_cl_mem, &b_mem_obj)
ret = clSetKernelArg(kernel, 2, size_of_cl_mem, &c_mem_obj)

// Execute the OpenCL kernel on the list
var global_item_size = LIST_SIZE
var local_item_size = 64
ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nil, &global_item_size,
  &local_item_size, 0, nil, nil)

// Read the memory buffer C on the device to the local variable C
let C = UnsafeMutablePointer<Int32>.allocate(capacity: LIST_SIZE)
ret = clEnqueueReadBuffer(command_queue, c_mem_obj, cl_bool(CL_TRUE), 0,
  LIST_SIZE * MemoryLayout<Int32>.stride, C, 0, nil, nil)

// Display the result to the screen
for i in 0..<LIST_SIZE {
  print("\(A[i]) + \(B[i]) = \(C[i])")
  
  /*
   Output:
   
   0 + 1024 = 1024
   1 + 1023 = 1024
   2 + 1022 = 1024
   3 + 1021 = 1024
   4 + 1020 = 1024
   5 + 1019 = 1024
   6 + 1018 = 1024
   ...
   1019 + 5 = 1024
   1020 + 4 = 1024
   1021 + 3 = 1024
   1022 + 2 = 1024
   1023 + 1 = 1024
   */
}

// Clean up
ret = clFlush(command_queue)
ret = clFinish(command_queue)
ret = clReleaseKernel(kernel)
ret = clReleaseProgram(program)
ret = clReleaseMemObject(a_mem_obj)
ret = clReleaseMemObject(b_mem_obj)
ret = clReleaseMemObject(c_mem_obj)
ret = clReleaseCommandQueue(command_queue)
ret = clReleaseContext(context)
free(A)
free(B)
free(C)
exit(0)
