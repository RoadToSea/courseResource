#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime.h>   
typedef unsigned char uchar;

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    

//check cuda error and set cuda device
void checkErr(cudaError err,int num);     /*check cuda errors*/
void setCudaDevice(int devNum);

//define gray-scale images
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);
//edge detection for gray-scale images
PGM_IMG edge_sobel(PGM_IMG img_in);
void sobel_cpu_kernel(uchar * img_in, uchar * img_out,int img_w, int img_h);
int sobel(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f);
//call edge detection
void run_cpu_sobel_test(PGM_IMG img_in);

void run_gpu_sobel_test(PGM_IMG img_in);
PGM_IMG gpu_edge_sobel(PGM_IMG img_in);
__global__ void sobel_gpu_kernel(uchar * img_in, uchar * img_out,int img_w, int img_h);
__device__ int sobel_gpu(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f);
int main()
{
	int devNum=0;   
    setCudaDevice(devNum);  
    PGM_IMG img_ibuf;
//	clock_t start, finish;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1,0);
//    double duration; 
//	start = clock();
    printf("Running edge detection for gray-scale images.\n\n");
	printf("---------------Read PGM-----------------\n");
    img_ibuf = read_pgm("lena.pgm");
    run_cpu_sobel_test(img_ibuf);
//	run_gpu_sobel_test(img_ibuf);
    free_pgm(img_ibuf); 
//	finish = clock();
     cudaEventRecord(stop1,0);
     cudaEventSynchronize(stop1);
     float time1;
     cudaEventElapsedTime(&time1,start1,stop1);
     
//	duration = (double)(finish - start) ; //CLOCKS_PER_SEC=1000,毫秒
    printf( "The time of calculating is :%f\n", time1); 

    return 0;
}
/*-----------------------------------------------------------*/
/*---------------------two useful functions------------------*/
/*----------------------do not modify------------------------*/
/*-----------------------------------------------------------*/
 void checkErr(cudaError err,int num)     /*check cuda errors*/
{
	 if( cudaSuccess != err) {  
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                __FILE__, num-1, cudaGetErrorString( err) );              
	 }
}
void setCudaDevice(int devNum)
{
	cudaError_t err = cudaSuccess;
	printf("\nCUDA Device #%d\n", devNum);
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, devNum);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
	printf("Major revision number:         %d\n",  devProp.major);
	printf("Minor revision number:         %d\n",  devProp.minor);
	err=cudaSetDevice(devNum);
	checkErr(err,__LINE__);
}
/*-----------------------------------------------------------*/
/*----------------------read and write pgm picture-----------*/
/*----------------------do not modify------------------------*/
/*-----------------------------------------------------------*/
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256]; 
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); /*  Skip the magic number,P2/P5   */
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);  
    printf("size:%s\n",sbuf);
    printf("Image size: %d x %d\n", result.w, result.h);
	printf("v_max:%d\n",v_max);
    
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file); //to result.img
	printf("Read the picture succeed!\n\n");
    fclose(in_file);
    
    return result;   //PGM_IMG
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
	printf("write the file............\n");
}
void free_pgm(PGM_IMG img)
{
    free(img.img);
}
/*-----------------------------------------------------------*/
/*-------------CPU sobel edge detection test-----------------*/
/*-----------------------------------------------------------*/
void run_cpu_sobel_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    printf("Starting CPU processing...\n");
    img_obuf = edge_sobel(img_in);  //edge detection
    printf("Edge detection of the picture!\n");
    write_pgm(img_obuf, "result_PGM.pgm");
	printf("Write new file succeed!\n\n");
    free_pgm(img_obuf);
}
/*-----------------------------------------------------------*/
/*-------------CPU sobel operator implementation-------------*/
/*-----------------------------------------------------------*/
PGM_IMG edge_sobel(PGM_IMG img_in)
{
    PGM_IMG result;
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	memset(result.img,0,result.w * result.h * sizeof(unsigned char));
    sobel_cpu_kernel(img_in.img, result.img,result.w,result.h);
    return result;
}
void sobel_cpu_kernel(uchar * img_in, uchar * img_out,int img_w, int img_h){
    for(int i=1;i<img_h-1;i++)  
    {  
        for(int j=1;j<img_w-1;j++)  
        {  
            //通过指针遍历图像上每一个像素  
			/*---------------------------------*/
			/*----------x1----x2----x3---------*/
			/*----------x4----x5----x6---------*/
			/*----------x7----x8----x9---------*/
			/*---------------------------------*/
			int x1, x2, x3, x4, x5, x6, x7, x8,x9;
			x1=img_in[(i-1)*img_w+(j-1)];
			x2=img_in[(i-1)*img_w+j];
			x3=img_in[(i-1)*img_w+j+1];
			x4=img_in[i*img_w+j-1];
			x5=img_in[i*img_w+j]; // never use x5
			x6=img_in[i*img_w+j+1];
			x7=img_in[(i+1)*img_w+j-1];
			x8=img_in[(i+1)*img_w+j];
			x9=img_in[(i+1)*img_w+j+1];
			int dfdy= sobel(x1, x2, x3, x7, x8, x9);
            int dfdx= sobel(x1, x4, x7, x3, x6, x9);	
			int gradient= sqrtf(dfdy*dfdy+dfdx*dfdx);
			img_out[i*img_w+j] = gradient;

		}  
    }
}
int sobel(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f) {
	return ((a + 2*b + c) - (d + 2*e + f));
}
/*-----------------------------------------------------------*/
/*-------------GPU sobel edge detection test-----------------*/
/*-----------------------------------------------------------*/
void run_gpu_sobel_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    printf("Starting GPU processing...\n");
    img_obuf = gpu_edge_sobel(img_in);  //edge detection
    printf("Edge detection of the picture!\n");
    write_pgm(img_obuf, "result_PGM.pgm");
	printf("Write new file succeed!\n\n");
    free_pgm(img_obuf);
}

/*-----------------------------------------------------------*/
/*-------------GPU sobel operator implementation-------------*/
/*-----------------------------------------------------------*/
PGM_IMG gpu_edge_sobel(PGM_IMG img_in)
{
    PGM_IMG img_out;
    img_out.w = img_in.w;
    img_out.h = img_in.h;
	size_t size = img_out.w * img_out.h * sizeof(uchar);
    img_out.img = (uchar *)malloc(size);
	memset(img_out.img,0,size);
	
    // Allocate the device memory for result.img
    uchar *img_temp=NULL, *d_img_in = NULL,*d_img_out = NULL;

	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_img_in, size);
    checkErr(err,__LINE__);
	err = cudaMemcpy(d_img_in, img_in.img, size, cudaMemcpyHostToDevice);
    checkErr(err,__LINE__);

	err = cudaMalloc((void **)&d_img_out, size);
    checkErr(err,__LINE__);
	err = cudaMemcpy(d_img_out, img_out.img, size, cudaMemcpyHostToDevice);
    checkErr(err,__LINE__);

	dim3 threadsPerBlock(16,16);
	dim3 blocksPerGrid((img_in.w+15)/16,(img_in.h+15)/16);

    sobel_gpu_kernel<<<blocksPerGrid,threadsPerBlock>>>(d_img_in, d_img_out,img_in.w,img_in.h);
	err = cudaGetLastError();

	err = cudaMemcpy(img_out.img, d_img_out, size, cudaMemcpyDeviceToHost);
    checkErr(err,__LINE__);
	cudaFree(d_img_in);
	cudaFree(d_img_out);
	//free(img_temp);
    return img_out;
}
__global__ void sobel_gpu_kernel(uchar * img_in, uchar * img_out,int img_w, int img_h)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

    if((row >= 0) && (row < img_h) && (col >= 0) && (col < img_w))
    {
			uchar x1, x2, x3, x4, x5, x6, x7, x8,x9;
			x1=img_in[(row-1)*img_w+(col-1)];
			x2=img_in[(row-1)*img_w+col];
			x3=img_in[(row-1)*img_w+col+1];
			x4=img_in[row*img_w+col-1];
			//x5=img_in[i*img_w+j]; // never use x5
			x6=img_in[row*img_w+col+1];
			x7=img_in[(row+1)*img_w+col-1];
			x8=img_in[(row+1)*img_w+col];
			x9=img_in[(row+1)*img_w+col+1];
			int dfdy= sobel_gpu(x1, x2, x3, x7, x8, x9);
            int dfdx= sobel_gpu(x1, x4, x7, x3, x6, x9);	
			int gradient= sqrtf(dfdy*dfdy+dfdx*dfdx);
			img_out[row*img_w+col] = gradient; 
    }
}
__device__ int sobel_gpu(uchar a, uchar b, uchar c, uchar d, uchar e, uchar f){
	return ((a + 2*b + c) - (d + 2*e + f));
}
