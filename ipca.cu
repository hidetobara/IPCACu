#include <stdio.h>
#include <jpeglib.h>
#include <cuda.h>

#define COUNT 300
#define DIMENSION 16
#define STRIDE 3072
#define AMNESIC 1.0
#define SIZE 32

double* loadJpeg(char* path, double* out=NULL);
void loadJpegs(char* path, double* out, int start, int count);
bool saveJpeg(char* path, double* i, int height, int width, double min, double max);
unsigned char stepInt(double v, double min = -1.0, double max = 1.0);
double searchMin(double* img, int height, int width);
double searchMax(double* img, int height, int width);
void calcStatistics(double* img, int height, int width, double* average, double* deviation);

__global__ void kernel(int* tableFrame)
{
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        tableFrame[tid] = tid;
}

__global__ void ipca_kernel( int current, int length, double* tableIn, double* tableU, double* tableV, int* tableFrame ) 
{
	double* imgA = (double*)malloc(sizeof(double)*STRIDE);
	double* imgB = (double*)malloc(sizeof(double)*STRIDE);
	double* imgC = (double*)malloc(sizeof(double)*STRIDE);

	///// thread id -> dimension id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	tableFrame[tid] = -1;

	for(int f = -tid; f < length; f++)
	{
		__syncthreads();

		int frame = current + f;
		if(tid == 0) tableFrame[0] = frame;
		double* strideIn = tableIn + STRIDE * f;
		double* strideV = tableV + STRIDE * tid;
		double* strideU = tableU + STRIDE * tid;

		if(tid == 0)
		{
			for(int s=0; s<STRIDE; s++) strideU[s] = strideIn[s];
		}
		if(tid == frame)
		{
			for(int s=0; s<STRIDE; s++) strideV[s] = strideU[s];
			//tableFrame[tid] = f;
			continue;
		}
		if(tid > frame) continue;

		if(f == length/2) tableFrame[tid] = tableFrame[0];

		///// Vi(n) = [a= (n-1-l)/n * Vi(n-1)] + [b= (1+l)/n * Ui(n)T Vi(n-1)/|Vi(n-1)| * Ui(n) ]
		double nrmV = 0;
		for(int s=0; s<STRIDE; s++){ nrmV += strideV[s]*strideV[s]; }
		nrmV = sqrt(nrmV);
		double dotUV = 0;
		for(int s=0; s<STRIDE; s++){ dotUV += strideV[s]*strideU[s]; }
		double scalerA = ((double)frame - 1.0 - AMNESIC) / frame;
		double scalerB = (1.0 + AMNESIC) * dotUV / ((double)frame * nrmV);

		for(int s=0; s<STRIDE; s++) imgA[s] = strideV[s] * scalerA;
		for(int s=0; s<STRIDE; s++) imgB[s] = strideU[s] * scalerB;
		for(int s=0; s<STRIDE; s++) strideV[s] = imgA[s] + imgB[s];

		///// Ui+1(n) = Ui(n) - [c= Ui(n)T Vi(n)/|Vi(n)| * Vi(n)/|Vi(n)| ]
		if(tid >= DIMENSION - 1) continue;

		nrmV = 0;
		for(int s=0; s<STRIDE; s++){ nrmV += strideV[s]*strideV[s]; }
		nrmV = sqrt(nrmV);
		dotUV = 0;
		for(int s=0; s<STRIDE; s++){ dotUV += strideV[s]*strideU[s]; }
		double scalerC = dotUV / (nrmV * nrmV);

		for(int s=0; s<STRIDE; s++) imgC[s] = strideV[s] * scalerC;
		for(int s=0; s<STRIDE; s++) strideU[STRIDE + s] = strideU[s] - imgC[s];	
	}
	//tableFrame[tid] = length;
	free(imgA);
	free(imgB);
	free(imgC);
}

int main(void)
{
	double* images = new double[STRIDE * COUNT];
	double* U = new double[STRIDE * DIMENSION];
	double* V = new double[STRIDE * DIMENSION];
	int* Frame = new int[DIMENSION];

	for(int p = 0; p < STRIDE * DIMENSION; p++)
	{
		U[p] = 0;
		V[p] = 0;
	}
	double *tableIn, *tableU, *tableV;
	int *tableFrame;
	size_t sizeIn = sizeof(double)*STRIDE*COUNT;
	size_t sizeU = sizeof(double)*STRIDE*DIMENSION;
	size_t sizeV = sizeof(double)*STRIDE*DIMENSION;
	size_t sizeFrame = sizeof(int)*DIMENSION;
 	cudaMalloc(&tableIn, sizeIn);
 	cudaMalloc(&tableU, sizeU);
 	cudaMalloc(&tableV, sizeV);
	cudaMalloc(&tableFrame, sizeFrame);
        printf("0. %s\n", cudaGetErrorString(cudaGetLastError()));

	cudaMemcpy(tableU, U, sizeU, cudaMemcpyHostToDevice);
	cudaMemcpy(tableV, V, sizeV, cudaMemcpyHostToDevice);
        printf("1. %s\n", cudaGetErrorString(cudaGetLastError()));

	clock_t start, end;
	start = clock();
	for(int bb = 0; bb < 12; bb++)
	{
		int b = bb % 4;
		loadJpegs("data/%04d.jpg", images, COUNT * b, COUNT);
		cudaMemcpy(tableIn, images, sizeIn, cudaMemcpyHostToDevice);

		dim3 grid(1,1,1);
		dim3 block(16,1,1);
		ipca_kernel<<<grid, block>>>(COUNT*b, COUNT, tableIn, tableU, tableV, tableFrame);
		printf("2.%d. %s\n", bb, cudaGetErrorString(cudaGetLastError()));
	}
	end = clock();
	printf("2. %f sec\n", (double)(end - start) / CLOCKS_PER_SEC);

	cudaMemcpy(U, tableU, sizeU, cudaMemcpyDeviceToHost);
	cudaMemcpy(V, tableV, sizeV, cudaMemcpyDeviceToHost);
	cudaMemcpy(Frame, tableFrame, sizeFrame, cudaMemcpyDeviceToHost);
        printf("3. %s\n", cudaGetErrorString(cudaGetLastError()));

	char path[256];
	double ave, dev;
	for(int d = 0; d < DIMENSION; d++)
	{
		double* img = V + STRIDE * d;
		double min = searchMin(img, SIZE, SIZE);
		double max = searchMax(img, SIZE, SIZE);
		calcStatistics(img, SIZE, SIZE, &ave, &dev);
		printf("%02d: min=%f, max=%f ave=%f dev=%f frame=%d\n", d, min, max, ave, dev, Frame[d]);
		sprintf(path, "result/%02d.jpg", d);
		saveJpeg(path, img, SIZE, SIZE, ave - dev * 2, ave + dev * 2);
	}

        cudaFree(tableIn);
        cudaFree(tableU);
        cudaFree(tableV);
        cudaFree(tableFrame);

	delete(images);
	delete(U);
	delete(V);
	delete(Frame);
	printf("end\n");
}

double* loadJpeg(char* path, double* out)
{
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	FILE *infile;

	JSAMPARRAY img;
	int i, j;
	int width;
	int height;

	// initialize JPEG object
	cinfo.err = jpeg_std_error( &jerr );
	jpeg_create_decompress( &cinfo );

	// open file
	infile = fopen( path, "rb" );
	if(infile == NULL) return NULL;
	jpeg_stdio_src( &cinfo, infile );

	// read header
	jpeg_read_header( &cinfo, TRUE );

	// start decompress
	jpeg_start_decompress( &cinfo );

	// get height, width
	width = cinfo.output_width;
	height = cinfo.output_height;

	// prepare memory
	img = (JSAMPARRAY)malloc( sizeof( JSAMPROW ) * height );
	for ( i = 0; i < height; i++ ) {
		img[i] = (JSAMPROW)calloc( sizeof( JSAMPLE ), 3 * width );
	}

	// retrieve
	while( cinfo.output_scanline < cinfo.output_height ) {
		jpeg_read_scanlines( &cinfo,
			img + cinfo.output_scanline,
			cinfo.output_height - cinfo.output_scanline
		);
	}

	// end decompress
	jpeg_finish_decompress( &cinfo );

	// destroy JPEG object
	jpeg_destroy_decompress( &cinfo );

	// close file
	fclose( infile );

	// to double array
	if(out == NULL) out = new double[width*height*3];
	for ( i = 0; i < height; i++ ){
		for ( j = 0; j < width; j++ ) {
			int loc = (width*i+j)*3;
			out[loc + 0] = (double)img[i][j*3+0] / 255.0;
			out[loc + 1] = (double)img[i][j*3+1] / 255.0;
			out[loc + 2] = (double)img[i][j*3+2] / 255.0;
		}
	}

	// free memory
	for ( i = 0; i < height; i++ ) free( img[i] );
	free( img );

	//printf("jpeg:%s (%d,%d) min=%f max=%f", path, height, width, searchMin(out, height, width), searchMax(out, height, width));
	return out;
}

void loadJpegs(char* tpl, double* images, int start, int count)
{
        char path[256];
        for(int c = 0; c < count; c++)
        {
                sprintf(path, tpl, start + c);
                loadJpeg(path, images + STRIDE * c);
        }
}

unsigned char stepInt(double v, double min, double max)
{
	if(v <= min) return 0;
	if(v >= max) return 255;
	return (unsigned char)( (v - min) / (max - min) * 255.0 );
}

bool saveJpeg(char* path, double* data, int height, int width, double min, double max)
{
	/* JPEG Object, Error handling */
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;

	/* Error handling for default */
	cinfo.err = jpeg_std_error(&jerr);

	/* initiazlie JPEG Object */
	jpeg_create_compress(&cinfo);

	/* open output file */
	FILE *fp = fopen(path, "wb");
	if (fp == NULL) {
		fprintf(stderr, "cannot open %s\n", path);
		return false;
	}
	jpeg_stdio_dest(&cinfo, fp);

	/* image settings */
	cinfo.image_width = width;
	cinfo.image_height = height;
	cinfo.input_components = 3;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, 75, TRUE);

	/* start compressing */
	jpeg_start_compress(&cinfo, TRUE);

	/* RGB */
	JSAMPARRAY img = (JSAMPARRAY) malloc(sizeof(JSAMPROW) * height);
	for (int i = 0; i < height; i++) {
		img[i] = (JSAMPROW) malloc(sizeof(JSAMPLE) * 3 * width);
		for (int j = 0; j < width; j++) {
			int loc = (i * width + j) * 3;
			img[i][j*3 + 0] = stepInt( data[loc + 0], min, max);
			img[i][j*3 + 1] = stepInt( data[loc + 1], min, max);
			img[i][j*3 + 2] = stepInt( data[loc + 2], min, max );
		}
	}
	/* write */
	jpeg_write_scanlines(&cinfo, img, height);

	/* end compressing */
	jpeg_finish_compress(&cinfo);

	/* destroy JPEG object */
	jpeg_destroy_compress(&cinfo);

	for (int i = 0; i < height; i++) {
		free(img[i]);
	}
	free(img);
	fclose(fp);
	return true;
}

double searchMin(double* img, int height, int width)
{
	double* pEnd = img + height * width * 3;
	double min = 0;
	for(double* p = img; p < pEnd; p++) if(min > *p) min = *p;
	return min;
}
double searchMax(double* img, int height, int width)
{
	double* pEnd = img + height * width * 3;
	double max = 0;
	for(double* p = img; p < pEnd; p++) if(max < *p) max = *p;
	return max;
}
void calcStatistics(double* img, int height, int width, double* average, double* deviation)
{
	double* pEnd = img + height * width * 3;
	double amount = 0;
	double distribution = 0;
	for(double* p = img; p < pEnd; p++)
	{
		amount += *p;
		distribution += (*p) * (*p);
	}
	
	amount = amount / (height * width * 3);
	distribution = distribution / (height * width * 3) - amount * amount;
	if(average != NULL) *average = amount;
	if(deviation != NULL) *deviation = sqrt(distribution); 
}
