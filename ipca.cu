#include <stdio.h>
#include <jpeglib.h>

#define COUNT 100
#define DIMENSION 16
#define STRIDE 3072
#define AMNESIC 2.0
#define SIZE 32

double* loadJpeg(char* path, double* out=NULL);
bool saveJpeg(char* path, double* i, int height, int width);
unsigned char stepInt(double v, double min = -1.0, double max = 1.0);
double searchMin(double* img, int height, int width);
double searchMax(double* img, int height, int width);

__global__ void ipca_kernel( int current, int length, double* tableIn, double* tableU, double* tableV, int* tableFrame ) 
{
	double imgA[STRIDE];
	double imgB[STRIDE];
	double imgC[STRIDE];

	///// thread id -> dimension id
	const unsigned int tid = threadIdx.x;
	tableFrame[tid] = -1;

	for(int f = 0; f < length; f++)
	{
		__syncthreads();

		int frame = current + f;
		if(tid > 0 && tableFrame[tid-1] >= f) continue;

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
			tableFrame[tid] = f;
			continue;
		}

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
		
		tableFrame[tid] = f;
	}
	tableFrame[tid] = length;
}

int main(void)
{
	double* images = new double[STRIDE * COUNT];
	double* U = new double[STRIDE * DIMENSION];
	double* V = new double[STRIDE * DIMENSION];
	int* Frame = new int[DIMENSION];

	printf("start count=%d\n", COUNT);
	char path[256];
	for(int c = 0; c < COUNT; c++)
	{
		sprintf(path, "data/%04d.jpg", c);
		loadJpeg(path, images + STRIDE * c);
	}
	for(int p = 0; p < STRIDE * DIMENSION; p++)
	{
		U[p] = 0;
		V[p] = 0;
	}
	double *tableIn, *tableU, *tableV;
	int *tableFrame;
 	cudaMalloc(&tableIn, sizeof(images));
 	cudaMalloc(&tableU, sizeof(U));
 	cudaMalloc(&tableV, sizeof(V));
	cudaMalloc(&tableFrame, sizeof(Frame));
	cudaMemcpy(images, tableIn, sizeof(images), cudaMemcpyHostToDevice);
	cudaMemcpy(U, tableU, sizeof(U), cudaMemcpyHostToDevice);
	cudaMemcpy(V, tableV, sizeof(V), cudaMemcpyHostToDevice);

	dim3 block (DIMENSION, 1, 1);
	dim3 grid  (1, 1, 1);
	ipca_kernel<<<grid, block>>>(0, COUNT, tableIn, tableU, tableV, tableFrame);

	cudaMemcpy(U, tableU, sizeof(U), cudaMemcpyDeviceToHost);
	cudaMemcpy(V, tableV, sizeof(V), cudaMemcpyDeviceToHost);
	cudaMemcpy(Frame, tableFrame, sizeof(Frame), cudaMemcpyDeviceToHost);

	cudaFree(tableIn);
	cudaFree(tableU);
	cudaFree(tableV);
	cudaFree(tableFrame);

	for(int d = 0; d < DIMENSION; d++)
	{
		double* img = V + STRIDE * d;
		printf("%02d: min=%f, max=%f frame=%d\n", d, searchMin(img, SIZE, SIZE), searchMax(img, SIZE, SIZE), Frame[d]);
		sprintf(path, "result/%02d.jpg", d);
		saveJpeg(path, img, SIZE, SIZE);
	}

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

	//printf("jpeg:%s (%d,%d) min=%f max=%f\n", path, height, width, searchMin(out, height, width), searchMax(out, height, width));
	return out;
}

unsigned char stepInt(double v, double min, double max)
{
	if(v <= min) return 0;
	if(v >= max) return 255;
	return (unsigned char)( (v - min) / (max - min) * 255.0 );
}

bool saveJpeg(char* path, double* data, int height, int width)
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
			img[i][j*3 + 0] = stepInt( data[loc + 0] );
			img[i][j*3 + 1] = stepInt( data[loc + 1] );
			img[i][j*3 + 2] = stepInt( data[loc + 2] );
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
