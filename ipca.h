#ifndef IPCA_H_
#define IPCA_H_

#define COUNT 300
#define DIMENSION 16
#define STRIDE 3072
#define AMNESIC 1.0

typedef struct ipca_t {
	int Count;
	int Stride;
	int Dimension;

	int frame;

	double* hostImages;
	double* hostU;
	double* hostV;
	int* hostFrames;

	double* deviceImages;
	double* deviceU;
	double* deviceV;
	int* deviceFrames;

	size_t sizeImages, sizeU, sizeV, sizeFrames;
} ipca_t;

ipca_t* ipca_initialize();
void ipca_run(ipca_t* t, double* images);
void ipca_sync(ipca_t* t);
void ipca_finalize(ipca_t* t);
void ipca_check(ipca_t* t);

#endif

