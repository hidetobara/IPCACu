#include <stdio.h>
#include <jpeglib.h>
#include <math.h>
#include "image.h"
#include "ipca.h"

#define SIZE 32

void loadJpegs(ipca_t* t, char* tpl, double* images, int start, int count);
void savePngs(ipca_t* t, char* dir, double* V);
unsigned char stepInt(double v, double min, double max);
double searchMin(double* img, int height, int width);
double searchMax(double* img, int height, int width);
void calcStatistics(double* img, int height, int width, double* average, double* deviation);


int main(int argc, char**argv)
{
	ipca_t* t = ipca_initialize();
	for(int ite = 0; ite < 2; ite++)
	{
		for(int c = 0; c < 4; c++)
		{
			loadJpegs(t, "data/%04d.jpg", t->hostImages, t->Count * c, t->Count);
			ipca_run(t, t->hostImages);
		}
	}
	ipca_sync(t);
	ipca_check(t);
	savePngs(t, "result", t->hostV);
	ipca_finalize(t);
}

void loadJpegs(ipca_t* t, char* tpl, double* images, int start, int count)
{
	printf("loadJpegs(): number=%d-%d\n", start, start+count);
        char path[256];
	image_t* it;
        for(int c = 0; c < count; c++)
        {
                sprintf(path, tpl, start + c);
		it = read_jpeg_file((const char*)path);
		if(it == NULL){ printf("WARN: not read jpeg path=%s\n", path); continue; }
		if(it->width * it->height * 3 != t->Stride){ printf("WARN: stride is invalid size path=%s\n", path); continue; }
		double* pi = images + t->Stride * c;
		for(int h = 0; h < it->height; h++)
		{
			for(int w = 0; w < it->width; w++, pi+=3)
			{
				pi[0] = it->map[h][w].c.r / 255.0;
				pi[1] = it->map[h][w].c.g / 255.0;
				pi[2] = it->map[h][w].c.b / 255.0;
			}
		}
		free_image(it);
        }
	//printf("loadJpegs(): OK\n");
}

void savePngs(ipca_t* t, char* dir, double* V)
{
	char path[256];
	image_t* it = allocate_image(SIZE, SIZE, COLOR_TYPE_RGB);
	for(int d = 0; d < DIMENSION; d++)
	{
		double* img = V + t->Stride * d;
		double ave, dev;
		double* pi = img;
		calcStatistics(pi, SIZE, SIZE, &ave, &dev);
		for(int h = 0; h < it->height; h++)
		{
			for(int w = 0; w < it->width; w++, pi+=3)
			{
				it->map[h][w].c.r = stepInt(pi[0], ave - dev * 2, ave + dev * 2);
				it->map[h][w].c.g = stepInt(pi[1], ave - dev * 2, ave + dev * 2);
				it->map[h][w].c.b = stepInt(pi[2], ave - dev * 2, ave + dev * 2);
			}
		}
		sprintf(path, "%s/%02d.png", dir, d);
		write_png_file(path, it);
	}
	free_image(it);
}

unsigned char stepInt(double v, double min, double max)
{
	if(v <= min) return 0;
	if(v >= max) return 255;
	return (unsigned char)( (v - min) / (max - min) * 255.0 );
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

