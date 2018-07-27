#nvcc -I ~/usr/include/ -L ~/usr/lib -ljpeg ipca.cu

#nvcc -c ipca.cu image.c jpeg.c png.c
#nvcc ipca.o image.o jpeg.o png.o -ljpeg -lpng

nvcc main.cu image.c jpeg.c png.c -ljpeg -lpng
