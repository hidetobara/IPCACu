#nvcc -I ~/usr/include/ -L ~/usr/lib -ljpeg ipca.cu

#nvcc -c ipca.cu image.c jpeg.c png.c
#nvcc ipca.o image.o jpeg.o png.o -ljpeg -lpng

rm a.out
nvcc image.cpp jpeg.cpp png.cpp ipca.cu main.cpp -ljpeg -lpng
