import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from PIL import Image
import math 

import numpy

import cv2  


#Read the image as an RGB otherwise it will have 3 channels  
image = Image.open("dummy.png").convert('RGB')


#Get the dimesions of the image
weidth,height  =  image.size


#Convert it into a nummpy array 
a = numpy.array(image) 
a = a.astype(numpy.float32)
#Design an image keeper for the cuda kernal 
a_gpu = cuda.mem_alloc(a.nbytes)

#We would require to transfer the image dimesions to the  kernal 

totalpixels = numpy.array(list([height*weidth]))
rows = numpy.array(list([height]))
cols = numpy.array(list([weidth]))

#Design a dummy output
output = numpy.full((height,weidth),1)
#print(output)

output = output.astype(numpy.float32) 
output_gpu = cuda.mem_alloc(output.nbytes)


#Design the gpu memory
totalpixels = totalpixels.astype(numpy.float32)
totalpixels_gpu =  cuda.mem_alloc(totalpixels.nbytes) 


#Design the gpu memory
rows = totalpixels.astype(numpy.float32)
rows_gpu =  cuda.mem_alloc(totalpixels.nbytes) 

#Design the gpu memory
cols = totalpixels.astype(numpy.float32)
cols_gpu =  cuda.mem_alloc(totalpixels.nbytes) 


#Copy the data to the device memory from the host memory
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(output_gpu, output)
cuda.memcpy_htod(totalpixels_gpu,totalpixels)
cuda.memcpy_htod(rows_gpu,rows)
cuda.memcpy_htod(cols_gpu,cols)

#Wrtie the kernal
mod = SourceModule("""
  __global__ void convertor(float *rgb, float *grey , float *rows , float *cols   )
  {


	long long int col = blockIdx.x * 32*32 + threadIdx.x; 
	

	//Compute for only those threads which map directly to 
	//image grid
  		long long int grey_offset =  col;
		  if(grey_offset< cols[0]*rows[0])
		  {
			  	
		long long int rgb_offset = grey_offset *3;
	
    	float r = rgb[rgb_offset + 0];
	    float g = rgb[rgb_offset + 1];
	    float b = rgb[rgb_offset + 2];
	
	    grey[grey_offset] =  0.21f*r + 0.61f*g + 0.06f*b;
 

		  }

  }
  """)

#Get the function
func = mod.get_function("convertor")


#We need to keep 256 elements in the bloclk
func(a_gpu, output_gpu, rows_gpu, cols_gpu ,block=( 32*32,1,1)  , grid=( math.ceil(totalpixels[0]/(32*32) ),1, 1 )   )


cuda.memcpy_dtoh(output,output_gpu)


output = output.astype(numpy.uint8)
print(output)
#	Convett the array into the PIL Image
outp = Image.fromarray(output)
#Save the image
outp.save("output_d.png")





