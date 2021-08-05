import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import sys
from PIL import Image

image = Image.open(sys.argv[-1])

def rgbtogray(image):
        h_imagePixel = numpy.array(image)
        width, height  = image.size
      
        
        totalPixel = width*height

        h_imagePixel = h_imagePixel.astype(numpy.float32)
        d_imagePixel = cuda.mem_alloc(h_imagePixel.nbytes)
        d_outPixel = cuda.mem_alloc(h_imagePixel.nbytes)
        cuda.memcpy_htod(d_imagePixel, h_imagePixel)
        
        mod = SourceModule(""" 
                __global__ void gray(unsigned char *h_imagePixel, unsigned char *d_outPixel)
                { 
                        unsigned char rgb;
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;  
                        rgb = h_imagePixel[0+idx*3]*0.299 + h_imagePixel[1+idx*3]*0.587 + h_imagePixel[2+idx*3]*0.114;
                        d_outPixel[0+idx*3] = rgb;
                        d_outPixel[1+idx*3] = rgb;
                        d_outPixel[2+idx*3] = rgb; 
                }
        """)
        
        func = mod.get_function("gray")
        func(d_imagePixel,d_outPixel, block=(width,1,1), grid=(height,1,1))
        '''
        h_imageOut = numpy.empty_like(h_imagePixel)
        cuda.memcpy_dtoh(h_imageOut, d_outPixel)

        image = Image.fromarray(h_imageOut)

        '''
        return image

image = rgbtogray(image)
image.save("./out.png")