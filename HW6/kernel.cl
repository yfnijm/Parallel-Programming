__kernel void convolution(int filterWidth, __global float* filter,
						int imageHeight, int imageWidth, __global float* inputImage, __global float* outputImage) 
{
	int gid = get_global_id(0); 
	int i = gid / imageWidth;
	int j = gid % imageWidth;
	int halffilterSize = filterWidth / 2;
	int k, l;
	float sum = 0.0;

	for (k = -halffilterSize; k <= halffilterSize; k++)
	{
		for (l = -halffilterSize; l <= halffilterSize; l++)
		{
			if (i + k >= 0 && i + k < imageHeight &&
					j + l >= 0 && j + l < imageWidth)
			{
				sum += inputImage[(i + k) * imageWidth + j + l] *
					filter[(k + halffilterSize) * filterWidth +
					l + halffilterSize];
			}
		}
	}
	outputImage[gid] = sum;
}
