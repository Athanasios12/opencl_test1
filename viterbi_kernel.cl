__kernel void viterbi_forward( __global const unsigned char * img, __global float *L, __global float *V, int img_height, int img_width, int column, int g_low, int g_high)
{
	int row = get_global_id(0);
	if (row >= img_height)
	{
		return;
	}
	float max_val = 0;
	unsigned char pixel_value = 0;
	for (int g = g_low; g <= g_high; g++)
	{
		if ((row + g) > (img_height - 1))
		{
			break;
		}
		if ((row + g) < 0)
		{
			continue;
		}
		pixel_value = img[(img_width * (row + g)) + column];
		if ((pixel_value + V[(row * img_width) + (column)]) > max_val)
		{
			max_val = pixel_value + V[(row * img_width) + (column)];
			L[(row * img_width) + column] = g;
		}
	}
	V[(row * img_width) + (column + 1)] = max_val;
}

__kernel void initV(__global float *V, int img_height, int img_width, int start_column)
{
	int row = get_global_id(0);
	if (row >= img_height)
	{
		return;
	}
	V[(row * img_width) + start_column] = 0;	
}

__kernel void viterbi_forward2(__global const unsigned char *img, 
								__global float *L, 
								__global int *line_x, 
								__global float* V_1,
								__global float* V_2,
								__global int *x_cord,
								int img_height,
								int img_width, 
								int g_high,
								int g_low)
{
	int start_column = get_global_id(0);
	long L_id = img_height * img_width * start_column; 
	int V_id = img_height *start_column;
	
	float P_max = 0;
	float x_max = 0;
	float max_val = 0;
	float pixel_value = 0;
	__global float *temp_buffer; // maybe may need to be passed with size as argument and set with clSetKernelArgs
	__global float *V_old = &V_1[V_id];
	__global float *V_new = &V_2[V_id];
	// init first column with zeros
	for (int m = 0; m < img_height; m++)
	{
		V_old[m] = 0;
	}
	for (int n = start_column; n < (img_width - 1); n++)
	{
		for (int j = 0; j < img_height; j++)
		{
			max_val = 0;
			for (int g = g_low; g <= g_high; g++)
			{
				if ((j + g) > (img_height - 1))
				{
					break;
				}
				if ((j + g) < 0)
				{
					continue;
				}
				pixel_value = img[((j + g) * img_width) + n];
				if ((pixel_value + V_old[j]) > max_val)
				{
					max_val = pixel_value + V_old[j];
					L[L_id + (j * img_width) + n] = g;
				}
			}
			V_new[j] = max_val;
		}
		temp_buffer = &V_old[0]; // have to do it or both pointers will have same adress
		V_old = &V_new[0];
		V_new = &temp_buffer[0];
	}
	//find biggest cost value in last column
	for (int j = 0; j < img_height; j++)
	{
		if (V_old[j] > P_max)
		{
			P_max = V_old[j];
			x_max = j;
		}
	}
	//backwards phase - retrace the path
	x_cord[(img_width - 1)] = x_max;
	for (int n = (img_width - 1); n > start_column; n--)
	{
		x_cord[n - 1] = x_cord[n] + L[L_id + (x_cord[n] * img_width) + (n - 1)]; //L[L_id][row][column]
	}
	// save only last pixel position
	line_x[start_column] = x_cord[start_column];
}
