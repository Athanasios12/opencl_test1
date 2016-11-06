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