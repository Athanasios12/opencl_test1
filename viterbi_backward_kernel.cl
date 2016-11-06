__kernel void viterbi_backward(__global const unsigned char * img, __global int **L, __global unsigned long int **V, int img_height, int img_width, int column, int g_low, int g_high)
{
	int row = get_global_id(0);
	if (row >= img_height)
	{
		return;
	}
	pVectorSum[glob_id] = pVector1[glob_id] + pVector2[glob_id];
	int max_val = 0;
	unsigned char pixel_value = 0;
	for (int g = g_low; g <= g_high; g++)
	{
		if (row + g > (img_height - 1))
		{
			break;
		}
		if (row + g < 0)
		{
			continue;
		}
		pixel_value = img[(img_width * row) + column];
		if ((pixel_value + V[row + g][column]) > max_val)
		{
			max_val = pixel_value + V[row + g][column];
			L[row][column] = g;
		}
	}
	V[row][column + 1] = max_val;
}