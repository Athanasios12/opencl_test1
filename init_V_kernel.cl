__kernel void init_V(__global unsigned long int *V, int img_height, int img_width, int start_column)
{
	int row = get_global_id(0);
	if (row >= (img_height * img_width))
	{
		return;
	}
	
	V[row] = 10;
	
}