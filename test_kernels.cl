__kernel void addVectors(__global float * pVector1, __global float * pVector2, __global float * result, int numOfElem)
{
	int glob_id = get_global_id(0);
	if (glob_id >= numOfElem)
	{
		return;
	}
	result[glob_id] = pVector1[glob_id] + pVector2[glob_id];
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