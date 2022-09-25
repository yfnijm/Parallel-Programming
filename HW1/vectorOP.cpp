#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
	//
	//  Ans: There are some lane active with id which larger than N
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

		__pp_mask pp_valid = _pp_init_ones((i + VECTOR_WIDTH <= N) ? VECTOR_WIDTH : N - i);
    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
		__pp_mask tmp =  _pp_mask_and(maskAll, pp_valid);
    _pp_vstore_float(output + i, result, tmp);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

	__pp_vec_float clamp_bound = _pp_vset_float(9.999999);
	__pp_vec_int pp_v1 = _pp_vset_int(1);
	__pp_vec_int pp_v0 = _pp_vset_int(0);


	__pp_vec_float pp_values;// pp_output;
  __pp_vec_int pp_exponents;
	for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
			__pp_mask pp_valid = _pp_init_ones((i + VECTOR_WIDTH <= N) ? VECTOR_WIDTH : N - i);
			//__pp_mask pp_mask = _pp_init_ones(VECTOR_WIDTH);
			_pp_vload_float(pp_values, &values[i], pp_valid);
			_pp_vload_int(pp_exponents, &exponents[i], pp_valid);

			__pp_vec_float pp_output = _pp_vset_float(1.0f);
		
			
			__pp_mask pp_mask = _pp_init_ones(VECTOR_WIDTH);
			_pp_mask_not(pp_mask);
			while(1){
				//pp_mask = _pp_init_ones(VECTOR_WIDTH);
				//_pp_mask_not(pp_mask);
				_pp_vgt_int(pp_mask, pp_exponents, pp_v0, pp_mask);
				pp_mask = _pp_mask_and(pp_mask, pp_valid);
				if(!_pp_cntbits(pp_mask)) break;
				
				_pp_vmult_float(pp_output, pp_output, pp_values, pp_mask);
				_pp_vsub_int(pp_exponents, pp_exponents, pp_v1, pp_mask);
			}

			pp_mask = _pp_init_ones(VECTOR_WIDTH);
			_pp_mask_not(pp_mask);
			_pp_vgt_float(pp_mask, pp_output, clamp_bound, pp_mask);

			_pp_vset_float(pp_output, 9.999999, pp_mask);
		
			//pp_mask = _pp_init_ones(VECTOR_WIDTH);
			_pp_vstore_float(&output[i], pp_output, pp_valid);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

	__pp_vec_float pp_values = _pp_vset_float(0.0f), pp_tmp;
	__pp_mask pp_mask = _pp_init_ones(VECTOR_WIDTH);
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
			_pp_vload_float(pp_tmp, &values[i], pp_mask);
			_pp_vadd_float(pp_values, pp_values, pp_tmp, pp_mask);
  }

	int iter = VECTOR_WIDTH;
	while(iter >>= 1){
		_pp_hadd_float(pp_tmp, pp_values);
		_pp_interleave_float(pp_values, pp_tmp);
	}

	float* res = new float[VECTOR_WIDTH];
	_pp_vstore_float(res, pp_values, pp_mask);

  return res[0];
}
