/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( int* reference, float* iraA, float* idecA, float* jraB, float* jdecB, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////

void computeGold( int* reference, float* iraA, float* idecA, float* jraB, float* jdecB, const unsigned int len) 
{
    const float f_len = static_cast<float>( len);
	float sep;
	for(int k = 0; k< 64; k++)
	{
		reference[k]=0;
	}
    for( unsigned int i = 0; i < len; ++i) 
    {
		for( unsigned int j = 0; j < len; ++j) 
		{
			
			sep = acos( sin(idecA[i])*sin(jdecB[j]) + cos(idecA[i])*cos(jdecB[j])*cos(fabs(iraA[i]-jraB[j])) );
			if(sep>3.2){sep=3.2;}
			reference[int(floor((sep/3.2)*64.))]++;
		}
	}
}

