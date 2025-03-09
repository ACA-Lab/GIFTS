#ifndef DG_MISC_H
#define DG_MISC_H

#include<math.h>

#if 1
#ifndef MAXBOUND 
   #define MAXBOUND 5.0 
#endif 
#ifndef SM_BOUND 
   #define SM_BOUND 5.0 
#endif 
#ifndef  SM_TABLE_SIZE
   #define SM_TABLE_SIZE 2048
#endif
#ifndef SM_RESOLUTION 
   #define SM_RESOLUTION SM_TABLE_SIZE/(2.0 * SM_BOUND)
#endif
#else 
   const float MAXBOUND = 5.0;
   const float SM_BOUND = 5.0;
   const int SM_TABLE_SIZE = 2048;
   const float SM_RESOLUTION = SM_TABLE_SIZE/(2.0 * SM_BOUND);

#endif
/* scalar scale function */
inline float sscale_SM(float val)
{
   float sval;
   /* hopefully compiler will figure out and replace it max min instruction */
   sval = (val > SM_BOUND) ? SM_BOUND : val;
   sval = (val < -SM_BOUND) ? -SM_BOUND : val;
   return(sval); 
}
/* not even parallel ?? */
void init_sSM_TABLE(float *sm_table)
{
   float x;
   for(INDEXTYPE i = 0; i < SM_TABLE_SIZE; i++)
   {
      x = 2.0 * SM_BOUND * i / SM_TABLE_SIZE - SM_BOUND;
      sm_table[i] = 1.0 / (1 + exp(-x));
   }
}
float fast_SM(float v, float *sm_table)
{
   if (v > SM_BOUND) return 1.0;
   else if (v < -SM_BOUND) return 0.0;
   return sm_table[(INDEXTYPE)((v + SM_BOUND) * SM_RESOLUTION)];
}

#endif
