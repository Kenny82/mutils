#include <stdio.h>
#include <math.h>

#define alpha 68.0*3.141597/180.

static double sinalpha = sin(alpha);


int main(int argc, char** argv)
{
printf("alpha = %2.5f \t sin(alpha) = %2.5f\n", alpha, sinalpha);
}
