/* DECPIC.C - MAIN PROGRAM FOR DECODING PICTURES. */

#include <stdio.h>

#define global 

#include "code.h"

#define Height 40			/* Height of images                 */
#define Width 40                        /* Width of images                  */

static int image[Height][Width];	/* The image to be encoded          */

static int freq0[2][2];			/* Frequencies of '0' in contexts   */
static int freq1[2][2];			/* Frequencies of '1' in contexts   */

static int inc[2][2];			/* Current increment                */

void main(void)
{   
    int i, j, a, l;

    start_inputing_bits();
    start_decoding();

    /* Initialize model. */

    for (a = 0; a<2; a++) {
        for (l = 0; l<2; l++) {
            inc[a][l] = Freq_half;
            freq0[a][l] = inc[a][l];		/* Set frequencies of 0's   */
            freq1[a][l] = inc[a][l];		/* and 1's to be equal.     */
        }
    }

    /* Decode and write image. */

    for (i = 0; i<Height; i++) {
        for (j = 0; j<Width; j++) {
            a = i==0 ? 0 : image[i-1][j];	/* Find current context.    */
            l = j==0 ? 0 : image[i][j-1];
            image[i][j] = 			/* Decode pixel.            */
              decode_bit(freq0[a][l],freq1[a][l]);
            printf("%c%c",image[i][j] ? '#' : '.', 
                          j==Width-1 ? '\n' : ' ');
            if (image[i][j]) {			/* Update frequencies for   */
                freq1[a][l] += inc[a][l];       /* this context.            */
            }
            else {
                freq0[a][l] += inc[a][l];
            }
            if (freq0[a][l]+freq1[a][l]>Freq_full) { 
                freq0[a][l] = (freq0[a][l]+1) >> 1; 
                freq1[a][l] = (freq1[a][l]+1) >> 1;
                if (inc[a][l]>1) inc[a][l] >>= 1;
            }
        }
    }

    exit(0);
}
