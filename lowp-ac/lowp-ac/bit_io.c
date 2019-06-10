/* BIT_IO.C - BIT INPUT/OUTPUT ROUTINES. */

#include <stdio.h>

#define global extern

#include "code.h"


/* THE BIT BUFFER. */

static int buffer;		/* Bits waiting to be input                 */
static int bits_to_go;		/* Number of bits still in buffer           */

static int garbage;		/* Number of garbage bytes after EOF        */


/* INITIALIZE FOR BIT OUTPUT. */

start_outputing_bits()
{
    buffer = 0;					/* Buffer is empty to start */
    bits_to_go = 8;				/* with.                    */
}


/* INITIALIZE FOR BIT INPUT. */

start_inputing_bits()
{
    bits_to_go = 0;				/* Buffer starts out with   */
    garbage = 0;				/* no bits in it.           */
}


/* OUTPUT A BIT. */

output_bit(bit)
    int bit;
{
    if (bits_to_go==0) {			/* Output buffer if it is   */
        putc(buffer,stdout);			/* full.                    */
        bits_to_go = 8;
    }

    buffer >>= 1;		 		/* Put bit in top of buffer.*/
    if (bit) buffer |= 0x80;
    bits_to_go -= 1;
}


/* INPUT A BIT. */

int input_bit()
{
    int t;

    if (bits_to_go==0) {			/* Read next byte if no     */
        buffer = getc(stdin);			/* bits left in the buffer. */
        bits_to_go = 8;
        if (buffer==EOF) {  			/* Return anything after    */
            if (garbage*8>=Code_bits) {		/* end-of-file, but not too */
                fprintf(stderr,"Bad input file (2)\n"); /* much of anything.*/
                exit(1);
            }
            garbage += 1;
        }
    }

    t = buffer&1;				/* Return the next bit from */
    buffer >>= 1;				/* the bottom of the byte.  */
    bits_to_go -= 1;
    return t;	
}


/* FLUSH OUT THE LAST BITS. */

done_outputing_bits()
{
    putc(buffer>>bits_to_go,stdout);
}
