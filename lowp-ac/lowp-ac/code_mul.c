/* CODE_MUL.C - ARITHMETIC ENCODING/DECODING, USING MULTIPLY/DIVIDE. */

#include <stdio.h>

#define global extern

#include "code.h"


/* CURRENT STATE OF THE ENCODING/DECODING. */

static code_value low;		/* Start of current coding region, in [0,1) */

static code_value range;	/* Size of region, normally kept in the     */
				/* interval [1/4,1/2]                       */

static code_value value;	/* Currently-seen code value, in [0,1)      */

static int bits_to_follow;	/* Number of opposite bits to output after  */
				/* the next bit                             */

static div_value F;		/* Common factor used, in interval [1/4,1)  */

static code_value split_region();


/* LOCAL PROCEDURES. */

static int        find_symbol   (freq_value []);
static void       narrow_region (freq_value [], int, int);
static code_value split_region  (freq_value, freq_value);
static void       push_out_bits (void);
static void       discard_bits  (void);


/* OUTPUT BIT PLUS FOLLOWING OPPOSITE BITS. */

#define bit_plus_follow(bit) \
do { \
    output_bit(bit);				/* Output the bit.          */\
    while (bits_to_follow>0) {			/* Output bits_to_follow    */\
        output_bit(!(bit));			/* opposite bits. Set       */\
        bits_to_follow -= 1;			/* bits_to_follow to zero.  */\
    } \
} while (0) 


/* START ENCODING A STREAM OF SYMBOLS. */

void start_encoding(void)
{   
    low = Code_half;				/* Use half the code region */
    range = Code_half;				/* (wastes first bit as 1). */
    bits_to_follow = 0;				/* No opposite bits follow. */
}


/* START DECODING A STREAM OF SYMBOLS. */

void start_decoding(void)
{   
    int i;

    value = input_bit();			/* Check that first bit     */
    if (value!=1) {				/* is a 1.                  */
        fprintf(stderr,"Bad input file (1)\n");
        exit(1);
    }

    for (i = 1; i<Code_bits; i++) { 		/* Fill the code value with */
        value += value;				/* input bits.              */
        value += input_bit();
    }

    low = Code_half;				/* Use half the code region */
    range = Code_half;				/* (first bit must be 1).   */
}


/* ENCODE A SYMBOL. */

void encode_symbol(symbol,cum_freq)
    int symbol;			/* Symbol to encode                         */
    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
{
    narrow_region(cum_freq,symbol,0);		/* Narrow coding region.    */
    push_out_bits();				/* Output determined bits.  */
}


/* ENCODE A BIT. */

void encode_bit(bit,freq0,freq1)
    int bit;			/* Bit to encode (0 or 1)                   */
    freq_value freq0;           /* Frequency for 0 bit                      */
    freq_value freq1;           /* Frequency for 1 bit                      */
{
    code_value split;

    if (freq1>freq0) {				/* Encode bit when most     */
        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
        if (bit) { low += split; range -= split; }
        else     { range = split;  }
    }

    else {					/* Encode bit when most     */
        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
        if (bit) { range = split;  }
        else     { low += split; range -= split; }
    }

    push_out_bits();				/* Output determined bits.  */
}


/* DECODE THE NEXT SYMBOL. */

int decode_symbol(cum_freq)
    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
{
    int symbol;			/* Symbol decoded                           */

    symbol = find_symbol(cum_freq);		/* Find decoded symbol.     */
    narrow_region(cum_freq,symbol,1);		/* Narrow coding region.    */
    discard_bits();				/* Discard output bits.     */

    return symbol;
}


/* DECODE A BIT. */

int decode_bit(freq0,freq1)
    freq_value freq0;           /* Frequency for 0 bit                      */
    freq_value freq1;           /* Frequency for 1 bit                      */
{
    code_value split;
    int bit;

    if (freq1>freq0) {				/* Decode bit when most     */
        split = split_region(freq0,freq1);	/* probable symbol is a 1.  */
        bit = value-low >= split;
        if (bit) { low += split; range -= split; }
        else     { range = split;  }
    }

    else {					/* Decode bit when most     */
        split = split_region(freq1,freq0);	/* probable symbol is a 0.  */
        bit = value-low < split;
        if (bit) { range = split;  }
        else     { low += split; range -= split; }
    }

    discard_bits();				/* Discard output bits.     */

    return bit;
}


/* DETERMINE DECODED SYMBOL FROM INPUT VALUE. */

static int find_symbol(cum_freq)
    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
{
    freq_value cum;		/* Cumulative frequency calculated          */
    int symbol;			/* Symbol decoded                           */

    F = range / cum_freq[0];			/* Compute common factor.   */

    cum = (value-low) / F;			/* Compute target cum freq. */
    for (symbol = 1; cum_freq[symbol]>cum; symbol++) ; /* Then find symbol. */

    return symbol;
}


/* NARROW CODING REGION TO THAT ALLOTTED TO SYMBOL. */

static void narrow_region(cum_freq,symbol,have_F)
    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
    int symbol;			/* Symbol decoded                           */
    int have_F;			/* Is F already computed?                   */
{
    code_value T;		/* Temporary value                          */

    if (!have_F) F = range / cum_freq[0];	/* Compute common factor.   */

    if (symbol==1) {				/* Narrow range for symbol  */
        T = F * cum_freq[symbol];		/* at the top.              */
        low += T;
        range -= T;
    }

    else {					/* Narrow range for symbol  */
        T = F * cum_freq[symbol];               /* not at the top.          */
        low += T;
        range = F * cum_freq[symbol-1] - T; 
    }
}


/* FIND BINARY SPLIT FOR CODING REGION. */

static code_value split_region(freq_b,freq_t)
    freq_value freq_b;		/* Frequency for symbol in bottom half      */
    freq_value freq_t;		/* Frequency for symbol in top half         */
{
    return freq_b * (range / (freq_b+freq_t)); 
}


/* OUTPUT BITS THAT ARE NOW DETERMINED. */

static void push_out_bits(void)
{
    while (range<=Code_quarter) {

        if (low>=Code_half) {			/* Output 1 if in top half.*/
            bit_plus_follow(1);
            low -= Code_half;			/* Subtract offset to top.  */
        }

        else if (low+range<=Code_half) {	/* Output 0 in bottom half. */
            bit_plus_follow(0);		
	} 

        else {			 		/* Output an opposite bit   */
            bits_to_follow += 1;		/* later if in middle half. */
            low -= Code_quarter;		/* Subtract offset to middle*/
        } 

        low += low;				/* Scale up code region.    */
        range += range;
    }
}


/* DISCARD BITS THE ENCODER WOULD HAVE OUTPUT. */

static void discard_bits(void)
{
    while (range<=Code_quarter) {

        if (low>=Code_half) {			/* Expand top half.         */
            low -= Code_half;			/* Subtract offset to top.  */
            value -= Code_half;
        }

        else if (low+range<=Code_half) {	/* Expand bottom half.      */
            /* nothing */
	} 

        else {			 		/* Expand middle half.      */
            low -= Code_quarter;		/* Subtract offset to middle*/
            value -= Code_quarter;
        } 

        low += low;				/* Scale up code region.    */
        range += range;

        value += value;				/* Move in next input bit.  */
        value += input_bit();
    }
}


/* FINISH ENCODING THE STREAM. */

void done_encoding(void)
{   
    for (;;) {

        if (low+(range>>1)>=Code_half) {	/* Output a 1 if mostly in  */
            bit_plus_follow(1);			/* top half.                */
            if (low<Code_half) {
                range -= Code_half-low;
                low = 0;
            }
            else {
                low -= Code_half;
            }
        }

        else {					/* Output a 0 if mostly in  */
            bit_plus_follow(0);			/* bottom half.             */
            if (low+range>Code_half) {
                range = Code_half-low;
            }
        }

        if (range==Code_half) break;		/* Quit when coding region  */
						/* becomes entire interval. */
        low += low;
        range += range;				/* Scale up code region.    */
    }
}
