/* CODE_SFT.C - ARITHMETIC ENCODING/DECODING, USING SHIFT/ADD. */

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

static code_value split_region();


/* LOCAL PROCEDURES. */

static int        find_symbol   (freq_value []);
static void       narrow_region (freq_value [], int);
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
    narrow_region(cum_freq,symbol);		/* Narrow coding region.    */
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
    narrow_region(cum_freq,symbol);		/* Narrow coding region.    */
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
    int symbol;			/* Symbol decoded                           */
    freq_value M, P, Q, B;	/* Temporary values                         */
    div_value F, G;
    code_value A;	
    int i;

    A = range; 					/* Compute the value of     */
    M = cum_freq[0] << (Code_bits-Freq_bits-1); /*   F = range/cum_freq[0]. */
    F = 0;

    if (A>=M) { A -= M; F += 1; }
#   if Code_bits-Freq_bits>2
    A <<= 1; F <<= 1;
    if (A>=M) { A -= M; F += 1; }
#   endif
#   if Code_bits-Freq_bits>3
    A <<= 1; F <<= 1;
    if (A>=M) { A -= M; F += 1; }
#   endif
#   if Code_bits-Freq_bits>4
    A <<= 1; F <<= 1;
    if (A>=M) { A -= M; F += 1; }
#   endif
#   if Code_bits-Freq_bits>5
    A <<= 1; F <<= 1;
    if (A>=M) { A -= M; F += 1; }
#   endif
#   if Code_bits-Freq_bits>6
    for (i = Code_bits-Freq_bits-6; i>0; i--) {
        A <<= 1; F <<= 1;
        if (A>=M) { A -= M; F += 1; }
    }
#   endif
    A <<= 1; F <<= 1;
    if (A>=M) { F += 1; }

    A = value - low;			  	/* Find the symbol that     */
    B = Freq_half;				/* fits the input. To do    */
    G = F << (Freq_bits-1);			/* so compute (value-low)/F */
    P = 0;                                      /* to as many bits as is    */
    Q = cum_freq[0];     			/* necessary.               */
    symbol = 1;

    while (cum_freq[symbol]>P) {
        if (A>=G) {
            A -= G; 
            P = P+B; 
        }
        else {
            Q = P+B;
            while (cum_freq[symbol]>=Q) symbol += 1;
        }
        B >>= 1; G >>= 1;
    }

    return symbol;
}


/* NARROW CODING REGION TO THAT ALLOTTED TO SYMBOL. */

static void narrow_region(cum_freq,symbol)
    freq_value cum_freq[];	/* Cumulative symbol frequencies            */
    int symbol;			/* Symbol decoded                           */
{
    code_value A, Ta, Tb;	/* Temporary values                         */
    freq_value M, Ba, Bb;
    int i;

    A = range; 
    M = cum_freq[0] << (Code_bits-Freq_bits-1);

    if (symbol==1) {				/* Narrow range for symbol  */
                                                /* at the top.              */
        Ba = cum_freq[symbol];
        Ta = 0;                                 /* Compute the value of     */
						/*   Ta = cum_freq[symbol]  */
        if (A>=M) { A -= M; Ta += Ba; }		/*     * (range/cum_freq[0])*/
#       if Code_bits-Freq_bits>2
        A <<= 1; Ta <<= 1;
        if (A>=M) { A -= M; Ta += Ba; }
#       endif
#       if Code_bits-Freq_bits>3
        A <<= 1; Ta <<= 1;
        if (A>=M) { A -= M; Ta += Ba; }
#       endif
#       if Code_bits-Freq_bits>4
        A <<= 1; Ta <<= 1;
        if (A>=M) { A -= M; Ta += Ba; }
#       endif
#       if Code_bits-Freq_bits>5
        A <<= 1; Ta <<= 1;
        if (A>=M) { A -= M; Ta += Ba; }
#       endif
#       if Code_bits-Freq_bits>6
        for (i = Code_bits-Freq_bits-6; i>0; i--) {
            A <<= 1; Ta <<= 1;
            if (A>=M) { A -= M; Ta += Ba; }
        }
#       endif
        A <<= 1; Ta <<= 1;
        if (A>=M) { Ta += Ba; }

        low += Ta;
        range -= Ta;
    }

    else {					/* Narrow range for symbol  */
                                                /* not at the top.          */
        Ba = cum_freq[symbol];
        Bb = cum_freq[symbol-1];	        /* Compute the value of     */
        Ta = Tb = 0;				/*   Ta = cum_freq[symbol]  */
						/*     * (range/cum_freq[0])*/
        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }/*and of                   */
#       if Code_bits-Freq_bits>2		/*   Tb = cum_freq[symbol-1]*/
        A <<= 1; Ta <<= 1; Tb <<= 1;		/*     * (range/cum_freq[0] */
        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
#       endif
#       if Code_bits-Freq_bits>3
        A <<= 1; Ta <<= 1; Tb <<= 1;
        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
#       endif
#       if Code_bits-Freq_bits>4
        A <<= 1; Ta <<= 1; Tb <<= 1;
        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
#       endif
#       if Code_bits-Freq_bits>5
        A <<= 1; Ta <<= 1; Tb <<= 1;
        if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
#       endif
#       if Code_bits-Freq_bits>6
        for (i = Code_bits-Freq_bits-6; i>0; i--) {
            A <<= 1; Ta <<= 1; Tb <<= 1;
            if (A>=M) { A -= M; Ta += Ba; Tb += Bb; }
        }
#       endif
        A <<= 1; Ta <<= 1; Tb <<= 1;
        if (A>=M) { Ta += Ba; Tb += Bb; }

        low += Ta;
        range = Tb - Ta;
    }
}


/* FIND BINARY SPLIT FOR CODING REGION. */

static code_value split_region(freq_b,freq_t)
    freq_value freq_b;		/* Frequency for symbol in bottom half      */
    freq_value freq_t;		/* Frequency for symbol in top half         */
{
    code_value A, T;		/* Temporary values                         */
    freq_value M, B;
    int i;

    A = range; 
    M = (freq_b+freq_t) << (Code_bits-Freq_bits-1);
    B = freq_b;
    T = 0;           	                        /* Compute the value of     */
						/* T = freq_b * (range      */
    if (A>=M) { A -= M; T += B; }		/*       / (freq_b+freq_t)) */
#   if Code_bits-Freq_bits>2
    A <<= 1; T <<= 1;
    if (A>=M) { A -= M; T += B; }
#   endif
#   if Code_bits-Freq_bits>3
    A <<= 1; T <<= 1;
    if (A>=M) { A -= M; T += B; }
#   endif
#   if Code_bits-Freq_bits>4
    A <<= 1; T <<= 1;
    if (A>=M) { A -= M; T += B; }
#   endif
#   if Code_bits-Freq_bits>5
    A <<= 1; T <<= 1;
    if (A>=M) { A -= M; T += B; }
#   endif
#   if Code_bits-Freq_bits>6
    for (i = Code_bits-Freq_bits-6; i>0; i--) {
        A <<= 1; T <<= 1;
        if (A>=M) { A -= M; T += B; }
    }
#   endif
    A <<= 1; T <<= 1;
    if (A>=M) { T += B; }

    return T;
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
