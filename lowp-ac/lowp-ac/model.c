/* MODEL.C - THE ADAPTIVE SOURCE MODEL */

#include <stdio.h>

#define global extern

#include "code.h"
#include "model.h"

static freq_value freq[No_of_symbols+1]; /* Symbol frequencies                */
static freq_value inc;		         /* Value to increment frequencies by */


/* INITIALIZE THE MODEL. */

void start_model(void)
{   
    int i;

    for (i = 0; i<No_of_chars; i++) {		/* Set up tables that       */
        char_to_index[i] = i+1;			/* translate between symbol */
        index_to_char[i+1] = i;			/* indexes and characters.  */
    }

    inc = 1;
    while (inc*No_of_symbols<=Freq_half) {	/* Find increment that puts */
        inc *= 2;				/* total in required range. */
    }

    cum_freq[No_of_symbols] = 0;		/* Set up initial frequency */
    for (i = No_of_symbols; i>0; i--) {		/* counts to be equal for   */
        freq[i] = inc;				/* all symbols.             */
        cum_freq[i-1] = cum_freq[i] + freq[i];	
    }

    freq[0] = 0;				/* Freq[0] must not be the  */
						/* same as freq[1].         */
}


/* UPDATE THE MODEL TO ACCOUNT FOR A NEW SYMBOL. */

void update_model(symbol)
    int symbol;			/* Index of new symbol                      */
{   
    int ch_i, ch_symbol;	/* Temporaries for exchanging symbols       */
    int i;

    for (i = symbol; freq[i]==freq[i-1]; i--) ;	/* Find symbol's new index. */

    if (i<symbol) {
        ch_i = index_to_char[i];		/* Update the translation   */
        ch_symbol = index_to_char[symbol];	/* tables if the symbol has */
        index_to_char[i] = ch_symbol;           /* moved.                   */
        index_to_char[symbol] = ch_i;
        char_to_index[ch_i] = symbol;
        char_to_index[ch_symbol] = i;
    }

    freq[i] += inc;				/* Increment the frequency  */
    while (i>0) {				/* count for the symbol and */
        i -= 1;					/* update the cumulative    */
        cum_freq[i] += inc;			/* frequencies.             */
    }

    if (cum_freq[0]>Freq_full) {		/* See if frequency counts  */
        cum_freq[No_of_symbols] = 0;		/* are past their maximum.  */
        for (i = No_of_symbols; i>0; i--) {	/* If so, halve all counts  */
            freq[i] = (freq[i]+1) >> 1;		/* (keeping them non-zero). */
            cum_freq[i-1] = cum_freq[i] + freq[i]; 
        }
        if (inc>1) inc >>= 1;			/* Halve increment if can.  */
    }
}
