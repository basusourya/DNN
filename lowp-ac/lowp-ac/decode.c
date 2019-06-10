/* DECODE.C - MAIN PROGRAM FOR DECODING. */

#include <stdio.h>

#define global

#include "code.h"
#include "model.h"

void main(void)
{
    int symbol;			/* Character to decoded as a symbol index    */
    int ch; 			/* Character to decoded as a character code  */

    start_model();				/* Set up other modules.    */
    start_inputing_bits();
    start_decoding();

    for (;;) {					/* Loop through characters. */

        symbol = decode_symbol(cum_freq);	/* Decode next symbol.      */
        if (symbol==EOF_symbol) break;		/* Exit loop if EOF symbol. */
        ch = index_to_char[symbol];		/* Translate to a character.*/
        putc(ch,stdout);			/* Write that character.    */
        update_model(symbol);			/* Update the model.        */
    }

    exit(0);
}
