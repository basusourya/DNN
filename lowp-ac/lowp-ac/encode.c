/* ENCODE.C - MAIN PROGRAM FOR ENCODING. */

#include <stdio.h>

#define global

#include "code.h"
#include "model.h"

void main(void)
{   
    int ch; 			/* Character to encode as a character code  */
    int symbol;			/* Character to encode as a symbol index    */

    start_model();				/* Set up other modules.    */
    start_outputing_bits();
    start_encoding();

    for (;;) {					/* Loop through characters. */

        ch = getc(stdin);			/* Read the next character. */
        if (ch==EOF) break;			/* Exit loop on end-of-file.*/
        symbol = char_to_index[ch];		/* Translate to an index.   */
        encode_symbol(symbol,cum_freq);		/* Encode that symbol.      */
        update_model(symbol);			/* Update the model.        */
    }

    encode_symbol(EOF_symbol,cum_freq);		/* Encode the EOF symbol.   */

    done_encoding();				/* Send the last few bits.  */
    done_outputing_bits();

    exit(0);
}
