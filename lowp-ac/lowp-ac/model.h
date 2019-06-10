/* MODEL.H - INTERFACE TO THE MODEL. */


/* THE SET OF SYMBOLS THAT MAY BE ENCODED. Symbols are indexed by integers
   from 1 to No_of_symbols. */

#define No_of_chars 256			/* Number of character symbols      */
#define EOF_symbol (No_of_chars+1)	/* Index of EOF symbol              */

#define No_of_symbols (No_of_chars+1)	/* Total number of symbols          */


/* TRANSLATION TABLES BETWEEN CHARACTERS AND SYMBOL INDEXES. */

global int char_to_index[No_of_chars];	/* To index from character          */
global unsigned char index_to_char[No_of_symbols+1]; /* To char from index  */


/* CUMULATIVE FREQUENCY TABLE. Cumulative frequencies are stored as
   partially normalized counts. The normalization factor is cum_freq[0],
   which must lie in the range (1/2,1]. */

global freq_value cum_freq[No_of_symbols+1]; /* Cumulative symbol frequencies */


/* PROCEDURES. */

void start_model  (void);
void update_model (int);
