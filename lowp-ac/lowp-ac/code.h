/* CODE.H - DECLARATIONS USED FOR ARITHMETIC ENCODING AND DECODING */


/* PRECISION OF CODING VALUES. Coding values are fixed-point binary 
   fractions in the range [0,1), represented as integers scaled by 
   2^Code_bits. */

#define Code_bits 32			/* Number of bits for code values   */
#define Code_quarter (1<<(Code_bits-2))	/* Quarter point for code values    */
#define Code_half (1<<(Code_bits-1))	/* Halfway point for code values    */

typedef unsigned code_value;		/* Data type holding code values    */


/* PRECISION OF SYMBOL PROBABILITIES. Symbol probabilities must be partially
   normalized, so that the total for all symbols is in the range (1/2,1].
   These probabilities are represented as integers scaled by 2^Freq_bits. */

#define Freq_bits 27			/* Number of bits for frequencies   */
#define Freq_half (1<<(Freq_bits-1))	/* Halfway point for frequencies    */
#define Freq_full (1<<Freq_bits)	/* Largest frequency value          */

typedef unsigned freq_value;		/* Data type holding frequencies    */


/* PRECISION OF DIVISION. Division of a code value by a frequency value 
   gives a result of precision (Code_bits-Freq_bits), which must be at
   least two for correct operation (larger differences give smaller code
   size). */

typedef unsigned div_value;		/* Data type for result of division */


/* PROCEDURES. */

void start_encoding (void);
void start_decoding (void);
void encode_symbol  (int, freq_value []);
void encode_bit     (int, freq_value, freq_value);
int  decode_symbol  (freq_value []);
int  decode_bit     (freq_value, freq_value);
void done_encoding  (void);
