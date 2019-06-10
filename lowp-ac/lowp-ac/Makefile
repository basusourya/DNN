# MAKEFILE FOR LOW-PRECISION ARITHMETIC CODING PROGRAMS.


CFLAGS = -O

all:			encode_mul decode_mul encode_sft decode_sft \
			encpic_mul decpic_mul encpic_sft decpic_sft

lowp_ac.shar:		Makefile tstpic model.h code.h \
			encode.c decode.c model.c encpic.c decpic.c \
			code_mul.c code_sft.c bit_io.c 
	shar -o lowp_ac.shar Makefile tstpic model.h code.h \
		encode.c decode.c model.c encpic.c decpic.c \
		code_mul.c code_sft.c bit_io.c redundancy.c


encode_mul:		encode.o model.o bit_io.o code_mul.o
	cc -O encode.o model.o bit_io.o code_mul.o -o encode_mul

decode_mul:		decode.o model.o bit_io.o code_mul.o
	cc -O decode.o model.o bit_io.o code_mul.o -o decode_mul

encode_sft:		encode.o model.o bit_io.o code_sft.o
	cc -O encode.o model.o bit_io.o code_sft.o -o encode_sft

decode_sft:		decode.o model.o bit_io.o code_sft.o
	cc -O decode.o model.o bit_io.o code_sft.o -o decode_sft


encpic_mul:		encpic.o bit_io.o code_mul.o
	cc -O encpic.o bit_io.o code_mul.o -o encpic_mul

decpic_mul:		decpic.o bit_io.o code_mul.o
	cc -O decpic.o bit_io.o code_mul.o -o decpic_mul

encpic_sft:		encpic.o bit_io.o code_sft.o
	cc -O encpic.o bit_io.o code_sft.o -o encpic_sft

decpic_sft:		decpic.o bit_io.o code_sft.o
	cc -O decpic.o bit_io.o code_sft.o -o decpic_sft


encode.o:		encode.c model.h code.h
decode.o:		decode.c model.h code.h
model.o:		model.c model.h code.h

encpic.o:		encpic.c model.h code.h
decpic.o:		decpic.c model.h code.h

code_mul.o:		code_mul.c code.h
code_sft.o:		code_sft.c code.h
bit_io.o:		bit_io.c code.h
