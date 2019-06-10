/* REDUNDANCY.C - PROGRAM TO CALCULATE MAXIMUM CODING INEFFICIENCY. */

#include <stdio.h>
#include <math.h>


/* This program calculates a bound on the expected number of extra bits
   produced as a result of doing arithmetic coding using divides of a given 
   precision, expressed as a percentage of the optimal coding size. The
   expectation is calculated on the assumption that the model's symbol
   probabilities are correct.

   The bound assumes that the coding range and total frequencies are such
   as to cause maximum truncation in the division, with the minimum true
   quotient. The bound varies as a function of the probability, p, of
   the most probable symbol. All but at most one of the remaining symbols
   are assumed to have probabilities equal to p (this gives minimum optimal
   coding size, and hence maximum relative inefficiency). */

main()
{
    double p;

    printf(
     "\nPrecision:    2      3      4      5      6      7      8      9\n\n");
   
    for (p = 0.001; p<0.0095; p+=0.001) do_p(p);
    for (p = 0.010; p<0.9905; p+=0.010) do_p(p);
    for (p = 0.991; p<0.9995; p+=0.001) do_p(p);
  
    exit(0);
}

do_p(p)
  double p;
{
    int precision, n;
    double e, opt, excess;

    printf("  p=%5.3f ",p);
    for (precision = 2; precision<10; precision++) {
        e = pow(2.0,-(double)precision)/(0.25+pow(2.0,-(double)precision));
        n = (int)(1/p);
        opt = - n*p*log(p);
        if (n*p!=1) opt -= (1-n*p)*log(1-n*p);
        excess = - (1-p)*log(1-e) - p*log(1-e+e/p);
        printf("  %5.2f",100*excess/opt);
    }
    printf("\n");
}
