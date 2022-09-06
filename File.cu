//#include <fstream>
//#include <iostream>
//#include <stdio.h>
//#include <cuComplex.h>
//#include <complex>
//#include "fftw3.h"
#include "Header.h"

using namespace std;

__constant__ cuconstParams coPar[1];


int main() {
	Params Par;
	Vectors Vec(&Par);
	comVectors cVec(Vec,Par);

	constParams cPar(Par);
	CUDA_CHECK(cudaMemcpyToSymbol(coPar, &cPar, sizeof(cuconstParams)));


	cudaParams cuPar(Par);
	cudaVectors cuVec(Vec, Par);

	cudaON(Par,cuPar, coPar);
	
	fftshift(cVec.E_a, cVec.Buf);
	fftw_execute(fftw_plan_dft_1d(NX, (fftw_complex*)(cVec.E_a), (fftw_complex*)(cVec.H_a), FFTW_FORWARD, FFTW_ESTIMATE));
	fftshift(cVec.H_a, cVec.Buf);
	fftshift(cVec.E_a, cVec.Buf);
	fprintf(cVec.E_a, "E_a.txt");
	Rcount(Par, Vec, cVec);
	Necount(Par, Vec);
	Jcount(Par, Vec, cVec);
	fprintf(Vec.J, "J1.txt");

	clock_t start = clock();
	for (int T = 0; T < 1; T++) {
		
		MulDiff(cVec, Vec, Par);
		
		fftshift(cVec.H_a, cVec.Buf);
fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.H_a), reinterpret_cast<fftw_complex*>(cVec.E_a), FFTW_BACKWARD, FFTW_ESTIMATE));
		iFFTnx(cVec.E_a);
		fftshift(cVec.E_a, cVec.Buf);
		fftshift(cVec.H_a, cVec.Buf);
		
		Poly(cVec);
		//cuPoly(cVec, cuVec, coPar, cuPar);
		
		Diff(cVec, Par);
		
		fftshift(cVec.Pol, cVec.Buf);
fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.Pol), reinterpret_cast<fftw_complex*>(cVec.Pol), FFTW_FORWARD, FFTW_ESTIMATE));
		fftshift(cVec.Pol, cVec.Buf);
		
		NextHalfDiff(cVec, Vec, Par);
		
		fftshift(cVec.H_b, cVec.Buf);
fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.H_b), reinterpret_cast<fftw_complex*>(cVec.E_a), FFTW_BACKWARD, FFTW_ESTIMATE));
		iFFTnx(cVec.E_a);
		fftshift(cVec.E_a, cVec.Buf);
		fftshift(cVec.H_b, cVec.Buf);

		Poly(cVec);
		//cuPoly(cVec, cuVec, coPar, cuPar);

		Diff(cVec, Par);
		fftshift(cVec.Pol, cVec.Buf);
fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.Pol), reinterpret_cast<fftw_complex*>(cVec.Pol), FFTW_FORWARD, FFTW_ESTIMATE));
		fftshift(cVec.Pol, cVec.Buf);
		
		NextFullDiff(cVec, Vec, Par); 
		
		Filter(cVec.H_b, Vec);
		fftshift(cVec.H_b, cVec.Buf);
		fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.H_b), reinterpret_cast<fftw_complex*>(cVec.E_a), FFTW_BACKWARD, FFTW_ESTIMATE));
		iFFTnx(cVec.E_a);
		fftshift(cVec.E_a, cVec.Buf);

		Rcount(Par, Vec, cVec);
		Necount(Par, Vec);
		Jcount(Par, Vec, cVec);
	}
	fprintf(Vec.R, "R.txt");
	fprintf(Vec.Ne, "Ne.txt");
	fprintf(Vec.J, "J.txt");
	fprintf(cVec.H_b, "H_b.txt");
	fprintf(cVec.H_a, "H_a.txt");
	fprintf(cVec.E_a, "E_b.txt");
	fftshift(cVec.H_b, cVec.Buf);
fftw_execute(fftw_plan_dft_1d(NX, reinterpret_cast<fftw_complex*>(cVec.H_b), reinterpret_cast<fftw_complex*>(cVec.E_a), FFTW_BACKWARD, FFTW_ESTIMATE));
	iFFTnx(cVec.E_a);
	fftshift(cVec.E_a, cVec.Buf);

	//fprintf(cVec.E_a);
	clock_t end = clock();
	cout << "Calculation: " << end - start << " ms" << endl;
	
}

