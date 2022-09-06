#pragma once
#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda/std/cmath"
#include "cufft.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

#include "helper_cuda.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuComplex.h>
#include <complex>
#include "fftw3.h"

using namespace std;

#define NX 65536   //Размер области решения шредингера
#define NT 65536   //Размер временной области счета поляризации
#define NZ 1000    //Количество шагов распространения

// Сборщик ошибок CUDA функций
#define CUDA_CHECK(call) if((call) != cudaSuccess)\ 
	{ cudaError_t err = cudaGetLastError(); printf("CUDA error calling \"%s\", error code is %d\n", #call, err); \
	printf("Error %d: %s\n", err, cudaGetErrorString(err)); exit(-1); }


struct Params {
	const double pi = 3.141592653589793;
	const double Tall = 1000;           //Длительность распространения fs
	const double t0 = 48;               //??? Длительность импульса fs
	const double dw = 2 * pi * t0 / Tall;
	const double W = 75;                
	const double w0 = 2 * pi * W * 1e-3 * t0;
	const double c = 3e-5;                
	const double A = 1.993e-4;            
	const double B0 = 5.58e-7 / (t0 * t0);
	const double p = 1;
	const double n2 = 1e-19;
	const double I0 = 1e+11; 
	const double Vc = 0; 
	const double N0 = sqrt(8*pi*I0/c);
	const double z = 2200 / c / t0;    
	const double dz = 1 / c / t0;
	const double dt = Tall / t0 / NT;  //В чем это?
	const double wmin = 0.3 * w0;
	const double wmax = 10 * w0;
	complex<double> im = complex<double>(0.0, 1.0);
	const double Xmax = 4096;
	const double dx = 2 * Xmax / NT;
	const double a = 2;
	const double e = -4.8032047e-10;   //СГСЭ ед.зар
	const double me = 9.1093837e-28;   //CГСЭ гр
	const double wa = 41.3;            //Атомная частота fs^-1
	const double Ea = 5.17e-3;         //Атомное поле ТВт/см

	const double Wi = 12.079;
	const double Wh = 13.6;
	const double r_n = Wi / Wh;

	double vg = NULL;
	Params() {
		cout << "pi  " << pi << endl;
		cout << "Tall  " << Tall << endl;
		cout << "t0  " << t0 << endl;
		cout << "dw  "<< dw << endl;
		cout << "w0  " << w0 << endl;
		cout << "c  " << c << endl;
		cout << "A  " << A << endl;
		cout << "B0  " << B0 << endl;
		cout << "n2  " << n2 << endl;
		cout << "I0  " << I0 << endl;
		cout << "z  " << z << endl;
		cout << "dz  " << dz << endl;
		cout << "dt  " << dt << endl;
	}
};

struct constParams {
	double pi;
	double dt;
	double two;
	double nx;

	constParams(Params P) {
		pi = P.pi;
		dt = P.dt;
		two = 2.0;
		nx = NX;
		}
};

struct cuconstParams {
	double pi;
	double dt;
	double two;
	double nx;
};

struct cudaParams {
	dim3 Thread;
	dim3 Block;
	cufftHandle plan1, plan2;
	cublasHandle_t handle1, handle2;

	cudaParams(Params P) {
		Block = dim3(NX / 128, 1, 1);
		Thread = dim3(128, 1, 1);
		CUDA_CHECK(cufftPlan1d(&plan1, NT, CUFFT_Z2Z, 1));
		CUDA_CHECK(cufftPlan1d(&plan2, NT, CUFFT_Z2Z, 1));
		CUDA_CHECK(cublasCreate_v2(&handle1));
		CUDA_CHECK(cublasCreate_v2(&handle2));
	}
};

struct Vectors {
	double* w;
	double* n;
	double* B;
	double* k;
	double* dB_dw;
	double* d2B_dw2;
	double* t;
	double* filter;
	double* R;            //скорость ионизации
	double* Ne;           //Доля ионизации
	double* dJfree;
	double* Jfree;
	double* Jion;
	double* J;
	double** psi_g;
	double* k1;
	double* x;
	double* U;
	double* Wconst;
	double* xdt;
	complex<double>* Ediff;
	complex<double>* constExp;

	Vectors(Params* P) {
		w = new double[NT];
		n = new double[NT];
		B = new double[NT];
		k = new double[NT];
		k1 = new double[NT];
		dB_dw = new double[NT];
		d2B_dw2 = new double[NT];
		t = new double[NT];
		filter = new double[NT];
		R = new double[NT];
		Ne = new double[NT];
		dJfree = new double[NT];
		Jfree = new double[NT];
		Jion = new double[NT];
		J = new double[NT];

		psi_g = new double * [3];
		for (int count = 0; count < 3; count++) {
			psi_g[count] = new double[NX];
		}

		x = new double[NX];
		xdt = new double[NX];
		U = new double[NX];
		Wconst = new double[NX];
		Ediff = new complex<double>[NX];
		constExp = new complex<double>[NX];

		for (int i = 0; i < NT; i++) {
			t[i] = (double(i) - double(NT/2)) / (*P).t0 * (*P).Tall / double(NT);
			w[i] = (double(i) - double(NT/2)) * (*P).dw;
			n[i] = 1 + (*P).p * ((*P).A + (*P).B0 * w[i] * w[i]);
			B[i] = w[i] * n[i];
			
			if (i < 7 * NT / 16 || i > 9 * NT / 16) {
				filter[i] = 0;
			}
			else {
				filter[i] = exp(-pow(((*P).wmin / w[i]), 6) - pow((w[i] / (*P).wmax), 4));
			}
			if (i < NT / 2) {
				k1[i + NX/2] = (*P).pi / (*P).Xmax * (i - NX / 2);
			}
			else {
				k1[i - NX / 2] = (*P).pi / (*P).Xmax * (i - NX / 2);
			}
		}
		for (int i = 1; i < (NT - 1); i++) {
			dB_dw[i] = (B[i + 1] - B[i - 1]) / 2 / (*P).dw;
		}
		dB_dw[0] = dB_dw[1];
		dB_dw[NT - 1] = dB_dw[NT - 2];
		for (int i = 0; i < NX; i++) {
			if (abs(w[i] - (*P).w0) < (*P).dw / 100) {
				(*P).vg = 1 / dB_dw[i];
				cout << "vg  " << (*P).vg << endl;
			}
		}
		for (int i = 0; i < NT; i++) {
			k[i] = B[i] - w[i] / (*P).vg;
		}
		for (int i = 1; i < (NT - 1); i++) {
			d2B_dw2[i] = (B[i - 1] + B[i + 1] - 2 * B[i]) / ((*P).dw * (*P).dw);
		}
		d2B_dw2[0] = d2B_dw2[1];
		d2B_dw2[NT - 1] = d2B_dw2[NT - 2];

		ifstream fin;
		fin.open("psi_g_lorentz_pit_1000.txt");
		for (int nx = 0; nx < NX; nx++) {
			fin >> psi_g[0][nx];
			fin >> psi_g[1][nx];
			fin >> psi_g[2][nx];
			
		}
		fin.close();

		for (int i = 0; i < NX; i++) {
			if (i < NX / 2) {
				x[i + NX / 2] = i - NX / 2;
				Wconst[i + NX / 2] = psi_g[2][i];
			}
			else {
				x[i - NX / 2] = i - NX / 2;
				Wconst[i - NX / 2] = psi_g[2][i];
			}
		}
		for (int i = 0; i < NX; i++) {
			U[i] = -0.146 * exp(-pow((x[i] / 4 / (*P).a), 16)) / sqrt(pow(x[i] / (*P).a / 1.8, 2) + pow(0.175, 2));
			Ediff[i] = exp(-(*P).im * (*P).dt * k1[i] * k1[i] / 2.0);
		}
		for (int i = 0; i < NX; i++) {
			constExp[i] = exp(-(*P).im * U[i] * (*P).dt / 2.0);
			xdt[i] = x[i] / (*P).dt / 2.0;
		}
	};
};

struct cudaVectors {
	double* x;
	double* xdt;
	double* U;
	cuDoubleComplex* Ediff;
	cuDoubleComplex* constExp;
	
	cuDoubleComplex* E;
	cuDoubleComplex* E_x;
	cuDoubleComplex* E_t;
	cuDoubleComplex* Wer;
	cuDoubleComplex* Pol;

	cudaVectors(Vectors V,Params P) {
		CUDA_CHECK(cudaMalloc(&x, NX * sizeof(double)));
		CUDA_CHECK(cudaMalloc(&xdt, NX * sizeof(double)));
		CUDA_CHECK(cudaMalloc(&U, NX * sizeof(double)));
		CUDA_CHECK(cudaMalloc(&Ediff, NX * sizeof(cuDoubleComplex)));
		CUDA_CHECK(cudaMalloc(&constExp, NX * sizeof(cuDoubleComplex)));

		CUDA_CHECK(cudaMemcpy(x, V.x, NX * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(xdt, V.xdt, NX * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(U, V.U, NX * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(Ediff, V.Ediff, NX * sizeof(double), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(constExp, V.constExp, NX * sizeof(double), cudaMemcpyHostToDevice));

		CUDA_CHECK(cudaMalloc(&E, NX * sizeof(cuDoubleComplex)));
		CUDA_CHECK(cudaMalloc(&E_x, NX * sizeof(cuDoubleComplex)));
		CUDA_CHECK(cudaMalloc(&E_t, NX * sizeof(cuDoubleComplex)));
		CUDA_CHECK(cudaMalloc(&Wer, NX * sizeof(cuDoubleComplex)));
		CUDA_CHECK(cudaMalloc(&Pol, NX * sizeof(cuDoubleComplex)));
	}
};

struct comVectors {
	complex<double>* E_a;
	complex<double>* H_a;
	complex<double>* H_b;
	complex<double>* Pol;
	complex<double>* Buf;
	
	comVectors(Vectors Vec, Params Par) {
		E_a = new complex<double>[NT];
		H_a = new complex<double>[NT];
		H_b = new complex<double>[NT];
		Pol = new complex<double>[NT];
		Buf = new complex<double>[NT];
		
		for (int i = 0; i < NT; i++) {
			E_a[i] = exp((-Vec.t[i] * Vec.t[i]) / 2) * cos(Vec.t[i] * Par.w0);  //5.17
		}
	};
};

__global__ void makecudaE_x(cudaVectors V, int T) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.x;
	__shared__ cuDoubleComplex buf[4][128];
	buf[0][y].x = V.constExp[x].x;
	buf[0][y].y = V.constExp[x].y;
	buf[1][y].x = V.E[T].x;
	buf[1][y].y = V.E[T].y;
	buf[2][y].x = V.xdt[x];
	buf[2][y].y = -buf[2][y].x;
	buf[3][y].x = exp(buf[2][y].y * buf[1][y].y) * cos(buf[2][y].x * buf[1][y].x);
	buf[3][y].y = exp(buf[2][y].y * buf[1][y].y) * sin(buf[2][y].x * buf[1][y].x);
	buf[2][y].x = V.E_t[x].x;
	buf[2][y].y = V.E_t[x].y;
	__syncthreads();

		buf[1][y].x = buf[0][y].x * buf[3][y].x - buf[0][y].y * buf[3][y].y;
	buf[1][y].y = buf[0][y].y * buf[3][y].x + buf[0][y].x * buf[3][y].y;

	V.E_t[x].x = buf[2][y].x * buf[1][y].x - buf[2][y].y * buf[1][y].y;
	V.E_t[x].y = buf[2][y].y * buf[1][y].x + buf[2][y].x * buf[1][y].y;

	V.E_x[x].x = buf[1][y].x;
	V.E_x[x].y = buf[1][y].y;
	__syncthreads();
}
__global__ void cuda_MUL(cuDoubleComplex* X, cuDoubleComplex* Y) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.x;
	__shared__ cuDoubleComplex buf[2][128];
	buf[0][y] = X[x];
	buf[1][y] = Y[x];
	__syncthreads();
	X[x].x = buf[0][y].x * buf[1][y].x - buf[0][y].y * buf[1][y].y;
	X[x].y = buf[0][y].y * buf[1][y].x + buf[0][y].x * buf[1][y].y;
}

__global__ void cuda_MUL(cuDoubleComplex* X, double* Y) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.x;
	__shared__ cuDoubleComplex buf[2][128];
	buf[0][y] = X[x];
	buf[1][y].x = Y[x];
	__syncthreads();
	X[x].x = buf[0][y].x * buf[1][y].x;
	X[x].y = buf[0][y].y * buf[1][y].x;
}

__global__ void cuda_MUL2(cuDoubleComplex* X, cuDoubleComplex* Y) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = threadIdx.x;
	__shared__ cuDoubleComplex buf[2][128];
	buf[0][y] = X[x];
	buf[1][y] = Y[x];
	__syncthreads();
	X[x].x = (buf[0][y].x * buf[1][y].x - buf[0][y].y * buf[1][y].y) / NX;
	X[x].y = (buf[0][y].y * buf[1][y].x + buf[0][y].x * buf[1][y].y) / NX;
}

void cuPoly(comVectors cV, cudaVectors V, cuconstParams* cP, cudaParams P) {
	cudaError_t err = cudaSuccess;

	CUDA_CHECK(cudaMemcpy(V.E, cV.E_a, 2 * NX * sizeof(double), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	cuDoubleComplex* Werp;
	Werp = new cuDoubleComplex[NX];
	for (int T = 0; T < NX; T++) {
		makecudaE_x <<< P.Block, P.Thread >>> (V, T);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch 1 (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

CUDA_CHECK(cufftExecZ2Z(P.plan1, (cufftDoubleComplex*)V.E_t, (cufftDoubleComplex*)V.E_t, CUFFT_FORWARD));
		cuda_MUL <<< P.Block, P.Thread >>> (V.E_t, V.Ediff);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch 2 (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
CUDA_CHECK(cufftExecZ2Z(P.plan2, (cufftDoubleComplex*)V.E_t, (cufftDoubleComplex*)V.E_t, CUFFT_INVERSE));
		cuda_MUL2 <<< P.Block, P.Thread >>> (V.E_t, V.E_x);
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch 3 (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
CUDA_CHECK(cublasZcopy_v2(P.handle1, NX, (cuDoubleComplex*)V.E_t, 1, (cuDoubleComplex*)V.E_x, 1));
err = cudaGetLastError();
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to launch 4 (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
CUDA_CHECK(cublasZdotc_v2(P.handle2, NX, V.E_t, 1, V.E_x, 1, (&(V.Wer[T]))));
err = cudaGetLastError();
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to launch 5 (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
		cuda_MUL <<< P.Block, P.Thread >>> (V.E_t, V.x);

checkCudaErrors(cublasZdotc_v2(P.handle1, 1, V.E_t, 1, V.E_x, 1, (&(V.Pol[T]))));
err = cudaGetLastError();
if (err != cudaSuccess)
{
	fprintf(stderr, "Failed to launch 6 (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}
	}
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaMemcpy(cV.Pol, V.Pol, 2 * NX * sizeof(double), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}

int cudaON(Params P,cudaParams cP, cuconstParams* cuconstPar) {
	
	int deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
		return 0;
	}
	CUDA_CHECK(cudaSetDevice(0));
	printf("cudaSetDevice\n");

	
	//CUDA_CHECK(cudaMalloc(&cuconstPar, sizeof(cuconstParams)));
	
	return 0;
}

int cudaOFF(cudaVectors V) {

	CUDA_CHECK(cudaFree(V.x));
	CUDA_CHECK(cudaFree(V.xdt));
	CUDA_CHECK(cudaFree(V.U));
	CUDA_CHECK(cudaFree(V.constExp));
	CUDA_CHECK(cudaFree(V.Ediff));

	
	CUDA_CHECK(cudaFree(V.E));
	CUDA_CHECK(cudaFree(V.E_t));
	CUDA_CHECK(cudaFree(V.E_x));
	CUDA_CHECK(cudaFree(V.Pol));
	CUDA_CHECK(cudaFree(V.Wer));

	CUDA_CHECK(cudaDeviceReset());
	printf("cudaResetDevice\n");
	return 0;

}

void MulDiff(comVectors cVec, Vectors Vec, Params Par) {
	//cout << "mul   " << Par.vg << "   " << Par.dz << "   "<<Par.im  << endl;
	for (int i = 0; i < NT; i++) {
		cVec.H_a[i] *= exp(-Par.im * (Vec.B[i] - Vec.w[i] / Par.vg) * Par.dz);
	}
}

void Filter(complex<double>* x, Vectors Vec) {
	for (int i = 0; i < NT; i++) {
		x[i] *= Vec.filter[i];
	}
}

void Poly(comVectors cVec) {
	for (int i = 0; i < NT; i++) {
		cVec.Pol[i] = pow(cVec.E_a[i], 3);
	}
}

void Diff(comVectors cVec, Params Par) {
	complex<double> buf[2];

	buf[0] = (cVec.Pol[0] - 2.0 * cVec.Pol[1] + cVec.Pol[2]) / pow(Par.dt, 2);

	for (int i = 2; i < (NT - 1); i++) {

		buf[1] = (cVec.Pol[i - 1] - 2.0 * cVec.Pol[i] + cVec.Pol[i + 1]) / pow(Par.dt, 2);

		cVec.Pol[i - 1] = buf[0];

		buf[0] = buf[1];
	}
	cVec.Pol[NT - 2] = buf[0];
	cVec.Pol[0] = cVec.Pol[1];
	cVec.Pol[NT-1] = cVec.Pol[NT-2];
}

void NextHalfDiff(comVectors cVec, Vectors Vec, Params Par) {
	complex<double> im(0, 1);
	for (int i = 0; i < NT; i++) {
		if (Vec.w[i] == 0) {
			cVec.H_b[i] = cVec.H_a[i];
			//cout << "w[" << i << "] = 0 " << endl;
		}
		else {
			cVec.H_b[i] = cVec.H_a[i] - cVec.Pol[i] * Par.n2 * Par.I0 * Par.dz / 2.0 / (im * Vec.w[i]);
		}

	}
}
void NextHalf(comVectors cVec, Vectors Vec, Params Par) {
	complex<double> im(0, 1);
	for (int i = 0; i < NT; i++) {
cVec.H_b[i] = cVec.H_a[i] - im * Vec.w[i] * cVec.Pol[i] * Par.n2 * Par.I0 * Par.dz / 2.0  ;
	}
}

void NextFullDiff(comVectors cVec, Vectors Vec, Params Par) {
	
	for (int i = 0; i < NT; i++) {
		if (Vec.w[i] == 0) {
			cVec.H_b[i] = cVec.H_a[i];
		}
		else {
			cVec.H_b[i] = cVec.H_a[i] - cVec.Pol[i] * Par.n2 * Par.I0 * Par.dz / (Par.im * Vec.w[i]);
		}
		//cVec.H_a[i] = cVec.H_b[i];
	}
}
void NextFull(comVectors cVec, Vectors Vec, Params Par) {
	for (int i = 0; i < NT; i++) {
cVec.H_b[i] = cVec.H_a[i] - Par.im * Vec.w[i] * cVec.Pol[i] * Par.n2 * Par.I0 * Par.dz;
cVec.H_a[i] = cVec.H_b[i];
	}
}

void iFFTnx(complex<double>* x) {
	complex<double> nx(NX, 0);
	for (int i = 0; i < NT; i++) {
		x[i] = x[i] / nx;
	}
};

void Rcount(Params P,Vectors V, comVectors cV) {
	//cout << "  r_n  " << P.r_n << "    " << pow(P.r_n, 2.5) << "    " << pow(P.r_n, 1.5) << endl;
	double r1 = 4 * pow(P.r_n, 2.5);
	double r2 = 2 * pow(P.r_n, 1.5) / 3;
	for (int i = 0; i < NT; i++) {
		if (abs(cV.E_a[i]) > 1e-7) {
			//cout << "E_a >1e-7   = " << abs(cV.E_a[i]) << endl;
			V.R[i] = (r1 / abs(cV.E_a[i])) * exp(- r2 / abs(cV.E_a[i]));   //Атомные коэффициенты
		}
		else {
			//cout << "E_a < 1e-7   = " << abs(cV.E_a[i]) << endl;
			V.R[i] = 0;
		}
	}
}

void Necount(Params P, Vectors V) {
	double sum = 0;
	for (int i = 0; i < NT; i++) {
		sum += V.R[i] * P.dt;
		V.Ne[i] = (1 - exp(-sum));
	}
}

void Jfreecount(Params P, Vectors V, comVectors cV) {
	double sum = 0;
	V.J[0] = 0;
	for (int i = 0; i < NT-1; i++) {
		if (P.Vc != 0) {
			V.Jfree[i + 1] = V.Jfree[i] + P.dt * V.Ne[i] * real(cV.E_a[i]) - P.dt * P.Vc * P.t0 * V.Jfree[i];
			V.dJfree[i] = V.Ne[i] * real(cV.E_a[i]) - P.Vc * P.t0 * V.Jfree[i];
		}
		else {
			V.dJfree[i] = V.Ne[i] * real(cV.E_a[i]);
			sum += V.dJfree[i] * P.dt;
			V.Jfree[i] = sum;
		}
		
		V.Jion[i] = V.R[i] / real(cV.E_a[i]);
		V.J[i] = (P.e*P.e *P.N0 *P.t0*P.t0/P.me) *V.Jfree[i] + 
	}
}
 
void fftshift(complex<double>* x,complex<double>* y) {
	for (int i = 0; i < NT; i++) {
		if (i < (NT / 2)) {
			y[i] = x[i];
		}
		else {
			x[i - NT / 2] = x[i];
			x[i] = y[i - NT / 2];
		}
	}
	
};
void fprintf(complex<double>* x, char* f) {
	FILE* fout;
	fout = fopen(f, "w");
	for (int nx = 0; nx < NT; nx++) {
		fprintf(fout, "%.10g  %.10g \n", real(x[nx]), imag(x[nx]));
	}
	fclose(fout);
}

void fprintf(double* x, char* f) {
	FILE* fout;
	fout = fopen(f, "w");
	for (int nx = 0; nx < NT; nx++) {
		fprintf(fout, "%.25g \n", x[nx]);
	}
	fclose(fout);
}

