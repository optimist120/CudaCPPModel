#include "cuda_runtime.h"
#include "cuda.h"
#include "cuda/std/cmath"
#include "cufft.h"
//include "math.h"
#include "device_launch_parameters.h"
#include <complex>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <vector> 
//#include <thrust\complex.h>
#include <cuComplex.h>
#include <cufftXt.h>
#include <cublas_v2.h>
#include "Header.h"

//#include <thrust\device_vector.h>
//#include <thrust\host_vector.h>

using namespace std;
struct Params {
    const double pi = 3.141592653589793;
    const double Tall = 1000;
    const double t0 = 48;
    const double dw = 2 * pi / Tall * t0;
    const double W = 75;
    const double w0 = 2 * pi * W * 1e-3 * t0;
    const double c = 3e-5;
    const double A = 1.993e-4;
    const double B0 = 5.58e-7 / (t0 * t0);
    const double p = 1;
    const double n2 = 1e-19;
    const double I0 = pow(10, 11);
    const double z = 2200 / c / t0;
    const double dz = 1 / c / t0;
    double vg = NULL;
};

#define CUDA_CHECK(call) if((call) != cudaSuccess) { cudaError_t err = cudaGetLastError(); printf("CUDA error calling \"%s\", error code is %d\n", #call, err); printf("Error %d: %s\n", err, cudaGetErrorString(err)); exit(-1); }

const int NX = pow(2, 16);
const int NT = pow(2, 16);
__constant__ int cudaNX = 65536;

__constant__ int cudaNT = 65536;
const int NN = 8;
const int WN = 36;
const int Ntask = NN * WN;
double* Intens;
double* cudaIntens;
double* freq;
double* cudafreq;
const double TWO = 2;
__constant__ __device__ double cudaTWO = 2;

const double pi = 3.1415926536;
__constant__ __device__ double cudapi = 3.1415926536;
const double Ea = sqrt(31000);
__constant__ __device__ double cudaEa = 55.677643628300220;
const double ta = 0.024;
__constant__ __device__ double cudata = 0.024;
const double Tall = 50 / ta;
__constant__ __device__ double cudaTall = 2083.333333333334;
const double t0 = 5 / ta;
__device__ double cudat0 = 5 / 0.024;

const double a = 2;
__constant__ __device__ double cudaa = 2;
const double w0 = 2 * pi * 0.001 * 900 * ta;
__device__ double cudaw0 = 2 * 3.1415926536 * 0.001 * 900 * 0.024;
const double dt = Tall / NT;
__device__ double cudadt = (50 / 0.024) / 65536;
const double Xmax = 1024 * 4;
__device__ double cudaXmax = 1024 * 4;
const double dx = 2 * Xmax / NX;
__constant__ __device__ double cudadx = 2 * 4096 / 65536;
const cufftDoubleComplex i = { 0, -1 };

double* t;
double* cudat;
double* tfs;
double* cudatfs;
double* x1;
double* x1s;
double* cudax1s;
double* k1;
double* cudak1;
double* U;
double* cudaU;

double* E;
cufftDoubleComplex* Pol; //P
cufftDoubleComplex* Wer; //Wn
cufftDoubleComplex** psi_g;
cufftDoubleComplex* psi_g_0;
cufftDoubleComplex** Population;

cufftDoubleComplex* cudaW, * cudaW2;
cufftDoubleComplex* cudaWconst;
double* cudaE;
cufftDoubleComplex* cudaE_t;
cufftDoubleComplex* cudaExp_diff;


//__constant__ cufftDoubleComplex cudaI;


void Massive() {


    for (int nt = 0; nt < NT; nt++) {
        t[nt] = nt - NT / 2;
        t[nt] = t[nt] / NT * Tall;
        tfs[nt] = t[nt] / ta;
    }
    for (int nx = 0; nx < NX; nx++) {
        x1[nx] = nx - (NX / 2);
        x1[nx] = dx * x1[nx];
        if (nx < NX / 2) {
            k1[nx + NX / 2] = pi / Xmax * (nx - NX / 2);
            //cout << nx + NX / 2 << "    " << k1[nx + NX / 2] << endl;
            x1s[nx + NX / 2] = x1[nx];
            U[nx + NX / 2] = -0.146 * exp(-pow((x1[nx] / 4 / 2), 16)) / sqrt(pow(x1[nx] / 2 / 1.8, 2) + pow(0.175, 2));
        }
        else {
            k1[nx - NX / 2] = pi / Xmax * (nx - NX / 2);
            //cout << nx - NX / 2 << "    " << k1[nx - NX / 2] << endl;
            x1s[nx - NX / 2] = x1[nx];
            U[nx - NX / 2] = -0.146 * exp(-pow((x1[nx] / 4 / 2), 16)) / sqrt(pow(x1[nx] / 2 / 1.8, 2) + pow(0.175, 2));
        }
    }

}

void write_const() {
    ofstream foutx;
    ofstream foutk1;
    ofstream foutt;
    ofstream fouttfs;
    ofstream foutU;
    ofstream foutConst;
    foutx.open("x.txt");
    foutk1.open("k1.txt");
    foutt.open("t.txt");
    fouttfs.open("tfs.txt");
    foutU.open("U.txt");
    foutConst.open("constants.txt");
    for (int nt = 0; nt < NT; nt++) {
        foutt << t[nt] << endl;
        fouttfs << tfs[nt] << endl;
    }
    for (int nx = 0; nx < NX; nx++) {
        foutx << x1s[nx] << endl;
        foutk1 << k1[nx] << endl;
        foutU << U[nx] << endl;
    }
    foutConst << "Ea = " << Ea << "ta = " << ta
        << "t0 = " << t0 << "w0 = " << w0
        << "Tall = " << Tall << "nt = " << NT
        << "nx = " << NX << endl;
    foutx.close();
    foutk1.close();
    foutt.close();
    fouttfs.close();
    foutU.close();
    foutConst.close();
}

void read_const() {

    ifstream fin;
    fin.open("psi_g_lorentz_pit_1000.txt");
    for (int nx = 0; nx < NX; nx++) {
        if (nx < NX / 2) {
            fin >> psi_g[0][nx + NX / 2].x;
            psi_g[0][nx + NX / 2].y = 0;
            fin >> psi_g[1][nx + NX / 2].x;
            psi_g[1][nx + NX / 2].y = 0;
            fin >> psi_g[2][nx + NX / 2].x;
            psi_g[2][nx + NX / 2].y = 0;
        }
        else {
            fin >> psi_g[0][nx - NX / 2].x;
            psi_g[0][nx - NX / 2].y = 0;
            fin >> psi_g[1][nx - NX / 2].x;
            psi_g[1][nx - NX / 2].y = 0;
            fin >> psi_g[2][nx - NX / 2].x;
            psi_g[2][nx - NX / 2].y = 0;
        }
    }
    ifstream fin2;
    fin2.open("cudaWtext.txt");
    for (int nx = 0; nx < NX; nx++) {
        if (nx < NX / 2) {
            fin2 >> psi_g_0[nx + NX / 2].x;
            //cout << psi_g_0[nx + NX / 2].x<<endl;
            psi_g_0[nx + NX / 2].y = 0;
        }
        else {
            fin2 >> psi_g_0[nx - NX / 2].x;
            psi_g_0[nx - NX / 2].y = 0;

        }
    }
    fin.close();
    fin2.close();
    ofstream fout;
    fout.open("psi_g_0.txt");
    for (int i = 0; i < NX; i++) {
        fout << psi_g[0][i].x << endl;
    }
}

void w_I() {

    Intens = new double[NN];
    freq = new double[WN];

    for (int nn = 0; nn < NN; nn++) {
        Intens[nn] = 15; //ion 0.001
    }

    for (int wn = 0; wn < WN; wn++) {
        freq[wn] = 900;
    }
    ofstream fout;
    fout.open("Tasks.txt");
    fout << "Nomer zadachi     Intensivnost     Chastota";
    for (int nn = 0; nn < NN; nn++) {
        for (int wn = 0; wn < WN; wn++) {
            fout << nn * wn << "  " << Intens[nn] << "  " << freq[wn] << endl;
        }
    }

}

__global__ void makecudaE(double* cudaE, const double* cudat, const double cudaIntens, const double cudafreq, const double t0) {
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    cudaE[t] = sqrt(cudaIntens) / cudaEa * exp(-(cudat[t] * cudat[t]) \
        / 2 / (cudat0 * cudat0)) * sin(cudaw0 * cudat[t]);
}

__global__ void makecudaE_t(cufftDoubleComplex* cudaE_t, const double* cudax1s, const  double* cudaU, const double* cudaE, const int T) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    cudaE_t[x] = make_cuDoubleComplex(cos((cudax1s[x] * cudaE[T] - cudaU[x]) * cudadt / cudaTWO), \
        sin((cudax1s[x] * cudaE[T] - cudaU[x]) * cudadt / cudaTWO));
}

__global__ void cuda_make_Exp_diff(cufftDoubleComplex* cudaExp_diff, double* cudak1) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    cudaExp_diff[x].x = cos(cudadt * cudak1[x] * cudak1[x] / cudaTWO);
    cudaExp_diff[x].y = sin(-cudadt * cudak1[x] * cudak1[x] / cudaTWO);
}

__global__ void cuda_MUL(cufftDoubleComplex* cudaW, const cufftDoubleComplex* cudaE_t) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    cudaW[x] = cuCmul(cudaW[x], cudaE_t[x]);
}
__global__ void cuda_MULdouble(cufftDoubleComplex* cudaW, const double* cudaE_t) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    cudaW[x].x = cudaW[x].x * cudaE_t[x];
    cudaW[x].y = cudaW[x].y * cudaE_t[x];
}

__global__ void cuda_MULAL(cufftDoubleComplex* cudaW, const cufftDoubleComplex* cudaE_t, const double Onx) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    cudaW[x].x = cuCmul(cudaW[x], cudaE_t[x]).x / Onx;
    cudaW[x].y = cuCmul(cudaW[x], cudaE_t[x]).y / Onx;
}
__global__ void cuda_TRAN(Params* cudapar) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    double A;
    A = (*cudapar).A;
}

int cudaON(comVectors) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
        return 0;
    }
    CUDA_CHECK(cudaSetDevice(0));
    printf("cudaSetDevice\n");

}