#include <stdlib.h>
#include <math.h>
#include "vhf.h"

void eval_boys(double *Rt, int l, double a, double *rpq);

#define LSUM_MAX        (LMAX*4)
#define RTIDX_MAX       12

static int Rt_idx[] = {
// l = 1
0,0,0,
// l = 2
0,0,0,0,0,0,0,0,0,
// l = 3
0,0,1,0,0,0,0,1,3,0,0,0,0,0,0,0,1,3,6,
// l = 4
0,0,1,2,0,0,0,0,0,1,2,4,5,7,0,0,0,0,0,0,
0,0,0,0,0,1,2,4,5,7,10,11,13,16,
// l = 5
0,0,1,2,3,0,0,0,0,0,0,1,2,3,5,6,7,9,10,12,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,5,
6,7,9,10,12,15,16,17,19,20,22,25,26,28,31,
// l = 6
0,0,1,2,3,4,0,0,0,0,0,0,0,1,2,3,4,6,7,8,
9,11,12,13,15,16,18,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,2,3,4,6,7,8,9,11,12,13,
15,16,18,21,22,23,24,26,27,28,30,31,33,36,37,38,40,41,43,46,
47,49,52,
// l = 7
0,0,1,2,3,4,5,0,0,0,0,0,0,0,0,1,2,3,4,5,
7,8,9,10,11,13,14,15,16,18,19,20,22,23,25,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,2,3,4,5,7,8,9,10,11,13,14,15,16,18,19,
20,22,23,25,28,29,30,31,32,34,35,36,37,39,40,41,43,44,46,49,
50,51,52,54,55,56,58,59,61,64,65,66,68,69,71,74,75,77,80,
// l = 8
0,0,1,2,3,4,5,6,0,0,0,0,0,0,0,0,0,1,2,3,
4,5,6,8,9,10,11,12,13,15,16,17,18,19,21,22,23,24,26,27,
28,30,31,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,21,22,
23,24,26,27,28,30,31,33,36,37,38,39,40,41,43,44,45,46,47,49,
50,51,52,54,55,56,58,59,61,64,65,66,67,68,70,71,72,73,75,76,
77,79,80,82,85,86,87,88,90,91,92,94,95,97,100,101,102,104,105,107,
110,111,113,116,
// l = 9
0,0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0,0,0,1,
2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,24,
25,26,27,28,30,31,32,33,35,36,37,39,40,42,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
1,2,3,4,5,6,7,9,10,11,12,13,14,15,17,18,19,20,21,22,
24,25,26,27,28,30,31,32,33,35,36,37,39,40,42,45,46,47,48,49,
50,51,53,54,55,56,57,58,60,61,62,63,64,66,67,68,69,71,72,73,
75,76,78,81,82,83,84,85,86,88,89,90,91,92,94,95,96,97,99,100,
101,103,104,106,109,110,111,112,113,115,116,117,118,120,121,122,124,125,127,130,
131,132,133,135,136,137,139,140,142,145,146,147,149,150,152,155,156,158,161,
// l = 10
0,0,1,2,3,4,5,6,7,8,0,0,0,0,0,0,0,0,0,0,
0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,
22,23,24,25,27,28,29,30,31,32,34,35,36,37,38,40,41,42,43,45,
46,47,49,50,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,19,20,21,
22,23,24,25,27,28,29,30,31,32,34,35,36,37,38,40,41,42,43,45,
46,47,49,50,52,55,56,57,58,59,60,61,62,64,65,66,67,68,69,70,
72,73,74,75,76,77,79,80,81,82,83,85,86,87,88,90,91,92,94,95,
97,100,101,102,103,104,105,106,108,109,110,111,112,113,115,116,117,118,119,121,
122,123,124,126,127,128,130,131,133,136,137,138,139,140,141,143,144,145,146,147,
149,150,151,152,154,155,156,158,159,161,164,165,166,167,168,170,171,172,173,175,
176,177,179,180,182,185,186,187,188,190,191,192,194,195,197,200,201,202,204,205,
207,210,211,213,216,
// l = 11
0,0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,0,0,0,0,
0,0,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,
19,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,38,39,40,41,
42,43,45,46,47,48,49,51,52,53,54,56,57,58,60,61,63,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,
18,19,21,22,23,24,25,26,27,28,30,31,32,33,34,35,36,38,39,40,
41,42,43,45,46,47,48,49,51,52,53,54,56,57,58,60,61,63,66,67,
68,69,70,71,72,73,74,76,77,78,79,80,81,82,83,85,86,87,88,89,
90,91,93,94,95,96,97,98,100,101,102,103,104,106,107,108,109,111,112,113,
115,116,118,121,122,123,124,125,126,127,128,130,131,132,133,134,135,136,138,139,
140,141,142,143,145,146,147,148,149,151,152,153,154,156,157,158,160,161,163,166,
167,168,169,170,171,172,174,175,176,177,178,179,181,182,183,184,185,187,188,189,
190,192,193,194,196,197,199,202,203,204,205,206,207,209,210,211,212,213,215,216,
217,218,220,221,222,224,225,227,230,231,232,233,234,236,237,238,239,241,242,243,
245,246,248,251,252,253,254,256,257,258,260,261,263,266,267,268,270,271,273,276,
277,279,282,
// l = 12
0,0,1,2,3,4,5,6,7,8,9,10,0,0,0,0,0,0,0,0,
0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,
17,18,19,20,21,23,24,25,26,27,28,29,30,31,33,34,35,36,37,38,
39,40,42,43,44,45,46,47,48,50,51,52,53,54,55,57,58,59,60,61,
63,64,65,66,68,69,70,72,73,75,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,12,
13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,33,34,
35,36,37,38,39,40,42,43,44,45,46,47,48,50,51,52,53,54,55,57,
58,59,60,61,63,64,65,66,68,69,70,72,73,75,78,79,80,81,82,83,
84,85,86,87,89,90,91,92,93,94,95,96,97,99,100,101,102,103,104,105,
106,108,109,110,111,112,113,114,116,117,118,119,120,121,123,124,125,126,127,129,
130,131,132,134,135,136,138,139,141,144,145,146,147,148,149,150,151,152,154,155,
156,157,158,159,160,161,163,164,165,166,167,168,169,171,172,173,174,175,176,178,
179,180,181,182,184,185,186,187,189,190,191,193,194,196,199,200,201,202,203,204,
205,206,208,209,210,211,212,213,214,216,217,218,219,220,221,223,224,225,226,227,
229,230,231,232,234,235,236,238,239,241,244,245,246,247,248,249,250,252,253,254,
255,256,257,259,260,261,262,263,265,266,267,268,270,271,272,274,275,277,280,281,
282,283,284,285,287,288,289,290,291,293,294,295,296,298,299,300,302,303,305,308,
309,310,311,312,314,315,316,317,319,320,321,323,324,326,329,330,331,332,334,335,
336,338,339,341,344,345,346,348,349,351,354,355,357,360,
};

// l*(l+1)*(l+2)*(l+3)//24 - l
int Rt_idx_offsets[] = {
0,0,3,12,31,65,120,203,322,486,705,990,1353,1807
};

static void iter_Rt_1(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = ry * Rt[0];
        out[3] = rx * Rt[0];
}

static void iter_Rt_2(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = rz * Rt[1] + Rt[0];
        out[3] = ry * Rt[0];
        out[4] = ry * Rt[1];
        out[5] = ry * Rt[2] + Rt[0];
        out[6] = rx * Rt[0];
        out[7] = rx * Rt[1];
        out[8] = rx * Rt[2];
        out[9] = rx * Rt[3] + Rt[0];
}

static void iter_Rt_3(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = rz * Rt[1] + Rt[0];
        out[3] = rz * Rt[2] + 2 * Rt[1];
        out[4] = ry * Rt[0];
        out[5] = ry * Rt[1];
        out[6] = ry * Rt[2];
        out[7] = ry * Rt[3] + Rt[0];
        out[8] = ry * Rt[4] + Rt[1];
        out[9] = ry * Rt[5] + 2 * Rt[3];
        out[10] = rx * Rt[0];
        out[11] = rx * Rt[1];
        out[12] = rx * Rt[2];
        out[13] = rx * Rt[3];
        out[14] = rx * Rt[4];
        out[15] = rx * Rt[5];
        out[16] = rx * Rt[6] + Rt[0];
        out[17] = rx * Rt[7] + Rt[1];
        out[18] = rx * Rt[8] + Rt[3];
        out[19] = rx * Rt[9] + 2 * Rt[6];
}

static void iter_Rt_4(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = rz * Rt[1] + Rt[0];
        out[3] = rz * Rt[2] + 2 * Rt[1];
        out[4] = rz * Rt[3] + 3 * Rt[2];
        out[5] = ry * Rt[0];
        out[6] = ry * Rt[1];
        out[7] = ry * Rt[2];
        out[8] = ry * Rt[3];
        out[9] = ry * Rt[4] + Rt[0];
        out[10] = ry * Rt[5] + Rt[1];
        out[11] = ry * Rt[6] + Rt[2];
        out[12] = ry * Rt[7] + 2 * Rt[4];
        out[13] = ry * Rt[8] + 2 * Rt[5];
        out[14] = ry * Rt[9] + 3 * Rt[7];
        out[15] = rx * Rt[0];
        out[16] = rx * Rt[1];
        out[17] = rx * Rt[2];
        out[18] = rx * Rt[3];
        out[19] = rx * Rt[4];
        out[20] = rx * Rt[5];
        out[21] = rx * Rt[6];
        out[22] = rx * Rt[7];
        out[23] = rx * Rt[8];
        out[24] = rx * Rt[9];
        out[25] = rx * Rt[10] + Rt[0];
        out[26] = rx * Rt[11] + Rt[1];
        out[27] = rx * Rt[12] + Rt[2];
        out[28] = rx * Rt[13] + Rt[4];
        out[29] = rx * Rt[14] + Rt[5];
        out[30] = rx * Rt[15] + Rt[7];
        out[31] = rx * Rt[16] + 2 * Rt[10];
        out[32] = rx * Rt[17] + 2 * Rt[11];
        out[33] = rx * Rt[18] + 2 * Rt[13];
        out[34] = rx * Rt[19] + 3 * Rt[16];
}

static void iter_Rt_5(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = rz * Rt[1] + Rt[0];
        out[3] = rz * Rt[2] + 2 * Rt[1];
        out[4] = rz * Rt[3] + 3 * Rt[2];
        out[5] = rz * Rt[4] + 4 * Rt[3];
        out[6] = ry * Rt[0];
        out[7] = ry * Rt[1];
        out[8] = ry * Rt[2];
        out[9] = ry * Rt[3];
        out[10] = ry * Rt[4];
        out[11] = ry * Rt[5] + Rt[0];
        out[12] = ry * Rt[6] + Rt[1];
        out[13] = ry * Rt[7] + Rt[2];
        out[14] = ry * Rt[8] + Rt[3];
        out[15] = ry * Rt[9] + 2 * Rt[5];
        out[16] = ry * Rt[10] + 2 * Rt[6];
        out[17] = ry * Rt[11] + 2 * Rt[7];
        out[18] = ry * Rt[12] + 3 * Rt[9];
        out[19] = ry * Rt[13] + 3 * Rt[10];
        out[20] = ry * Rt[14] + 4 * Rt[12];
        out[21] = rx * Rt[0];
        out[22] = rx * Rt[1];
        out[23] = rx * Rt[2];
        out[24] = rx * Rt[3];
        out[25] = rx * Rt[4];
        out[26] = rx * Rt[5];
        out[27] = rx * Rt[6];
        out[28] = rx * Rt[7];
        out[29] = rx * Rt[8];
        out[30] = rx * Rt[9];
        out[31] = rx * Rt[10];
        out[32] = rx * Rt[11];
        out[33] = rx * Rt[12];
        out[34] = rx * Rt[13];
        out[35] = rx * Rt[14];
        out[36] = rx * Rt[15] + Rt[0];
        out[37] = rx * Rt[16] + Rt[1];
        out[38] = rx * Rt[17] + Rt[2];
        out[39] = rx * Rt[18] + Rt[3];
        out[40] = rx * Rt[19] + Rt[5];
        out[41] = rx * Rt[20] + Rt[6];
        out[42] = rx * Rt[21] + Rt[7];
        out[43] = rx * Rt[22] + Rt[9];
        out[44] = rx * Rt[23] + Rt[10];
        out[45] = rx * Rt[24] + Rt[12];
        out[46] = rx * Rt[25] + 2 * Rt[15];
        out[47] = rx * Rt[26] + 2 * Rt[16];
        out[48] = rx * Rt[27] + 2 * Rt[17];
        out[49] = rx * Rt[28] + 2 * Rt[19];
        out[50] = rx * Rt[29] + 2 * Rt[20];
        out[51] = rx * Rt[30] + 2 * Rt[22];
        out[52] = rx * Rt[31] + 3 * Rt[25];
        out[53] = rx * Rt[32] + 3 * Rt[26];
        out[54] = rx * Rt[33] + 3 * Rt[28];
        out[55] = rx * Rt[34] + 4 * Rt[31];
}

static void iter_Rt_6(double *out, double *Rt, double *rpq)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out[1] = rz * Rt[0];
        out[2] = rz * Rt[1] + Rt[0];
        out[3] = rz * Rt[2] + 2 * Rt[1];
        out[4] = rz * Rt[3] + 3 * Rt[2];
        out[5] = rz * Rt[4] + 4 * Rt[3];
        out[6] = rz * Rt[5] + 5 * Rt[4];
        out[7] = ry * Rt[0];
        out[8] = ry * Rt[1];
        out[9] = ry * Rt[2];
        out[10] = ry * Rt[3];
        out[11] = ry * Rt[4];
        out[12] = ry * Rt[5];
        out[13] = ry * Rt[6] + Rt[0];
        out[14] = ry * Rt[7] + Rt[1];
        out[15] = ry * Rt[8] + Rt[2];
        out[16] = ry * Rt[9] + Rt[3];
        out[17] = ry * Rt[10] + Rt[4];
        out[18] = ry * Rt[11] + 2 * Rt[6];
        out[19] = ry * Rt[12] + 2 * Rt[7];
        out[20] = ry * Rt[13] + 2 * Rt[8];
        out[21] = ry * Rt[14] + 2 * Rt[9];
        out[22] = ry * Rt[15] + 3 * Rt[11];
        out[23] = ry * Rt[16] + 3 * Rt[12];
        out[24] = ry * Rt[17] + 3 * Rt[13];
        out[25] = ry * Rt[18] + 4 * Rt[15];
        out[26] = ry * Rt[19] + 4 * Rt[16];
        out[27] = ry * Rt[20] + 5 * Rt[18];
        out[28] = rx * Rt[0];
        out[29] = rx * Rt[1];
        out[30] = rx * Rt[2];
        out[31] = rx * Rt[3];
        out[32] = rx * Rt[4];
        out[33] = rx * Rt[5];
        out[34] = rx * Rt[6];
        out[35] = rx * Rt[7];
        out[36] = rx * Rt[8];
        out[37] = rx * Rt[9];
        out[38] = rx * Rt[10];
        out[39] = rx * Rt[11];
        out[40] = rx * Rt[12];
        out[41] = rx * Rt[13];
        out[42] = rx * Rt[14];
        out[43] = rx * Rt[15];
        out[44] = rx * Rt[16];
        out[45] = rx * Rt[17];
        out[46] = rx * Rt[18];
        out[47] = rx * Rt[19];
        out[48] = rx * Rt[20];
        out[49] = rx * Rt[21] + Rt[0];
        out[50] = rx * Rt[22] + Rt[1];
        out[51] = rx * Rt[23] + Rt[2];
        out[52] = rx * Rt[24] + Rt[3];
        out[53] = rx * Rt[25] + Rt[4];
        out[54] = rx * Rt[26] + Rt[6];
        out[55] = rx * Rt[27] + Rt[7];
        out[56] = rx * Rt[28] + Rt[8];
        out[57] = rx * Rt[29] + Rt[9];
        out[58] = rx * Rt[30] + Rt[11];
        out[59] = rx * Rt[31] + Rt[12];
        out[60] = rx * Rt[32] + Rt[13];
        out[61] = rx * Rt[33] + Rt[15];
        out[62] = rx * Rt[34] + Rt[16];
        out[63] = rx * Rt[35] + Rt[18];
        out[64] = rx * Rt[36] + 2 * Rt[21];
        out[65] = rx * Rt[37] + 2 * Rt[22];
        out[66] = rx * Rt[38] + 2 * Rt[23];
        out[67] = rx * Rt[39] + 2 * Rt[24];
        out[68] = rx * Rt[40] + 2 * Rt[26];
        out[69] = rx * Rt[41] + 2 * Rt[27];
        out[70] = rx * Rt[42] + 2 * Rt[28];
        out[71] = rx * Rt[43] + 2 * Rt[30];
        out[72] = rx * Rt[44] + 2 * Rt[31];
        out[73] = rx * Rt[45] + 2 * Rt[33];
        out[74] = rx * Rt[46] + 3 * Rt[36];
        out[75] = rx * Rt[47] + 3 * Rt[37];
        out[76] = rx * Rt[48] + 3 * Rt[38];
        out[77] = rx * Rt[49] + 3 * Rt[40];
        out[78] = rx * Rt[50] + 3 * Rt[41];
        out[79] = rx * Rt[51] + 3 * Rt[43];
        out[80] = rx * Rt[52] + 4 * Rt[46];
        out[81] = rx * Rt[53] + 4 * Rt[47];
        out[82] = rx * Rt[54] + 4 * Rt[49];
        out[83] = rx * Rt[55] + 5 * Rt[52];
}

static void iter_Rt_n(double *out, double *Rt, double *rpq, int l)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        out++;
        int *p1 = Rt_idx + Rt_idx_offsets[l];
        int t, u, v, i, k;
        k = 0;
        i = 0;
#pragma GCC ivdep
        for (v = 0; v < l; v++) {
                out[k] = rz * Rt[i] + v * Rt[p1[k]];
                k++; i++;
        }
        i = 0;
        for (u = 0; u < l; u++) {
#pragma GCC ivdep
        for (v = 0; v < l-u; v++) {
                out[k] = ry * Rt[i] + u * Rt[p1[k]];
                k++; i++;
        } }
        i = 0;
        for (t = 0; t < l; t++) {
                // corresponding to the nested loops
                // for (u = 0; u < l-t; u++) for (v = 0; v < l-t-u; v++)
                int uv;
#pragma GCC ivdep
                for (uv = 0; uv < (l-t) * (l-t+1) / 2; uv++) {
                        out[k] = rx * Rt[i] + t * Rt[p1[k]];
                        k++; i++;
                }
        }
}

#define ADDR(l, t, u, v) \
        (lll - ((l)-(t)+1)*((l)-(t)+2)*((l)-(t)+3)/6 + \
         ((l)-(t)+1)*((l)-(t)+2)/2 - ((l)-(t)-(u)+1)*((l)-(t)-(u)+2)/2 + (v))
#define ADDR1(l, t, u, v) \
        (lll1 - ((l)-(t)+1)*((l)-(t)+2)*((l)-(t)+3)/6 + \
         ((l)-(t)+1)*((l)-(t)+2)/2 - ((l)-(t)-(u)+1)*((l)-(t)-(u)+2)/2 + (v))

static void iter_Rt_iter(double *out, double *Rt, double *rpq, int l)
{
        double rx = rpq[0];
        double ry = rpq[1];
        double rz = rpq[2];
        int l1 = l - 1;
        int lll1 = (l1+1)*(l1+2)*(l1+3)/6;
        out++;
        int t, u, v, i, k;
        k = 0;
        i = 0;
        out[k] = rz * Rt[i]; // v = 0
        k++; i++;
        for (v = 1; v < l; v++) {
                out[k] = rz * Rt[i] + v * Rt[ADDR1(l1,0,0,v-1)];
                k++; i++;
        }
        i = 0;
        for (v = 0; v < l; v++) { // u = 0
                out[k] = ry * Rt[i];
                k++; i++;
        }
        for (u = 1; u < l; u++) {
        for (v = 0; v < l-u; v++) {
                out[k] = ry * Rt[i] + u * Rt[ADDR1(l1,0,u-1,v)];
                k++; i++;
        } }
        i = 0;
        for (u = 0; u < l; u++) { // t = 0;
        for (v = 0; v < l-u; v++) {
                out[k] = rx * Rt[i];
                k++; i++;
        } }
        for (t = 1; t < l; t++) {
                for (u = 0; u < l-t; u++) {
                for (v = 0; v < l-t-u; v++) {
                        out[k] = rx * Rt[i] + t * Rt[ADDR1(l1,t-1,u,v)];
                        k++; i++;
                } }
        }
}

int get_R_tensor(double *Rt, int l, double a, double fac, double *rpq,
                 double *buf)
{
        if (l > LSUM_MAX) {
                return -1;
        }
        if (l == 0) {
                eval_boys(Rt, l, a, rpq);
                Rt[0] *= fac;
                return 0;
        }

        double boys[LSUM_MAX+1];
        eval_boys(boys, l, a, rpq);
        for (int n = 0; n <= l; n++) {
                boys[n] *= fac;
        }
        if (l == 1) {
                Rt[0] = boys[0];
                iter_Rt_1(Rt, boys+1, rpq);
                return 0;
        }

        double *tmp;
        if (l % 2 == 0) {
                tmp = buf;
                buf = Rt;
                Rt = tmp;
        }
        buf[0] = boys[l];

        for (int n = 1; n <= l; n++) {
                Rt[0] = boys[l-n];
                switch (n) {
                case 1: iter_Rt_1(Rt, buf, rpq); break;
                case 2: iter_Rt_2(Rt, buf, rpq); break;
                case 3: iter_Rt_3(Rt, buf, rpq); break;
                case 4: iter_Rt_4(Rt, buf, rpq); break;
                case 5: iter_Rt_5(Rt, buf, rpq); break;
                case 6: iter_Rt_6(Rt, buf, rpq); break;
                default:
                        if (n <= RTIDX_MAX) {
                                iter_Rt_n(Rt, buf, rpq, n);
                        } else {
                                iter_Rt_iter(Rt, buf, rpq, n);
                        }
                }
                // swap input and output
                tmp = buf;
                buf = Rt;
                Rt = tmp;
        }
        return 0;
}

/*
int get_Rt2(double *Rt2, int l1, int l2, double a, double fac, double *rpq,
            double *buf)
{
        int l = l1 + l2;
        int info = get_R_tensor(buf, l, a, fac, rpq, Rt2);
        int lll = (l+1)*(l+2)*(l+3)/6;
        int nf2 = (l2+1)*(l2+2)*(l2+3)/6;
        int e, f, g, t, u, v;
        int i, j;
        for (i = 0, e = 0; e <= l1; e++) {
        for (f = 0; f <= l1-e; f++) {
        for (g = 0; g <= l1-e-f; g++, i++) {
                if ((e + f + g) % 2 == 0) {
                        for (j = 0, t = 0; t <= l2; t++) {
                        for (u = 0; u <= l2-t; u++) {
                        for (v = 0; v <= l2-t-u; v++, j++) {
                                Rt2[i*nf2+j] = buf[ADDR(l,e+t,u+f,v+g)];
                        } } }
                } else {
                        for (j = 0, t = 0; t <= l2; t++) {
                        for (u = 0; u <= l2-t; u++) {
                        for (v = 0; v <= l2-t-u; v++, j++) {
                                Rt2[i*nf2+j] = -buf[ADDR(l,e+t,u+f,v+g)];
                        } } }
                }
        } } }
        return info;
}
*/

int get_Rt2(double *Rt2, int l1, int l2, double a, double fac, double *rpq,
            double *buf)
{
        int l = l1 + l2;
        int info = get_R_tensor(Rt2, l, a, fac, rpq, buf);
        if (l1 == 0) {
                return info;
        }
        int e, f, g, t, u, v, n;
        if (l2 == 0) {
                for (n = 0, e = 0; e <= l1; e++) {
                for (f = 0; f <= l1-e; f++) {
                for (g = 0; g <= l1-e-f; g++, n++) {
                        if ((e + f + g) % 2 == 1) {
                                Rt2[n] = -Rt2[n];
                        }
                } } }
                return info;
        }

        int stride_l = (l+1);
        int stride_ll = stride_l * (l+1);
        double *Rsub;
        for (n = 0, t = 0; t <= l; t++) {
                Rsub = buf + t*stride_ll;
                for (u = 0; u <= l-t; u++) {
#pragma GCC ivdep
                for (v = 0; v <= l-t-u; v++, n++) {
                        Rsub[u*stride_l+v] = Rt2[n];
                } }
        }

        for (n = 0, e = 0; e <= l1; e++) {
        for (f = 0; f <= l1-e; f++) {
        for (g = 0; g <= l1-e-f; g++) {
                if ((e + f + g) % 2 == 0) {
                        for (t = 0; t <= l2; t++) {
                                Rsub = buf + (e+t)*stride_ll + f*stride_l + g;
                                for (u = 0; u <= l2-t; u++) {
#pragma GCC ivdep
                                for (v = 0; v <= l2-t-u; v++, n++) {
                                        Rt2[n] = Rsub[u*stride_l+v];
                                } }
                        }
                } else {
                        for (t = 0; t <= l2; t++) {
                                Rsub = buf + (e+t)*stride_ll + f*stride_l + g;
                                for (u = 0; u <= l2-t; u++) {
#pragma GCC ivdep
                                for (v = 0; v <= l2-t-u; v++, n++) {
                                        Rt2[n] = -Rsub[u*stride_l+v];
                                } }
                        }
                }
        } } }
        return info;
}

#define Ex_at(i,j,t)    Ex[(i)*stride1+(j)*stride2+t]
#define Ey_at(i,j,t)    Ey[(i)*stride1+(j)*stride2+t]
#define Ez_at(i,j,t)    Ez[(i)*stride1+(j)*stride2+t]

void get_E_cart_components(double *Ecart, int li, int lj, double ai, double aj,
                           double *Ra, double *Rb)
{
        double aij = ai + aj;
        double xixj = Ra[0] - Rb[0];
        double yiyj = Ra[1] - Rb[1];
        double zizj = Ra[2] - Rb[2];
        double theta_ij = ai * aj / aij;
        double Kab = exp(-theta_ij * (xixj*xixj + yiyj*yiyj + zizj*zizj));
        double Xp = (ai * Ra[0] + aj * Rb[0]) / aij;
        double Yp = (ai * Ra[1] + aj * Rb[1]) / aij;
        double Zp = (ai * Ra[2] + aj * Rb[2]) / aij;
        double Xpa = Xp - Ra[0];
        double Ypa = Yp - Ra[1];
        double Zpa = Zp - Ra[2];
        double Xpb = Xp - Rb[0];
        double Ypb = Yp - Rb[1];
        double Zpb = Zp - Rb[2];
        int lij = li + lj;
        int stride2 = lij+1;
        int stride1 = (lj+1) * stride2;
        int Ex_size = (li+1) * stride1;
        double *Ex = Ecart;
        double *Ey = Ex + Ex_size;
        double *Ez = Ey + Ex_size;
        int i, j, t;
        double fac, fac1;

        Ex_at(0,0,0) = 1.;
        Ey_at(0,0,0) = 1.;
        Ez_at(0,0,0) = Kab;
        for (t = 1; t <= lij; t++) {
                Ex_at(0,0,t) = 0.;
                Ey_at(0,0,t) = 0.;
                Ez_at(0,0,t) = 0.;
        }

        for (j = 1; j <= lj; j++) {
                Ex_at(0,j,0) = Xpb * Ex_at(0,j-1,0) + Ex_at(0,j-1,1);
                Ey_at(0,j,0) = Ypb * Ey_at(0,j-1,0) + Ey_at(0,j-1,1);
                Ez_at(0,j,0) = Zpb * Ez_at(0,j-1,0) + Ez_at(0,j-1,1);
                for (t = 1; t <= lij; t++) {
                        fac = j/(2*aij*t);
                        Ex_at(0,j,t) = fac * Ex_at(0,j-1,t-1);
                        Ey_at(0,j,t) = fac * Ey_at(0,j-1,t-1);
                        Ez_at(0,j,t) = fac * Ez_at(0,j-1,t-1);
                }
        }

        for (i = 1; i <= li; i++) {
                Ex_at(i,0,0) = Xpa * Ex_at(i-1,0,0) + Ex_at(i-1,0,1);
                Ey_at(i,0,0) = Ypa * Ey_at(i-1,0,0) + Ey_at(i-1,0,1);
                Ez_at(i,0,0) = Zpa * Ez_at(i-1,0,0) + Ez_at(i-1,0,1);
                for (t = 1; t <= lij; t++) {
                        fac = i/(2*aij*t);
                        Ex_at(i,0,t) = fac * Ex_at(i-1,0,t-1);
                        Ey_at(i,0,t) = fac * Ey_at(i-1,0,t-1);
                        Ez_at(i,0,t) = fac * Ez_at(i-1,0,t-1);
                }
        }

        for (i = 1; i <= li; i++) {
                for (j = 1; j <= lj; j++) {
                        Ex_at(i,j,0) = Xpb * Ex_at(i,j-1,0) + Ex_at(i,j-1,1);
                        Ey_at(i,j,0) = Ypb * Ey_at(i,j-1,0) + Ey_at(i,j-1,1);
                        Ez_at(i,j,0) = Zpb * Ez_at(i,j-1,0) + Ez_at(i,j-1,1);
                        for (t = 1; t <= lij; t++) {
                                fac = i/(2*aij*t);
                                fac1 = j/(2*aij*t);
                                Ex_at(i,j,t) = fac*Ex_at(i-1,j,t-1) + fac1*Ex_at(i,j-1,t-1);
                                Ey_at(i,j,t) = fac*Ey_at(i-1,j,t-1) + fac1*Ey_at(i,j-1,t-1);
                                Ez_at(i,j,t) = fac*Ez_at(i-1,j,t-1) + fac1*Ez_at(i,j-1,t-1);
                        }
                }
        }
}

// Shape of E tensor is [:li+lj,:li,:lj]
void get_E_tensor(double *Et, int li, int lj, double ai, double aj,
                  double *Ra, double *Rb, double *buf)
{
        get_E_cart_components(buf, li, lj, ai, aj, Ra, Rb);
        int lij = li + lj;
        int stride2 = lij+1;
        int stride1 = (lj+1) * stride2;
        int Ex_size = (li+1) * stride1;
        double *Ex = buf;
        double *Ey = Ex + Ex_size;
        double *Ez = Ey + Ex_size;
        int t, u, v, n;
        int ix, iy, iz;
        int jx, jy, jz;

        n = 0;
        // products subject to t+u+v <= li+lj
        for (t = 0; t <= lij; t++) {
        for (u = 0; u <= lij-t; u++) {
        for (v = 0; v <= lij-t-u; v++) {
                for (ix = li; ix >= 0; ix--) {
                for (iy = li-ix; iy >= 0; iy--) {
                        iz = li - ix - iy;
                        for (jx = lj; jx >= 0; jx--) {
                        for (jy = lj-jx; jy >= 0; jy--) {
                                jz = lj - jx - jy;
                                Et[n] = Ex_at(ix,jx,t) * Ey_at(iy,jy,u) * Ez_at(iz,jz,v);
                                n++;
                        } }
                } }
        } } }
}
