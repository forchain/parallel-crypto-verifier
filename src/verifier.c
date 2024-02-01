#include <omp.h>
#include <mpi.h>

#if defined HAVE_CONFIG_H
#include "libsecp256k1-config.h"
#endif

#include <stdint.h>


#define NUM_OF_SIGS 100000

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else

#include <CL/cl.h>

#endif

#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include "secp256k1.c"
#include "include/secp256k1.h"
#include "testrand_impl.h"
#include <stdint.h>
#include <string.h>
#include "argparse.h"
#include <pthread.h>
#include "log.h"

#define MAX_SOURCE_SIZE (0x100000)

typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;

#include <unistd.h>

# define SECP256K1_RESTRICT

#define ECMULT_WINDOW_SIZE 15
#  define WINDOW_G ECMULT_WINDOW_SIZE
#  define WINDOW_A 5

#define ECMULT_TABLE_SIZE(w) (1 << ((w)-2))

#define SECP256K1_N_0 ((uint64_t)0xBFD25E8CD0364141ULL)
#define SECP256K1_N_1 ((uint64_t)0xBAAEDCE6AF48A03BULL)
#define SECP256K1_N_2 ((uint64_t)0xFFFFFFFFFFFFFFFEULL)
#define SECP256K1_N_3 ((uint64_t)0xFFFFFFFFFFFFFFFFULL)

#define SECP256K1_N_C_0 (~SECP256K1_N_0 + 1)
#define SECP256K1_N_C_1 (~SECP256K1_N_1)
#define SECP256K1_N_C_2 (1)

/* Limbs of half the secp256k1 order. */
#define SECP256K1_N_H_0 ((uint64_t)0xDFE92F46681B20A0ULL)
#define SECP256K1_N_H_1 ((uint64_t)0x5D576E7357A4501DULL)
#define SECP256K1_N_H_2 ((uint64_t)0xFFFFFFFFFFFFFFFFULL)
#define SECP256K1_N_H_3 ((uint64_t)0x7FFFFFFFFFFFFFFFULL)

#define SECP256K1_FE_CONST_INNER(d7, d6, d5, d4, d3, d2, d1, d0) { \
    (d0) | (((uint64_t)(d1) & 0xFFFFFUL) << 32), \
    ((uint64_t)(d1) >> 20) | (((uint64_t)(d2)) << 12) | (((uint64_t)(d3) & 0xFFUL) << 44), \
    ((uint64_t)(d3) >> 8) | (((uint64_t)(d4) & 0xFFFFFFFUL) << 24), \
    ((uint64_t)(d4) >> 28) | (((uint64_t)(d5)) << 4) | (((uint64_t)(d6) & 0xFFFFUL) << 36), \
    ((uint64_t)(d6) >> 16) | (((uint64_t)(d7)) << 16) \
}

#define SECP256K1_FE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {SECP256K1_FE_CONST_INNER((d7), (d6), (d5), (d4), (d3), (d2), (d1), (d0)), 1, 1}

#define ECMULT_TABLE_GET_GE_STORAGE(r, pre, n, w) do { \
    if ((n) > 0) { \
        secp256k1_ge_from_storageX_m((r), (pre)[((n)-1)/2]); \
    } else { \
        secp256k1_ge_from_storageX_m((r), (pre)[(-(n)-1)/2]); \
        secp256k1_fe_negateX(&((r)->y), &((r)->y), 1); \
    } \
} while(0)

#define ECMULT_TABLE_GET_GE(r, pre, n, w) do { \
    if ((n) > 0) { \
        *(r) = (pre)[((n)-1)/2]; \
    } else { \
        *(r) = (pre)[(-(n)-1)/2]; \
        secp256k1_fe_negateX(&((r)->y), &((r)->y), 1); \
    } \
} while(0)

#define muladd(a, b) { \
    uint64_t tl, th; \
    { \
        uint128_t t = (uint128_t)a * b; \
        th = t >> 64;         /* at most 0xFFFFFFFFFFFFFFFE */ \
        tl = t; \
    } \
    c0 += tl;                 /* overflow is handled on the next line */ \
    th += (c0 < tl) ? 1 : 0;  /* at most 0xFFFFFFFFFFFFFFFF */ \
    c1 += th;                 /* overflow is handled on the next line */ \
    c2 += (c1 < th) ? 1 : 0;  /* never overflows by contract (verified in the next line) */ \
}

/** Add a*b to the number defined by (c0,c1). c1 must never overflow. */
#define muladd_fast(a, b) { \
    uint64_t tl, th; \
    { \
        uint128_t t = (uint128_t)a * b; \
        th = t >> 64;         /* at most 0xFFFFFFFFFFFFFFFE */ \
        tl = t; \
    } \
    c0 += tl;                 /* overflow is handled on the next line */ \
    th += (c0 < tl) ? 1 : 0;  /* at most 0xFFFFFFFFFFFFFFFF */ \
    c1 += th;                 /* never overflows by contract (verified in the next line) */ \
}

/** Add 2*a*b to the number defined by (c0,c1,c2). c2 must never overflow. */
#define muladd2(a, b) { \
    uint64_t tl, th, th2, tl2; \
    { \
        uint128_t t = (uint128_t)a * b; \
        th = t >> 64;               /* at most 0xFFFFFFFFFFFFFFFE */ \
        tl = t; \
    } \
    th2 = th + th;                  /* at most 0xFFFFFFFFFFFFFFFE (in case th was 0x7FFFFFFFFFFFFFFF) */ \
    c2 += (th2 < th) ? 1 : 0;       /* never overflows by contract (verified the next line) */ \
    tl2 = tl + tl;                  /* at most 0xFFFFFFFFFFFFFFFE (in case the lowest 63 bits of tl were 0x7FFFFFFFFFFFFFFF) */ \
    th2 += (tl2 < tl) ? 1 : 0;      /* at most 0xFFFFFFFFFFFFFFFF */ \
    c0 += tl2;                      /* overflow is handled on the next line */ \
    th2 += (c0 < tl2) ? 1 : 0;      /* second overflow is handled on the next line */ \
    c2 += (c0 < tl2) & (th2 == 0);  /* never overflows by contract (verified the next line) */ \
    c1 += th2;                      /* overflow is handled on the next line */ \
    c2 += (c1 < th2) ? 1 : 0;       /* never overflows by contract (verified the next line) */ \
}

/** Add a to the number defined by (c0,c1,c2). c2 must never overflow. */
#define sumadd(a) { \
    unsigned int over; \
    c0 += (a);                  /* overflow is handled on the next line */ \
    over = (c0 < (a)) ? 1 : 0; \
    c1 += over;                 /* overflow is handled on the next line */ \
    c2 += (c1 < over) ? 1 : 0;  /* never overflows by contract */ \
}

/** Add a to the number defined by (c0,c1). c1 must never overflow, c2 must be zero. */
#define sumadd_fast(a) { \
    c0 += (a);                 /* overflow is handled on the next line */ \
    c1 += (c0 < (a)) ? 1 : 0;  /* never overflows by contract (verified the next line) */ \
}

/** Extract the lowest 64 bits of (c0,c1,c2) into n, and left shift the number 64 bits. */
#define extract(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = c2; \
    c2 = 0; \
}

/** Extract the lowest 64 bits of (c0,c1,c2) into n, and left shift the number 64 bits. c2 is required to be zero. */
#define extract_fast(n) { \
    (n) = c0; \
    c0 = c1; \
    c1 = 0; \
}

static const char *const usages[] = {
        "verifier [options] [[--] args]",
        "verifier [options]",
        NULL,
};

// mode
#define MODE_CPU 0
#define MODE_OPENMP 1
#define MODE_GPU 2
#define MODE_OPENMP_GPU 3
#define MODE_ALL 4
#define MODE_MPI_CPU 5
#define MODE_MPI_OPENMP 6
#define MODE_MPI_GPU 7
#define MODE_MPI_OPENMP_GPU 8

static int count = 64;
static secp256k1_context *ctx = NULL;

typedef struct {
    uint64_t d[4];
} secp256k1_scalarX;

typedef struct {
    uint64_t n[5];
#ifdef VERIFY
    int magnitude;
    int normalized;
#endif
} secp256k1_feX;

typedef struct {
    secp256k1_feX x;
    secp256k1_feX y;
    int infinity; /* whether this represents the point at infinity */
} secp256k1_geX;

typedef struct {
    secp256k1_feX x; /* actual X: x/z^2 */
    secp256k1_feX y; /* actual Y: y/z^3 */
    secp256k1_feX z;
    int infinity; /* whether this represents the point at infinity */
} secp256k1_gejX;

struct secp256k1_strauss_point_stateX {
    int wnaf_na[256];
    int bits_na;
    size_t input_pos;
};

struct secp256k1_strauss_stateX {
    secp256k1_gejX *prej;
    secp256k1_feX *zr;
    secp256k1_geX *pre_a;
    struct secp256k1_strauss_point_stateX *ps;
};

typedef struct {
    uint64_t n[4];
} secp256k1_fe_storageX;

typedef struct {
    secp256k1_fe_storageX x;
    secp256k1_fe_storageX y;
} secp256k1_ge_storageX;

typedef struct {
    secp256k1_ge_storageX (*pre_g)[];
} secp256k1_ecmult_contextX;

typedef struct {
    void (*fn)(const char *text, void *data);

    const void *data;
} secp256k1_callbackX;

typedef struct {
    secp256k1_ge_storageX (*prec)[64][16];
    secp256k1_scalarX blind;
    secp256k1_gejX initial;
} secp256k1_ecmult_gen_contextX;

typedef struct secp256k1_context_structX secp256k1_contextX;

struct secp256k1_context_structX {
    secp256k1_ecmult_contextX ecmult_ctx;
    secp256k1_ecmult_gen_contextX ecmult_gen_ctx;
    secp256k1_callbackX illegal_callback;
    secp256k1_callbackX error_callback;
};

typedef struct {
    unsigned char data[64];
} secp256k1_ecdsa_signatureX;

typedef struct {
    unsigned char data[64];
} secp256k1_pubkeyX;

typedef unsigned char secp256k1_msgX[32];

int secp256k1_scalar_is_zeroX(const secp256k1_scalarX *a) {
    return (a->d[0] | a->d[1] | a->d[2] | a->d[3]) == 0;
}

void secp256k1_scalar_set_intX(secp256k1_scalarX *r, unsigned int v) {
    r->d[0] = v;
    r->d[1] = 0;
    r->d[2] = 0;
    r->d[3] = 0;
}

static const secp256k1_feX secp256k1_ecdsa_const_order_as_feX = SECP256K1_FE_CONST(
        0xFFFFFFFFUL, 0xFFFFFFFFUL, 0xFFFFFFFFUL, 0xFFFFFFFEUL,
        0xBAAEDCE6UL, 0xAF48A03BUL, 0xBFD25E8CUL, 0xD0364141UL
);

static const secp256k1_feX secp256k1_ecdsa_const_p_minus_orderX = SECP256K1_FE_CONST(
        0, 0, 0, 1, 0x45512319UL, 0x50B75FC4UL, 0x402DA172UL, 0x2FC9BAEEUL
);

void secp256k1_scalar_inverseX(secp256k1_scalarX *r, const secp256k1_scalarX *x);

static void secp256k1_scalar_inverse_varX(secp256k1_scalarX *r, const secp256k1_scalarX *x);

static void secp256k1_scalar_reduce_512X(secp256k1_scalarX *r, const uint64_t *l);

void secp256k1_scalar_sqrX(secp256k1_scalarX *r, const secp256k1_scalarX *a);

int secp256k1_scalar_is_evenX(const secp256k1_scalarX *a);

void secp256k1_scalar_sqr_512X(uint64_t l[8], const secp256k1_scalarX *a);

int secp256k1_scalar_reduceX(secp256k1_scalarX *r, unsigned int overflow);

int secp256k1_scalar_check_overflowX(const secp256k1_scalarX *a);

static void secp256k1_scalar_mul_512X(uint64_t l[8], const secp256k1_scalarX *a, const secp256k1_scalarX *b);

void secp256k1_scalar_mulX(secp256k1_scalarX *r, const secp256k1_scalarX *a, const secp256k1_scalarX *b);

static void secp256k1_gej_set_geX(secp256k1_gejX *r, const secp256k1_geX *a);

static void secp256k1_fe_verifyX(const secp256k1_feX *a);

void secp256k1_fe_set_intX(secp256k1_feX *r, int a);

unsigned int secp256k1_scalar_get_bitsX(const secp256k1_scalarX *a, unsigned int offset, unsigned int count);

static void secp256k1_scalar_negateX(secp256k1_scalarX *r, const secp256k1_scalarX *a);

unsigned int secp256k1_scalar_get_bits_varX(const secp256k1_scalarX *a, unsigned int offset, unsigned int count);

static int secp256k1_ecmult_wnafX(int *wnaf, int len, const secp256k1_scalarX *a, int w);

void secp256k1_fe_clearX(secp256k1_feX *a);

void secp256k1_ecmult_strauss_wnafX(const secp256k1_ecmult_contextX *ctx, const struct secp256k1_strauss_stateX *state,
                                    secp256k1_gejX *r, int num, const secp256k1_gejX *a, const secp256k1_scalarX *na,
                                    const secp256k1_scalarX *ng);

int secp256k1_fe_equal_varX(const secp256k1_feX *a, const secp256k1_feX *b);


void secp256k1_scalar_inverseX(secp256k1_scalarX *r, const secp256k1_scalarX *x) {
    secp256k1_scalarX *t;
    int i;
    /* First compute xN as x ^ (2^N - 1) for some values of N,
     * and uM as x ^ M for some values of M. */
    secp256k1_scalarX x2, x3, x6, x8, x14, x28, x56, x112, x126;
    secp256k1_scalarX u2, u5, u9, u11, u13;

    secp256k1_scalar_sqrX(&u2, x);
    secp256k1_scalar_mulX(&x2, &u2, x);
    secp256k1_scalar_mulX(&u5, &u2, &x2);
    secp256k1_scalar_mulX(&x3, &u5, &u2);
    secp256k1_scalar_mulX(&u9, &x3, &u2);
    secp256k1_scalar_mulX(&u11, &u9, &u2);
    secp256k1_scalar_mulX(&u13, &u11, &u2);

    secp256k1_scalar_sqrX(&x6, &u13);
    secp256k1_scalar_sqrX(&x6, &x6);
    secp256k1_scalar_mulX(&x6, &x6, &u11);

    secp256k1_scalar_sqrX(&x8, &x6);
    secp256k1_scalar_sqrX(&x8, &x8);
    secp256k1_scalar_mulX(&x8, &x8, &x2);

    secp256k1_scalar_sqrX(&x14, &x8);
    for (i = 0; i < 5; i++) {
        secp256k1_scalar_sqrX(&x14, &x14);
    }
    secp256k1_scalar_mulX(&x14, &x14, &x6);

    secp256k1_scalar_sqrX(&x28, &x14);
    for (i = 0; i < 13; i++) {
        secp256k1_scalar_sqrX(&x28, &x28);
    }
    secp256k1_scalar_mulX(&x28, &x28, &x14);

    secp256k1_scalar_sqrX(&x56, &x28);
    for (i = 0; i < 27; i++) {
        secp256k1_scalar_sqrX(&x56, &x56);
    }
    secp256k1_scalar_mulX(&x56, &x56, &x28);

    secp256k1_scalar_sqrX(&x112, &x56);
    for (i = 0; i < 55; i++) {
        secp256k1_scalar_sqrX(&x112, &x112);
    }
    secp256k1_scalar_mulX(&x112, &x112, &x56);

    secp256k1_scalar_sqrX(&x126, &x112);
    for (i = 0; i < 13; i++) {
        secp256k1_scalar_sqrX(&x126, &x126);
    }
    secp256k1_scalar_mulX(&x126, &x126, &x14);

    /* Then accumulate the final result (t starts at x126). */
    t = &x126;
    for (i = 0; i < 3; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u5); /* 101 */
    for (i = 0; i < 4; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 4; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u5); /* 101 */
    for (i = 0; i < 5; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u11); /* 1011 */
    for (i = 0; i < 4; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u11); /* 1011 */
    for (i = 0; i < 4; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 5; i++) { /* 00 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 6; i++) { /* 00 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u13); /* 1101 */
    for (i = 0; i < 4; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u5); /* 101 */
    for (i = 0; i < 3; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 5; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u9); /* 1001 */
    for (i = 0; i < 6; i++) { /* 000 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u5); /* 101 */
    for (i = 0; i < 10; i++) { /* 0000000 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 4; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x3); /* 111 */
    for (i = 0; i < 9; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x8); /* 11111111 */
    for (i = 0; i < 5; i++) { /* 0 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u9); /* 1001 */
    for (i = 0; i < 6; i++) { /* 00 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u11); /* 1011 */
    for (i = 0; i < 4; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u13); /* 1101 */
    for (i = 0; i < 5; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &x2); /* 11 */
    for (i = 0; i < 6; i++) { /* 00 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u13); /* 1101 */
    for (i = 0; i < 10; i++) { /* 000000 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u13); /* 1101 */
    for (i = 0; i < 4; i++) {
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, &u9); /* 1001 */
    for (i = 0; i < 6; i++) { /* 00000 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(t, t, x); /* 1 */
    for (i = 0; i < 8; i++) { /* 00 */
        secp256k1_scalar_sqrX(t, t);
    }
    secp256k1_scalar_mulX(r, t, &x6); /* 111111 */
}


static void secp256k1_scalar_inverse_varX(secp256k1_scalarX *r, const secp256k1_scalarX *x) {
    secp256k1_scalar_inverseX(r, x);
}

static void secp256k1_scalar_reduce_512X(secp256k1_scalarX *r, const uint64_t *l) {
    uint128_t c;
    uint64_t c0, c1, c2;
    uint64_t n0 = l[4], n1 = l[5], n2 = l[6], n3 = l[7];
    uint64_t m0, m1, m2, m3, m4, m5;
    uint32_t m6;
    uint64_t p0, p1, p2, p3;
    uint32_t p4;

    /* Reduce 512 bits into 385. */
    /* m[0..6] = l[0..3] + n[0..3] * SECP256K1_N_C. */
    c0 = l[0];
    c1 = 0;
    c2 = 0;
    muladd_fast(n0, SECP256K1_N_C_0);
    extract_fast(m0);
    sumadd_fast(l[1]);
    muladd(n1, SECP256K1_N_C_0);
    muladd(n0, SECP256K1_N_C_1);
    extract(m1);
    sumadd(l[2]);
    muladd(n2, SECP256K1_N_C_0);
    muladd(n1, SECP256K1_N_C_1);
    sumadd(n0);
    extract(m2);
    sumadd(l[3]);
    muladd(n3, SECP256K1_N_C_0);
    muladd(n2, SECP256K1_N_C_1);
    sumadd(n1);
    extract(m3);
    muladd(n3, SECP256K1_N_C_1);
    sumadd(n2);
    extract(m4);
    sumadd_fast(n3);
    extract_fast(m5);
    m6 = c0;

    /* Reduce 385 bits into 258. */
    /* p[0..4] = m[0..3] + m[4..6] * SECP256K1_N_C. */
    c0 = m0;
    c1 = 0;
    c2 = 0;
    muladd_fast(m4, SECP256K1_N_C_0);
    extract_fast(p0);
    sumadd_fast(m1);
    muladd(m5, SECP256K1_N_C_0);
    muladd(m4, SECP256K1_N_C_1);
    extract(p1);
    sumadd(m2);
    muladd(m6, SECP256K1_N_C_0);
    muladd(m5, SECP256K1_N_C_1);
    sumadd(m4);
    extract(p2);
    sumadd_fast(m3);
    muladd_fast(m6, SECP256K1_N_C_1);
    sumadd_fast(m5);
    extract_fast(p3);
    p4 = c0 + m6;

    /* Reduce 258 bits into 256. */
    /* r[0..3] = p[0..3] + p[4] * SECP256K1_N_C. */
    c = p0 + (uint128_t) SECP256K1_N_C_0 * p4;
    r->d[0] = c & 0xFFFFFFFFFFFFFFFFULL;
    c >>= 64;
    c += p1 + (uint128_t) SECP256K1_N_C_1 * p4;
    r->d[1] = c & 0xFFFFFFFFFFFFFFFFULL;
    c >>= 64;
    c += p2 + (uint128_t) p4;
    r->d[2] = c & 0xFFFFFFFFFFFFFFFFULL;
    c >>= 64;
    c += p3;
    r->d[3] = c & 0xFFFFFFFFFFFFFFFFULL;
    c >>= 64;

    /* Final reduction of r. */
    secp256k1_scalar_reduceX(r, c + secp256k1_scalar_check_overflowX(r));
}

void secp256k1_scalar_sqrX(secp256k1_scalarX *r, const secp256k1_scalarX *a) {
    uint64_t l[8];
    secp256k1_scalar_sqr_512X(l, a);
    secp256k1_scalar_reduce_512X(r, l);
}

int secp256k1_scalar_is_evenX(const secp256k1_scalarX *a) {
    return !(a->d[0] & 1);
}

void secp256k1_scalar_sqr_512X(uint64_t l[8], const secp256k1_scalarX *a) {
    /* 160 bit accumulator. */
    uint64_t c0 = 0, c1 = 0;
    uint32_t c2 = 0;

    /* l[0..7] = a[0..3] * b[0..3]. */
    muladd_fast(a->d[0], a->d[0]);
    extract_fast(l[0]);
    muladd2(a->d[0], a->d[1]);
    extract(l[1]);
    muladd2(a->d[0], a->d[2]);
    muladd(a->d[1], a->d[1]);
    extract(l[2]);
    muladd2(a->d[0], a->d[3]);
    muladd2(a->d[1], a->d[2]);
    extract(l[3]);
    muladd2(a->d[1], a->d[3]);
    muladd(a->d[2], a->d[2]);
    extract(l[4]);
    muladd2(a->d[2], a->d[3]);
    extract(l[5]);
    muladd_fast(a->d[3], a->d[3]);
    extract_fast(l[6]);
    l[7] = c0;
}

int secp256k1_scalar_reduceX(secp256k1_scalarX *r, unsigned int overflow) {
    uint128_t t;
    t = (uint128_t) r->d[0] + overflow * SECP256K1_N_C_0;
    r->d[0] = t & 0xFFFFFFFFFFFFFFFFULL;
    t >>= 64;
    t += (uint128_t) r->d[1] + overflow * SECP256K1_N_C_1;
    r->d[1] = t & 0xFFFFFFFFFFFFFFFFULL;
    t >>= 64;
    t += (uint128_t) r->d[2] + overflow * SECP256K1_N_C_2;
    r->d[2] = t & 0xFFFFFFFFFFFFFFFFULL;
    t >>= 64;
    t += (uint64_t) r->d[3];
    r->d[3] = t & 0xFFFFFFFFFFFFFFFFULL;
    return overflow;
}

int secp256k1_scalar_check_overflowX(const secp256k1_scalarX *a) {
    int yes = 0;
    int no = 0;
    no |= (a->d[3] < SECP256K1_N_3);
    no |= (a->d[2] < SECP256K1_N_2);
    yes |= (a->d[2] > SECP256K1_N_2) & ~no;
    no |= (a->d[1] < SECP256K1_N_1);
    yes |= (a->d[1] > SECP256K1_N_1) & ~no;
    yes |= (a->d[0] >= SECP256K1_N_0) & ~no;
    return yes;
}

static void secp256k1_scalar_mul_512X(uint64_t l[8], const secp256k1_scalarX *a, const secp256k1_scalarX *b) {
    uint64_t c0 = 0, c1 = 0;
    uint32_t c2 = 0;

    muladd_fast(a->d[0], b->d[0]);
    extract_fast(l[0]);
    muladd(a->d[0], b->d[1]);
    muladd(a->d[1], b->d[0]);
    extract(l[1]);
    muladd(a->d[0], b->d[2]);
    muladd(a->d[1], b->d[1]);
    muladd(a->d[2], b->d[0]);
    extract(l[2]);
    muladd(a->d[0], b->d[3]);
    muladd(a->d[1], b->d[2]);
    muladd(a->d[2], b->d[1]);
    muladd(a->d[3], b->d[0]);
    extract(l[3]);
    muladd(a->d[1], b->d[3]);
    muladd(a->d[2], b->d[2]);
    muladd(a->d[3], b->d[1]);
    extract(l[4]);
    muladd(a->d[2], b->d[3]);
    muladd(a->d[3], b->d[2]);
    extract(l[5]);
    muladd_fast(a->d[3], b->d[3]);
    extract_fast(l[6]);
    l[7] = c0;
}


void secp256k1_scalar_mulX(secp256k1_scalarX *r, const secp256k1_scalarX *a, const secp256k1_scalarX *b) {
    uint64_t l[8];
    secp256k1_scalar_mul_512X(l, a, b);
    secp256k1_scalar_reduce_512X(r, l);
}

static void secp256k1_gej_set_geX(secp256k1_gejX *r, const secp256k1_geX *a) {
    r->infinity = a->infinity;
    r->x = a->x;
    r->y = a->y;
    secp256k1_fe_set_intX(&r->z, 1);
}

static void secp256k1_fe_verifyX(const secp256k1_feX *a) {
    const uint32_t *d = a->n;
    int m = a->normalized ? 1 : 2 * a->magnitude, r = 1;
    r &= (d[0] <= 0x3FFFFFFUL * m);
    r &= (d[1] <= 0x3FFFFFFUL * m);
    r &= (d[2] <= 0x3FFFFFFUL * m);
    r &= (d[3] <= 0x3FFFFFFUL * m);
    r &= (d[4] <= 0x3FFFFFFUL * m);
    r &= (d[5] <= 0x3FFFFFFUL * m);
    r &= (d[6] <= 0x3FFFFFFUL * m);
    r &= (d[7] <= 0x3FFFFFFUL * m);
    r &= (d[8] <= 0x3FFFFFFUL * m);
    r &= (d[9] <= 0x03FFFFFUL * m);
    r &= (a->magnitude >= 0);
    r &= (a->magnitude <= 32);
    if (a->normalized) {
        r &= (a->magnitude <= 1);
        if (r && (d[9] == 0x03FFFFFUL)) {
            uint32_t mid = d[8] & d[7] & d[6] & d[5] & d[4] & d[3] & d[2];
            if (mid == 0x3FFFFFFUL) {
                r &= ((d[1] + 0x40UL + ((d[0] + 0x3D1UL) >> 26)) <= 0x3FFFFFFUL);
            }
        }
    }
}

void secp256k1_fe_set_intX(secp256k1_feX *r, int a) {
    r->n[0] = a;
    r->n[1] = r->n[2] = r->n[3] = r->n[4] = 0;
    r->magnitude = 1;
    r->normalized = 1;
    secp256k1_fe_verifyX(r);
}

unsigned int secp256k1_scalar_get_bitsX(const secp256k1_scalarX *a, unsigned int offset, unsigned int count) {
    return (a->d[offset >> 6] >> (offset & 0x3F)) & ((((uint64_t) 1) << count) - 1);
}

static void secp256k1_scalar_negateX(secp256k1_scalarX *r, const secp256k1_scalarX *a) {
    uint64_t nonzero = 0xFFFFFFFFFFFFFFFFULL * (secp256k1_scalar_is_zeroX(a) == 0);
    uint128_t t = (uint128_t) (~a->d[0]) + SECP256K1_N_0 + 1;
    r->d[0] = t & nonzero;
    t >>= 64;
    t += (uint128_t) (~a->d[1]) + SECP256K1_N_1;
    r->d[1] = t & nonzero;
    t >>= 64;
    t += (uint128_t) (~a->d[2]) + SECP256K1_N_2;
    r->d[2] = t & nonzero;
    t >>= 64;
    t += (uint128_t) (~a->d[3]) + SECP256K1_N_3;
    r->d[3] = t & nonzero;
}

unsigned int secp256k1_scalar_get_bits_varX(const secp256k1_scalarX *a, unsigned int offset, unsigned int count) {
    if ((offset + count - 1) >> 6 == offset >> 6) {
        return secp256k1_scalar_get_bitsX(a, offset, count);
    } else {
        return ((a->d[offset >> 6] >> (offset & 0x3F)) | (a->d[(offset >> 6) + 1] << (64 - (offset & 0x3F)))) &
               ((((uint64_t) 1) << count) - 1);
    }
}

static int secp256k1_ecmult_wnafX(int *wnaf, int len, const secp256k1_scalarX *a, int w) {
    secp256k1_scalarX s;
    int last_set_bit = -1;
    int bit = 0;
    int sign = 1;
    int carry = 0;

    memset(wnaf, 0, len * sizeof(wnaf[0]));

    s = *a;
    if (secp256k1_scalar_get_bitsX(&s, 255, 1)) {
        secp256k1_scalar_negateX(&s, &s);
        sign = -1;
    }

    while (bit < len) {
        int now;
        int word;
        if (secp256k1_scalar_get_bitsX(&s, bit, 1) == (unsigned int) carry) {
            bit++;
            continue;
        }

        now = w;
        if (now > len - bit) {
            now = len - bit;
        }

        word = secp256k1_scalar_get_bits_varX(&s, bit, now) + carry;

        carry = (word >> (w - 1)) & 1;
        word -= carry << w;

        wnaf[bit] = sign * word;
        last_set_bit = bit;

        bit += now;
    }
    while (bit < 256) {
    }
    return last_set_bit + 1;
}

void secp256k1_fe_clearX(secp256k1_feX *a) {
    int i;
    a->magnitude = 0;
    a->normalized = 1;
    for (i = 0; i < 5; i++) {
        a->n[i] = 0;
    }
}


void secp256k1_fe_from_storageX_m(secp256k1_feX *r, secp256k1_fe_storageX a) {
    r->n[0] = a.n[0] & 0xFFFFFFFFFFFFFULL;
    r->n[1] = a.n[0] >> 52 | ((a.n[1] << 12) & 0xFFFFFFFFFFFFFULL);
    r->n[2] = a.n[1] >> 40 | ((a.n[2] << 24) & 0xFFFFFFFFFFFFFULL);
    r->n[3] = a.n[2] >> 28 | ((a.n[3] << 36) & 0xFFFFFFFFFFFFFULL);
    r->n[4] = a.n[3] >> 16;
    r->magnitude = 1;
    r->normalized = 1;
}

static void secp256k1_ge_from_storageX_m(secp256k1_geX *r, secp256k1_ge_storageX a) {
    secp256k1_fe_from_storageX_m(&r->x, a.x);
    secp256k1_fe_from_storageX_m(&r->y, a.y);
    /*printf("C %ul %ul %ul %ul \n", r->x.n[0], r->y.n[0], a->x.n[0], a->y.n[0]);*/
    r->infinity = 0;
}

void secp256k1_fe_from_storageX(secp256k1_feX *r, const secp256k1_fe_storageX *a) {
    r->n[0] = a->n[0] & 0xFFFFFFFFFFFFFULL;
    r->n[1] = a->n[0] >> 52 | ((a->n[1] << 12) & 0xFFFFFFFFFFFFFULL);
    r->n[2] = a->n[1] >> 40 | ((a->n[2] << 24) & 0xFFFFFFFFFFFFFULL);
    r->n[3] = a->n[2] >> 28 | ((a->n[3] << 36) & 0xFFFFFFFFFFFFFULL);
    r->n[4] = a->n[3] >> 16;
    r->magnitude = 1;
    r->normalized = 1;
}

static void secp256k1_ge_from_storageX(secp256k1_geX *r, const secp256k1_ge_storageX *a) {
    /*printf("ge C %ul %ul \n", a->x.n[0], a->y.n[0]);*/
    secp256k1_fe_from_storageX(&r->x, &a->x);
    secp256k1_fe_from_storageX(&r->y, &a->y);
    /*printf("C %ul %ul %ul %ul \n", r->x.n[0], r->y.n[0], a->x.n[0], a->y.n[0]);*/
    r->infinity = 0;
}

static void secp256k1_ecmultX(const secp256k1_ecmult_contextX *ctx, secp256k1_gejX *r, const secp256k1_gejX *a,
                              const secp256k1_scalarX *na, const secp256k1_scalarX *ng) {
    secp256k1_gejX prej[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_feX zr[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_geX pre_a[ECMULT_TABLE_SIZE(WINDOW_A)];
    struct secp256k1_strauss_point_stateX ps[1];
    struct secp256k1_strauss_stateX state;

    state.prej = prej;
    state.zr = zr;
    state.pre_a = pre_a;
    state.ps = ps;
    secp256k1_ecmult_strauss_wnafX(ctx, &state, r, 1, a, na, ng);
}

static int secp256k1_gej_is_infinityX(const secp256k1_gejX *a) {
    return a->infinity;
}

static void secp256k1_fe_normalize_weakX(secp256k1_feX *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    /* Reduce t4 at the start so there will be at most a single carry from the first pass */
    uint64_t x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52);
    t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52);
    t1 &= 0xFFFFFFFFFFFFFULL;
    t3 += (t2 >> 52);
    t2 &= 0xFFFFFFFFFFFFFULL;
    t4 += (t3 >> 52);
    t3 &= 0xFFFFFFFFFFFFFULL;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */

    r->n[0] = t0;
    r->n[1] = t1;
    r->n[2] = t2;
    r->n[3] = t3;
    r->n[4] = t4;

#ifdef VERIFY
    r->magnitude = 1;
    secp256k1_fe_verifyX(r);
#endif
}

void secp256k1_fe_mul_intX(secp256k1_feX *r, int a) {
    r->n[0] *= a;
    r->n[1] *= a;
    r->n[2] *= a;
    r->n[3] *= a;
    r->n[4] *= a;
    r->magnitude *= a;
    r->normalized = 0;
    secp256k1_fe_verifyX(r);
}

void secp256k1_fe_mul_innerX_z(uint64_t *r, const uint64_t *a, const uint64_t *SECP256K1_RESTRICT b) {
    uint64_t c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;


    /*  [... a b c] is a shorthand for ... + a<<104 + b<<52 + c<<0 mod n.
     *  for 0 <= x <= 4, px is a shorthand for sum(a[i]*b[x-i], i=0..x).
     *  for 4 <= x <= 8, px is a shorthand for sum(a[i]*b[x-i], i=(x-4)..4)
     *  Note that [x 0 0 0 0 0] = [x*R].
     */

    d = (uint128_t) a0 * b[3]
        + (uint128_t) a1 * b[2]
        + (uint128_t) a2 * b[1]
        + (uint128_t) a3 * b[0];
    /* [d 0 0 0] = [p3 0 0 0] */
    c = (uint128_t) a4 * b[4];
    /* [c 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    d += (c & M) * R;
    c >>= 52;
    /* [c 0 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    t3 = d & M;
    d >>= 52;
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */

    d += (uint128_t) a0 * b[4]
         + (uint128_t) a1 * b[3]
         + (uint128_t) a2 * b[2]
         + (uint128_t) a3 * b[1]
         + (uint128_t) a4 * b[0];
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    d += c * R;
    /* [d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    t4 = d & M;
    d >>= 52;
    /* [d t4 t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    tx = (t4 >> 48);
    t4 &= (M >> 4);
    /* [d t4+(tx<<48) t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */

    c = (uint128_t) a0 * b[0];
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 0 p4 p3 0 0 p0] */
    d += (uint128_t) a1 * b[4]
         + (uint128_t) a2 * b[3]
         + (uint128_t) a3 * b[2]
         + (uint128_t) a4 * b[1];
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = d & M;
    d >>= 52;
    /* [d u0 t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    /* [d 0 t4+(tx<<48)+(u0<<52) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = (u0 << 4) | tx;
    /* [d 0 t4+(u0<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    c += (uint128_t) u0 * (R >> 4);
    /* [d 0 t4 t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    r[0] = c & M;
    c >>= 52;
    /*printf("\n??? C %lu", c);
    printf("\n??? M %lu", M);
    printf("\n??? cM %lu", c & M);*/
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 0 p0] */

    c += (uint128_t) a0 * b[1]
         + (uint128_t) a1 * b[0];
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 p1 p0] */
    d += (uint128_t) a2 * b[4]
         + (uint128_t) a3 * b[3]
         + (uint128_t) a4 * b[2];
    /* [d 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    r[1] = c & M;
    c >>= 52;
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */

    c += (uint128_t) a0 * b[2]
         + (uint128_t) a1 * b[1]
         + (uint128_t) a2 * b[0];
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 p2 p1 p0] */
    d += (uint128_t) a3 * b[4]
         + (uint128_t) a4 * b[3];
    /* [d 0 0 t4 t3 c t1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */

    /* [d 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[2] = c & M;
    c >>= 52;
    /* [d 0 0 0 t4 t3+c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += d * R + t3;
    /* [t4 c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[3] = c & M;
    c >>= 52;
    /* [t4+c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += t4;
    /* [c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[4] = c;
    /* [r4 r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}


void secp256k1_fe_mul_innerX(uint64_t *r, const uint64_t *a, const uint64_t *SECP256K1_RESTRICT b) {
    uint128_t c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;


    /*  [... a b c] is a shorthand for ... + a<<104 + b<<52 + c<<0 mod n.
     *  for 0 <= x <= 4, px is a shorthand for sum(a[i]*b[x-i], i=0..x).
     *  for 4 <= x <= 8, px is a shorthand for sum(a[i]*b[x-i], i=(x-4)..4)
     *  Note that [x 0 0 0 0 0] = [x*R].
     */

    d = (uint128_t) a0 * b[3]
        + (uint128_t) a1 * b[2]
        + (uint128_t) a2 * b[1]
        + (uint128_t) a3 * b[0];
    /* [d 0 0 0] = [p3 0 0 0] */
    c = (uint128_t) a4 * b[4];
    /* [c 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    d += (c & M) * R;
    c >>= 52;
    /* [c 0 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    t3 = d & M;
    d >>= 52;
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */

    d += (uint128_t) a0 * b[4]
         + (uint128_t) a1 * b[3]
         + (uint128_t) a2 * b[2]
         + (uint128_t) a3 * b[1]
         + (uint128_t) a4 * b[0];
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    d += c * R;
    /* [d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    t4 = d & M;
    d >>= 52;
    /* [d t4 t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    tx = (t4 >> 48);
    t4 &= (M >> 4);
    /* [d t4+(tx<<48) t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */

    c = (uint128_t) a0 * b[0];
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 0 p4 p3 0 0 p0] */
    d += (uint128_t) a1 * b[4]
         + (uint128_t) a2 * b[3]
         + (uint128_t) a3 * b[2]
         + (uint128_t) a4 * b[1];
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = d & M;
    d >>= 52;
    /* [d u0 t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    /* [d 0 t4+(tx<<48)+(u0<<52) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = (u0 << 4) | tx;
    /* [d 0 t4+(u0<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    c += (uint128_t) u0 * (R >> 4);
    /* [d 0 t4 t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    r[0] = c & M;
    c >>= 52;
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 0 p0] */

    c += (uint128_t) a0 * b[1]
         + (uint128_t) a1 * b[0];
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 p1 p0] */
    d += (uint128_t) a2 * b[4]
         + (uint128_t) a3 * b[3]
         + (uint128_t) a4 * b[2];
    /* [d 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    r[1] = c & M;
    c >>= 52;
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */

    c += (uint128_t) a0 * b[2]
         + (uint128_t) a1 * b[1]
         + (uint128_t) a2 * b[0];
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 p2 p1 p0] */
    d += (uint128_t) a3 * b[4]
         + (uint128_t) a4 * b[3];
    /* [d 0 0 t4 t3 c t1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */

    /* [d 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[2] = c & M;
    c >>= 52;
    /* [d 0 0 0 t4 t3+c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += d * R + t3;
    /* [t4 c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[3] = c & M;
    c >>= 52;
    /* [t4+c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += t4;
    /* [c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[4] = c;
    /* [r4 r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}


static void secp256k1_fe_mulX(secp256k1_feX *r, const secp256k1_feX *a, const secp256k1_feX *SECP256K1_RESTRICT b) {

    secp256k1_fe_verifyX(a);
    secp256k1_fe_verifyX(b);
    secp256k1_fe_mul_innerX(r->n, a->n, b->n);
    r->magnitude = 1;
    r->normalized = 0;
    secp256k1_fe_verifyX(r);
}

void secp256k1_fe_sqr_innerX(uint64_t *r, const uint64_t *a) {
    uint128_t c, d;
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    int64_t t3, t4, tx, u0;
    const uint64_t M = 0xFFFFFFFFFFFFFULL, R = 0x1000003D10ULL;

    /**  [... a b c] is a shorthand for ... + a<<104 + b<<52 + c<<0 mod n.
     *  px is a shorthand for sum(a[i]*a[x-i], i=0..x).
     *  Note that [x 0 0 0 0 0] = [x*R].
     */

    d = (uint128_t) (a0 * 2) * a3
        + (uint128_t) (a1 * 2) * a2;
    /* [d 0 0 0] = [p3 0 0 0] */
    c = (uint128_t) a4 * a4;
    /* [c 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    d += (c & M) * R;
    c >>= 52;
    /* [c 0 0 0 0 0 d 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */
    t3 = d & M;
    d >>= 52;
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 0 p3 0 0 0] */

    a4 *= 2;
    d += (uint128_t) a0 * a4
         + (uint128_t) (a1 * 2) * a3
         + (uint128_t) a2 * a2;
    /* [c 0 0 0 0 d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    d += c * R;
    /* [d t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    t4 = d & M;
    d >>= 52;
    /* [d t4 t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */
    tx = (t4 >> 48);
    t4 &= (M >> 4);
    /* [d t4+(tx<<48) t3 0 0 0] = [p8 0 0 0 p4 p3 0 0 0] */

    c = (uint128_t) a0 * a0;
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 0 p4 p3 0 0 p0] */
    d += (uint128_t) a1 * a4
         + (uint128_t) (a2 * 2) * a3;
    /* [d t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = d & M;
    d >>= 52;
    /* [d u0 t4+(tx<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    /* [d 0 t4+(tx<<48)+(u0<<52) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    u0 = (u0 << 4) | tx;
    /* [d 0 t4+(u0<<48) t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    c += (uint128_t) u0 * (R >> 4);
    /* [d 0 t4 t3 0 0 c] = [p8 0 0 p5 p4 p3 0 0 p0] */
    r[0] = c & M;
    c >>= 52;
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 0 p0] */

    a0 *= 2;
    c += (uint128_t) a0 * a1;
    /* [d 0 t4 t3 0 c r0] = [p8 0 0 p5 p4 p3 0 p1 p0] */
    d += (uint128_t) a2 * a4
         + (uint128_t) a3 * a3;
    /* [d 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 t4 t3 0 c r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */
    r[1] = c & M;
    c >>= 52;
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 0 p1 p0] */

    c += (uint128_t) a0 * a2
         + (uint128_t) a1 * a1;
    /* [d 0 0 t4 t3 c r1 r0] = [p8 0 p6 p5 p4 p3 p2 p1 p0] */
    d += (uint128_t) a3 * a4;
    /* [d 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += (d & M) * R;
    d >>= 52;
    /* [d 0 0 0 t4 t3 c r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[2] = c & M;
    c >>= 52;
    /* [d 0 0 0 t4 t3+c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */

    c += d * R + t3;
    /* [t4 c r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[3] = c & M;
    c >>= 52;
    /* [t4+c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    c += t4;
    /* [c r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
    r[4] = c;
    /* [r4 r3 r2 r1 r0] = [p8 p7 p6 p5 p4 p3 p2 p1 p0] */
}

static void secp256k1_fe_sqrX(secp256k1_feX *r, const secp256k1_feX *a) {
    secp256k1_fe_verifyX(a);
    secp256k1_fe_sqr_innerX(r->n, a->n);
    r->magnitude = 1;
    r->normalized = 0;
    secp256k1_fe_verifyX(r);
}

void secp256k1_fe_negateX(secp256k1_feX *r, const secp256k1_feX *a, int m) {
    secp256k1_fe_verifyX(a);
    r->n[0] = 0xFFFFEFFFFFC2FULL * 2 * (m + 1) - a->n[0];
    r->n[1] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[1];
    r->n[2] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[2];
    r->n[3] = 0xFFFFFFFFFFFFFULL * 2 * (m + 1) - a->n[3];
    r->n[4] = 0x0FFFFFFFFFFFFULL * 2 * (m + 1) - a->n[4];
    r->magnitude = m + 1;
    r->normalized = 0;
    secp256k1_fe_verifyX(r);
}

void secp256k1_fe_addX(secp256k1_feX *r, const secp256k1_feX *a) {
    secp256k1_fe_verifyX(a);
    r->n[0] += a->n[0];
    r->n[1] += a->n[1];
    r->n[2] += a->n[2];
    r->n[3] += a->n[3];
    r->n[4] += a->n[4];
    r->magnitude += a->magnitude;
    r->normalized = 0;
    secp256k1_fe_verifyX(r);
}

static void secp256k1_gej_double_varX(secp256k1_gejX *r, const secp256k1_gejX *a, secp256k1_feX *rzr) {

    secp256k1_feX t1, t2, t3, t4;
    r->infinity = a->infinity;
    if (r->infinity) {
        if (rzr != NULL) {
            secp256k1_fe_set_intX(rzr, 1);
        }
        return;
    }

    if (rzr != NULL) {
        *rzr = a->y;
        secp256k1_fe_normalize_weakX(rzr);
        secp256k1_fe_mul_intX(rzr, 2);
    }

    secp256k1_fe_mulX(&r->z, &a->z, &a->y);
    secp256k1_fe_mul_intX(&r->z, 2);       /* Z' = 2*Y*Z (2) */
    secp256k1_fe_sqrX(&t1, &a->x);
    secp256k1_fe_mul_intX(&t1, 3);         /* T1 = 3*X^2 (3) */
    secp256k1_fe_sqrX(&t2, &t1);           /* T2 = 9*X^4 (1) */
    secp256k1_fe_sqrX(&t3, &a->y);
    secp256k1_fe_mul_intX(&t3, 2);         /* T3 = 2*Y^2 (2) */
    secp256k1_fe_sqrX(&t4, &t3);
    secp256k1_fe_mul_intX(&t4, 2);         /* T4 = 8*Y^4 (2) */
    secp256k1_fe_mulX(&t3, &t3, &a->x);    /* T3 = 2*X*Y^2 (1) */
    r->x = t3;
    secp256k1_fe_mul_intX(&r->x, 4);       /* X' = 8*X*Y^2 (4) */
    secp256k1_fe_negateX(&r->x, &r->x, 4); /* X' = -8*X*Y^2 (5) */
    secp256k1_fe_addX(&r->x, &t2);         /* X' = 9*X^4 - 8*X*Y^2 (6) */
    secp256k1_fe_negateX(&t2, &t2, 1);     /* T2 = -9*X^4 (2) */
    secp256k1_fe_mul_intX(&t3, 6);         /* T3 = 12*X*Y^2 (6) */
    secp256k1_fe_addX(&t3, &t2);           /* T3 = 12*X*Y^2 - 9*X^4 (8) */
    secp256k1_fe_mulX(&r->y, &t1, &t3);    /* Y' = 36*X^3*Y^2 - 27*X^6 (1) */
    secp256k1_fe_negateX(&t2, &t4, 2);     /* T2 = -8*Y^4 (3) */
    secp256k1_fe_addX(&r->y, &t2);         /* Y' = 36*X^3*Y^2 - 27*X^6 - 8*Y^4 (4) */
}

static void secp256k1_ge_set_gej_zinvX(secp256k1_geX *r, const secp256k1_gejX *a, const secp256k1_feX *zi) {
    secp256k1_feX zi2;
    secp256k1_feX zi3;
    secp256k1_fe_sqrX(&zi2, zi);
    secp256k1_fe_mulX(&zi3, &zi2, zi);
    secp256k1_fe_mulX(&r->x, &a->x, &zi2);
    secp256k1_fe_mulX(&r->y, &a->y, &zi3);
    r->infinity = a->infinity;
}

static int secp256k1_fe_normalizes_to_zero_varX(secp256k1_feX *r) {
    uint64_t t0, t1, t2, t3, t4;
    uint64_t z0, z1;
    uint64_t x;

    t0 = r->n[0];
    t4 = r->n[4];

    x = t4 >> 48;

    t0 += x * 0x1000003D1ULL;

    z0 = t0 & 0xFFFFFFFFFFFFFULL;
    z1 = z0 ^ 0x1000003D0ULL;

    if ((z0 != 0ULL) & (z1 != 0xFFFFFFFFFFFFFULL)) {
        return 0;
    }

    t1 = r->n[1];
    t2 = r->n[2];
    t3 = r->n[3];

    t4 &= 0x0FFFFFFFFFFFFULL;

    t1 += (t0 >> 52);
    t2 += (t1 >> 52);
    t1 &= 0xFFFFFFFFFFFFFULL;
    z0 |= t1;
    z1 &= t1;
    t3 += (t2 >> 52);
    t2 &= 0xFFFFFFFFFFFFFULL;
    z0 |= t2;
    z1 &= t2;
    t4 += (t3 >> 52);
    t3 &= 0xFFFFFFFFFFFFFULL;
    z0 |= t3;
    z1 &= t3;
    z0 |= t4;
    z1 &= t4 ^ 0xF000000000000ULL;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */

    return (z0 == 0) | (z1 == 0xFFFFFFFFFFFFFULL);
}

static void
secp256k1_gej_add_ge_varX(secp256k1_gejX *r, const secp256k1_gejX *a, const secp256k1_geX *b, secp256k1_feX *rzr) {
    /* 8 mul, 3 sqr, 4 normalize, 12 mul_int/add/negate */
    secp256k1_feX z12, u1, u2, s1, s2, h, i, i2, h2, h3, t;
    if (a->infinity) {
        secp256k1_gej_set_geX(r, b);
        return;
    }
    if (b->infinity) {
        if (rzr != NULL) {
            secp256k1_fe_set_intX(rzr, 1);
        }
        *r = *a;
        return;
    }
    r->infinity = 0;

    secp256k1_fe_sqrX(&z12, &a->z);
    u1 = a->x;
    secp256k1_fe_normalize_weakX(&u1);
    secp256k1_fe_mulX(&u2, &b->x, &z12);
    s1 = a->y;
    secp256k1_fe_normalize_weakX(&s1);
    secp256k1_fe_mulX(&s2, &b->y, &z12);
    secp256k1_fe_mulX(&s2, &s2, &a->z);
    secp256k1_fe_negateX(&h, &u1, 1);
    secp256k1_fe_addX(&h, &u2);
    secp256k1_fe_negateX(&i, &s1, 1);
    secp256k1_fe_addX(&i, &s2);
    if (secp256k1_fe_normalizes_to_zero_varX(&h)) {
        if (secp256k1_fe_normalizes_to_zero_varX(&i)) {
            secp256k1_gej_double_varX(r, a, rzr);
        } else {
            if (rzr != NULL) {
                secp256k1_fe_set_intX(rzr, 0);
            }
            r->infinity = 1;
        }
        return;
    }
    secp256k1_fe_sqrX(&i2, &i);
    secp256k1_fe_sqrX(&h2, &h);
    secp256k1_fe_mulX(&h3, &h, &h2);
    if (rzr != NULL) {
        *rzr = h;
    }
    secp256k1_fe_mulX(&r->z, &a->z, &h);
    secp256k1_fe_mulX(&t, &u1, &h2);
    r->x = t;
    secp256k1_fe_mul_intX(&r->x, 2);
    secp256k1_fe_addX(&r->x, &h3);
    secp256k1_fe_negateX(&r->x, &r->x, 3);
    secp256k1_fe_addX(&r->x, &i2);
    secp256k1_fe_negateX(&r->y, &r->x, 5);
    secp256k1_fe_addX(&r->y, &t);
    secp256k1_fe_mulX(&r->y, &r->y, &i);
    secp256k1_fe_mulX(&h3, &h3, &s1);
    secp256k1_fe_negateX(&h3, &h3, 1);
    secp256k1_fe_addX(&r->y, &h3);
}

static void
secp256k1_ecmult_odd_multiples_tableX(int n, secp256k1_gejX *prej, secp256k1_feX *zr, const secp256k1_gejX *a) {
    secp256k1_gejX d;
    secp256k1_geX a_ge, d_ge;
    int i;

    secp256k1_gej_double_varX(&d, a, NULL);

    d_ge.x = d.x;
    d_ge.y = d.y;
    d_ge.infinity = 0;

    secp256k1_ge_set_gej_zinvX(&a_ge, a, &d.z);
    prej[0].x = a_ge.x;
    prej[0].y = a_ge.y;
    prej[0].z = a->z;
    prej[0].infinity = 0;

    zr[0] = d.z;
    for (i = 1; i < n; i++) {
        secp256k1_gej_add_ge_varX(&prej[i], &prej[i - 1], &d_ge, &zr[i]);
    }

    /*
     * Each point in 'prej' has a z coordinate too small by a factor of 'd.z'. Only
     * the final point's z coordinate is actually used though, so just update that.
     */
    secp256k1_fe_mulX(&prej[n - 1].z, &prej[n - 1].z, &d.z);
}

static void secp256k1_fe_normalize_varX(secp256k1_feX *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];

    uint64_t m;
    uint64_t x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFULL;

    /* The first pass ensures the magnitude is 1, ... */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52);
    t0 &= 0xFFFFFFFFFFFFFULL;
    t2 += (t1 >> 52);
    t1 &= 0xFFFFFFFFFFFFFULL;
    m = t1;
    t3 += (t2 >> 52);
    t2 &= 0xFFFFFFFFFFFFFULL;
    m &= t2;
    t4 += (t3 >> 52);
    t3 &= 0xFFFFFFFFFFFFFULL;
    m &= t3;

    /* ... except for a possible carry at bit 48 of t4 (i.e. bit 256 of the field element) */

    /* At most a single final reduction is needed; check if the value is >= the field characteristic */
    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == 0xFFFFFFFFFFFFFULL)
                      & (t0 >= 0xFFFFEFFFFFC2FULL));

    if (x) {
        t0 += 0x1000003D1ULL;
        t1 += (t0 >> 52);
        t0 &= 0xFFFFFFFFFFFFFULL;
        t2 += (t1 >> 52);
        t1 &= 0xFFFFFFFFFFFFFULL;
        t3 += (t2 >> 52);
        t2 &= 0xFFFFFFFFFFFFFULL;
        t4 += (t3 >> 52);
        t3 &= 0xFFFFFFFFFFFFFULL;

        /* If t4 didn't carry to bit 48 already, then it should have after any final reduction */

        /* Mask off the possible multiple of 2^256 from the final reduction */
        t4 &= 0x0FFFFFFFFFFFFULL;
    }

    r->n[0] = t0;
    r->n[1] = t1;
    r->n[2] = t2;
    r->n[3] = t3;
    r->n[4] = t4;

    r->magnitude = 1;
    r->normalized = 1;
    secp256k1_fe_verifyX(r);
}

static void secp256k1_gej_rescaleX(secp256k1_gejX *r, const secp256k1_feX *s) {
    /* Operations: 4 mul, 1 sqr */
    secp256k1_feX zz;
    secp256k1_fe_sqrX(&zz, s);
    secp256k1_fe_mulX(&r->x, &r->x, &zz);                /* r->x *= s^2 */
    secp256k1_fe_mulX(&r->y, &r->y, &zz);
    secp256k1_fe_mulX(&r->y, &r->y, s);                  /* r->y *= s^3 */
    secp256k1_fe_mulX(&r->z, &r->z, s);                  /* r->z *= s   */
}

static void
secp256k1_ge_globalz_set_table_gejX(size_t len, secp256k1_geX *r, secp256k1_feX *globalz, const secp256k1_gejX *a,
                                    const secp256k1_feX *zr) {
    size_t i = len - 1;
    secp256k1_feX zs;

    if (len > 0) {
        /* The z of the final point gives us the "global Z" for the table. */
        r[i].x = a[i].x;
        r[i].y = a[i].y;
        /* Ensure all y values are in weak normal form for fast negation of points */
        secp256k1_fe_normalize_weakX(&r[i].y);
        *globalz = a[i].z;
        r[i].infinity = 0;
        zs = zr[i];

        /* Work our way backwards, using the z-ratios to scale the x/y values. */
        while (i > 0) {
            if (i != len - 1) {
                secp256k1_fe_mulX(&zs, &zs, &zr[i]);
            }
            i--;
            secp256k1_ge_set_gej_zinvX(&r[i], &a[i], &zs);
        }
    }
}

static void secp256k1_gej_set_infinityX(secp256k1_gejX *r) {
    r->infinity = 1;
    secp256k1_fe_clearX(&r->x);
    secp256k1_fe_clearX(&r->y);
    secp256k1_fe_clearX(&r->z);
}

static void secp256k1_fe_mulX_z(secp256k1_feX *r, secp256k1_feX *a, const secp256k1_feX *SECP256K1_RESTRICT b) {

    secp256k1_fe_verifyX(a);
    secp256k1_fe_verifyX(b);
    uint64_t rn[5];
    secp256k1_fe_mul_innerX_z(r, a->n, b->n);
    /*printf("\nOOO %lu", rn[0]);*/
    r->magnitude = 1;
    r->normalized = 0;
}


static void secp256k1_gej_add_zinv_varX(secp256k1_gejX *r, const secp256k1_gejX *a, const secp256k1_geX *b,
                                        const secp256k1_feX *bzinv) {
    /* 9 mul, 3 sqr, 4 normalize, 12 mul_int/add/negate */
    secp256k1_feX az, z12, u1, u2, s1, s2, h, i, i2, h2, h3, t;
    if (b->infinity) {
        *r = *a;
        return;
    }


    if (a->infinity) {
        secp256k1_feX bzinv2, bzinv3;
        r->infinity = b->infinity;
        secp256k1_fe_sqrX(&bzinv2, bzinv);
        secp256k1_fe_mulX(&bzinv3, &bzinv2, bzinv);
        secp256k1_fe_mulX(&r->x, &b->x, &bzinv2);
        secp256k1_fe_mulX(&r->y, &b->y, &bzinv3);
        secp256k1_fe_set_intX(&r->z, 1);

        return;
    }
    r->infinity = 0;

    secp256k1_fe_mulX(&az, &a->z, bzinv);

    secp256k1_fe_sqrX(&z12, &az);
    u1 = a->x;
    secp256k1_fe_normalize_weakX(&u1);
    secp256k1_fe_mulX(&u2, &b->x, &z12);
    s1 = a->y;
    secp256k1_fe_normalize_weakX(&s1);
    secp256k1_fe_mulX(&s2, &b->y, &z12);
    secp256k1_fe_mulX(&s2, &s2, &az);
    secp256k1_fe_negateX(&h, &u1, 1);
    secp256k1_fe_addX(&h, &u2);
    secp256k1_fe_negateX(&i, &s1, 1);
    secp256k1_fe_addX(&i, &s2);
    if (secp256k1_fe_normalizes_to_zero_varX(&h)) {
        if (secp256k1_fe_normalizes_to_zero_varX(&i)) {
            secp256k1_gej_double_varX(r, a, NULL);
        } else {
            r->infinity = 1;
        }
        return;
    }
    secp256k1_fe_sqrX(&i2, &i);
    secp256k1_fe_sqrX(&h2, &h);
    secp256k1_fe_mulX(&h3, &h, &h2);
    r->z = a->z;
    secp256k1_fe_mulX(&r->z, &r->z, &h);
    secp256k1_fe_mulX(&t, &u1, &h2);
    r->x = t;
    secp256k1_fe_mul_intX(&r->x, 2);
    secp256k1_fe_addX(&r->x, &h3);
    secp256k1_fe_negateX(&r->x, &r->x, 3);
    secp256k1_fe_addX(&r->x, &i2);
    secp256k1_fe_negateX(&r->y, &r->x, 5);
    secp256k1_fe_addX(&r->y, &t);
    secp256k1_fe_mulX(&r->y, &r->y, &i);
    secp256k1_fe_mulX(&h3, &h3, &s1);
    secp256k1_fe_negateX(&h3, &h3, 1);
    secp256k1_fe_addX(&r->y, &h3);
}

void secp256k1_ecmult_strauss_wnafX(const secp256k1_ecmult_contextX *ctx, const struct secp256k1_strauss_stateX *state,
                                    secp256k1_gejX *r, int num, const secp256k1_gejX *a, const secp256k1_scalarX *na,
                                    const secp256k1_scalarX *ng) {
    secp256k1_geX tmpa;
    secp256k1_feX Z;
    int wnaf_ng[256];
    int bits_ng = 0;
    int i;
    int bits = 0;
    int np;
    int no = 0;

    for (np = 0; np < num; ++np) {
        if (secp256k1_scalar_is_zeroX(&na[np]) || secp256k1_gej_is_infinityX(&a[np])) {
            continue;
        }
        state->ps[no].input_pos = np;
        state->ps[no].bits_na = secp256k1_ecmult_wnafX(state->ps[no].wnaf_na, 256, &na[np], WINDOW_A);
        if (state->ps[no].bits_na > bits) {
            bits = state->ps[no].bits_na;
        }
        ++no;
    }

    if (no > 0) {
        secp256k1_ecmult_odd_multiples_tableX(ECMULT_TABLE_SIZE(WINDOW_A), state->prej, state->zr,
                                              &a[state->ps[0].input_pos]);
        for (np = 1; np < no; ++np) {
            secp256k1_gejX tmp = a[state->ps[np].input_pos];
            secp256k1_fe_normalize_varX(
                    &(state->prej[(np - 1) * ECMULT_TABLE_SIZE(WINDOW_A) + ECMULT_TABLE_SIZE(WINDOW_A) - 1].z));
            secp256k1_gej_rescaleX(&tmp,
                                   &(state->prej[(np - 1) * ECMULT_TABLE_SIZE(WINDOW_A) + ECMULT_TABLE_SIZE(WINDOW_A) -
                                                 1].z));
            secp256k1_ecmult_odd_multiples_tableX(ECMULT_TABLE_SIZE(WINDOW_A),
                                                  state->prej + np * ECMULT_TABLE_SIZE(WINDOW_A),
                                                  state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A), &tmp);
            secp256k1_fe_mulX(state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A),
                              state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A), &(a[state->ps[np].input_pos].z));
        }
        secp256k1_ge_globalz_set_table_gejX(ECMULT_TABLE_SIZE(WINDOW_A) * no, state->pre_a, &Z, state->prej, state->zr);
    } else {
        secp256k1_fe_set_intX(&Z, 1);
    }

    if (ng) {
        bits_ng = secp256k1_ecmult_wnafX(wnaf_ng, 256, ng, WINDOW_G);
        if (bits_ng > bits) {
            bits = bits_ng;
        }
    }

    secp256k1_gej_set_infinityX(r);

    for (i = bits - 1; i >= 0; i--) {
        int n;
        secp256k1_gej_double_varX(r, r, NULL);
        for (np = 0; np < no; ++np) {
            if (i < state->ps[np].bits_na && (n = state->ps[np].wnaf_na[i])) {
                ECMULT_TABLE_GET_GE(&tmpa, state->pre_a + np * ECMULT_TABLE_SIZE(WINDOW_A), n, WINDOW_A);
                secp256k1_gej_add_ge_varX(r, r, &tmpa, NULL);
            }
        }

        if (i < bits_ng && (n = wnaf_ng[i])) {
            /*printf("\ni %d ", i);
            printf("A C x %ul ", &(*ctx->pre_g)[i].x.n[0]);
            printf("A C y %ul ", &(*ctx->pre_g)[i].y.n[0]);*/
            ECMULT_TABLE_GET_GE_STORAGE(&tmpa, *ctx->pre_g, n, WINDOW_G);
            secp256k1_gej_add_zinv_varX(r, r, &tmpa, &Z);

        }
    }
    if (!r->infinity) {
        secp256k1_fe_mulX(&r->z, &r->z, &Z);
    }
}

static void secp256k1_scalar_get_b32X(unsigned char *bin, const secp256k1_scalarX *a) {
    bin[0] = a->d[3] >> 56;
    bin[1] = a->d[3] >> 48;
    bin[2] = a->d[3] >> 40;
    bin[3] = a->d[3] >> 32;
    bin[4] = a->d[3] >> 24;
    bin[5] = a->d[3] >> 16;
    bin[6] = a->d[3] >> 8;
    bin[7] = a->d[3];
    bin[8] = a->d[2] >> 56;
    bin[9] = a->d[2] >> 48;
    bin[10] = a->d[2] >> 40;
    bin[11] = a->d[2] >> 32;
    bin[12] = a->d[2] >> 24;
    bin[13] = a->d[2] >> 16;
    bin[14] = a->d[2] >> 8;
    bin[15] = a->d[2];
    bin[16] = a->d[1] >> 56;
    bin[17] = a->d[1] >> 48;
    bin[18] = a->d[1] >> 40;
    bin[19] = a->d[1] >> 32;
    bin[20] = a->d[1] >> 24;
    bin[21] = a->d[1] >> 16;
    bin[22] = a->d[1] >> 8;
    bin[23] = a->d[1];
    bin[24] = a->d[0] >> 56;
    bin[25] = a->d[0] >> 48;
    bin[26] = a->d[0] >> 40;
    bin[27] = a->d[0] >> 32;
    bin[28] = a->d[0] >> 24;
    bin[29] = a->d[0] >> 16;
    bin[30] = a->d[0] >> 8;
    bin[31] = a->d[0];
}

static int secp256k1_fe_set_b32X(secp256k1_feX *r, const unsigned char *a) {
    int i;
    r->n[0] = r->n[1] = r->n[2] = r->n[3] = r->n[4] = 0;
    for (i = 0; i < 32; i++) {
        int j;
        for (j = 0; j < 2; j++) {
            int limb = (8 * i + 4 * j) / 52;
            int shift = (8 * i + 4 * j) % 52;
            r->n[limb] |= (uint64_t)((a[31 - i] >> (4 * j)) & 0xF) << shift;
        }
    }
    if (r->n[4] == 0x0FFFFFFFFFFFFULL && (r->n[3] & r->n[2] & r->n[1]) == 0xFFFFFFFFFFFFFULL &&
        r->n[0] >= 0xFFFFEFFFFFC2FULL) {
        return 0;
    }
    r->magnitude = 1;
    r->normalized = 1;
    secp256k1_fe_verifyX(r);
    return 1;
}

static int secp256k1_gej_eq_x_varX(const secp256k1_feX *x, const secp256k1_gejX *a) {
    secp256k1_feX r, r2;
    secp256k1_fe_sqrX(&r, &a->z);
    secp256k1_fe_mulX(&r, &r, x);
    r2 = a->x;
    secp256k1_fe_normalize_weakX(&r2);
    return secp256k1_fe_equal_varX(&r, &r2);
}

int secp256k1_fe_equal_varX(const secp256k1_feX *a, const secp256k1_feX *b) {
    secp256k1_feX na;
    secp256k1_fe_negateX(&na, a, 1);
    secp256k1_fe_addX(&na, b);
    return secp256k1_fe_normalizes_to_zero_varX(&na);
}

static int secp256k1_fe_cmp_varX(const secp256k1_feX *a, const secp256k1_feX *b) {
    int i;
    secp256k1_fe_verifyX(a);
    secp256k1_fe_verifyX(b);
    for (i = 4; i >= 0; i--) {
        if (a->n[i] > b->n[i]) {
            return 1;
        }
        if (a->n[i] < b->n[i]) {
            return -1;
        }
    }
    return 0;
}

static void secp256k1_scalar_set_b32X(secp256k1_scalarX *r, const unsigned char *b32, int *overflow) {
    int over;
    r->d[0] = (uint64_t) b32[31] | (uint64_t) b32[30] << 8 | (uint64_t) b32[29] << 16 | (uint64_t) b32[28] << 24 |
              (uint64_t) b32[27] << 32 | (uint64_t) b32[26] << 40 | (uint64_t) b32[25] << 48 | (uint64_t) b32[24] << 56;
    r->d[1] = (uint64_t) b32[23] | (uint64_t) b32[22] << 8 | (uint64_t) b32[21] << 16 | (uint64_t) b32[20] << 24 |
              (uint64_t) b32[19] << 32 | (uint64_t) b32[18] << 40 | (uint64_t) b32[17] << 48 | (uint64_t) b32[16] << 56;
    r->d[2] = (uint64_t) b32[15] | (uint64_t) b32[14] << 8 | (uint64_t) b32[13] << 16 | (uint64_t) b32[12] << 24 |
              (uint64_t) b32[11] << 32 | (uint64_t) b32[10] << 40 | (uint64_t) b32[9] << 48 | (uint64_t) b32[8] << 56;
    r->d[3] = (uint64_t) b32[7] | (uint64_t) b32[6] << 8 | (uint64_t) b32[5] << 16 | (uint64_t) b32[4] << 24 |
              (uint64_t) b32[3] << 32 | (uint64_t) b32[2] << 40 | (uint64_t) b32[1] << 48 | (uint64_t) b32[0] << 56;
    over = secp256k1_scalar_reduceX(r, secp256k1_scalar_check_overflowX(r));
    if (overflow) {
        *overflow = over;
    }
}


void memcpyX(void *dest, const void *src, size_t len) {
    char *d = dest;
    const char *s = src;
    while (len--)
        *d++ = *s++;
}

static void secp256k1_ecdsa_signature_loadX(const secp256k1_contextX *ctx, secp256k1_scalarX *r, secp256k1_scalarX *s,
                                            secp256k1_ecdsa_signatureX *sig) {
    (void) ctx;
    if (sizeof(secp256k1_scalarX) == 32) {
        memcpyX(r, &sig->data[0], 32);
        memcpyX(s, &sig->data[32], 32);
    } else {
        secp256k1_scalar_set_b32X(r, &sig->data[0], NULL);
        secp256k1_scalar_set_b32X(s, &sig->data[32], NULL);
    }
}

static void secp256k1_ge_set_xyX(secp256k1_geX *r, const secp256k1_feX *x, const secp256k1_feX *y) {
    r->infinity = 0;
    r->x = *x;
    r->y = *y;
}

static int secp256k1_pubkey_loadX(const secp256k1_contextX *ctx, secp256k1_geX *ge, const secp256k1_pubkeyX *pubkey) {
    if (sizeof(secp256k1_ge_storageX) == 64) {
        /* When the secp256k1_ge_storageX type is exactly 64 byte, use its
         * representation inside secp256k1_pubkeyX, as conversion is very fast.
         * Note that secp256k1_pubkey_save must use the same representation. */
        secp256k1_ge_storageX s;
        memcpy(&s, &pubkey->data[0], 64);
        secp256k1_ge_from_storageX(ge, &s);
    } else {
        /* Otherwise, fall back to 32-byte big endian for X and Y. */
        secp256k1_feX x, y;
        secp256k1_fe_set_b32X(&x, pubkey->data);
        secp256k1_fe_set_b32X(&y, pubkey->data + 32);
        secp256k1_ge_set_xyX(ge, &x, &y);
    }
    return 1;
}

static int secp256k1_ecdsa_sig_verifyX(const secp256k1_ecmult_contextX *ctx, const secp256k1_scalarX *sigr,
                                       const secp256k1_scalarX *sigs, const secp256k1_geX *pubkey,
                                       const secp256k1_scalarX *message) {
    unsigned char c[32];
    secp256k1_scalarX sn, u1, u2;
    secp256k1_feX xr;
    secp256k1_gejX pubkeyj;
    secp256k1_gejX pr;

    if (secp256k1_scalar_is_zeroX(sigr) || secp256k1_scalar_is_zeroX(sigs)) {
        return 0;
    }
    secp256k1_scalar_inverse_varX(&sn, sigs);
    secp256k1_scalar_mulX(&u1, &sn, message);
    secp256k1_scalar_mulX(&u2, &sn, sigr);
    secp256k1_gej_set_geX(&pubkeyj, pubkey);
    secp256k1_ecmultX(ctx, &pr, &pubkeyj, &u2, &u1);
    if (secp256k1_gej_is_infinityX(&pr)) {
        return 0;
    }
    secp256k1_scalar_get_b32X(c, sigr);
    secp256k1_fe_set_b32X(&xr, c);
    if (secp256k1_gej_eq_x_varX(&xr, &pr)) {
        return 1;
    }
    if (secp256k1_fe_cmp_varX(&xr, &secp256k1_ecdsa_const_p_minus_orderX) >= 0) {
        return 0;
    }
    secp256k1_fe_addX(&xr, &secp256k1_ecdsa_const_order_as_feX);
    if (secp256k1_gej_eq_x_varX(&xr, &pr)) {
        return 1;
    }
    return 0;
}

int secp256k1_ecdsa_verifyX(const secp256k1_contextX *ctx, const secp256k1_ecdsa_signatureX *sig,
                            const unsigned char *msg32, const secp256k1_pubkeyX *pubkey) {

    secp256k1_geX q;
    secp256k1_gejX pubj;
    secp256k1_scalarX r1, s1;
    secp256k1_scalarX m;
    secp256k1_scalar_set_b32(&m, msg32, NULL);
    secp256k1_ecdsa_signature_load(ctx, &r1, &s1, sig);
    return (secp256k1_pubkey_load(ctx, &q, pubkey) && secp256k1_ecdsa_sig_verifyX(&ctx->ecmult_ctx, &r1, &s1, &q, &m));
}

int
secp256k1_ecdsa_verify_gpu(const secp256k1_contextX *ctx, secp256k1_ecdsa_signatureX *sig,
                           const secp256k1_msgX *msg32,
                           const secp256k1_pubkeyX *pubkey,int sigsNum) {

    const int LIST_SIZE = sigsNum;
    FILE *fp;
    char *source_str;
    size_t source_size;
    fp = fopen("src/kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
                         &device_id, &ret_num_devices);

    secp256k1_ge_storageX *arr = (secp256k1_ge_storageX *) malloc(8192 * sizeof(secp256k1_ge_storageX));
    int idx, nnn;
    for (idx = 0; idx < 8192; idx++) {
        for (nnn = 0; nnn < 4; nnn++) {
            arr[idx].x.n[nnn] = (*ctx->ecmult_ctx.pre_g)[idx].x.n[nnn];
            arr[idx].y.n[nnn] = (*ctx->ecmult_ctx.pre_g)[idx].y.n[nnn];
        }
    }


    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      8192 * sizeof(secp256k1_ge_storageX), NULL, &ret);

    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * sizeof(secp256k1_ecdsa_signatureX), NULL, &ret);

    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * 32 * sizeof(char), NULL, &ret);

    cl_mem d_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * sizeof(secp256k1_pubkeyX), NULL, &ret);
    cl_mem e_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      LIST_SIZE * sizeof(int), NULL, &ret);

    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                               8192 * sizeof(secp256k1_ge_storageX), arr, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(secp256k1_ecdsa_signatureX), sig, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(secp256k1_msgX), msg32, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, d_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(secp256k1_pubkeyX), pubkey, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, (const size_t *) &source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "secp256k1_ecdsa_verifyX", &ret);
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_mem_obj);
    ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *) &e_mem_obj);

    size_t global_item_size = LIST_SIZE;
    size_t local_item_size = 1;
    if (LIST_SIZE > 100) {
        local_item_size = 40;
    }
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
                                 &global_item_size, &local_item_size, 0, NULL, NULL);

    int *res = (int *) malloc(LIST_SIZE * sizeof(int));
    ret = clEnqueueReadBuffer(command_queue, e_mem_obj, CL_TRUE, 0,
                              LIST_SIZE * sizeof(int), res, 0, NULL, NULL);


    size_t log_size;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("\%s\n", log);

    /*
    secp256k1_geX q;
    secp256k1_gejX pubj;
    secp256k1_scalarX r1, s1;
    secp256k1_scalarX m;
    secp256k1_scalar_set_b32(&m, msg32, NULL);
    secp256k1_ecdsa_signature_load(ctx, &r1, &s1, sig);
	 int tmp = (secp256k1_pubkey_load(ctx, &q, pubkey) && secp256k1_ecdsa_sig_verifyX(&ctx->ecmult_ctx, &r1, &s1, &q, &m));

	 int k;
	 printf("\n# CPU GPU\n");
	 for(k=0;k<LIST_SIZE;k++)
	 {
	 	printf("\n%d ", k+1);
	 	printf(" %d   %d", tmp, res[k]);
	 }
	 printf("\n# CPU GPU\n");
	 printf("--------------\n");
	 */
    char c;
    /*scanf("%c", &c);*/
    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    return 1;
}

#define MAX_LINE_LENGTH 1024

int hexToByteArray(char* hexString, unsigned char* byteArray) {
    int length = strlen(hexString);

    if (length % 2 != 0) { // 确保输入的十六进制字符串长度为偶数
        printf("Invalid input.\n");
        return 0;
    }

    for (int i = 0; i < length / 2; ++i) {
        sscanf(&hexString[i * 2], "%2hhx", &byteArray[i]);
    }
    return 1;
}

double readECDSADataFromCSV(const char *filename, secp256k1_ecdsa_signature *signatures, secp256k1_pubkey *public_keys, secp256k1_msgX *messages, int maxRecords, int *dataCount) {
    double time_taken = omp_get_wtime();

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        log_error("无法打开文件:%s",filename);
        return omp_get_wtime() - time_taken;
    }

    char line[MAX_LINE_LENGTH];
    int count = 0;

    const size_t siglen=72;
    unsigned char sig[siglen];

    const size_t pubkeyclen=65;
    unsigned char pubkeyc[pubkeyclen];

    while (count < maxRecords && fgets(line, MAX_LINE_LENGTH, file)) {
        line[strcspn(line, "\n")] = 0;

        char *signatureStr = strtok(line, ",");
        char *publicKeyStr = strtok(NULL, ",");
        char *messageStr = strtok(NULL, ",");

        if (signatureStr && publicKeyStr && messageStr) {
            memset(sig,0,sizeof(unsigned char)*siglen);
            CHECK(hexToByteArray(signatureStr,sig) == 1);
            CHECK(secp256k1_ecdsa_signature_parse_der(ctx, &signatures[count], sig, strlen(signatureStr)/2) == 1);
            memset(pubkeyc,0,sizeof(unsigned char)*pubkeyclen);
            CHECK(hexToByteArray(publicKeyStr,pubkeyc) == 1);
            CHECK(secp256k1_ec_pubkey_parse(ctx, &public_keys[count], pubkeyc,strlen(publicKeyStr)/2) == 1);
            CHECK(hexToByteArray(messageStr,messages[count]) == 1);
            count++;
        }
    }

    fclose(file);
    *dataCount = count;

    return omp_get_wtime() - time_taken;
}

typedef struct {
   secp256k1_ecdsa_signature *signatures;
   secp256k1_pubkey *public_keys;
   secp256k1_msgX *messages;
   int startIdx;
   int endIdx;
   int taskNum;
   int nodeID;
   int nodeNum;
   int usempi;
   int dataCount;
   int maxthreads;
   float taskratio;
}verifyData;

int verifyBitcoinSigsByCPU(verifyData *vdata) {
    int failed = 0;
    double time_taken = omp_get_wtime();
    for (int i = vdata->startIdx; i <=vdata->endIdx; i++) {
        int verify_result = secp256k1_ecdsa_verify(ctx, &vdata->signatures[i], vdata->messages[i], &vdata->public_keys[i]);
        if (verify_result != 1) {
            failed++;
        }
    }

    log_trace("%d/%d failed", failed, vdata->taskNum);

    int MAXLEN =32;
    char mess[MAXLEN];

    if (vdata->usempi == 1 && vdata->nodeNum>1) {
        if (vdata->nodeID==0) {
            MPI_Status status;
            for (int i = 1; i < vdata->nodeNum; i++)
            {
                memset(mess,0,MAXLEN);
                MPI_Recv(mess, MAXLEN, MPI_CHAR,i, 0, MPI_COMM_WORLD, &status);
                log_trace("Received cost cpu(%s) from node(%d)",mess,i);
            }
            time_taken = omp_get_wtime() - time_taken;
            log_info("[CPU] %d verify took %f seconds to execute (MPI:%d)", vdata->dataCount, time_taken,vdata->nodeNum);
        }else{
            time_taken = omp_get_wtime() - time_taken;
            memset(mess,0,MAXLEN);
            sprintf(mess, "%f",time_taken);
            int ret = MPI_Send(mess, strlen(mess), MPI_CHAR,0, 0,MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("[CPU] Failed to return task result");
            }
        }
    }else{
        time_taken = omp_get_wtime() - time_taken;
        log_info("[CPU] %d verify took %f seconds to execute", vdata->taskNum, time_taken);
    }

    return 0;
}

int verifyBitcoinSigsByOMP(verifyData *vdata) {
    double time_taken;
    int MAXLEN =32;
    char mess[MAXLEN];
    int i;
    /* with OpenMP  */
    int step=vdata->taskNum/vdata->maxthreads;
    time_taken = omp_get_wtime();
#pragma omp parallel num_threads(vdata->maxthreads) firstprivate(step,vdata)
    {
        int id = omp_get_thread_num();
        int sIdx=id*step+vdata->startIdx;
        int eIdx=vdata->startIdx;
        if (id==vdata->maxthreads-1) {
            eIdx=eIdx+vdata->taskNum-1;
        }else{
            eIdx=eIdx+(id+1)*step-1;
        }
        for (int i = sIdx; i <=eIdx; i++) {
            secp256k1_ecdsa_verify(ctx, &vdata->signatures[i], vdata->messages[i], &vdata->public_keys[i]);
        }
    }

    if (vdata->usempi == 1 && vdata->nodeNum>1) {
        if (vdata->nodeID==0) {
            MPI_Status status;
            for (int i = 1; i < vdata->nodeNum; i++)
            {
                memset(mess,0,MAXLEN);
                MPI_Recv(mess, MAXLEN, MPI_CHAR,i, 1, MPI_COMM_WORLD, &status);
                log_trace("Received cost omp(%s) from node(%d)",mess,i);
            }
            time_taken = omp_get_wtime() - time_taken;
            log_info("[OpenMP] %d verify took %f seconds to execute (MPI:%d)",vdata->dataCount, time_taken,vdata->nodeNum);
        }else{
            time_taken = omp_get_wtime() - time_taken;
            memset(mess,0,MAXLEN);
            sprintf(mess, "%f",time_taken);
            int ret = MPI_Send(mess, strlen(mess), MPI_CHAR,0, 1,MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("[OpenMP] Failed to return task result");
            }
        }
    }else{
        time_taken = omp_get_wtime() - time_taken;
        log_info("[OpenMP] %d verify took %f seconds to execute (cores:%d)",vdata->taskNum, time_taken,vdata->maxthreads);
    }

    return 0;
}

int verifyBitcoinSigsByGPU(verifyData *vdata) {
    double time_taken;
    int MAXLEN =32;
    char mess[MAXLEN];
    int i;
    // GPU
    if (vdata->usempi == 1 && vdata->nodeNum>1)
    {
        secp256k1_ecdsa_signature *sigs = malloc(vdata->taskNum * sizeof(secp256k1_ecdsa_signature));
        secp256k1_pubkey *pubks = malloc(vdata->taskNum * sizeof(secp256k1_pubkey));
        secp256k1_msgX *msgs = malloc(vdata->taskNum * sizeof(secp256k1_msgX));

        if (!sigs || !pubks || !msgs) {
            log_error("Memory allocation failed.");
            return -1;
        }
        int idx =0;
        for (i = vdata->startIdx; i <=vdata->endIdx; i++) {
            sigs[idx] = vdata->signatures[i];
            pubks[idx] = vdata->public_keys[i]; 
        
            for (int ii = 0; ii < sizeof(vdata->messages[i]); ii++)
            {
                msgs[idx][ii] = vdata->messages[i][ii];
            }
            
            idx++;
        }
        time_taken = omp_get_wtime();
        secp256k1_ecdsa_verify_gpu(ctx, sigs, msgs, pubks,vdata->taskNum);

        if (vdata->nodeID==0) {
            MPI_Status status;
            for (int i = 1; i < vdata->nodeNum; i++)
            {
                memset(mess,0,MAXLEN);
                MPI_Recv(mess, MAXLEN, MPI_CHAR,i, 2, MPI_COMM_WORLD, &status);
                log_trace("Received cost gpu(%s) from node(%d)",mess,i);
            }
            time_taken = omp_get_wtime() - time_taken;
            log_info("[GPU] %d verify took %f seconds to execute (MPI:%d)", vdata->dataCount, time_taken,vdata->nodeNum);
        }else{
            time_taken = omp_get_wtime() - time_taken;
            memset(mess,0,MAXLEN);
            sprintf(mess, "%f",time_taken);
            int ret = MPI_Send(mess, strlen(mess), MPI_CHAR,0, 2,MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("[GPU] Failed to return task result");
            }
        }
        free(sigs);
        free(pubks);
        free(msgs);
    }else{
        time_taken = omp_get_wtime();
        secp256k1_ecdsa_verify_gpu(ctx, vdata->signatures, vdata->messages, vdata->public_keys,vdata->taskNum);
        time_taken = omp_get_wtime() - time_taken;
        log_info("[GPU] %d verify took %f seconds to execute ", vdata->taskNum, time_taken);
    }
}

void* verifyBitcoinSigsByOMPThread(void *arg) {
    verifyData *vdata=(verifyData*)arg;
    int MAXLEN =32;
    char mess[MAXLEN];
    int gtaskNum = (int)(vdata->taskratio*(float)vdata->taskNum);
    int otaskNum = vdata->taskNum-gtaskNum;
    if (gtaskNum <= 0 || otaskNum <= 0)
    {
        log_error("Task assignment error");
        return NULL;
    }
    log_info("[OpenMP+GPU]OMPThread:%d.",otaskNum);

    /* with OpenMP  */
    int step=otaskNum/vdata->maxthreads;
#pragma omp parallel num_threads(vdata->maxthreads) firstprivate(step,otaskNum,vdata)
    {
        int id = omp_get_thread_num();
        int sIdx=id*step+vdata->startIdx;
        int eIdx=vdata->startIdx;
        if (id==vdata->maxthreads-1) {
            eIdx=eIdx+otaskNum-1;
        }else{
            eIdx=eIdx+(id+1)*step-1;
        }
        for (int i = sIdx; i <=eIdx; i++) {
            secp256k1_ecdsa_verify(ctx, &vdata->signatures[i], vdata->messages[i], &vdata->public_keys[i]);
        }
    }
    return NULL;
}

void* verifyBitcoinSigsByGPUThread(void *arg) {
    log_info("[OpenMP+GPU]GPUThread.");
    verifyData *vdata=(verifyData*)arg;
    secp256k1_ecdsa_verify_gpu(ctx, vdata->signatures, vdata->messages, vdata->public_keys,vdata->taskNum);
    return NULL;
}

double verifyBitcoinSigsByOMPGPU(verifyData *vdata) {
    double time_taken;
    int MAXLEN =32;
    char mess[MAXLEN];
    int gtaskNum = (int)(vdata->taskratio*(float)vdata->taskNum);
    int otaskNum = vdata->taskNum-gtaskNum;
    if (gtaskNum <= 0 || otaskNum <= 0)
    {
        log_error("Task assignment error");
       return 0;
    }
    log_info("[OpenMP+GPU]OpenMP tasks to distribute: %d ; GPU tasks to distribute: %d.",otaskNum,gtaskNum);
    // gpu 分配
    secp256k1_ecdsa_signature *sigs = malloc(gtaskNum * sizeof(secp256k1_ecdsa_signature));
    secp256k1_pubkey *pubks = malloc(gtaskNum * sizeof(secp256k1_pubkey));
    secp256k1_msgX *msgs = malloc(gtaskNum * sizeof(secp256k1_msgX));

    if (!sigs || !pubks || !msgs) {
        log_error("Memory allocation failed.");
        return 0;
    }
    int idx =0;
    for (int i = vdata->startIdx+otaskNum; i <=vdata->endIdx; i++) {
        sigs[idx] = vdata->signatures[i];
        pubks[idx] = vdata->public_keys[i]; 
    
        for (int ii = 0; ii < sizeof(vdata->messages[i]); ii++)
        {
            msgs[idx][ii] = vdata->messages[i][ii];
        }
        idx++;
    }
    time_taken = omp_get_wtime();
    int res;
    pthread_t ompthread, gputhread;
    /* with OpenMP  */
    res = pthread_create(&ompthread, NULL, verifyBitcoinSigsByOMPThread,(void*)vdata);
    if (res != 0) {
        log_error("[OpenMP+GPU] OMP Create thread fail");
        free(sigs);
        free(pubks);
        free(msgs);
        return 0;
    }
    verifyData gvdata;
    gvdata.signatures=sigs;
    gvdata.public_keys=pubks;
    gvdata.messages=msgs;
    gvdata.taskNum=gtaskNum;
    res = pthread_create(&gputhread, NULL, verifyBitcoinSigsByGPUThread,(void*)&gvdata);
    if (res != 0) {
        log_error("[OpenMP+GPU] GPU Create thread fail");
        return 0;
    }

    res = pthread_join(ompthread, NULL);
    if (res != 0) {
        log_error("[OpenMP+GPU] OMP Thread failure");
        return 0;
    }
    res = pthread_join(gputhread, NULL);
    if (res != 0) {
        log_error("[OpenMP+GPU] GPU Thread failure");
        return 0;
    }

    if (vdata->usempi == 1 && vdata->nodeNum>1) {
        if (vdata->nodeID==0) {
            MPI_Status status;
            for (int i = 1; i < vdata->nodeNum; i++)
            {
                memset(mess,0,MAXLEN);
                MPI_Recv(mess, MAXLEN, MPI_CHAR,i, 1, MPI_COMM_WORLD, &status);
                log_trace("Received cost ompgpu(%s) from node(%d)",mess,i);
            }
            time_taken = omp_get_wtime() - time_taken;
            log_info("[OpenMP+GPU] %d verify took %f seconds to execute (MPI:%d)",vdata->dataCount, time_taken,vdata->nodeNum);
        }else{
            time_taken = omp_get_wtime() - time_taken;
            memset(mess,0,MAXLEN);
            sprintf(mess, "%f",time_taken);
            int ret = MPI_Send(mess, strlen(mess), MPI_CHAR,0, 1,MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("[OpenMP+GPU] Failed to return task result");
            }
        }
    }else{
        time_taken = omp_get_wtime() - time_taken;
        log_info("[OpenMP+GPU] %d verify took %f seconds to execute (cores:%d)",vdata->taskNum, time_taken,vdata->maxthreads);
    }

    free(sigs);
    free(pubks);
    free(msgs);
    return time_taken;
}

int verifyBitcoinSigs(const char *path,int maxsigs,int maxthreads,float taskratio,int mode) {
    const int maxRecords = maxsigs;
    secp256k1_ecdsa_signature *signatures = malloc(maxRecords * sizeof(secp256k1_ecdsa_signature));
    secp256k1_pubkey *public_keys = malloc(maxRecords * sizeof(secp256k1_pubkey));
    secp256k1_msgX *messages = malloc(maxRecords * sizeof(secp256k1_msgX));


    if (!signatures || !public_keys || !messages) {
        log_error("Memory allocation failed.");
        return -1;
    }
    int dataCount;

    double time = readECDSADataFromCSV(path, signatures, public_keys, messages, maxRecords, &dataCount);
    log_trace("%d bitcoin transaction signatures loaded,Total spending %f 秒", dataCount,time);

    int usempi = 0;
    if (mode == MODE_MPI_CPU ||
        mode == MODE_MPI_OPENMP || 
        mode == MODE_MPI_GPU ||
        mode == MODE_MPI_OPENMP_GPU) {
        usempi = 1;
    }

    int startIdx=0;
    int endIdx=dataCount-1;
    int taskNum=dataCount;
    int nodeID, nodeNum;
    if (usempi == 1) {
        MPI_Init(NULL,NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);
        MPI_Comm_size(MPI_COMM_WORLD, &nodeNum);

        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == -1) {
            log_error("Error getting the host name");
            return -1;
        }
        log_info("%s:Start MPI id=%i of %i",hostname, nodeID, nodeNum);
        int step=dataCount/nodeNum;
        startIdx=nodeID*step;
        if (nodeID==nodeNum-1)
        {
            endIdx=dataCount-1;
        }else{
            endIdx=(nodeID+1)*step-1;
        }
        taskNum=endIdx-startIdx+1;
    }
    verifyData vdata;
    vdata.signatures=signatures;
    vdata.public_keys=public_keys;
    vdata.messages=messages;
    vdata.startIdx=startIdx;
    vdata.endIdx=endIdx;
    vdata.taskNum=taskNum;
    vdata.nodeID=nodeID;
    vdata.nodeNum=nodeNum;
    vdata.usempi=usempi; 
    vdata.dataCount=dataCount;
    vdata.maxthreads=maxthreads;
    vdata.taskratio=taskratio;

    if (mode == MODE_ALL ||
        mode == MODE_CPU ||
        mode == MODE_MPI_CPU)
    {
        verifyBitcoinSigsByCPU(&vdata);
    }
    if (mode == MODE_ALL ||
        mode == MODE_OPENMP ||
        mode == MODE_MPI_OPENMP)
    {
        verifyBitcoinSigsByOMP(&vdata);
    }
    if (mode == MODE_ALL ||
        mode == MODE_GPU ||
        mode == MODE_MPI_GPU)
    {
        verifyBitcoinSigsByGPU(&vdata);
    }
    if (mode == MODE_ALL ||
        mode == MODE_OPENMP_GPU ||
        mode == MODE_MPI_OPENMP_GPU)
    {
        verifyBitcoinSigsByOMPGPU(&vdata);
    }

    if (usempi == 1) {
        MPI_Finalize();
        log_info("End MPI id=%i of %i",nodeID, nodeNum);
    }
    free(signatures);
    free(public_keys);
    free(messages);
    return 0;
}

#define GET_MSG_TYPE 0
#define SET_MSG_TYPE 1

void* VBSDMasterThread(void *arg) {
    verifyData *vdata = (verifyData*)arg;
    int taskpackSize = vdata->taskNum;
    int taskpackNum = vdata->dataCount/taskpackSize;
    if (vdata->dataCount%taskpackSize != 0) {
        taskpackNum+=1;
    }

    verifyData *vdatas= malloc(taskpackNum*sizeof(verifyData));
    for (int i = 0; i < taskpackNum; ++i) {
        vdatas[i].startIdx=i*taskpackSize;
        if (i==taskpackNum-1)
        {
            vdatas[i].endIdx=vdata->dataCount-1;
        }else{
            vdatas[i].endIdx=(i+1)*taskpackSize-1;
        }
        vdatas[i].taskNum=vdatas[i].endIdx-vdatas[i].startIdx+1;
        vdatas[i].nodeID=-1;
    }
    log_info("Enter master thread: Task package distribution minimum unit=%d Total task package number=%d",taskpackSize,taskpackNum);

    int completed =0;

    int maxSetOptNum =1+4+4+(64+64+32)*taskpackSize+4;
    unsigned char *setOptMsg = malloc(maxSetOptNum*sizeof(unsigned char));

    int maxGetOptNum =4+4+4;
    unsigned char getOptMsg[maxGetOptNum];

    int position=0;
    int completedIdx=-1;
    double costTime=0;

    int ret;
    while (completed<taskpackNum) {
        MPI_Status status;
        position=0;
        completedIdx=-1;
        costTime=0;

        memset(getOptMsg,0,maxGetOptNum);
        ret = MPI_Recv(getOptMsg, maxGetOptNum, MPI_UNSIGNED_CHAR,MPI_ANY_SOURCE,GET_MSG_TYPE, MPI_COMM_WORLD, &status);
        if (ret != MPI_SUCCESS) {
            log_error("Failed to receive the message nodeID(%d)",vdata->nodeID);
            break;
        }
        MPI_Unpack(getOptMsg, maxGetOptNum, &position, &completedIdx, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(getOptMsg, maxGetOptNum, &position, &costTime, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        int nodeID = status.MPI_SOURCE;
        if (completedIdx != -1 && completedIdx < taskpackNum) {
            log_trace("Received request command from Node ID(%d), completed task number %d (%d signatures), took %f seconds.",nodeID,completedIdx,vdatas[completedIdx].taskNum,costTime);
            if (vdatas[completedIdx].nodeID == -1) {
                vdatas[completedIdx].nodeID=nodeID;
                completed++;
            }else{
                log_warn("You (ID:%d) take someone's (ID:%d) task which has been done",nodeID,vdatas[completedIdx].nodeID);
            }

        }else{
            log_trace("Received request instruction from node ID(%d), will start distributing tasks to it. (completedIdx=%d)",nodeID,completedIdx);
        }
        // 分配任务
        int willTaskIdx=-1;
        for (int i = 0; i < taskpackNum; ++i) {
            if (vdatas[i].nodeID == -1 && !vdatas[i].usempi) {
                willTaskIdx=i;
                break;
            }
        }
        position=0;
        unsigned char end=0;
        memset(setOptMsg,0,maxSetOptNum);

        if (willTaskIdx != -1) {
            log_trace("Master gonna assign the %d th task(contains %d signatures) to worker %d, current task finishing status: %d/%d",willTaskIdx,vdatas[willTaskIdx].taskNum,nodeID,completed,taskpackNum);

            MPI_Pack(&end, 1, MPI_UNSIGNED_CHAR,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
            MPI_Pack(&willTaskIdx, 1, MPI_INT,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
            MPI_Pack(&vdatas[willTaskIdx].taskNum, 1, MPI_INT,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
            for (int i = vdatas[willTaskIdx].startIdx; i <= vdatas[willTaskIdx].endIdx; ++i) {
                MPI_Pack(&vdata->signatures[i].data, 64,MPI_UNSIGNED_CHAR ,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
                MPI_Pack(&vdata->public_keys[i].data, 64,MPI_UNSIGNED_CHAR ,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
                MPI_Pack(&vdata->messages[i], 32,MPI_UNSIGNED_CHAR ,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
            }
            ret = MPI_Send (setOptMsg, position, MPI_UNSIGNED_CHAR, nodeID,SET_MSG_TYPE, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("Message sending failed nodeID(%d)",vdata->nodeID);
                break;
            }
            vdatas[willTaskIdx].usempi=true;
        }else{
            log_info("All the task is done, please quit (%d)",nodeID);
            end=1;
            MPI_Pack(&end, 1, MPI_UNSIGNED_CHAR,setOptMsg, maxSetOptNum, &position, MPI_COMM_WORLD);
            ret = MPI_Send (setOptMsg, position, MPI_UNSIGNED_CHAR, nodeID,SET_MSG_TYPE, MPI_COMM_WORLD);
            if (ret != MPI_SUCCESS) {
                log_error("Message sending failed nodeID(%d)",vdata->nodeID);
                break;
            }
        }
    }

    free(vdatas);
    free(setOptMsg);
    return NULL;
}

void* VBSDWorkerThread(void *arg) {
    verifyData *vdata = (verifyData *) arg;
    int taskpackSize = vdata->taskNum;

    log_info("[ID:%d]Enter worker thread: the minimum unit of receiving task packages=%d",vdata->nodeID,taskpackSize);

    int maxSetOptNum =1+4+4+(64+64+32)*taskpackSize+4;
    unsigned char *setOptMsg = malloc(maxSetOptNum*sizeof(unsigned char));

    int maxGetOptNum =4+4+4;
    unsigned char getOptMsg[maxGetOptNum];

    int position=0;
    int completedIdx=-1;
    double costTime=0;
    int ret;

    while (1) {
        position=0;
        memset(getOptMsg,0,maxGetOptNum);

        MPI_Pack(&completedIdx, 1, MPI_INT,getOptMsg, maxGetOptNum, &position, MPI_COMM_WORLD);
        MPI_Pack(&costTime, 1, MPI_DOUBLE,getOptMsg, maxGetOptNum, &position, MPI_COMM_WORLD);

        ret = MPI_Send (getOptMsg, position, MPI_UNSIGNED_CHAR, 0,GET_MSG_TYPE, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
            log_error("[ID:%d]Message sending failed",vdata->nodeID);
            break;
        }
        if (completedIdx != -1) {
            log_trace("[ID:%d]Send request instruction to Master, completed task number %d (signature %d), costing %f seconds.",vdata->nodeID,completedIdx,vdata->taskNum,costTime);
        }else{
            log_trace("[ID:%d]Instruction to start sending task requests to Master",vdata->nodeID);
        }
        // get
        MPI_Status status;
        position=0;
        completedIdx=-1;
        costTime=0;

        if (vdata->signatures || vdata->public_keys || vdata->messages) {
            free(vdata->signatures);
            free(vdata->public_keys);
            free(vdata->messages);
            vdata->signatures=NULL;
            vdata->public_keys=NULL;
            vdata->messages=NULL;
        }
        vdata->dataCount=0;

        memset(setOptMsg,0,maxSetOptNum);
        ret = MPI_Recv(setOptMsg, maxSetOptNum, MPI_UNSIGNED_CHAR,0,SET_MSG_TYPE, MPI_COMM_WORLD, &status);
        if (ret != MPI_SUCCESS) {
            log_error("[ID:%d]Failed to receive the message",vdata->nodeID);
            break;
        }
        unsigned char end=0;
        MPI_Unpack(setOptMsg, maxSetOptNum, &position, &end, 1, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
        if (end != 0) {
            log_info("[ID:%d]Exit task message queue",vdata->nodeID);
            break;
        }
        MPI_Unpack(setOptMsg, maxSetOptNum, &position, &completedIdx, 1, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(setOptMsg, maxSetOptNum, &position, &vdata->dataCount, 1, MPI_INT, MPI_COMM_WORLD);
        if (vdata->dataCount > 0) {
            vdata->signatures = malloc(vdata->dataCount * sizeof(secp256k1_ecdsa_signature));
            vdata->public_keys = malloc(vdata->dataCount * sizeof(secp256k1_pubkey));
            vdata->messages = malloc(vdata->dataCount * sizeof(secp256k1_msgX));

            if (!vdata->signatures || !vdata->public_keys || !vdata->messages) {
                log_info("[ID:%d]malloc",vdata->nodeID);
                break;
            }
            for (int i = 0; i < vdata->dataCount; ++i) {
                MPI_Unpack(setOptMsg, maxSetOptNum, &position, &vdata->signatures[i].data, 64, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
                MPI_Unpack(setOptMsg, maxSetOptNum, &position, &vdata->public_keys[i].data, 64, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
                MPI_Unpack(setOptMsg, maxSetOptNum, &position, &vdata->messages[i], 32, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
            }
        }
        log_trace("[ID:%d]Receive the %d task (%d signatures) from Master",vdata->nodeID,completedIdx,vdata->dataCount);
        vdata->usempi=false;
        vdata->startIdx=0;
        vdata->endIdx=vdata->dataCount-1;
        vdata->taskNum=vdata->dataCount;
        costTime = verifyBitcoinSigsByOMPGPU(vdata);
    }
    free(setOptMsg);
    return NULL;
}

int verifyBitcoinSigsDynamic(const char *path,int maxsigs,int maxthreads,float taskratio,int taskpack) {
    const int maxRecords = maxsigs;
    int nodeID, nodeNum;
    //
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &nodeID);
    MPI_Comm_size(MPI_COMM_WORLD, &nodeNum);

    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == -1) {
        log_error("Error getting the host name");
        return -1;
    }
    log_info("%s:Start MPI id=%d of %d",hostname, nodeID, nodeNum);

    if (nodeID == 0) {

        verifyData vdata;
        vdata.signatures = malloc(maxRecords * sizeof(secp256k1_ecdsa_signature));
        vdata.public_keys = malloc(maxRecords * sizeof(secp256k1_pubkey));
        vdata.messages = malloc(maxRecords * sizeof(secp256k1_msgX));

        if (!vdata.signatures || !vdata.public_keys || !vdata.messages) {
            log_error("Memory allocation failed.");
            MPI_Finalize();
            return -1;
        }

        int dataCount;
        double time = readECDSADataFromCSV(path, vdata.signatures, vdata.public_keys, vdata.messages, maxRecords, &dataCount);
        log_trace("%d bitcoin transaction signatures loaded,Total spent %f seconds", dataCount,time);
        if (dataCount <= 0) {
            log_error("Task cannot be empty");
            return -1;
        }
        double time_taken = omp_get_wtime();
        //
        vdata.nodeID=nodeID;
        vdata.nodeNum=nodeNum;
        vdata.usempi=true;
        vdata.dataCount=dataCount;
        vdata.maxthreads=maxthreads;
        vdata.taskratio=taskratio;
        vdata.taskNum=taskpack;
        int res;
        pthread_t masterthread;
        res = pthread_create(&masterthread, NULL, VBSDMasterThread,(void*)&vdata);
        if (res != 0) {
            log_error("master Create thread fail");
            return -1;
        }
        res = pthread_join(masterthread, NULL);
        if (res != 0) {
            log_error("master Thread failure");
            return 0;
        }

        free(vdata.signatures);
        free(vdata.public_keys);
        free(vdata.messages);

        time_taken = omp_get_wtime() - time_taken;
        log_info("Master statistics time: total time1(include data read) = %f s, total time2(exclude data read) = %f s",time+time_taken,time_taken);
    }else{
        verifyData vdata;
        vdata.signatures=NULL;
        vdata.public_keys=NULL;
        vdata.messages=NULL;
        vdata.nodeID=nodeID;
        vdata.nodeNum=nodeNum;
        vdata.usempi=false;
        vdata.maxthreads=maxthreads;
        vdata.taskratio=taskratio;
        vdata.taskNum=taskpack;
        int res;
        pthread_t workerthread;
        res = pthread_create(&workerthread, NULL, VBSDWorkerThread,(void*)&vdata);
        if (res != 0) {
            log_error("worker Create thread fail");
            return -1;
        }
        res = pthread_join(workerthread, NULL);
        if (res != 0) {
            log_error("worker Thread failure");
            return 0;
        }
    }

    MPI_Finalize();
    log_info("%s:End MPI id=%d of %d",hostname,nodeID, nodeNum);
    return 0;
}

int main(int argc, char **argv) {
    const char *path = "./data/signature_data.csv";
    int maxsigs = NUM_OF_SIGS;
    int maxthreads = omp_get_max_threads();
    float taskratio = 0.5f;
    int cpu = 0;
    int omp = 0;
    int gpu = 0;
    int ompgpu = 0;
    int mpicpu = 0;
    int mpiomp = 0;
    int mpigpu = 0;
    int mpiompgpu = 0;
    int all = 0;
    int loglevel = 0;
    int dynamic = 0;
    int taskpack = 10000;

    struct argparse_option options[] = {
            OPT_HELP(),
            OPT_GROUP("Basic Options"),
            OPT_STRING('p', "path", &path, "Data file path (default:./data/signature_data.csv)", NULL, 0, 0),
            OPT_INTEGER('s', "maxsigs", &maxsigs, "Maximum number of signature check operations in a block (default: 100000)", NULL, 0, 0),
            OPT_INTEGER('t', "maxthreads", &maxthreads, "Max number of CPU cores to use (default: allow all)", NULL, 0, 0),
            OPT_FLOAT('r', "taskratio", &taskratio, "Percentage of tasks that are offloaded to the GPU when OpenMP works together with the GPU (default: 0.5 ie. half and half)", NULL, 0, 0),
            OPT_BOOLEAN('a', "all", &all, "Whether to use the stand-alone mode continuously(same as -c -o -g -x)", NULL, 0, 0),
            OPT_INTEGER('l', "loglevel", &loglevel, "Logging level to print (default: 0=LOG_TRACE)", NULL, 0, 0),
            OPT_INTEGER('k', "taskpack", &taskpack, "Distributed Cluster Network Task Distribution Package Size (default: 10,000 signatures)", NULL, 0, 0),
            OPT_GROUP("Single player mode"),
            OPT_BOOLEAN('c', "cpu", &cpu, "CPU only", NULL, 0, 0),
            OPT_BOOLEAN('o', "omp", &omp, "Use OpenMP only", NULL, 0, 0),
            OPT_BOOLEAN('g', "gpu", &gpu, "GPU only", NULL, 0, 0),
            OPT_BOOLEAN('x', "ompgpu", &ompgpu, "Use OpenMP+GPU (collaboration percentage needs to be set)", NULL, 0, 0),
            OPT_GROUP("Cluster mode"),
            OPT_BOOLEAN('C', "mpicpu", &mpicpu, "Using CPU with MPI", NULL, 0, 0),
            OPT_BOOLEAN('O', "mpiomp", &mpiomp, "Using OpenMP with MPI", NULL, 0, 0),
            OPT_BOOLEAN('G', "mpigpu", &mpigpu, "Using GPUs with MPI", NULL, 0, 0),
            OPT_BOOLEAN('X', "mpiompgpu", &mpiompgpu, "Using OpenMP+GPU(Collaboration ratio setting required) via MPI", NULL, 0, 0),
            OPT_BOOLEAN('D', "dynamic", &dynamic, "Use MPI master-slave patterns to dynamically distribute tasks to target machines", NULL, 0, 0),
            OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usages, 0);
    argparse_describe(&argparse, "\nLoading the bitcoin signature data, the efficiency of statistic verification signature is counted for different resource allocation.", "\nCopyright (c) 2023-2024 AUS Bitcoin Signature Verifier");
    argc = argparse_parse(&argparse, argc, argv);

    int mode = MODE_CPU;
    if (cpu != 0) {
       mode = MODE_CPU; 
    }else if (omp != 0) {
       mode = MODE_OPENMP; 
    }else if (gpu != 0) {
       mode = MODE_GPU; 
    }else if (ompgpu != 0) {
       mode = MODE_OPENMP_GPU; 
    }else if (mpicpu != 0) {
       mode = MODE_MPI_CPU; 
    }else if (mpiomp != 0) {
       mode = MODE_MPI_OPENMP; 
    }else if (mpigpu != 0) {
       mode = MODE_MPI_GPU; 
    }else if (mpiompgpu != 0) {
       mode = MODE_MPI_OPENMP_GPU; 
    }
    if (all != 0) {
       mode= MODE_ALL;
    }
    if (loglevel != 0) {
        log_set_level(loglevel);
    }
    log_info("Begin Verifier");
    /* initialize */
    ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (dynamic == 0) {
        verifyBitcoinSigs(path,maxsigs,maxthreads,taskratio,mode);
    }else{
        verifyBitcoinSigsDynamic(path,maxsigs,maxthreads,taskratio,taskpack);
    }
    /* shutdown */
    secp256k1_context_destroy(ctx);
    log_info("Exit Verifier");
    return 0;
}
