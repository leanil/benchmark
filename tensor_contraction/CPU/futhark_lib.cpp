#include "futhark_lib.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#undef NDEBUG
#include <assert.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

static void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* For generating arbitrary-sized error messages.  It is the callers
   responsibility to free the buffer at some point. */
static char* msgprintf(const char *s, ...) {
  va_list vl;
  va_start(vl, s);
  size_t needed = 1 + vsnprintf(NULL, 0, s, vl);
  char *buffer = (char*)malloc(needed);
  va_start(vl, s); /* Must re-init. */
  vsnprintf(buffer, needed, s, vl);
  return buffer;
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

static int64_t get_wall_time(void) {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

static int64_t get_wall_time(void) {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#ifdef _MSC_VER
#define inline __inline
#endif
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
/* A very simple cross-platform implementation of locks.  Uses
   pthreads on Unix and some Windows thing there.  Futhark's
   host-level code is not multithreaded, but user code may be, so we
   need some mechanism for ensuring atomic access to API functions.
   This is that mechanism.  It is not exposed to user code at all, so
   we do not have to worry about name collisions. */

#ifdef _WIN32

typedef HANDLE lock_t;

static lock_t create_lock(lock_t *lock) {
  return *lock = CreateMutex(NULL,  /* Default security attributes. */
                      FALSE, /* Initially unlocked. */
                      NULL); /* Unnamed. */
}

static void lock_lock(lock_t *lock) {
  assert(WaitForSingleObject(*lock, INFINITE) == WAIT_OBJECT_0);
}

static void lock_unlock(lock_t *lock) {
  assert(ReleaseMutex(*lock));
}

static void free_lock(lock_t *lock) {
  CloseHandle(*lock);
}

#else
/* Assuming POSIX */

#include <pthread.h>

typedef pthread_mutex_t lock_t;

static void create_lock(lock_t *lock) {
  int r = pthread_mutex_init(lock, NULL);
  assert(r == 0);
}

static void lock_lock(lock_t *lock) {
  int r = pthread_mutex_lock(lock);
  assert(r == 0);
}

static void lock_unlock(lock_t *lock) {
  int r = pthread_mutex_unlock(lock);
  assert(r == 0);
}

static void free_lock(lock_t *lock) {
  /* Nothing to do for pthreads. */
  lock = lock;
}

#endif

struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
struct futhark_context_config {
    int debugging;
} ;
struct futhark_context_config *futhark_context_config_new()
{
    struct futhark_context_config *cfg =
        (futhark_context_config *) malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->debugging = 0;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int detail)
{
    cfg->debugging = detail;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int detail)
{
    /* Does nothing for this backend. */
    cfg = cfg;
    detail = detail;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
} ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx = (futhark_context *)malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->debugging;
    ctx->debugging = cfg->debugging;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    ctx = ctx;
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
static void memblock_unref(struct futhark_context *ctx, struct memblock *block,
                           const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "default space", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_default -= block->size;
            free(block->mem);
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_default);
        }
        block->references = NULL;
    }
}
static void memblock_alloc(struct futhark_context *ctx, struct memblock *block,
                           int64_t size, const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "default space",
              ctx->cur_mem_usage_default);
    memblock_unref(ctx, block, desc);
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_default += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "default space",
                (long long) ctx->cur_mem_usage_default);
    if (ctx->cur_mem_usage_default > ctx->peak_mem_usage_default) {
        ctx->peak_mem_usage_default = ctx->cur_mem_usage_default;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set(struct futhark_context *ctx, struct memblock *lhs,
                         struct memblock *rhs, const char *lhs_desc)
{
    memblock_unref(ctx, lhs, lhs_desc);
    (*rhs->references)++;
    *lhs = *rhs;
}
void futhark_debugging_report(struct futhark_context *ctx)
{
    if (ctx->detail_memory) {
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) { }
}
static int futrts_BaselineSum(struct futhark_context *ctx,
                              float *out_scalar_out_8567,
                              int64_t A_mem_sizze_8538,
                              struct memblock A_mem_8539, int32_t sizze_8286);
static int futrts_BaselineProd(struct futhark_context *ctx,
                               int64_t *out_out_memsizze_8568,
                               struct memblock *out_mem_p_8569,
                               int32_t *out_out_arrsizze_8570,
                               int64_t A_mem_sizze_8538,
                               struct memblock A_mem_8539, int32_t sizze_8293,
                               float c_8294);
static int futrts_Dot(struct futhark_context *ctx, float *out_scalar_out_8571,
                      int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                      int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                      int32_t sizze_8299, int32_t sizze_8300);
static int futrts_Dot1(struct futhark_context *ctx, float *out_scalar_out_8572,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8317, int32_t sizze_8318,
                       int32_t sizze_8319);
static int futrts_Dot2(struct futhark_context *ctx, float *out_scalar_out_8573,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8345, int32_t sizze_8346,
                       int32_t sizze_8347);
static int futrts_Dot3(struct futhark_context *ctx, float *out_scalar_out_8574,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8373, int32_t sizze_8374,
                       int32_t sizze_8375, int32_t sizze_8376);
static int futrts_Dot4(struct futhark_context *ctx, float *out_scalar_out_8575,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8411, int32_t sizze_8412,
                       int32_t sizze_8413, int32_t sizze_8414, float a_8415,
                       float b_8417, float c_8419, float d_8421);
static int futrts_Dot5(struct futhark_context *ctx, float *out_scalar_out_8576,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8467, int32_t sizze_8468,
                       int32_t sizze_8469);
static int futrts_Dot6(struct futhark_context *ctx, float *out_scalar_out_8577,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8496, int32_t sizze_8497,
                       int32_t sizze_8498, int32_t sizze_8499);
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline int8_t smin8(int8_t x, int8_t y)
{
    return x < y ? x : y;
}
static inline int16_t smin16(int16_t x, int16_t y)
{
    return x < y ? x : y;
}
static inline int32_t smin32(int32_t x, int32_t y)
{
    return x < y ? x : y;
}
static inline int64_t smin64(int64_t x, int64_t y)
{
    return x < y ? x : y;
}
static inline uint8_t umin8(uint8_t x, uint8_t y)
{
    return x < y ? x : y;
}
static inline uint16_t umin16(uint16_t x, uint16_t y)
{
    return x < y ? x : y;
}
static inline uint32_t umin32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}
static inline uint64_t umin64(uint64_t x, uint64_t y)
{
    return x < y ? x : y;
}
static inline int8_t smax8(int8_t x, int8_t y)
{
    return x < y ? y : x;
}
static inline int16_t smax16(int16_t x, int16_t y)
{
    return x < y ? y : x;
}
static inline int32_t smax32(int32_t x, int32_t y)
{
    return x < y ? y : x;
}
static inline int64_t smax64(int64_t x, int64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t umax8(uint8_t x, uint8_t y)
{
    return x < y ? y : x;
}
static inline uint16_t umax16(uint16_t x, uint16_t y)
{
    return x < y ? y : x;
}
static inline uint32_t umax32(uint32_t x, uint32_t y)
{
    return x < y ? y : x;
}
static inline uint64_t umax64(uint64_t x, uint64_t y)
{
    return x < y ? y : x;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fmin32(float x, float y)
{
    return x < y ? x : y;
}
static inline float fmax32(float x, float y)
{
    return x < y ? y : x;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fmin64(double x, double y)
{
    return x < y ? x : y;
}
static inline double fmax64(double x, double y)
{
    return x < y ? y : x;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static inline float futrts_log32(float x)
{
    return log(x);
}
static inline float futrts_log2_32(float x)
{
    return log2(x);
}
static inline float futrts_log10_32(float x)
{
    return log10(x);
}
static inline float futrts_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futrts_exp32(float x)
{
    return exp(x);
}
static inline float futrts_cos32(float x)
{
    return cos(x);
}
static inline float futrts_sin32(float x)
{
    return sin(x);
}
static inline float futrts_tan32(float x)
{
    return tan(x);
}
static inline float futrts_acos32(float x)
{
    return acos(x);
}
static inline float futrts_asin32(float x)
{
    return asin(x);
}
static inline float futrts_atan32(float x)
{
    return atan(x);
}
static inline float futrts_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline float futrts_round32(float x)
{
    return rint(x);
}
static inline char futrts_isnan32(float x)
{
    return isnan(x);
}
static inline char futrts_isinf32(float x)
{
    return isinf(x);
}
static inline int32_t futrts_to_bits32(float x)
{
    union {
        float f;
        int32_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline float futrts_from_bits32(int32_t x)
{
    union {
        int32_t f;
        float t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_log64(double x)
{
    return log(x);
}
static inline double futrts_log2_64(double x)
{
    return log2(x);
}
static inline double futrts_log10_64(double x)
{
    return log10(x);
}
static inline double futrts_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futrts_exp64(double x)
{
    return exp(x);
}
static inline double futrts_cos64(double x)
{
    return cos(x);
}
static inline double futrts_sin64(double x)
{
    return sin(x);
}
static inline double futrts_tan64(double x)
{
    return tan(x);
}
static inline double futrts_acos64(double x)
{
    return acos(x);
}
static inline double futrts_asin64(double x)
{
    return asin(x);
}
static inline double futrts_atan64(double x)
{
    return atan(x);
}
static inline double futrts_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline double futrts_round64(double x)
{
    return rint(x);
}
static inline char futrts_isnan64(double x)
{
    return isnan(x);
}
static inline char futrts_isinf64(double x)
{
    return isinf(x);
}
static inline int64_t futrts_to_bits64(double x)
{
    union {
        double f;
        int64_t t;
    } p;
    
    p.f = x;
    return p.t;
}
static inline double futrts_from_bits64(int64_t x)
{
    union {
        int64_t f;
        double t;
    } p;
    
    p.f = x;
    return p.t;
}
static int futrts_BaselineSum(struct futhark_context *ctx,
                              float *out_scalar_out_8567,
                              int64_t A_mem_sizze_8538,
                              struct memblock A_mem_8539, int32_t sizze_8286)
{
    float scalar_out_8547;
    float res_8288;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8286; i_8535++) {
        float x_8292 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float res_8291 = x_8292 + redout_8534;
        float redout_tmp_8548 = res_8291;
        
        redout_8534 = redout_tmp_8548;
    }
    res_8288 = redout_8534;
    scalar_out_8547 = res_8288;
    *out_scalar_out_8567 = scalar_out_8547;
    return 0;
}
static int futrts_BaselineProd(struct futhark_context *ctx,
                               int64_t *out_out_memsizze_8568,
                               struct memblock *out_mem_p_8569,
                               int32_t *out_out_arrsizze_8570,
                               int64_t A_mem_sizze_8538,
                               struct memblock A_mem_8539, int32_t sizze_8293,
                               float c_8294)
{
    int64_t out_memsizze_8550;
    struct memblock out_mem_8549;
    
    out_mem_8549.references = NULL;
    
    int32_t out_arrsizze_8551;
    int64_t binop_x_8541 = sext_i32_i64(sizze_8293);
    int64_t bytes_8540 = 4 * binop_x_8541;
    struct memblock mem_8542;
    
    mem_8542.references = NULL;
    memblock_alloc(ctx, &mem_8542, bytes_8540, "mem_8542");
    for (int32_t i_8536 = 0; i_8536 < sizze_8293; i_8536++) {
        float x_8297 = *(float *) &A_mem_8539.mem[i_8536 * 4];
        float res_8298 = c_8294 * x_8297;
        
        *(float *) &mem_8542.mem[i_8536 * 4] = res_8298;
    }
    out_arrsizze_8551 = sizze_8293;
    out_memsizze_8550 = bytes_8540;
    memblock_set(ctx, &out_mem_8549, &mem_8542, "mem_8542");
    *out_out_memsizze_8568 = out_memsizze_8550;
    (*out_mem_p_8569).references = NULL;
    memblock_set(ctx, &*out_mem_p_8569, &out_mem_8549, "out_mem_8549");
    *out_out_arrsizze_8570 = out_arrsizze_8551;
    memblock_unref(ctx, &mem_8542, "mem_8542");
    memblock_unref(ctx, &out_mem_8549, "out_mem_8549");
    return 0;
}
static int futrts_Dot(struct futhark_context *ctx, float *out_scalar_out_8571,
                      int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                      int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                      int32_t sizze_8299, int32_t sizze_8300)
{
    float scalar_out_8553;
    bool dim_zzero_8303 = 0 == sizze_8300;
    bool dim_zzero_8304 = 0 == sizze_8299;
    bool both_empty_8305 = dim_zzero_8303 && dim_zzero_8304;
    bool dim_match_8306 = sizze_8299 == sizze_8300;
    bool empty_or_match_8307 = both_empty_8305 || dim_match_8306;
    bool empty_or_match_cert_8308;
    
    if (!empty_or_match_8307) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:7:1-8:29 -> .\\BaselineSum.fut:8:17-8:28",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8310;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8299; i_8535++) {
        float x_8314 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8315 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float res_8316 = x_8314 * x_8315;
        float res_8313 = res_8316 + redout_8534;
        float redout_tmp_8554 = res_8313;
        
        redout_8534 = redout_tmp_8554;
    }
    res_8310 = redout_8534;
    scalar_out_8553 = res_8310;
    *out_scalar_out_8571 = scalar_out_8553;
    return 0;
}
static int futrts_Dot1(struct futhark_context *ctx, float *out_scalar_out_8572,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8317, int32_t sizze_8318,
                       int32_t sizze_8319)
{
    float scalar_out_8555;
    bool dim_zzero_8323 = 0 == sizze_8318;
    bool dim_zzero_8324 = 0 == sizze_8317;
    bool both_empty_8325 = dim_zzero_8323 && dim_zzero_8324;
    bool dim_match_8326 = sizze_8317 == sizze_8318;
    bool empty_or_match_8327 = both_empty_8325 || dim_match_8326;
    bool empty_or_match_cert_8328;
    
    if (!empty_or_match_8327) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:10:1-11:42 -> .\\BaselineSum.fut:11:27-11:38",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8329 = 0 == sizze_8319;
    bool both_empty_8330 = dim_zzero_8324 && dim_zzero_8329;
    bool dim_match_8331 = sizze_8317 == sizze_8319;
    bool empty_or_match_8332 = both_empty_8330 || dim_match_8331;
    bool empty_or_match_cert_8333;
    
    if (!empty_or_match_8332) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:10:1-11:42 -> .\\BaselineSum.fut:11:17-11:41",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8336;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8317; i_8535++) {
        float x_8340 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8341 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float x_8342 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float res_8343 = x_8340 * x_8341;
        float res_8344 = x_8342 * res_8343;
        float res_8339 = res_8344 + redout_8534;
        float redout_tmp_8556 = res_8339;
        
        redout_8534 = redout_tmp_8556;
    }
    res_8336 = redout_8534;
    scalar_out_8555 = res_8336;
    *out_scalar_out_8572 = scalar_out_8555;
    return 0;
}
static int futrts_Dot2(struct futhark_context *ctx, float *out_scalar_out_8573,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8345, int32_t sizze_8346,
                       int32_t sizze_8347)
{
    float scalar_out_8557;
    bool dim_zzero_8351 = 0 == sizze_8346;
    bool dim_zzero_8352 = 0 == sizze_8345;
    bool both_empty_8353 = dim_zzero_8351 && dim_zzero_8352;
    bool dim_match_8354 = sizze_8345 == sizze_8346;
    bool empty_or_match_8355 = both_empty_8353 || dim_match_8354;
    bool empty_or_match_cert_8356;
    
    if (!empty_or_match_8355) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:13:1-14:42 -> .\\BaselineSum.fut:14:27-14:38",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8357 = 0 == sizze_8347;
    bool both_empty_8358 = dim_zzero_8352 && dim_zzero_8357;
    bool dim_match_8359 = sizze_8345 == sizze_8347;
    bool empty_or_match_8360 = both_empty_8358 || dim_match_8359;
    bool empty_or_match_cert_8361;
    
    if (!empty_or_match_8360) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:13:1-14:42 -> .\\BaselineSum.fut:14:17-14:41",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8364;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8345; i_8535++) {
        float x_8368 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8369 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float x_8370 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float res_8371 = x_8368 + x_8369;
        float res_8372 = x_8370 * res_8371;
        float res_8367 = res_8372 + redout_8534;
        float redout_tmp_8558 = res_8367;
        
        redout_8534 = redout_tmp_8558;
    }
    res_8364 = redout_8534;
    scalar_out_8557 = res_8364;
    *out_scalar_out_8573 = scalar_out_8557;
    return 0;
}
static int futrts_Dot3(struct futhark_context *ctx, float *out_scalar_out_8574,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8373, int32_t sizze_8374,
                       int32_t sizze_8375, int32_t sizze_8376)
{
    float scalar_out_8559;
    bool dim_zzero_8381 = 0 == sizze_8374;
    bool dim_zzero_8382 = 0 == sizze_8373;
    bool both_empty_8383 = dim_zzero_8381 && dim_zzero_8382;
    bool dim_match_8384 = sizze_8373 == sizze_8374;
    bool empty_or_match_8385 = both_empty_8383 || dim_match_8384;
    bool empty_or_match_cert_8386;
    
    if (!empty_or_match_8385) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:16:1-17:55 -> .\\BaselineSum.fut:17:27-17:38",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8387 = 0 == sizze_8376;
    bool dim_zzero_8388 = 0 == sizze_8375;
    bool both_empty_8389 = dim_zzero_8387 && dim_zzero_8388;
    bool dim_match_8390 = sizze_8375 == sizze_8376;
    bool empty_or_match_8391 = both_empty_8389 || dim_match_8390;
    bool empty_or_match_cert_8392;
    
    if (!empty_or_match_8391) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:16:1-17:55 -> .\\BaselineSum.fut:17:42-17:53",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool both_empty_8393 = dim_zzero_8382 && dim_zzero_8388;
    bool dim_match_8394 = sizze_8373 == sizze_8375;
    bool empty_or_match_8395 = both_empty_8393 || dim_match_8394;
    bool empty_or_match_cert_8396;
    
    if (!empty_or_match_8395) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:16:1-17:55 -> .\\BaselineSum.fut:17:17-17:54",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8400;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8373; i_8535++) {
        float x_8404 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8405 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float x_8406 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float x_8407 = *(float *) &D_mem_8545.mem[i_8535 * 4];
        float res_8408 = x_8404 + x_8405;
        float res_8409 = x_8406 - x_8407;
        float res_8410 = res_8408 * res_8409;
        float res_8403 = res_8410 + redout_8534;
        float redout_tmp_8560 = res_8403;
        
        redout_8534 = redout_tmp_8560;
    }
    res_8400 = redout_8534;
    scalar_out_8559 = res_8400;
    *out_scalar_out_8574 = scalar_out_8559;
    return 0;
}
static int futrts_Dot4(struct futhark_context *ctx, float *out_scalar_out_8575,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8411, int32_t sizze_8412,
                       int32_t sizze_8413, int32_t sizze_8414, float a_8415,
                       float b_8417, float c_8419, float d_8421)
{
    float scalar_out_8561;
    bool dim_zzero_8423 = 0 == sizze_8412;
    bool dim_zzero_8424 = 0 == sizze_8411;
    bool both_empty_8425 = dim_zzero_8423 && dim_zzero_8424;
    bool dim_match_8426 = sizze_8411 == sizze_8412;
    bool empty_or_match_8427 = both_empty_8425 || dim_match_8426;
    bool empty_or_match_cert_8428;
    
    if (!empty_or_match_8427) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:19:1-20:99 -> .\\BaselineSum.fut:20:27-20:60",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8435 = 0 == sizze_8414;
    bool dim_zzero_8436 = 0 == sizze_8413;
    bool both_empty_8437 = dim_zzero_8435 && dim_zzero_8436;
    bool dim_match_8438 = sizze_8413 == sizze_8414;
    bool empty_or_match_8439 = both_empty_8437 || dim_match_8438;
    bool empty_or_match_cert_8440;
    
    if (!empty_or_match_8439) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:19:1-20:99 -> .\\BaselineSum.fut:20:64-20:97",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool both_empty_8441 = dim_zzero_8424 && dim_zzero_8436;
    bool dim_match_8442 = sizze_8411 == sizze_8413;
    bool empty_or_match_8443 = both_empty_8441 || dim_match_8442;
    bool empty_or_match_cert_8444;
    
    if (!empty_or_match_8443) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:19:1-20:99 -> .\\BaselineSum.fut:20:17-20:98",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8451;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8411; i_8535++) {
        float x_8455 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float x_8456 = *(float *) &D_mem_8545.mem[i_8535 * 4];
        float x_8457 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8458 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float res_8459 = c_8419 * x_8455;
        float res_8460 = d_8421 * x_8456;
        float res_8461 = a_8415 * x_8457;
        float res_8462 = b_8417 * x_8458;
        float res_8463 = res_8461 + res_8462;
        float res_8464 = res_8459 + res_8460;
        float res_8465 = res_8463 * res_8464;
        float res_8454 = res_8465 + redout_8534;
        float redout_tmp_8562 = res_8454;
        
        redout_8534 = redout_tmp_8562;
    }
    res_8451 = redout_8534;
    scalar_out_8561 = res_8451;
    *out_scalar_out_8575 = scalar_out_8561;
    return 0;
}
static int futrts_Dot5(struct futhark_context *ctx, float *out_scalar_out_8576,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int32_t sizze_8467, int32_t sizze_8468,
                       int32_t sizze_8469)
{
    float scalar_out_8563;
    bool dim_zzero_8473 = 0 == sizze_8468;
    bool dim_zzero_8474 = 0 == sizze_8467;
    bool both_empty_8475 = dim_zzero_8473 && dim_zzero_8474;
    bool dim_match_8476 = sizze_8467 == sizze_8468;
    bool empty_or_match_8477 = both_empty_8475 || dim_match_8476;
    bool empty_or_match_cert_8478;
    
    if (!empty_or_match_8477) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:22:1-23:55 -> .\\BaselineSum.fut:23:27-23:38",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8479 = 0 == sizze_8469;
    bool both_empty_8480 = dim_zzero_8474 && dim_zzero_8479;
    bool dim_match_8481 = sizze_8467 == sizze_8469;
    bool empty_or_match_8482 = both_empty_8480 || dim_match_8481;
    bool empty_or_match_cert_8483;
    
    if (!empty_or_match_8482) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:22:1-23:55 -> .\\BaselineSum.fut:23:42-23:53",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8486;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8467; i_8535++) {
        float x_8490 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8491 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float x_8492 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float res_8493 = x_8490 + x_8491;
        float res_8494 = x_8490 - x_8492;
        float res_8495 = res_8493 * res_8494;
        float res_8489 = res_8495 + redout_8534;
        float redout_tmp_8564 = res_8489;
        
        redout_8534 = redout_tmp_8564;
    }
    res_8486 = redout_8534;
    scalar_out_8563 = res_8486;
    *out_scalar_out_8576 = scalar_out_8563;
    return 0;
}
static int futrts_Dot6(struct futhark_context *ctx, float *out_scalar_out_8577,
                       int64_t A_mem_sizze_8538, struct memblock A_mem_8539,
                       int64_t B_mem_sizze_8540, struct memblock B_mem_8541,
                       int64_t C_mem_sizze_8542, struct memblock C_mem_8543,
                       int64_t D_mem_sizze_8544, struct memblock D_mem_8545,
                       int32_t sizze_8496, int32_t sizze_8497,
                       int32_t sizze_8498, int32_t sizze_8499)
{
    float scalar_out_8565;
    bool dim_zzero_8504 = 0 == sizze_8497;
    bool dim_zzero_8505 = 0 == sizze_8496;
    bool both_empty_8506 = dim_zzero_8504 && dim_zzero_8505;
    bool dim_match_8507 = sizze_8496 == sizze_8497;
    bool empty_or_match_8508 = both_empty_8506 || dim_match_8507;
    bool empty_or_match_cert_8509;
    
    if (!empty_or_match_8508) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:25:1-28:31 -> .\\BaselineSum.fut:26:12-26:23",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8510 = 0 == sizze_8499;
    bool dim_zzero_8511 = 0 == sizze_8498;
    bool both_empty_8512 = dim_zzero_8510 && dim_zzero_8511;
    bool dim_match_8513 = sizze_8498 == sizze_8499;
    bool empty_or_match_8514 = both_empty_8512 || dim_match_8513;
    bool empty_or_match_cert_8515;
    
    if (!empty_or_match_8514) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:25:1-28:31 -> .\\BaselineSum.fut:27:12-27:23",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool both_empty_8516 = dim_zzero_8505 && dim_zzero_8511;
    bool dim_match_8517 = sizze_8496 == sizze_8498;
    bool empty_or_match_8518 = both_empty_8516 || dim_match_8517;
    bool empty_or_match_cert_8519;
    
    if (!empty_or_match_8518) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               ".\\BaselineSum.fut:25:1-28:31 -> .\\BaselineSum.fut:28:17-28:30",
                               "function arguments of wrong shape");
        return 1;
    }
    
    float res_8523;
    float redout_8534 = 0.0F;
    
    for (int32_t i_8535 = 0; i_8535 < sizze_8496; i_8535++) {
        float x_8527 = *(float *) &A_mem_8539.mem[i_8535 * 4];
        float x_8528 = *(float *) &B_mem_8541.mem[i_8535 * 4];
        float x_8529 = *(float *) &C_mem_8543.mem[i_8535 * 4];
        float x_8530 = *(float *) &D_mem_8545.mem[i_8535 * 4];
        float res_8531 = x_8527 + x_8528;
        float res_8532 = x_8529 - x_8530;
        float res_8533 = res_8531 * res_8532;
        float res_8526 = res_8533 + redout_8534;
        float redout_tmp_8566 = res_8526;
        
        redout_8534 = redout_tmp_8566;
    }
    res_8523 = redout_8534;
    scalar_out_8565 = res_8523;
    *out_scalar_out_8577 = scalar_out_8565;
    return 0;
}
struct futhark_f32_1d {
    struct memblock mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0)
{
    struct futhark_f32_1d *arr = (futhark_f32_1d *)malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem");
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + 0, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset, int dim0)
{
    struct futhark_f32_1d *arr = (futhark_f32_1d *)malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    memblock_alloc(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem");
    arr->shape[0] = dim0;
    memmove(arr->mem.mem + 0, data + offset, dim0 * sizeof(float));
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    memblock_unref(ctx, &arr->mem, "arr->mem");
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    memmove(data + 0, arr->mem.mem + 0, arr->shape[0] * sizeof(float));
    lock_unlock(&ctx->lock);
    return 0;
}
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr)
{
    return arr->mem.mem;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    return arr->shape;
}
int futhark_entry_BaselineSum(struct futhark_context *ctx, float *out0, const
                              struct futhark_f32_1d *in0)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int32_t sizze_8286;
    float scalar_out_8547;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8286 = in0->shape[0];
    
    int ret = futrts_BaselineSum(ctx, &scalar_out_8547, A_mem_sizze_8538,
                                 A_mem_8539, sizze_8286);
    
    if (ret == 0) {
        *out0 = scalar_out_8547;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_BaselineProd(struct futhark_context *ctx,
                               struct futhark_f32_1d **out0, const float in0,
                               const struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int32_t sizze_8293;
    float c_8294;
    int64_t out_memsizze_8550;
    struct memblock out_mem_8549;
    
    out_mem_8549.references = NULL;
    
    int32_t out_arrsizze_8551;
    
    lock_lock(&ctx->lock);
    c_8294 = in0;
    A_mem_8539 = in1->mem;
    A_mem_sizze_8538 = in1->mem.size;
    sizze_8293 = in1->shape[0];
    
    int ret = futrts_BaselineProd(ctx, &out_memsizze_8550, &out_mem_8549,
                                  &out_arrsizze_8551, A_mem_sizze_8538,
                                  A_mem_8539, sizze_8293, c_8294);
    
    if (ret == 0) {
        assert((*out0 = (futhark_f32_1d *)malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_8549;
        (*out0)->shape[0] = out_arrsizze_8551;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int32_t sizze_8299;
    int32_t sizze_8300;
    float scalar_out_8553;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8299 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8300 = in1->shape[0];
    
    int ret = futrts_Dot(ctx, &scalar_out_8553, A_mem_sizze_8538, A_mem_8539,
                         B_mem_sizze_8540, B_mem_8541, sizze_8299, sizze_8300);
    
    if (ret == 0) {
        *out0 = scalar_out_8553;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int32_t sizze_8317;
    int32_t sizze_8318;
    int32_t sizze_8319;
    float scalar_out_8555;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8317 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8318 = in1->shape[0];
    C_mem_8543 = in2->mem;
    C_mem_sizze_8542 = in2->mem.size;
    sizze_8319 = in2->shape[0];
    
    int ret = futrts_Dot1(ctx, &scalar_out_8555, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, sizze_8317, sizze_8318, sizze_8319);
    
    if (ret == 0) {
        *out0 = scalar_out_8555;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot2(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int32_t sizze_8345;
    int32_t sizze_8346;
    int32_t sizze_8347;
    float scalar_out_8557;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8345 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8346 = in1->shape[0];
    C_mem_8543 = in2->mem;
    C_mem_sizze_8542 = in2->mem.size;
    sizze_8347 = in2->shape[0];
    
    int ret = futrts_Dot2(ctx, &scalar_out_8557, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, sizze_8345, sizze_8346, sizze_8347);
    
    if (ret == 0) {
        *out0 = scalar_out_8557;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot3(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int64_t D_mem_sizze_8544;
    struct memblock D_mem_8545;
    
    D_mem_8545.references = NULL;
    
    int32_t sizze_8373;
    int32_t sizze_8374;
    int32_t sizze_8375;
    int32_t sizze_8376;
    float scalar_out_8559;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8373 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8374 = in1->shape[0];
    C_mem_8543 = in2->mem;
    C_mem_sizze_8542 = in2->mem.size;
    sizze_8375 = in2->shape[0];
    D_mem_8545 = in3->mem;
    D_mem_sizze_8544 = in3->mem.size;
    sizze_8376 = in3->shape[0];
    
    int ret = futrts_Dot3(ctx, &scalar_out_8559, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, D_mem_sizze_8544, D_mem_8545, sizze_8373,
                          sizze_8374, sizze_8375, sizze_8376);
    
    if (ret == 0) {
        *out0 = scalar_out_8559;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot4(struct futhark_context *ctx, float *out0, const
                       float in0, const struct futhark_f32_1d *in1, const
                       float in2, const struct futhark_f32_1d *in3, const
                       float in4, const struct futhark_f32_1d *in5, const
                       float in6, const struct futhark_f32_1d *in7)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int64_t D_mem_sizze_8544;
    struct memblock D_mem_8545;
    
    D_mem_8545.references = NULL;
    
    int32_t sizze_8411;
    int32_t sizze_8412;
    int32_t sizze_8413;
    int32_t sizze_8414;
    float a_8415;
    float b_8417;
    float c_8419;
    float d_8421;
    float scalar_out_8561;
    
    lock_lock(&ctx->lock);
    a_8415 = in0;
    A_mem_8539 = in1->mem;
    A_mem_sizze_8538 = in1->mem.size;
    sizze_8411 = in1->shape[0];
    b_8417 = in2;
    B_mem_8541 = in3->mem;
    B_mem_sizze_8540 = in3->mem.size;
    sizze_8412 = in3->shape[0];
    c_8419 = in4;
    C_mem_8543 = in5->mem;
    C_mem_sizze_8542 = in5->mem.size;
    sizze_8413 = in5->shape[0];
    d_8421 = in6;
    D_mem_8545 = in7->mem;
    D_mem_sizze_8544 = in7->mem.size;
    sizze_8414 = in7->shape[0];
    
    int ret = futrts_Dot4(ctx, &scalar_out_8561, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, D_mem_sizze_8544, D_mem_8545, sizze_8411,
                          sizze_8412, sizze_8413, sizze_8414, a_8415, b_8417,
                          c_8419, d_8421);
    
    if (ret == 0) {
        *out0 = scalar_out_8561;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot5(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int32_t sizze_8467;
    int32_t sizze_8468;
    int32_t sizze_8469;
    float scalar_out_8563;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8467 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8468 = in1->shape[0];
    C_mem_8543 = in2->mem;
    C_mem_sizze_8542 = in2->mem.size;
    sizze_8469 = in2->shape[0];
    
    int ret = futrts_Dot5(ctx, &scalar_out_8563, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, sizze_8467, sizze_8468, sizze_8469);
    
    if (ret == 0) {
        *out0 = scalar_out_8563;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_Dot6(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_8538;
    struct memblock A_mem_8539;
    
    A_mem_8539.references = NULL;
    
    int64_t B_mem_sizze_8540;
    struct memblock B_mem_8541;
    
    B_mem_8541.references = NULL;
    
    int64_t C_mem_sizze_8542;
    struct memblock C_mem_8543;
    
    C_mem_8543.references = NULL;
    
    int64_t D_mem_sizze_8544;
    struct memblock D_mem_8545;
    
    D_mem_8545.references = NULL;
    
    int32_t sizze_8496;
    int32_t sizze_8497;
    int32_t sizze_8498;
    int32_t sizze_8499;
    float scalar_out_8565;
    
    lock_lock(&ctx->lock);
    A_mem_8539 = in0->mem;
    A_mem_sizze_8538 = in0->mem.size;
    sizze_8496 = in0->shape[0];
    B_mem_8541 = in1->mem;
    B_mem_sizze_8540 = in1->mem.size;
    sizze_8497 = in1->shape[0];
    C_mem_8543 = in2->mem;
    C_mem_sizze_8542 = in2->mem.size;
    sizze_8498 = in2->shape[0];
    D_mem_8545 = in3->mem;
    D_mem_sizze_8544 = in3->mem.size;
    sizze_8499 = in3->shape[0];
    
    int ret = futrts_Dot6(ctx, &scalar_out_8565, A_mem_sizze_8538, A_mem_8539,
                          B_mem_sizze_8540, B_mem_8541, C_mem_sizze_8542,
                          C_mem_8543, D_mem_sizze_8544, D_mem_8545, sizze_8496,
                          sizze_8497, sizze_8498, sizze_8499);
    
    if (ret == 0) {
        *out0 = scalar_out_8565;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
