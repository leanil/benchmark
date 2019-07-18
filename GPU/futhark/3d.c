#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Initialisation
*/

int futhark_get_num_sizes(void);
const char *futhark_get_size_name(int);
const char *futhark_get_size_class(int);
const char *futhark_get_size_entry(int);
struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s);
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s);
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path);
void
futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                         const char *path);
void
futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                              int size);
void
futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                              int num);
void
futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                             int num);
void
futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                             int num);
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);

/*
 * Arrays
*/

struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1);
int futhark_free_f32_2d(struct futhark_context *ctx,
                        struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data);
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr);
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1,
                                          int dim2);
int futhark_free_f32_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d *arr);
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data);
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr);
struct futhark_f32_1d ;
struct futhark_f32_2d ;
struct futhark_f32_3d ;

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_t9(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2);
int futhark_entry_t1(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1);
int futhark_entry_t2(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2);
int futhark_entry_t10(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_3d *in0, const
                      struct futhark_f32_3d *in1, const
                      struct futhark_f32_1d *in2);
int futhark_entry_t11(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1, const
                      struct futhark_f32_1d *in2, const
                      struct futhark_f32_3d *in3, const
                      struct futhark_f32_1d *in4);
int futhark_entry_t3(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3);
int futhark_entry_t4(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_3d *in1, const
                     struct futhark_f32_2d *in2);
int futhark_entry_t5(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_3d *in1, const
                     struct futhark_f32_2d *in2, const
                     struct futhark_f32_2d *in3);
int futhark_entry_t6(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const float in0, const struct futhark_f32_3d *in1, const
                     float in2, const struct futhark_f32_3d *in3, const
                     float in4, const struct futhark_f32_2d *in5, const
                     float in6, const struct futhark_f32_2d *in7);
int futhark_entry_t7(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const float in0, const struct futhark_f32_3d *in1, const
                     float in2, const struct futhark_f32_3d *in3, const
                     float in4, const struct futhark_f32_2d *in5, const
                     float in6, const struct futhark_f32_2d *in7, const
                     float in8, const struct futhark_f32_1d *in9, const
                     float in10, const struct futhark_f32_1d *in11, const
                     float in12, const struct futhark_f32_1d *in13, const
                     float in14, const struct futhark_f32_1d *in15);
int futhark_entry_t8(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_2d *in2);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
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
  char *buffer = malloc(needed);
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

#include <string.h>
#include <inttypes.h>
#include <errno.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
//// Text I/O

typedef int (*writer)(FILE*, void*);
typedef int (*bin_reader)(void*);
typedef int (*str_reader)(const char *, void*);

struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  str_reader elem_reader;
};

static void skipspaces() {
  int c;
  do {
    c = getchar();
  } while (isspace(c));

  if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int constituent(char c) {
  return isalnum(c) || c == '.' || c == '-' || c == '+' || c == '_';
}

// Produces an empty token only on EOF.
static void next_token(char *buf, int bufsize) {
 start:
  skipspaces();

  int i = 0;
  while (i < bufsize) {
    int c = getchar();
    buf[i] = c;

    if (c == EOF) {
      buf[i] = 0;
      return;
    } else if (c == '-' && i == 1 && buf[0] == '-') {
      // Line comment, so skip to end of line and start over.
      for (; c != '\n' && c != EOF; c = getchar());
      goto start;
    } else if (!constituent(c)) {
      if (i == 0) {
        // We permit single-character tokens that are not
        // constituents; this lets things like ']' and ',' be
        // tokens.
        buf[i+1] = 0;
        return;
      } else {
        ungetc(c, stdin);
        buf[i] = 0;
        return;
      }
    }

    i++;
  }

  buf[bufsize-1] = 0;
}

static int next_token_is(char *buf, int bufsize, const char* expected) {
  next_token(buf, bufsize);
  return strcmp(buf, expected) == 0;
}

static void remove_underscores(char *buf) {
  char *w = buf;

  for (char *r = buf; *r; r++) {
    if (*r != '_') {
      *w++ = *r;
    }
  }

  *w++ = 0;
}

static int read_str_elem(char *buf, struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(buf, reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_str_array_elems(char *buf, int bufsize,
                                struct array_reader *reader, int dims) {
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));

  while (1) {
    next_token(buf, bufsize);

    if (strcmp(buf, "]") == 0) {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (strcmp(buf, ",") == 0) {
      next_token(buf, bufsize);
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (strlen(buf) == 0) {
      // EOF
      ret = 1;
      break;
    } else if (first) {
      if (strcmp(buf, "[") == 0) {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ret = read_str_elem(buf, reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_str_empty_array(char *buf, int bufsize,
                                const char *type_name, int64_t *shape, int64_t dims) {
  if (strlen(buf) == 0) {
    // EOF
    return 1;
  }

  if (strcmp(buf, "empty") != 0) {
    return 1;
  }

  if (!next_token_is(buf, bufsize, "(")) {
    return 1;
  }

  for (int i = 0; i < dims-1; i++) {
    if (!next_token_is(buf, bufsize, "[")) {
      return 1;
    }

    if (!next_token_is(buf, bufsize, "]")) {
      return 1;
    }
  }

  if (!next_token_is(buf, bufsize, type_name)) {
    return 1;
  }


  if (!next_token_is(buf, bufsize, ")")) {
    return 1;
  }

  for (int i = 0; i < dims; i++) {
    shape[i] = 0;
  }

  return 0;
}

static int read_str_array(int64_t elem_size, str_reader elem_reader,
                          const char *type_name,
                          void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  char buf[100];

  int dims_seen;
  for (dims_seen = 0; dims_seen < dims; dims_seen++) {
    if (!next_token_is(buf, sizeof(buf), "[")) {
      break;
    }
  }

  if (dims_seen == 0) {
    return read_str_empty_array(buf, sizeof(buf), type_name, shape, dims);
  }

  if (dims_seen != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_str_array_elems(buf, sizeof(buf), &reader, dims);

  *data = reader.elems;

  return ret;
}

#define READ_STR(MACRO, PTR, SUFFIX)                                   \
  remove_underscores(buf);                                              \
  int j;                                                                \
  if (sscanf(buf, "%"MACRO"%n", (PTR*)dest, &j) == 1) {                 \
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, SUFFIX) == 0);     \
  } else {                                                              \
    return 1;                                                           \
  }

static int read_str_i8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNi8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%d%n", &x, &j) == 1) {
    *(int8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "i8") == 0);
  } else {
    return 1;
  }
}

static int read_str_u8(char *buf, void* dest) {
  /* Some platforms (WINDOWS) does not support scanf %hhd or its
     cousin, %SCNu8.  Read into int first to avoid corrupting
     memory.

     https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63417  */
  remove_underscores(buf);
  int j, x;
  if (sscanf(buf, "%d%n", &x, &j) == 1) {
    *(uint8_t*)dest = x;
    return !(strcmp(buf+j, "") == 0 || strcmp(buf+j, "u8") == 0);
  } else {
    return 1;
  }
}

static int read_str_i16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "i16");
}

static int read_str_u16(char *buf, void* dest) {
  READ_STR(SCNi16, int16_t, "u16");
}

static int read_str_i32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "i32");
}

static int read_str_u32(char *buf, void* dest) {
  READ_STR(SCNi32, int32_t, "u32");
}

static int read_str_i64(char *buf, void* dest) {
  READ_STR(SCNi64, int64_t, "i64");
}

static int read_str_u64(char *buf, void* dest) {
  // FIXME: This is not correct, as SCNu64 only permits decimal
  // literals.  However, SCNi64 does not handle very large numbers
  // correctly (it's really for signed numbers, so that's fair).
  READ_STR(SCNu64, uint64_t, "u64");
}

static int read_str_f32(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f32.nan") == 0) {
    *(float*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f32.inf") == 0) {
    *(float*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f32.inf") == 0) {
    *(float*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("f", float, "f32");
  }
}

static int read_str_f64(char *buf, void* dest) {
  remove_underscores(buf);
  if (strcmp(buf, "f64.nan") == 0) {
    *(double*)dest = NAN;
    return 0;
  } else if (strcmp(buf, "f64.inf") == 0) {
    *(double*)dest = INFINITY;
    return 0;
  } else if (strcmp(buf, "-f64.inf") == 0) {
    *(double*)dest = -INFINITY;
    return 0;
  } else {
    READ_STR("lf", double, "f64");
  }
}

static int read_str_bool(char *buf, void* dest) {
  if (strcmp(buf, "true") == 0) {
    *(char*)dest = 1;
    return 0;
  } else if (strcmp(buf, "false") == 0) {
    *(char*)dest = 0;
    return 0;
  } else {
    return 1;
  }
}

static int write_str_i8(FILE *out, int8_t *src) {
  return fprintf(out, "%hhdi8", *src);
}

static int write_str_u8(FILE *out, uint8_t *src) {
  return fprintf(out, "%hhuu8", *src);
}

static int write_str_i16(FILE *out, int16_t *src) {
  return fprintf(out, "%hdi16", *src);
}

static int write_str_u16(FILE *out, uint16_t *src) {
  return fprintf(out, "%huu16", *src);
}

static int write_str_i32(FILE *out, int32_t *src) {
  return fprintf(out, "%di32", *src);
}

static int write_str_u32(FILE *out, uint32_t *src) {
  return fprintf(out, "%uu32", *src);
}

static int write_str_i64(FILE *out, int64_t *src) {
  return fprintf(out, "%"PRIi64"i64", *src);
}

static int write_str_u64(FILE *out, uint64_t *src) {
  return fprintf(out, "%"PRIu64"u64", *src);
}

static int write_str_f32(FILE *out, float *src) {
  float x = *src;
  if (isnan(x)) {
    return fprintf(out, "f32.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f32.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f32.inf");
  } else {
    return fprintf(out, "%.6ff32", x);
  }
}

static int write_str_f64(FILE *out, double *src) {
  double x = *src;
  if (isnan(x)) {
    return fprintf(out, "f64.nan");
  } else if (isinf(x) && x >= 0) {
    return fprintf(out, "f64.inf");
  } else if (isinf(x)) {
    return fprintf(out, "-f64.inf");
  } else {
    return fprintf(out, "%.6ff64", *src);
  }
}

static int write_str_bool(FILE *out, void *src) {
  return fprintf(out, *(char*)src ? "true" : "false");
}

//// Binary I/O

#define BINARY_FORMAT_VERSION 2
#define IS_BIG_ENDIAN (!*(unsigned char *)&(uint16_t){1})

// Reading little-endian byte sequences.  On big-endian hosts, we flip
// the resulting bytes.

static int read_byte(void* dest) {
  int num_elems_read = fread(dest, 1, 1, stdin);
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_2byte(void* dest) {
  uint16_t x;
  int num_elems_read = fread(&x, 2, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  *(uint16_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_4byte(void* dest) {
  uint32_t x;
  int num_elems_read = fread(&x, 4, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  *(uint32_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int read_le_8byte(void* dest) {
  uint64_t x;
  int num_elems_read = fread(&x, 8, 1, stdin);
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  *(uint64_t*)dest = x;
  return num_elems_read == 1 ? 0 : 1;
}

static int write_byte(void* dest) {
  int num_elems_written = fwrite(dest, 1, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_2byte(void* dest) {
  uint16_t x = *(uint16_t*)dest;
  if (IS_BIG_ENDIAN) {
    x = (x>>8) | (x<<8);
  }
  int num_elems_written = fwrite(&x, 2, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_4byte(void* dest) {
  uint32_t x = *(uint32_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>24)&0xFF) |
      ((x>>8) &0xFF00) |
      ((x<<8) &0xFF0000) |
      ((x<<24)&0xFF000000);
  }
  int num_elems_written = fwrite(&x, 4, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

static int write_le_8byte(void* dest) {
  uint64_t x = *(uint64_t*)dest;
  if (IS_BIG_ENDIAN) {
    x =
      ((x>>56)&0xFFull) |
      ((x>>40)&0xFF00ull) |
      ((x>>24)&0xFF0000ull) |
      ((x>>8) &0xFF000000ull) |
      ((x<<8) &0xFF00000000ull) |
      ((x<<24)&0xFF0000000000ull) |
      ((x<<40)&0xFF000000000000ull) |
      ((x<<56)&0xFF00000000000000ull);
  }
  int num_elems_written = fwrite(&x, 8, 1, stdin);
  return num_elems_written == 1 ? 0 : 1;
}

//// Types

struct primtype_info_t {
  const char binname[4]; // Used for parsing binary data.
  const char* type_name; // Same name as in Futhark.
  const int size; // in bytes
  const writer write_str; // Write in text format.
  const str_reader read_str; // Read in text format.
  const writer write_bin; // Write in binary format.
  const bin_reader read_bin; // Read in binary format.
};

static const struct primtype_info_t i8_info =
  {.binname = "  i8", .type_name = "i8",   .size = 1,
   .write_str = (writer)write_str_i8, .read_str = (str_reader)read_str_i8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t i16_info =
  {.binname = " i16", .type_name = "i16",  .size = 2,
   .write_str = (writer)write_str_i16, .read_str = (str_reader)read_str_i16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t i32_info =
  {.binname = " i32", .type_name = "i32",  .size = 4,
   .write_str = (writer)write_str_i32, .read_str = (str_reader)read_str_i32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t i64_info =
  {.binname = " i64", .type_name = "i64",  .size = 8,
   .write_str = (writer)write_str_i64, .read_str = (str_reader)read_str_i64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t u8_info =
  {.binname = "  u8", .type_name = "u8",   .size = 1,
   .write_str = (writer)write_str_u8, .read_str = (str_reader)read_str_u8,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};
static const struct primtype_info_t u16_info =
  {.binname = " u16", .type_name = "u16",  .size = 2,
   .write_str = (writer)write_str_u16, .read_str = (str_reader)read_str_u16,
   .write_bin = (writer)write_le_2byte, .read_bin = (bin_reader)read_le_2byte};
static const struct primtype_info_t u32_info =
  {.binname = " u32", .type_name = "u32",  .size = 4,
   .write_str = (writer)write_str_u32, .read_str = (str_reader)read_str_u32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t u64_info =
  {.binname = " u64", .type_name = "u64",  .size = 8,
   .write_str = (writer)write_str_u64, .read_str = (str_reader)read_str_u64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t f32_info =
  {.binname = " f32", .type_name = "f32",  .size = 4,
   .write_str = (writer)write_str_f32, .read_str = (str_reader)read_str_f32,
   .write_bin = (writer)write_le_4byte, .read_bin = (bin_reader)read_le_4byte};
static const struct primtype_info_t f64_info =
  {.binname = " f64", .type_name = "f64",  .size = 8,
   .write_str = (writer)write_str_f64, .read_str = (str_reader)read_str_f64,
   .write_bin = (writer)write_le_8byte, .read_bin = (bin_reader)read_le_8byte};
static const struct primtype_info_t bool_info =
  {.binname = "bool", .type_name = "bool", .size = 1,
   .write_str = (writer)write_str_bool, .read_str = (str_reader)read_str_bool,
   .write_bin = (writer)write_byte, .read_bin = (bin_reader)read_byte};

static const struct primtype_info_t* primtypes[] = {
  &i8_info, &i16_info, &i32_info, &i64_info,
  &u8_info, &u16_info, &u32_info, &u64_info,
  &f32_info, &f64_info,
  &bool_info,
  NULL // NULL-terminated
};

// General value interface.  All endian business taken care of at
// lower layers.

static int read_is_binary() {
  skipspaces();
  int c = getchar();
  if (c == 'b') {
    int8_t bin_version;
    int ret = read_byte(&bin_version);

    if (ret != 0) { panic(1, "binary-input: could not read version.\n"); }

    if (bin_version != BINARY_FORMAT_VERSION) {
      panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
            bin_version, BINARY_FORMAT_VERSION);
    }

    return 1;
  }
  ungetc(c, stdin);
  return 0;
}

static const struct primtype_info_t* read_bin_read_type_enum() {
  char read_binname[4];

  int num_matched = scanf("%4c", read_binname);
  if (num_matched != 1) { panic(1, "binary-input: Couldn't read element type.\n"); }

  const struct primtype_info_t **type = primtypes;

  for (; *type != NULL; type++) {
    // I compare the 4 characters manually instead of using strncmp because
    // this allows any value to be used, also NULL bytes
    if (memcmp(read_binname, (*type)->binname, 4) == 0) {
      return *type;
    }
  }
  panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname);
  return NULL;
}

static void read_bin_ensure_scalar(const struct primtype_info_t *expected_type) {
  int8_t bin_dims;
  int ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != 0) {
    panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n",
          bin_dims);
  }

  const struct primtype_info_t *bin_type = read_bin_read_type_enum();
  if (bin_type != expected_type) {
    panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
          expected_type->type_name,
          bin_type->type_name);
  }
}

//// High-level interface

static int read_bin_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  int ret;

  int8_t bin_dims;
  ret = read_byte(&bin_dims);
  if (ret != 0) { panic(1, "binary-input: Couldn't get dims.\n"); }

  if (bin_dims != dims) {
    panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
          dims, bin_dims);
  }

  const struct primtype_info_t *bin_primtype = read_bin_read_type_enum();
  if (expected_type != bin_primtype) {
    panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
          dims, expected_type->type_name, dims, bin_primtype->type_name);
  }

  uint64_t elem_count = 1;
  for (int i=0; i<dims; i++) {
    uint64_t bin_shape;
    ret = read_le_8byte(&bin_shape);
    if (ret != 0) { panic(1, "binary-input: Couldn't read size for dimension %i of array.\n", i); }
    elem_count *= bin_shape;
    shape[i] = (int64_t) bin_shape;
  }

  size_t elem_size = expected_type->size;
  void* tmp = realloc(*data, elem_count * elem_size);
  if (tmp == NULL) {
    panic(1, "binary-input: Failed to allocate array of size %i.\n",
          elem_count * elem_size);
  }
  *data = tmp;

  size_t num_elems_read = fread(*data, elem_size, elem_count, stdin);
  if (num_elems_read != elem_count) {
    panic(1, "binary-input: tried to read %i elements of an array, but only got %i elements.\n",
          elem_count, num_elems_read);
  }

  // If we're on big endian platform we must change all multibyte elements
  // from using little endian to big endian
  if (IS_BIG_ENDIAN && elem_size != 1) {
    char* elems = (char*) *data;
    for (uint64_t i=0; i<elem_count; i++) {
      char* elem = elems+(i*elem_size);
      for (unsigned int j=0; j<elem_size/2; j++) {
        char head = elem[j];
        int tail_index = elem_size-1-j;
        elem[j] = elem[tail_index];
        elem[tail_index] = head;
      }
    }
  }

  return 0;
}

static int read_array(const struct primtype_info_t *expected_type, void **data, int64_t *shape, int64_t dims) {
  if (!read_is_binary()) {
    return read_str_array(expected_type->size, (str_reader)expected_type->read_str, expected_type->type_name, data, shape, dims);
  } else {
    return read_bin_array(expected_type, data, shape, dims);
  }
}

static int write_str_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  if (rank==0) {
    elem_type->write_str(out, (void*)data);
  } else {
    int64_t len = shape[0];
    int64_t slice_size = 1;

    int64_t elem_size = elem_type->size;
    for (int64_t i = 1; i < rank; i++) {
      slice_size *= shape[i];
    }

    if (len*slice_size == 0) {
      printf("empty(");
      for (int64_t i = 1; i < rank; i++) {
        printf("[]");
      }
      printf("%s", elem_type->type_name);
      printf(")");
    } else if (rank==1) {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        elem_type->write_str(out, (void*) (data + i * elem_size));
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    } else {
      putchar('[');
      for (int64_t i = 0; i < len; i++) {
        write_str_array(out, elem_type, data + i * slice_size * elem_size, shape+1, rank-1);
        if (i != len-1) {
          printf(", ");
        }
      }
      putchar(']');
    }
  }
  return 0;
}

static int write_bin_array(FILE *out, const struct primtype_info_t *elem_type, unsigned char *data, int64_t *shape, int8_t rank) {
  int64_t num_elems = 1;
  for (int64_t i = 0; i < rank; i++) {
    num_elems *= shape[i];
  }

  fputc('b', out);
  fputc((char)BINARY_FORMAT_VERSION, out);
  fwrite(&rank, sizeof(int8_t), 1, out);
  fputs(elem_type->binname, out);
  fwrite(shape, sizeof(int64_t), rank, out);

  if (IS_BIG_ENDIAN) {
    for (int64_t i = 0; i < num_elems; i++) {
      unsigned char *elem = data+i*elem_type->size;
      for (int64_t j = 0; j < elem_type->size; j++) {
        fwrite(&elem[elem_type->size-j], 1, 1, out);
      }
    }
  } else {
    fwrite(data, elem_type->size, num_elems, out);
  }

  return 0;
}

static int write_array(FILE *out, int write_binary,
                       const struct primtype_info_t *elem_type, void *data, int64_t *shape, int8_t rank) {
  if (write_binary) {
    return write_bin_array(out, elem_type, data, shape, rank);
  } else {
    return write_str_array(out, elem_type, data, shape, rank);
  }
}

static int read_scalar(const struct primtype_info_t *expected_type, void *dest) {
  if (!read_is_binary()) {
    char buf[100];
    next_token(buf, sizeof(buf));
    return expected_type->read_str(buf, dest);
  } else {
    read_bin_ensure_scalar(expected_type);
    return expected_type->read_bin(dest);
  }
}

static int write_scalar(FILE *out, int write_binary, const struct primtype_info_t *type, void *src) {
  if (write_binary) {
    return write_bin_array(out, type, src, NULL, 0);
  } else {
    return type->write_str(out, src);
  }
}

static int binary_output = 0;
static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
static const char *entry_point = "main";
int parse_options(struct futhark_context_config *cfg, int argc,
                  char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"debugging", no_argument, NULL, 3},
                                           {"log", no_argument, NULL, 4},
                                           {"entry-point", required_argument,
                                            NULL, 5}, {"binary-output",
                                                       no_argument, NULL, 6},
                                           {"platform", required_argument, NULL,
                                            7}, {"device", required_argument,
                                                 NULL, 8},
                                           {"default-group-size",
                                            required_argument, NULL, 9},
                                           {"default-num-groups",
                                            required_argument, NULL, 10},
                                           {"default-tile-size",
                                            required_argument, NULL, 11},
                                           {"default-threshold",
                                            required_argument, NULL, 12},
                                           {"dump-opencl", required_argument,
                                            NULL, 13}, {"load-opencl",
                                                        required_argument, NULL,
                                                        14}, {"print-sizes",
                                                              no_argument, NULL,
                                                              15}, {"size",
                                                                    required_argument,
                                                                    NULL, 16},
                                           {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:DLe:bp:d:", long_options,
                             NULL)) != -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s\n", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s\n", optarg);
        }
        if (ch == 3 || ch == 'D')
            futhark_context_config_set_debugging(cfg, 1);
        if (ch == 4 || ch == 'L')
            futhark_context_config_set_logging(cfg, 1);
        if (ch == 5 || ch == 'e')
            entry_point = optarg;
        if (ch == 6 || ch == 'b')
            binary_output = 1;
        if (ch == 7 || ch == 'p')
            futhark_context_config_set_platform(cfg, optarg);
        if (ch == 8 || ch == 'd')
            futhark_context_config_set_device(cfg, optarg);
        if (ch == 9)
            futhark_context_config_set_default_group_size(cfg, atoi(optarg));
        if (ch == 10)
            futhark_context_config_set_default_num_groups(cfg, atoi(optarg));
        if (ch == 11)
            futhark_context_config_set_default_tile_size(cfg, atoi(optarg));
        if (ch == 12)
            futhark_context_config_set_default_threshold(cfg, atoi(optarg));
        if (ch == 13)
            futhark_context_config_dump_program_to(cfg, optarg);
        if (ch == 14)
            futhark_context_config_load_program_from(cfg, optarg);
        if (ch == 15) {
            int n = futhark_get_num_sizes();
            
            for (int i = 0; i < n; i++) {
                if (strcmp(futhark_get_size_entry(i), entry_point) == 0)
                    printf("%s (%s)\n", futhark_get_size_name(i),
                           futhark_get_size_class(i));
            }
            exit(0);
        }
        if (ch == 16) {
            char *name = optarg;
            char *equals = strstr(optarg, "=");
            char *value_str = equals != NULL ? equals + 1 : optarg;
            int value = atoi(value_str);
            
            if (equals != NULL) {
                *equals = 0;
                if (futhark_context_config_set_size(cfg, name, value) != 0)
                    panic(1, "Unknown size: %s\n", name);
            } else
                panic(1, "Invalid argument for size option: %s\n", optarg);
        }
        if (ch == ':')
            panic(-1, "Missing argument for option %s\n", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s\n", argv[optind - 1]);
    }
    return optind;
}
static void futrts_cli_entry_t9(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31399;
    int64_t read_shape_31400[3];
    float *read_arr_31401 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31401, read_shape_31400, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31402;
    int64_t read_shape_31403[2];
    float *read_arr_31404 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31404, read_shape_31403, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31405;
    int64_t read_shape_31406[1];
    float *read_arr_31407 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31407, read_shape_31406, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_31408;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31399 = futhark_new_f32_3d(ctx, read_arr_31401,
                                                      read_shape_31400[0],
                                                      read_shape_31400[1],
                                                      read_shape_31400[2])) !=
            0);
        assert((read_value_31402 = futhark_new_f32_2d(ctx, read_arr_31404,
                                                      read_shape_31403[0],
                                                      read_shape_31403[1])) !=
            0);
        assert((read_value_31405 = futhark_new_f32_1d(ctx, read_arr_31407,
                                                      read_shape_31406[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t9(ctx, &result_31408, read_value_31399,
                             read_value_31402, read_value_31405);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31399) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31402) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31405) == 0);
        assert(futhark_free_f32_2d(ctx, result_31408) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31399 = futhark_new_f32_3d(ctx, read_arr_31401,
                                                      read_shape_31400[0],
                                                      read_shape_31400[1],
                                                      read_shape_31400[2])) !=
            0);
        assert((read_value_31402 = futhark_new_f32_2d(ctx, read_arr_31404,
                                                      read_shape_31403[0],
                                                      read_shape_31403[1])) !=
            0);
        assert((read_value_31405 = futhark_new_f32_1d(ctx, read_arr_31407,
                                                      read_shape_31406[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t9(ctx, &result_31408, read_value_31399,
                             read_value_31402, read_value_31405);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31399) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31402) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31405) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_31408) == 0);
        }
    }
    free(read_arr_31401);
    free(read_arr_31404);
    free(read_arr_31407);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_31408)[0] *
                            futhark_shape_f32_2d(ctx, result_31408)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_31408, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_31408), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_31408) == 0);
}
static void futrts_cli_entry_t1(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31409;
    int64_t read_shape_31410[3];
    float *read_arr_31411 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31411, read_shape_31410, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31412;
    int64_t read_shape_31413[2];
    float *read_arr_31414 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31414, read_shape_31413, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31415;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31409 = futhark_new_f32_3d(ctx, read_arr_31411,
                                                      read_shape_31410[0],
                                                      read_shape_31410[1],
                                                      read_shape_31410[2])) !=
            0);
        assert((read_value_31412 = futhark_new_f32_2d(ctx, read_arr_31414,
                                                      read_shape_31413[0],
                                                      read_shape_31413[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1(ctx, &result_31415, read_value_31409,
                             read_value_31412);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31409) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31412) == 0);
        assert(futhark_free_f32_3d(ctx, result_31415) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31409 = futhark_new_f32_3d(ctx, read_arr_31411,
                                                      read_shape_31410[0],
                                                      read_shape_31410[1],
                                                      read_shape_31410[2])) !=
            0);
        assert((read_value_31412 = futhark_new_f32_2d(ctx, read_arr_31414,
                                                      read_shape_31413[0],
                                                      read_shape_31413[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1(ctx, &result_31415, read_value_31409,
                             read_value_31412);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31409) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31412) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31415) == 0);
        }
    }
    free(read_arr_31411);
    free(read_arr_31414);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31415)[0] *
                            futhark_shape_f32_3d(ctx, result_31415)[1] *
                            futhark_shape_f32_3d(ctx, result_31415)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31415, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31415), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31415) == 0);
}
static void futrts_cli_entry_t2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31416;
    int64_t read_shape_31417[3];
    float *read_arr_31418 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31418, read_shape_31417, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31419;
    int64_t read_shape_31420[1];
    float *read_arr_31421 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31421, read_shape_31420, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31422;
    int64_t read_shape_31423[1];
    float *read_arr_31424 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31424, read_shape_31423, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31425;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31416 = futhark_new_f32_3d(ctx, read_arr_31418,
                                                      read_shape_31417[0],
                                                      read_shape_31417[1],
                                                      read_shape_31417[2])) !=
            0);
        assert((read_value_31419 = futhark_new_f32_1d(ctx, read_arr_31421,
                                                      read_shape_31420[0])) !=
            0);
        assert((read_value_31422 = futhark_new_f32_1d(ctx, read_arr_31424,
                                                      read_shape_31423[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t2(ctx, &result_31425, read_value_31416,
                             read_value_31419, read_value_31422);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31416) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31419) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31422) == 0);
        assert(futhark_free_f32_3d(ctx, result_31425) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31416 = futhark_new_f32_3d(ctx, read_arr_31418,
                                                      read_shape_31417[0],
                                                      read_shape_31417[1],
                                                      read_shape_31417[2])) !=
            0);
        assert((read_value_31419 = futhark_new_f32_1d(ctx, read_arr_31421,
                                                      read_shape_31420[0])) !=
            0);
        assert((read_value_31422 = futhark_new_f32_1d(ctx, read_arr_31424,
                                                      read_shape_31423[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t2(ctx, &result_31425, read_value_31416,
                             read_value_31419, read_value_31422);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31416) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31419) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31422) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31425) == 0);
        }
    }
    free(read_arr_31418);
    free(read_arr_31421);
    free(read_arr_31424);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31425)[0] *
                            futhark_shape_f32_3d(ctx, result_31425)[1] *
                            futhark_shape_f32_3d(ctx, result_31425)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31425, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31425), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31425) == 0);
}
static void futrts_cli_entry_t10(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31426;
    int64_t read_shape_31427[3];
    float *read_arr_31428 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31428, read_shape_31427, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31429;
    int64_t read_shape_31430[3];
    float *read_arr_31431 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31431, read_shape_31430, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31432;
    int64_t read_shape_31433[1];
    float *read_arr_31434 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31434, read_shape_31433, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_31435;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31426 = futhark_new_f32_3d(ctx, read_arr_31428,
                                                      read_shape_31427[0],
                                                      read_shape_31427[1],
                                                      read_shape_31427[2])) !=
            0);
        assert((read_value_31429 = futhark_new_f32_3d(ctx, read_arr_31431,
                                                      read_shape_31430[0],
                                                      read_shape_31430[1],
                                                      read_shape_31430[2])) !=
            0);
        assert((read_value_31432 = futhark_new_f32_1d(ctx, read_arr_31434,
                                                      read_shape_31433[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t10(ctx, &result_31435, read_value_31426,
                              read_value_31429, read_value_31432);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31426) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31429) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31432) == 0);
        assert(futhark_free_f32_2d(ctx, result_31435) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31426 = futhark_new_f32_3d(ctx, read_arr_31428,
                                                      read_shape_31427[0],
                                                      read_shape_31427[1],
                                                      read_shape_31427[2])) !=
            0);
        assert((read_value_31429 = futhark_new_f32_3d(ctx, read_arr_31431,
                                                      read_shape_31430[0],
                                                      read_shape_31430[1],
                                                      read_shape_31430[2])) !=
            0);
        assert((read_value_31432 = futhark_new_f32_1d(ctx, read_arr_31434,
                                                      read_shape_31433[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t10(ctx, &result_31435, read_value_31426,
                              read_value_31429, read_value_31432);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31426) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31429) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31432) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_31435) == 0);
        }
    }
    free(read_arr_31428);
    free(read_arr_31431);
    free(read_arr_31434);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_31435)[0] *
                            futhark_shape_f32_2d(ctx, result_31435)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_31435, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_31435), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_31435) == 0);
}
static void futrts_cli_entry_t11(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_31436;
    int64_t read_shape_31437[1];
    float *read_arr_31438 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31438, read_shape_31437, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31439;
    int64_t read_shape_31440[1];
    float *read_arr_31441 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31441, read_shape_31440, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31442;
    int64_t read_shape_31443[1];
    float *read_arr_31444 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31444, read_shape_31443, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31445;
    int64_t read_shape_31446[3];
    float *read_arr_31447 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31447, read_shape_31446, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31448;
    int64_t read_shape_31449[1];
    float *read_arr_31450 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31450, read_shape_31449, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 4, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *result_31451;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31436 = futhark_new_f32_1d(ctx, read_arr_31438,
                                                      read_shape_31437[0])) !=
            0);
        assert((read_value_31439 = futhark_new_f32_1d(ctx, read_arr_31441,
                                                      read_shape_31440[0])) !=
            0);
        assert((read_value_31442 = futhark_new_f32_1d(ctx, read_arr_31444,
                                                      read_shape_31443[0])) !=
            0);
        assert((read_value_31445 = futhark_new_f32_3d(ctx, read_arr_31447,
                                                      read_shape_31446[0],
                                                      read_shape_31446[1],
                                                      read_shape_31446[2])) !=
            0);
        assert((read_value_31448 = futhark_new_f32_1d(ctx, read_arr_31450,
                                                      read_shape_31449[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t11(ctx, &result_31451, read_value_31436,
                              read_value_31439, read_value_31442,
                              read_value_31445, read_value_31448);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_31436) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31439) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31442) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31445) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31448) == 0);
        assert(futhark_free_f32_2d(ctx, result_31451) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31436 = futhark_new_f32_1d(ctx, read_arr_31438,
                                                      read_shape_31437[0])) !=
            0);
        assert((read_value_31439 = futhark_new_f32_1d(ctx, read_arr_31441,
                                                      read_shape_31440[0])) !=
            0);
        assert((read_value_31442 = futhark_new_f32_1d(ctx, read_arr_31444,
                                                      read_shape_31443[0])) !=
            0);
        assert((read_value_31445 = futhark_new_f32_3d(ctx, read_arr_31447,
                                                      read_shape_31446[0],
                                                      read_shape_31446[1],
                                                      read_shape_31446[2])) !=
            0);
        assert((read_value_31448 = futhark_new_f32_1d(ctx, read_arr_31450,
                                                      read_shape_31449[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t11(ctx, &result_31451, read_value_31436,
                              read_value_31439, read_value_31442,
                              read_value_31445, read_value_31448);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_31436) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31439) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31442) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31445) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31448) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_2d(ctx, result_31451) == 0);
        }
    }
    free(read_arr_31438);
    free(read_arr_31441);
    free(read_arr_31444);
    free(read_arr_31447);
    free(read_arr_31450);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_2d(ctx,
                                                                result_31451)[0] *
                            futhark_shape_f32_2d(ctx, result_31451)[1]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_2d(ctx, result_31451, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_2d(ctx, result_31451), 2);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_2d(ctx, result_31451) == 0);
}
static void futrts_cli_entry_t3(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31452;
    int64_t read_shape_31453[3];
    float *read_arr_31454 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31454, read_shape_31453, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31455;
    int64_t read_shape_31456[2];
    float *read_arr_31457 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31457, read_shape_31456, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31458;
    int64_t read_shape_31459[1];
    float *read_arr_31460 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31460, read_shape_31459, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31461;
    int64_t read_shape_31462[1];
    float *read_arr_31463 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31463, read_shape_31462, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31464;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31452 = futhark_new_f32_3d(ctx, read_arr_31454,
                                                      read_shape_31453[0],
                                                      read_shape_31453[1],
                                                      read_shape_31453[2])) !=
            0);
        assert((read_value_31455 = futhark_new_f32_2d(ctx, read_arr_31457,
                                                      read_shape_31456[0],
                                                      read_shape_31456[1])) !=
            0);
        assert((read_value_31458 = futhark_new_f32_1d(ctx, read_arr_31460,
                                                      read_shape_31459[0])) !=
            0);
        assert((read_value_31461 = futhark_new_f32_1d(ctx, read_arr_31463,
                                                      read_shape_31462[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t3(ctx, &result_31464, read_value_31452,
                             read_value_31455, read_value_31458,
                             read_value_31461);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31452) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31455) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31458) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31461) == 0);
        assert(futhark_free_f32_3d(ctx, result_31464) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31452 = futhark_new_f32_3d(ctx, read_arr_31454,
                                                      read_shape_31453[0],
                                                      read_shape_31453[1],
                                                      read_shape_31453[2])) !=
            0);
        assert((read_value_31455 = futhark_new_f32_2d(ctx, read_arr_31457,
                                                      read_shape_31456[0],
                                                      read_shape_31456[1])) !=
            0);
        assert((read_value_31458 = futhark_new_f32_1d(ctx, read_arr_31460,
                                                      read_shape_31459[0])) !=
            0);
        assert((read_value_31461 = futhark_new_f32_1d(ctx, read_arr_31463,
                                                      read_shape_31462[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t3(ctx, &result_31464, read_value_31452,
                             read_value_31455, read_value_31458,
                             read_value_31461);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31452) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31455) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31458) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_31461) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31464) == 0);
        }
    }
    free(read_arr_31454);
    free(read_arr_31457);
    free(read_arr_31460);
    free(read_arr_31463);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31464)[0] *
                            futhark_shape_f32_3d(ctx, result_31464)[1] *
                            futhark_shape_f32_3d(ctx, result_31464)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31464, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31464), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31464) == 0);
}
static void futrts_cli_entry_t4(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31465;
    int64_t read_shape_31466[3];
    float *read_arr_31467 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31467, read_shape_31466, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31468;
    int64_t read_shape_31469[3];
    float *read_arr_31470 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31470, read_shape_31469, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31471;
    int64_t read_shape_31472[2];
    float *read_arr_31473 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31473, read_shape_31472, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31474;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31465 = futhark_new_f32_3d(ctx, read_arr_31467,
                                                      read_shape_31466[0],
                                                      read_shape_31466[1],
                                                      read_shape_31466[2])) !=
            0);
        assert((read_value_31468 = futhark_new_f32_3d(ctx, read_arr_31470,
                                                      read_shape_31469[0],
                                                      read_shape_31469[1],
                                                      read_shape_31469[2])) !=
            0);
        assert((read_value_31471 = futhark_new_f32_2d(ctx, read_arr_31473,
                                                      read_shape_31472[0],
                                                      read_shape_31472[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t4(ctx, &result_31474, read_value_31465,
                             read_value_31468, read_value_31471);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31465) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31468) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31471) == 0);
        assert(futhark_free_f32_3d(ctx, result_31474) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31465 = futhark_new_f32_3d(ctx, read_arr_31467,
                                                      read_shape_31466[0],
                                                      read_shape_31466[1],
                                                      read_shape_31466[2])) !=
            0);
        assert((read_value_31468 = futhark_new_f32_3d(ctx, read_arr_31470,
                                                      read_shape_31469[0],
                                                      read_shape_31469[1],
                                                      read_shape_31469[2])) !=
            0);
        assert((read_value_31471 = futhark_new_f32_2d(ctx, read_arr_31473,
                                                      read_shape_31472[0],
                                                      read_shape_31472[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t4(ctx, &result_31474, read_value_31465,
                             read_value_31468, read_value_31471);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31465) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31468) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31471) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31474) == 0);
        }
    }
    free(read_arr_31467);
    free(read_arr_31470);
    free(read_arr_31473);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31474)[0] *
                            futhark_shape_f32_3d(ctx, result_31474)[1] *
                            futhark_shape_f32_3d(ctx, result_31474)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31474, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31474), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31474) == 0);
}
static void futrts_cli_entry_t5(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31475;
    int64_t read_shape_31476[3];
    float *read_arr_31477 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31477, read_shape_31476, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31478;
    int64_t read_shape_31479[3];
    float *read_arr_31480 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31480, read_shape_31479, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31481;
    int64_t read_shape_31482[2];
    float *read_arr_31483 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31483, read_shape_31482, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31484;
    int64_t read_shape_31485[2];
    float *read_arr_31486 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31486, read_shape_31485, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31487;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31475 = futhark_new_f32_3d(ctx, read_arr_31477,
                                                      read_shape_31476[0],
                                                      read_shape_31476[1],
                                                      read_shape_31476[2])) !=
            0);
        assert((read_value_31478 = futhark_new_f32_3d(ctx, read_arr_31480,
                                                      read_shape_31479[0],
                                                      read_shape_31479[1],
                                                      read_shape_31479[2])) !=
            0);
        assert((read_value_31481 = futhark_new_f32_2d(ctx, read_arr_31483,
                                                      read_shape_31482[0],
                                                      read_shape_31482[1])) !=
            0);
        assert((read_value_31484 = futhark_new_f32_2d(ctx, read_arr_31486,
                                                      read_shape_31485[0],
                                                      read_shape_31485[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t5(ctx, &result_31487, read_value_31475,
                             read_value_31478, read_value_31481,
                             read_value_31484);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31475) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31478) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31481) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31484) == 0);
        assert(futhark_free_f32_3d(ctx, result_31487) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31475 = futhark_new_f32_3d(ctx, read_arr_31477,
                                                      read_shape_31476[0],
                                                      read_shape_31476[1],
                                                      read_shape_31476[2])) !=
            0);
        assert((read_value_31478 = futhark_new_f32_3d(ctx, read_arr_31480,
                                                      read_shape_31479[0],
                                                      read_shape_31479[1],
                                                      read_shape_31479[2])) !=
            0);
        assert((read_value_31481 = futhark_new_f32_2d(ctx, read_arr_31483,
                                                      read_shape_31482[0],
                                                      read_shape_31482[1])) !=
            0);
        assert((read_value_31484 = futhark_new_f32_2d(ctx, read_arr_31486,
                                                      read_shape_31485[0],
                                                      read_shape_31485[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t5(ctx, &result_31487, read_value_31475,
                             read_value_31478, read_value_31481,
                             read_value_31484);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31475) == 0);
        assert(futhark_free_f32_3d(ctx, read_value_31478) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31481) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31484) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31487) == 0);
        }
    }
    free(read_arr_31477);
    free(read_arr_31480);
    free(read_arr_31483);
    free(read_arr_31486);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31487)[0] *
                            futhark_shape_f32_3d(ctx, result_31487)[1] *
                            futhark_shape_f32_3d(ctx, result_31487)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31487, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31487), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31487) == 0);
}
static void futrts_cli_entry_t6(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    float read_value_31488;
    
    if (read_scalar(&f32_info, &read_value_31488) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31489;
    int64_t read_shape_31490[3];
    float *read_arr_31491 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31491, read_shape_31490, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    float read_value_31492;
    
    if (read_scalar(&f32_info, &read_value_31492) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31493;
    int64_t read_shape_31494[3];
    float *read_arr_31495 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31495, read_shape_31494, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
              "[][][]", f32_info.type_name, strerror(errno));
    
    float read_value_31496;
    
    if (read_scalar(&f32_info, &read_value_31496) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31497;
    int64_t read_shape_31498[2];
    float *read_arr_31499 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31499, read_shape_31498, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[][]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31500;
    
    if (read_scalar(&f32_info, &read_value_31500) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 6,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31501;
    int64_t read_shape_31502[2];
    float *read_arr_31503 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31503, read_shape_31502, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31504;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        assert((read_value_31489 = futhark_new_f32_3d(ctx, read_arr_31491,
                                                      read_shape_31490[0],
                                                      read_shape_31490[1],
                                                      read_shape_31490[2])) !=
            0);
        ;
        assert((read_value_31493 = futhark_new_f32_3d(ctx, read_arr_31495,
                                                      read_shape_31494[0],
                                                      read_shape_31494[1],
                                                      read_shape_31494[2])) !=
            0);
        ;
        assert((read_value_31497 = futhark_new_f32_2d(ctx, read_arr_31499,
                                                      read_shape_31498[0],
                                                      read_shape_31498[1])) !=
            0);
        ;
        assert((read_value_31501 = futhark_new_f32_2d(ctx, read_arr_31503,
                                                      read_shape_31502[0],
                                                      read_shape_31502[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t6(ctx, &result_31504, read_value_31488,
                             read_value_31489, read_value_31492,
                             read_value_31493, read_value_31496,
                             read_value_31497, read_value_31500,
                             read_value_31501);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31489) == 0);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31493) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31497) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31501) == 0);
        assert(futhark_free_f32_3d(ctx, result_31504) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        assert((read_value_31489 = futhark_new_f32_3d(ctx, read_arr_31491,
                                                      read_shape_31490[0],
                                                      read_shape_31490[1],
                                                      read_shape_31490[2])) !=
            0);
        ;
        assert((read_value_31493 = futhark_new_f32_3d(ctx, read_arr_31495,
                                                      read_shape_31494[0],
                                                      read_shape_31494[1],
                                                      read_shape_31494[2])) !=
            0);
        ;
        assert((read_value_31497 = futhark_new_f32_2d(ctx, read_arr_31499,
                                                      read_shape_31498[0],
                                                      read_shape_31498[1])) !=
            0);
        ;
        assert((read_value_31501 = futhark_new_f32_2d(ctx, read_arr_31503,
                                                      read_shape_31502[0],
                                                      read_shape_31502[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t6(ctx, &result_31504, read_value_31488,
                             read_value_31489, read_value_31492,
                             read_value_31493, read_value_31496,
                             read_value_31497, read_value_31500,
                             read_value_31501);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31489) == 0);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31493) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31497) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31501) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31504) == 0);
        }
    }
    ;
    free(read_arr_31491);
    ;
    free(read_arr_31495);
    ;
    free(read_arr_31499);
    ;
    free(read_arr_31503);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31504)[0] *
                            futhark_shape_f32_3d(ctx, result_31504)[1] *
                            futhark_shape_f32_3d(ctx, result_31504)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31504, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31504), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31504) == 0);
}
static void futrts_cli_entry_t7(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    float read_value_31505;
    
    if (read_scalar(&f32_info, &read_value_31505) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31506;
    int64_t read_shape_31507[3];
    float *read_arr_31508 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31508, read_shape_31507, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1,
              "[][][]", f32_info.type_name, strerror(errno));
    
    float read_value_31509;
    
    if (read_scalar(&f32_info, &read_value_31509) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *read_value_31510;
    int64_t read_shape_31511[3];
    float *read_arr_31512 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31512, read_shape_31511, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3,
              "[][][]", f32_info.type_name, strerror(errno));
    
    float read_value_31513;
    
    if (read_scalar(&f32_info, &read_value_31513) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31514;
    int64_t read_shape_31515[2];
    float *read_arr_31516 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31516, read_shape_31515, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[][]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31517;
    
    if (read_scalar(&f32_info, &read_value_31517) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 6,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31518;
    int64_t read_shape_31519[2];
    float *read_arr_31520 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31520, read_shape_31519, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[][]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31521;
    
    if (read_scalar(&f32_info, &read_value_31521) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 8,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31522;
    int64_t read_shape_31523[1];
    float *read_arr_31524 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31524, read_shape_31523, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 9, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31525;
    
    if (read_scalar(&f32_info, &read_value_31525) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 10,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31526;
    int64_t read_shape_31527[1];
    float *read_arr_31528 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31528, read_shape_31527, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 11, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31529;
    
    if (read_scalar(&f32_info, &read_value_31529) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 12,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31530;
    int64_t read_shape_31531[1];
    float *read_arr_31532 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31532, read_shape_31531, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 13, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_31533;
    
    if (read_scalar(&f32_info, &read_value_31533) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 14,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_31534;
    int64_t read_shape_31535[1];
    float *read_arr_31536 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31536, read_shape_31535, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 15, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31537;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        assert((read_value_31506 = futhark_new_f32_3d(ctx, read_arr_31508,
                                                      read_shape_31507[0],
                                                      read_shape_31507[1],
                                                      read_shape_31507[2])) !=
            0);
        ;
        assert((read_value_31510 = futhark_new_f32_3d(ctx, read_arr_31512,
                                                      read_shape_31511[0],
                                                      read_shape_31511[1],
                                                      read_shape_31511[2])) !=
            0);
        ;
        assert((read_value_31514 = futhark_new_f32_2d(ctx, read_arr_31516,
                                                      read_shape_31515[0],
                                                      read_shape_31515[1])) !=
            0);
        ;
        assert((read_value_31518 = futhark_new_f32_2d(ctx, read_arr_31520,
                                                      read_shape_31519[0],
                                                      read_shape_31519[1])) !=
            0);
        ;
        assert((read_value_31522 = futhark_new_f32_1d(ctx, read_arr_31524,
                                                      read_shape_31523[0])) !=
            0);
        ;
        assert((read_value_31526 = futhark_new_f32_1d(ctx, read_arr_31528,
                                                      read_shape_31527[0])) !=
            0);
        ;
        assert((read_value_31530 = futhark_new_f32_1d(ctx, read_arr_31532,
                                                      read_shape_31531[0])) !=
            0);
        ;
        assert((read_value_31534 = futhark_new_f32_1d(ctx, read_arr_31536,
                                                      read_shape_31535[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t7(ctx, &result_31537, read_value_31505,
                             read_value_31506, read_value_31509,
                             read_value_31510, read_value_31513,
                             read_value_31514, read_value_31517,
                             read_value_31518, read_value_31521,
                             read_value_31522, read_value_31525,
                             read_value_31526, read_value_31529,
                             read_value_31530, read_value_31533,
                             read_value_31534);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31506) == 0);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31510) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31514) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31518) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31522) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31526) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31530) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31534) == 0);
        assert(futhark_free_f32_3d(ctx, result_31537) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        assert((read_value_31506 = futhark_new_f32_3d(ctx, read_arr_31508,
                                                      read_shape_31507[0],
                                                      read_shape_31507[1],
                                                      read_shape_31507[2])) !=
            0);
        ;
        assert((read_value_31510 = futhark_new_f32_3d(ctx, read_arr_31512,
                                                      read_shape_31511[0],
                                                      read_shape_31511[1],
                                                      read_shape_31511[2])) !=
            0);
        ;
        assert((read_value_31514 = futhark_new_f32_2d(ctx, read_arr_31516,
                                                      read_shape_31515[0],
                                                      read_shape_31515[1])) !=
            0);
        ;
        assert((read_value_31518 = futhark_new_f32_2d(ctx, read_arr_31520,
                                                      read_shape_31519[0],
                                                      read_shape_31519[1])) !=
            0);
        ;
        assert((read_value_31522 = futhark_new_f32_1d(ctx, read_arr_31524,
                                                      read_shape_31523[0])) !=
            0);
        ;
        assert((read_value_31526 = futhark_new_f32_1d(ctx, read_arr_31528,
                                                      read_shape_31527[0])) !=
            0);
        ;
        assert((read_value_31530 = futhark_new_f32_1d(ctx, read_arr_31532,
                                                      read_shape_31531[0])) !=
            0);
        ;
        assert((read_value_31534 = futhark_new_f32_1d(ctx, read_arr_31536,
                                                      read_shape_31535[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t7(ctx, &result_31537, read_value_31505,
                             read_value_31506, read_value_31509,
                             read_value_31510, read_value_31513,
                             read_value_31514, read_value_31517,
                             read_value_31518, read_value_31521,
                             read_value_31522, read_value_31525,
                             read_value_31526, read_value_31529,
                             read_value_31530, read_value_31533,
                             read_value_31534);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31506) == 0);
        ;
        assert(futhark_free_f32_3d(ctx, read_value_31510) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31514) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_31518) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31522) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31526) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31530) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_31534) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31537) == 0);
        }
    }
    ;
    free(read_arr_31508);
    ;
    free(read_arr_31512);
    ;
    free(read_arr_31516);
    ;
    free(read_arr_31520);
    ;
    free(read_arr_31524);
    ;
    free(read_arr_31528);
    ;
    free(read_arr_31532);
    ;
    free(read_arr_31536);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31537)[0] *
                            futhark_shape_f32_3d(ctx, result_31537)[1] *
                            futhark_shape_f32_3d(ctx, result_31537)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31537, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31537), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31537) == 0);
}
static void futrts_cli_entry_t8(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_3d *read_value_31538;
    int64_t read_shape_31539[3];
    float *read_arr_31540 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31540, read_shape_31539, 3) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0,
              "[][][]", f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31541;
    int64_t read_shape_31542[2];
    float *read_arr_31543 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31543, read_shape_31542, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_31544;
    int64_t read_shape_31545[2];
    float *read_arr_31546 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_31546, read_shape_31545, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_3d *result_31547;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_31538 = futhark_new_f32_3d(ctx, read_arr_31540,
                                                      read_shape_31539[0],
                                                      read_shape_31539[1],
                                                      read_shape_31539[2])) !=
            0);
        assert((read_value_31541 = futhark_new_f32_2d(ctx, read_arr_31543,
                                                      read_shape_31542[0],
                                                      read_shape_31542[1])) !=
            0);
        assert((read_value_31544 = futhark_new_f32_2d(ctx, read_arr_31546,
                                                      read_shape_31545[0],
                                                      read_shape_31545[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t8(ctx, &result_31547, read_value_31538,
                             read_value_31541, read_value_31544);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31538) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31541) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31544) == 0);
        assert(futhark_free_f32_3d(ctx, result_31547) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_31538 = futhark_new_f32_3d(ctx, read_arr_31540,
                                                      read_shape_31539[0],
                                                      read_shape_31539[1],
                                                      read_shape_31539[2])) !=
            0);
        assert((read_value_31541 = futhark_new_f32_2d(ctx, read_arr_31543,
                                                      read_shape_31542[0],
                                                      read_shape_31542[1])) !=
            0);
        assert((read_value_31544 = futhark_new_f32_2d(ctx, read_arr_31546,
                                                      read_shape_31545[0],
                                                      read_shape_31545[1])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t8(ctx, &result_31547, read_value_31538,
                             read_value_31541, read_value_31544);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_3d(ctx, read_value_31538) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31541) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_31544) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_3d(ctx, result_31547) == 0);
        }
    }
    free(read_arr_31540);
    free(read_arr_31543);
    free(read_arr_31546);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_3d(ctx,
                                                                result_31547)[0] *
                            futhark_shape_f32_3d(ctx, result_31547)[1] *
                            futhark_shape_f32_3d(ctx, result_31547)[2]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_3d(ctx, result_31547, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_3d(ctx, result_31547), 3);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_3d(ctx, result_31547) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="t9", .fun =
                                                futrts_cli_entry_t9}, {.name =
                                                                       "t1",
                                                                       .fun =
                                                                       futrts_cli_entry_t1},
                                               {.name ="t2", .fun =
                                                futrts_cli_entry_t2}, {.name =
                                                                       "t10",
                                                                       .fun =
                                                                       futrts_cli_entry_t10},
                                               {.name ="t11", .fun =
                                                futrts_cli_entry_t11}, {.name =
                                                                        "t3",
                                                                        .fun =
                                                                        futrts_cli_entry_t3},
                                               {.name ="t4", .fun =
                                                futrts_cli_entry_t4}, {.name =
                                                                       "t5",
                                                                       .fun =
                                                                       futrts_cli_entry_t5},
                                               {.name ="t6", .fun =
                                                futrts_cli_entry_t6}, {.name =
                                                                       "t7",
                                                                       .fun =
                                                                       futrts_cli_entry_t7},
                                               {.name ="t8", .fun =
                                                futrts_cli_entry_t8}};
    struct futhark_context_config *cfg = futhark_context_config_new();
    
    assert(cfg != NULL);
    
    int parsed_options = parse_options(cfg, argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    if (argc != 0)
        panic(1, "Excess non-option: %s\n", argv[0]);
    
    struct futhark_context *ctx = futhark_context_new(cfg);
    
    assert(ctx != NULL);
    
    int num_entry_points = sizeof(entry_points) / sizeof(entry_points[0]);
    entry_point_fun *entry_point_fun = NULL;
    
    for (int i = 0; i < num_entry_points; i++) {
        if (strcmp(entry_points[i].name, entry_point) == 0) {
            entry_point_fun = entry_points[i].fun;
            break;
        }
    }
    if (entry_point_fun == NULL) {
        fprintf(stderr,
                "No entry point '%s'.  Select another with --entry-point.  Options are:\n",
                entry_point);
        for (int i = 0; i < num_entry_points; i++)
            fprintf(stderr, "%s\n", entry_points[i].name);
        return 1;
    }
    entry_point_fun(ctx);
    if (runtime_file != NULL)
        fclose(runtime_file);
    futhark_debugging_report(ctx);
    futhark_context_free(ctx);
    futhark_context_config_free(cfg);
    return 0;
}
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
  *lock = CreateMutex(NULL,  /* Default security attributes. */
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

/* The simple OpenCL runtime framework used by Futhark. */

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define OPENCL_SUCCEED(e) opencl_succeed(e, #e, __FILE__, __LINE__)

struct opencl_config {
  int debugging;
  int logging;
  int preferred_device_num;
  const char *preferred_platform;
  const char *preferred_device;

  const char* dump_program_to;
  const char* load_program_from;

  size_t default_group_size;
  size_t default_num_groups;
  size_t default_tile_size;
  size_t default_threshold;
  size_t transpose_block_dim;

  int default_group_size_changed;
  int default_tile_size_changed;

  int num_sizes;
  const char **size_names;
  size_t *size_values;
  const char **size_classes;
  const char **size_entry_points;
};

void opencl_config_init(struct opencl_config *cfg,
                        int num_sizes,
                        const char *size_names[],
                        size_t *size_values,
                        const char *size_classes[],
                        const char *size_entry_points[]) {
  cfg->debugging = 0;
  cfg->logging = 0;
  cfg->preferred_device_num = 0;
  cfg->preferred_platform = "";
  cfg->preferred_device = "";
  cfg->dump_program_to = NULL;
  cfg->load_program_from = NULL;

  cfg->default_group_size = 256;
  cfg->default_num_groups = 128;
  cfg->default_tile_size = 32;
  cfg->default_threshold = 32*1024;
  cfg->transpose_block_dim = 16;

  cfg->default_group_size_changed = 0;
  cfg->default_tile_size_changed = 0;

  cfg->num_sizes = num_sizes;
  cfg->size_names = size_names;
  cfg->size_values = size_values;
  cfg->size_classes = size_classes;
  cfg->size_entry_points = size_entry_points;
}

/* An entry in the free list.  May be invalid, to avoid having to
   deallocate entries as soon as they are removed.  There is also a
   tag, to help with memory reuse. */
struct opencl_free_list_entry {
  size_t size;
  cl_mem mem;
  const char *tag;
  unsigned char valid;
};

struct opencl_free_list {
  struct opencl_free_list_entry *entries; // Pointer to entries.
  int capacity;                           // Number of entries.
  int used;                               // Number of valid entries.
};

void free_list_init(struct opencl_free_list *l) {
  l->capacity = 30; // Picked arbitrarily.
  l->used = 0;
  l->entries = malloc(sizeof(struct opencl_free_list_entry) * l->capacity);
  for (int i = 0; i < l->capacity; i++) {
    l->entries[i].valid = 0;
  }
}

/* Remove invalid entries from the free list. */
void free_list_pack(struct opencl_free_list *l) {
  int p = 0;
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[p] = l->entries[i];
      p++;
    }
  }
  // Now p == l->used.
  l->entries = realloc(l->entries, l->used * sizeof(struct opencl_free_list_entry));
  l->capacity = l->used;
}

void free_list_destroy(struct opencl_free_list *l) {
  assert(l->used == 0);
  free(l->entries);
}

int free_list_find_invalid(struct opencl_free_list *l) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (!l->entries[i].valid) {
      break;
    }
  }
  return i;
}

void free_list_insert(struct opencl_free_list *l, size_t size, cl_mem mem, const char *tag) {
  int i = free_list_find_invalid(l);

  if (i == l->capacity) {
    // List is full; so we have to grow it.
    int new_capacity = l->capacity * 2 * sizeof(struct opencl_free_list_entry);
    l->entries = realloc(l->entries, new_capacity);
    for (int j = 0; j < l->capacity; j++) {
      l->entries[j+l->capacity].valid = 0;
    }
    l->capacity *= 2;
  }

  // Now 'i' points to the first invalid entry.
  l->entries[i].valid = 1;
  l->entries[i].size = size;
  l->entries[i].mem = mem;
  l->entries[i].tag = tag;

  l->used++;
}

/* Find and remove a memory block of at least the desired size and
   tag.  Returns 0 on success.  */
int free_list_find(struct opencl_free_list *l, const char *tag, size_t *size_out, cl_mem *mem_out) {
  int i;
  for (i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid && l->entries[i].tag == tag) {
      l->entries[i].valid = 0;
      *size_out = l->entries[i].size;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

/* Remove the first block in the free list.  Returns 0 if a block was
   removed, and nonzero if the free list was already empty. */
int free_list_first(struct opencl_free_list *l, cl_mem *mem_out) {
  for (int i = 0; i < l->capacity; i++) {
    if (l->entries[i].valid) {
      l->entries[i].valid = 0;
      *mem_out = l->entries[i].mem;
      l->used--;
      return 0;
    }
  }

  return 1;
}

struct opencl_context {
  cl_platform_id platform;
  cl_device_id device;
  cl_context ctx;
  cl_command_queue queue;

  struct opencl_config cfg;

  struct opencl_free_list free_list;

  size_t max_group_size;
  size_t max_num_groups;
  size_t max_tile_size;
  size_t max_threshold;

  size_t lockstep_width;
};

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_context*, struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed(unsigned int ret,
                           const char *call,
                           const char *file,
                           int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

void set_preferred_platform(struct opencl_config *cfg, const char *s) {
  cfg->preferred_platform = s;
}

void set_preferred_device(struct opencl_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  cfg->preferred_device = s;
  cfg->preferred_device_num = x;
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static int is_blacklisted(const char *platform_name, const char *device_name,
                          const struct opencl_config *cfg) {
  if (strcmp(cfg->preferred_platform, "") != 0 ||
      strcmp(cfg->preferred_device, "") != 0) {
    return 0;
  } else if (strstr(platform_name, "Apple") != NULL &&
             strstr(device_name, "Intel(R) Core(TM)") != NULL) {
    return 1;
  } else {
    return 0;
  }
}

static struct opencl_device_option get_preferred_device(const struct opencl_config *cfg) {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  int num_device_matches = 0;

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (!is_blacklisted(device.platform_name, device.device_name, cfg) &&
        strstr(device.platform_name, cfg->preferred_platform) != NULL &&
        strstr(device.device_name, cfg->preferred_device) != NULL &&
        num_device_matches++ == cfg->preferred_device_num) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.\n");
  exit(1); // Never reached
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int ret_val = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE) {
    assert(ret_val == 0);
  }

  cl_build_status build_status;
  ret_val = clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status),
                                  &build_status,
                                  NULL);
  assert(ret_val == 0);

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    ret_val = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    assert(ret_val == 0);

    build_log = malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    assert(ret_val == 0);

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

/* Fields in a bitmask indicating which types we must be sure are
   available. */
enum opencl_required_type { OPENCL_F64 = 1 };

// We take as input several strings representing the program, because
// C does not guarantee that the compiler supports particularly large
// literals.  Notably, Visual C has a limit of 2048 characters.  The
// array must be NULL-terminated.
static cl_program setup_opencl(struct opencl_context *ctx,
                               const char *srcs[],
                               int required_types) {

  cl_int error;
  cl_platform_id platform;
  cl_device_id device;
  size_t max_group_size;

  ctx->lockstep_width = 0;

  free_list_init(&ctx->free_list);

  struct opencl_device_option device_option = get_preferred_device(&ctx->cfg);

  if (ctx->cfg.logging) {
    describe_device_option(device_option);
  }

  ctx->device = device = device_option.device;
  ctx->platform = platform = device_option.platform;

  if (required_types & OPENCL_F64) {
    cl_uint supported;
    OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                   sizeof(cl_uint), &supported, NULL));
    if (!supported) {
      panic(1,
            "Program uses double-precision floats, but this is not supported on chosen device: %s\n",
            device_option.device_name);
    }
  }

  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  size_t max_tile_size = sqrt(max_group_size);

  if (max_group_size < ctx->cfg.default_group_size) {
    if (ctx->cfg.default_group_size_changed) {
      fprintf(stderr, "Note: Device limits default group size to %zu (down from %zu).\n",
              max_group_size, ctx->cfg.default_group_size);
    }
    ctx->cfg.default_group_size = max_group_size;
  }

  if (max_tile_size < ctx->cfg.default_tile_size) {
    if (ctx->cfg.default_tile_size_changed) {
      fprintf(stderr, "Note: Device limits default tile size to %zu (down from %zu).\n",
              max_tile_size, ctx->cfg.default_tile_size);
    }
    ctx->cfg.default_tile_size = max_tile_size;
  }

  ctx->max_group_size = max_group_size;
  ctx->max_tile_size = max_tile_size; // No limit.
  ctx->max_threshold = ctx->max_num_groups = 0; // No limit.

  // Now we go through all the sizes, clamp them to the valid range,
  // or set them to the default.
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    const char *size_class = ctx->cfg.size_classes[i];
    size_t *size_value = &ctx->cfg.size_values[i];
    const char* size_name = ctx->cfg.size_names[i];
    size_t max_value, default_value;
    if (strstr(size_class, "group_size") == size_class) {
      max_value = max_group_size;
      default_value = ctx->cfg.default_group_size;
    } else if (strstr(size_class, "num_groups") == size_class) {
      max_value = max_group_size; // Futhark assumes this constraint.
      default_value = ctx->cfg.default_num_groups;
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = sqrt(max_group_size);
      default_value = ctx->cfg.default_tile_size;
    } else if (strstr(size_class, "threshold") == size_class) {
      max_value = 0; // No limit.
      default_value = ctx->cfg.default_threshold;
    } else {
      panic(1, "Unknown size class for size '%s': %s\n", size_name, size_class);
    }
    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %d (down from %d)\n",
              size_name, (int)max_value, (int)*size_value);
      *size_value = max_value;
    }
  }

  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };
  // Note that nVidia's OpenCL requires the platform property
  ctx->ctx = clCreateContext(properties, 1, &device, NULL, NULL, &error);
  assert(error == 0);

  ctx->queue = clCreateCommandQueue(ctx->ctx, device, 0, &error);
  assert(error == 0);

  // Make sure this function is defined.
  post_opencl_setup(ctx, &device_option);

  if (ctx->cfg.logging) {
    fprintf(stderr, "Lockstep width: %d\n", (int)ctx->lockstep_width);
    fprintf(stderr, "Default group size: %d\n", (int)ctx->cfg.default_group_size);
    fprintf(stderr, "Default number of groups: %d\n", (int)ctx->cfg.default_num_groups);
  }

  char *fut_opencl_src = NULL;
  size_t src_size = 0;

  // Maybe we have to read OpenCL source from somewhere else (used for debugging).
  if (ctx->cfg.load_program_from != NULL) {
    FILE *f = fopen(ctx->cfg.load_program_from, "r");
    assert(f != NULL);
    fseek(f, 0, SEEK_END);
    src_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    fut_opencl_src = malloc(src_size);
    assert(fread(fut_opencl_src, 1, src_size, f) == src_size);
    fclose(f);
  } else {
    // Build the OpenCL program.  First we have to concatenate all the fragments.
    for (const char **src = srcs; src && *src; src++) {
      src_size += strlen(*src);
    }

    fut_opencl_src = malloc(src_size + 1);

    size_t n, i;
    for (i = 0, n = 0; srcs && srcs[i]; i++) {
      strncpy(fut_opencl_src+n, srcs[i], src_size-n);
      n += strlen(srcs[i]);
    }
    fut_opencl_src[src_size] = 0;

  }

  cl_program prog;
  error = 0;
  const char* src_ptr[] = {fut_opencl_src};

  if (ctx->cfg.dump_program_to != NULL) {
    FILE *f = fopen(ctx->cfg.dump_program_to, "w");
    assert(f != NULL);
    fputs(fut_opencl_src, f);
    fclose(f);
  }

  prog = clCreateProgramWithSource(ctx->ctx, 1, src_ptr, &src_size, &error);
  assert(error == 0);

  int compile_opts_size = 1024;
  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    compile_opts_size += strlen(ctx->cfg.size_names[i]) + 20;
  }
  char *compile_opts = malloc(compile_opts_size);

  int w = snprintf(compile_opts, compile_opts_size,
                   "-DFUT_BLOCK_DIM=%d -DLOCKSTEP_WIDTH=%d ",
                   (int)ctx->cfg.transpose_block_dim,
                   (int)ctx->lockstep_width);

  for (int i = 0; i < ctx->cfg.num_sizes; i++) {
    w += snprintf(compile_opts+w, compile_opts_size-w,
                  "-D%s=%d ", ctx->cfg.size_names[i],
                  (int)ctx->cfg.size_values[i]);
  }

  OPENCL_SUCCEED(build_opencl_program(prog, device, compile_opts));
  free(compile_opts);
  free(fut_opencl_src);

  return prog;
}

// Allocate memory from driver. The problem is that OpenCL may perform
// lazy allocation, so we cannot know whether an allocation succeeded
// until the first time we try to use it.  Hence we immediately
// perform a write to see if the allocation succeeded.  This is slow,
// but the assumption is that this operation will be rare (most things
// will go through the free list).
int opencl_alloc_actual(struct opencl_context *ctx, size_t size, cl_mem *mem_out) {
  int error;
  *mem_out = clCreateBuffer(ctx->ctx, CL_MEM_READ_WRITE, size, NULL, &error);

  if (error != CL_SUCCESS) {
    return error;
  }

  int x = 2;
  error = clEnqueueWriteBuffer(ctx->queue, *mem_out, 1, 0, sizeof(x), &x, 0, NULL, NULL);

  // No need to wait for completion here. clWaitForEvents() cannot
  // return mem object allocation failures. This implies that the
  // buffer is faulted onto the device on enqueue. (Observation by
  // Andreas Kloeckner.)

  return error;
}

int opencl_alloc(struct opencl_context *ctx, size_t min_size, const char *tag, cl_mem *mem_out) {
  assert(min_size >= 0);
  if (min_size < sizeof(int)) {
    min_size = sizeof(int);
  }

  size_t size;

  if (free_list_find(&ctx->free_list, tag, &size, mem_out) == 0) {
    // Successfully found a free block.  Is it big enough?
    //
    // FIXME: we might also want to check whether the block is *too
    // big*, to avoid internal fragmentation.  However, this can
    // sharply impact performance on programs where arrays change size
    // frequently.  Fortunately, such allocations are usually fairly
    // short-lived, as they are necessarily within a loop, so the risk
    // of internal fragmentation resulting in an OOM situation is
    // limited.  However, it would be preferable if we could go back
    // and *shrink* oversize allocations when we encounter an OOM
    // condition.  That is technically feasible, since we do not
    // expose OpenCL pointer values directly to the application, but
    // instead rely on a level of indirection.
    if (size >= min_size) {
      return CL_SUCCESS;
    } else {
      // Not just right - free it.
      int error = clReleaseMemObject(*mem_out);
      if (error != CL_SUCCESS) {
        return error;
      }
    }
  }

  // We have to allocate a new block from the driver.  If the
  // allocation does not succeed, then we might be in an out-of-memory
  // situation.  We now start freeing things from the free list until
  // we think we have freed enough that the allocation will succeed.
  // Since we don't know how far the allocation is from fitting, we
  // have to check after every deallocation.  This might be pretty
  // expensive.  Let's hope that this case is hit rarely.

  int error = opencl_alloc_actual(ctx, min_size, mem_out);

  while (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
    cl_mem mem;
    if (free_list_first(&ctx->free_list, &mem) == 0) {
      error = clReleaseMemObject(mem);
      if (error != CL_SUCCESS) {
        return error;
      }
    } else {
      break;
    }
    error = opencl_alloc_actual(ctx, min_size, mem_out);
  }

  return error;
}

int opencl_free(struct opencl_context *ctx, cl_mem mem, const char *tag) {
  size_t size;
  cl_mem existing_mem;

  // If there is already a block with this tag, then remove it.
  if (free_list_find(&ctx->free_list, tag, &size, &existing_mem) == 0) {
    int error = clReleaseMemObject(existing_mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  int error = clGetMemObjectInfo(mem, CL_MEM_SIZE, sizeof(size_t), &size, NULL);

  if (error == CL_SUCCESS) {
    free_list_insert(&ctx->free_list, size, mem, tag);
  }

  return error;
}

int opencl_free_all(struct opencl_context *ctx) {
  cl_mem mem;
  free_list_pack(&ctx->free_list);
  while (free_list_first(&ctx->free_list, &mem) == 0) {
    int error = clReleaseMemObject(mem);
    if (error != CL_SUCCESS) {
      return error;
    }
  }

  return CL_SUCCESS;
}

const char *opencl_program[] =
           {"#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable\n__kernel void dummy_kernel(__global unsigned char *dummy, int n)\n{\n    const int thread_gid = get_global_id(0);\n    \n    if (thread_gid >= n)\n        return;\n}\ntypedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\n#define ALIGNED_LOCAL_MEMORY(m,size) __local unsigned char m[size] __attribute__ ((align))\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n",
            "{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t smin8(int8_t x, int8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int16_t smin16(int16_t x, int16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int32_t smin32(int32_t x, int32_t y)\n{\n    ret",
            "urn x < y ? x : y;\n}\nstatic inline int64_t smin64(int64_t x, int64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint8_t umin8(uint8_t x, uint8_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint16_t umin16(uint16_t x, uint16_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint32_t umin32(uint32_t x, uint32_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline uint64_t umin64(uint64_t x, uint64_t y)\n{\n    return x < y ? x : y;\n}\nstatic inline int8_t smax8(int8_t x, int8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int16_t smax16(int16_t x, int16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int32_t smax32(int32_t x, int32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline int64_t smax64(int64_t x, int64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t umax8(uint8_t x, uint8_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint16_t umax16(uint16_t x, uint16_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint32_t umax32(uint32_t x, uint32_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint64_t umax64(uint64_t x, uint64_t y)\n{\n    return x < y ? y : x;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint",
            "8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8",
            "_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_",
            "t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fmin32(float x, float y)\n{\n    return x < y ? x : y;\n}\nstatic inline float fmax32(float x, float y)\n{\n    return x < y ? y : x;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    ret",
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define tile_sizze_30924 (tile_size_30923)\n#define tiled_group_sizze_30925 (tile_size_30923 * tile_size_30923)\n#define tile_sizze_30924 (tile_size_30923)\n#define tiled_group_",
            "sizze_30925 (tile_size_30923 * tile_size_30923)\n#define group_sizze_30373 (group_size_30372)\n#define tile_sizze_30924 (tile_size_30923)\n#define tiled_group_sizze_30925 (tile_size_30923 * tile_size_30923)\n#define tile_sizze_30924 (tile_size_30923)\n#define tiled_group_sizze_30925 (tile_size_30923 * tile_size_30923)\n#define tile_sizze_30924 (tile_size_30923)\n#define tiled_group_sizze_30925 (tile_size_30923 * tile_size_30923)\n#define group_sizze_30874 (group_size_30873)\n__kernel void fut_kernel_map_transpose_f32(__global float *odata,\n                                           uint odata_offset, __global\n                                           float *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size",
            ")\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_f32(__global float *odata,\n                                                     uint odata_offset, __global\n                                                     float *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index",
            ";\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_f32(__global float *odata,\n                                                    uint odata_offset, __global\n                                                    float *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index ",
            "* height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_small_f32(__global float *odata,\n                                                 uint odata_offset, __global\n                                                 float *idata,\n                                                 uint idata_offset,\n                                                 uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n                                                                   width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void map_kernel_30102(int32_t sizze_29398, int32_t sizze_29399,\n                               int32_t sizze_29400, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30984, __global\n                               unsigned char *mem_30987)\n{\n    int32_t wave_sizze_31039;\n    int32_t group_sizze_31040;\n    bool thread_active_31041;\n    int32_t gtid_30095;\n    int32_t global_tid_30102;\n    int32_t local_tid_30103;\n    int32_t group_id_30104;\n    \n    global_tid_30102 ",
            "= get_global_id(0);\n    local_tid_30103 = get_local_id(0);\n    group_sizze_31040 = get_local_size(0);\n    wave_sizze_31039 = LOCKSTEP_WIDTH;\n    group_id_30104 = get_group_id(0);\n    gtid_30095 = global_tid_30102;\n    thread_active_31041 = slt32(gtid_30095, sizze_29398);\n    \n    float res_30106;\n    \n    if (thread_active_31041) {\n        float x_30109 = 0.0F;\n        \n        for (int32_t chunk_offset_30108 = 0; chunk_offset_30108 < sizze_29400;\n             chunk_offset_30108++) {\n            float x_30118 = *(__global float *) &mem_30984[(chunk_offset_30108 *\n                                                            sizze_29399 +\n                                                            gtid_30095) * 4];\n            float x_30119 = *(__global\n                              float *) &C_mem_30980[chunk_offset_30108 * 4];\n            float res_30121 = x_30118 * x_30119;\n            float res_30123 = x_30109 + res_30121;\n            float x_tmp_31042 = res_30123;\n            \n            x_30109 = x_tmp_31042;\n        }\n        res_30106 = x_30109;\n    }\n    if (thread_active_31041) {\n        *(__global float *) &mem_30987[gtid_30095 * 4] = res_30106;\n    }\n}\n__kernel void map_kernel_30144(int32_t sizze_29396, int32_t sizze_29397,\n                               int32_t sizze_29398, __global\n                               unsigned char *mem_30987, __global\n                               unsigned char *mem_30992, __global\n                               unsigned char *mem_30996)\n{\n    int32_t wave_sizze_31043;\n    int32_t group_sizze_31044;\n    bool thread_active_31045;\n    int32_t gtid_30135;\n    int32_t gtid_30136;\n    int32_t global_tid_30144;\n    int32_t local_tid_30145;\n    int32_t group_id_30146;\n    \n    global_tid_30144 = get_global_id(0);\n    local_tid_30145 = get_local_id(0);\n    group_sizze_31044 = get_local_size(0);\n    wave_sizze_31043 = LOCKSTEP_WIDTH;\n    group_id_30146 = get_group_id(0);\n    gtid_30135 = squot32(global_tid_30144, sizze_29397);\n    gt",
            "id_30136 = global_tid_30144 - squot32(global_tid_30144, sizze_29397) *\n        sizze_29397;\n    thread_active_31045 = slt32(gtid_30135, sizze_29396) && slt32(gtid_30136,\n                                                                  sizze_29397);\n    \n    float res_30148;\n    \n    if (thread_active_31045) {\n        float x_30151 = 0.0F;\n        \n        for (int32_t chunk_offset_30150 = 0; chunk_offset_30150 < sizze_29398;\n             chunk_offset_30150++) {\n            float x_30160 = *(__global float *) &mem_30992[(chunk_offset_30150 *\n                                                            (sizze_29396 *\n                                                             sizze_29397) +\n                                                            gtid_30135 *\n                                                            sizze_29397 +\n                                                            gtid_30136) * 4];\n            float x_30161 = *(__global float *) &mem_30987[chunk_offset_30150 *\n                                                           4];\n            float res_30163 = x_30160 * x_30161;\n            float res_30165 = x_30151 + res_30163;\n            float x_tmp_31046 = res_30165;\n            \n            x_30151 = x_tmp_31046;\n        }\n        res_30148 = x_30151;\n    }\n    if (thread_active_31045) {\n        *(__global float *) &mem_30996[(gtid_30135 * sizze_29397 + gtid_30136) *\n                                       4] = res_30148;\n    }\n}\n__kernel void map_kernel_30188(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29441, int32_t sizze_29442,\n                               int32_t sizze_29443, int32_t sizze_29445,\n                               __global unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *mem_30991)\n{\n    __local volatile char *r",
            "estrict mem_30982 = mem_aligned_0;\n    __local volatile char *restrict mem_30986 = mem_aligned_1;\n    int32_t wave_sizze_31052;\n    int32_t group_sizze_31053;\n    bool thread_active_31054;\n    int32_t gtid_30177;\n    int32_t gtid_30178;\n    int32_t gtid_30179;\n    int32_t global_tid_30188;\n    int32_t local_tid_30189;\n    int32_t group_id_30190;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t ltid_30928;\n    \n    global_tid_30188 = get_global_id(0);\n    local_tid_30189 = get_local_id(0);\n    group_sizze_31053 = get_local_size(0);\n    wave_sizze_31052 = LOCKSTEP_WIDTH;\n    group_id_30190 = get_group_id(0);\n    gtid_30177 = squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924) + squot32(squot32(global_tid_30188,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29442 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29445 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924));\n    gtid_30178 = squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                ",
            "      tile_sizze_30924),\n                         tile_sizze_30924) + squot32(squot32(global_tid_30188,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924) -\n                                                     squot32(squot32(global_tid_30188,\n                                                                     tile_sizze_30924 *\n                                                                     tile_sizze_30924),\n                                                             squot32(sizze_29442 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924) *\n                                                             squot32(sizze_29445 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924)) *\n                                                     (squot32(sizze_29442 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924) *\n                                                      squot32(sizze_29445 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924)),\n                                                     squot32(sizze_29445 +\n                                                             tile_sizze_30924 -\n                                            ",
            "                 1,\n                                                             tile_sizze_30924)) *\n        tile_sizze_30924;\n    gtid_30179 = srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924 +\n        (squot32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924) -\n         squot32(squot32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924),\n                 squot32(sizze_29442 + tile_sizze_30924 - 1, tile_sizze_30924) *\n                 squot32(sizze_29445 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * (squot32(sizze_29442 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924) *\n                                               squot32(sizze_29445 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924)) -\n         squot32(squot32(global_tid_30188, tile_sizze_30924 *\n                         tile_sizze_30924) - squot32(squot32(global_tid_30188,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29442 +\n                                             ",
            "                tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29445 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n                 (squot32(sizze_29442 + tile_sizze_30924 - 1,\n                          tile_sizze_30924) * squot32(sizze_29445 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)),\n                 squot32(sizze_29445 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * squot32(sizze_29445 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)) *\n        tile_sizze_30924;\n    ltid_30926 = squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924);\n    ltid_30927 = squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924);\n    ltid_30928 = srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                 ",
            "       tile_sizze_30924) -\n        squot32(srem32(global_tid_30188, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30188, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924;\n    thread_active_31054 = (slt32(gtid_30177, sizze_29441) && slt32(gtid_30178,\n                                                                   sizze_29442)) &&\n        slt32(gtid_30179, sizze_29445);\n    if (thread_active_31054) { }\n    \n    float res_30193;\n    float x_30196 = 0.0F;\n    int32_t chunk_sizze_30194;\n    int32_t chunk_offset_30195 = 0;\n    \n    while (slt32(chunk_offset_30195, sizze_29443)) {\n        if (slt32(sizze_29443 - chunk_offset_30195, tile_sizze_30924)) {\n            chunk_sizze_30194 = sizze_29443 - chunk_offset_30195;\n        } else {\n            chunk_sizze_30194 = tile_sizze_30924;\n        }\n        for (int32_t comb_iter_31055 = 0; comb_iter_31055 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31055++) {\n            int32_t cid_30944;\n            int32_t cid_30945;\n            int32_t flat_comb_id_31056 = comb_iter_31055 *\n                    tiled_group_sizze_30925 + local_tid_30189;\n            \n            cid_30944 = squot32(flat_comb_id_31056, tile_sizze_30924);\n            cid_30945 = flat_comb_id_31056 - squot32(flat_comb_id_31056,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30944, tile_sizze_30924) && slt32(cid_30945,\n                                                             chunk_sizze_30194)) &&\n                slt32(gtid_30178, sizze_29442)) {\n                float x_chunk_outer_elem_30943 = *(__global\n  ",
            "                                                 float *) &A_mem_30976[(gtid_30177 *\n                                                                          (sizze_29442 *\n                                                                           sizze_29443) +\n                                                                          gtid_30178 *\n                                                                          sizze_29443 +\n                                                                          (chunk_offset_30195 +\n                                                                           ltid_30928)) *\n                                                                         4];\n                \n                *(__local float *) &mem_30982[(cid_30944 * tile_sizze_30924 +\n                                               cid_30945) * 4] =\n                    x_chunk_outer_elem_30943;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31054) { }\n        for (int32_t comb_iter_31057 = 0; comb_iter_31057 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31057++) {\n            int32_t cid_30949;\n            int32_t cid_30950;\n            int32_t flat_comb_id_31058 = comb_iter_31057 *\n                    tiled_group_sizze_30925 + local_tid_30189;\n            \n            cid_30949 = squot32(flat_comb_id_31058, tile_sizze_30924);\n            cid_30950 = flat_comb_id_31058 - squot32(flat_comb_id_31058,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30949, chunk_sizze_30194) && slt32(cid_30950,\n                                                              tile_sizze_30924)) &&\n                slt32(gtid_30179, sizze_29445)) {\n                float x_chunk_outer_elem_30948 = *(__global\n                                                 ",
            "  float *) &B_mem_30978[((chunk_offset_30195 +\n                                                                           ltid_30927) *\n                                                                          sizze_29445 +\n                                                                          gtid_30179) *\n                                                                         4];\n                \n                *(__local float *) &mem_30986[(cid_30949 * tile_sizze_30924 +\n                                               cid_30950) * 4] =\n                    x_chunk_outer_elem_30948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31054) { }\n        \n        float res_30199;\n        float sync_30952;\n        float acc_30202 = x_30196;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30200 = 1;\n        \n        if (thread_active_31054) {\n            if (chunk_sizze_30194 == tile_sizze_30924) {\n                for (int32_t i_30201 = 0; i_30201 < tile_sizze_30924;\n                     i_30201++) {\n                    float x_30205 = *(__local float *) &mem_30982[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30201) *\n                                                                  4];\n                    float x_30206 = *(__local float *) &mem_30986[(i_30201 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30208 = x_30205 * x_30206;\n                    float res_30210 = acc_30202 + res_30208;\n                    float acc_tmp_31059 = res_30210;\n                    \n                    acc_30202 = acc_tmp_31059;\n                }\n            } else {\n                for (int32_t i_302",
            "01 = 0; i_30201 < chunk_sizze_30194;\n                     i_30201++) {\n                    float x_30205 = *(__local float *) &mem_30982[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30201) *\n                                                                  4];\n                    float x_30206 = *(__local float *) &mem_30986[(i_30201 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30208 = x_30205 * x_30206;\n                    float res_30210 = acc_30202 + res_30208;\n                    float acc_tmp_31059 = res_30210;\n                    \n                    acc_30202 = acc_tmp_31059;\n                }\n            }\n        }\n        res_30199 = acc_30202;\n        sync_30952 = res_30199;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30196 = sync_30952;\n        chunk_offset_30195 += tile_sizze_30924;\n    }\n    res_30193 = x_30196;\n    if (thread_active_31054) {\n        *(__global float *) &mem_30991[(gtid_30177 * (sizze_29442 *\n                                                      sizze_29445) +\n                                        gtid_30178 * sizze_29445 + gtid_30179) *\n                                       4] = res_30193;\n    }\n}\n__kernel void map_kernel_30220(int32_t sizze_29474, int32_t sizze_29476,\n                               __global unsigned char *B_mem_30978, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30984)\n{\n    int32_t wave_sizze_31065;\n    int32_t group_sizze_31066;\n    bool thread_active_31067;\n    int32_t gtid_30211;\n    int32_t gtid_30212;\n    int32_t global_tid_30220;\n    int32_t local_tid_30221;\n    int32_t group_id_30",
            "222;\n    \n    global_tid_30220 = get_global_id(0);\n    local_tid_30221 = get_local_id(0);\n    group_sizze_31066 = get_local_size(0);\n    wave_sizze_31065 = LOCKSTEP_WIDTH;\n    group_id_30222 = get_group_id(0);\n    gtid_30211 = squot32(global_tid_30220, sizze_29476);\n    gtid_30212 = global_tid_30220 - squot32(global_tid_30220, sizze_29476) *\n        sizze_29476;\n    thread_active_31067 = slt32(gtid_30211, sizze_29474) && slt32(gtid_30212,\n                                                                  sizze_29476);\n    \n    float x_30223;\n    float x_30224;\n    float res_30225;\n    \n    if (thread_active_31067) {\n        x_30223 = *(__global float *) &B_mem_30978[gtid_30211 * 4];\n        x_30224 = *(__global float *) &C_mem_30980[gtid_30212 * 4];\n        res_30225 = x_30223 * x_30224;\n    }\n    if (thread_active_31067) {\n        *(__global float *) &mem_30984[(gtid_30211 * sizze_29476 + gtid_30212) *\n                                       4] = res_30225;\n    }\n}\n__kernel void map_kernel_30248(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29472, int32_t sizze_29473,\n                               int32_t sizze_29474, int32_t sizze_29476,\n                               __global unsigned char *A_mem_30976, __global\n                               unsigned char *mem_30984, __global\n                               unsigned char *mem_30997)\n{\n    __local volatile char *restrict mem_30988 = mem_aligned_0;\n    __local volatile char *restrict mem_30992 = mem_aligned_1;\n    int32_t wave_sizze_31068;\n    int32_t group_sizze_31069;\n    bool thread_active_31070;\n    int32_t gtid_30237;\n    int32_t gtid_30238;\n    int32_t gtid_30239;\n    int32_t global_tid_30248;\n    int32_t local_tid_30249;\n    int32_t group_id_30250;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t ltid_30928;\n    \n    global_tid_30248 = get_global_id(0);\n    local_tid_30249 = get_local_id(",
            "0);\n    group_sizze_31069 = get_local_size(0);\n    wave_sizze_31068 = LOCKSTEP_WIDTH;\n    group_id_30250 = get_group_id(0);\n    gtid_30237 = squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924) + squot32(squot32(global_tid_30248,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29473 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29476 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924));\n    gtid_30238 = squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924) + squot32(squot32(global_tid_30248,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924) -\n                                                     squot32(squot32(global_tid_30248,\n                                                                     tile_sizze_30924 *\n                                                             ",
            "        tile_sizze_30924),\n                                                             squot32(sizze_29473 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924) *\n                                                             squot32(sizze_29476 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924)) *\n                                                     (squot32(sizze_29473 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924) *\n                                                      squot32(sizze_29476 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924)),\n                                                     squot32(sizze_29476 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n        tile_sizze_30924;\n    gtid_30239 = srem32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30248, tile_sizze_30924 * ti",
            "le_sizze_30924) -\n                squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924 +\n        (squot32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924) -\n         squot32(squot32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924),\n                 squot32(sizze_29473 + tile_sizze_30924 - 1, tile_sizze_30924) *\n                 squot32(sizze_29476 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * (squot32(sizze_29473 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924) *\n                                               squot32(sizze_29476 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924)) -\n         squot32(squot32(global_tid_30248, tile_sizze_30924 *\n                         tile_sizze_30924) - squot32(squot32(global_tid_30248,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29473 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29476 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n                 (sq",
            "uot32(sizze_29473 + tile_sizze_30924 - 1,\n                          tile_sizze_30924) * squot32(sizze_29476 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)),\n                 squot32(sizze_29476 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * squot32(sizze_29476 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)) *\n        tile_sizze_30924;\n    ltid_30926 = squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924);\n    ltid_30927 = squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924);\n    ltid_30928 = srem32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30248, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30248, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924;\n    thread_active_31070 = (slt32(gtid_30237, sizze_29472) && slt32(gtid_302",
            "38,\n                                                                   sizze_29473)) &&\n        slt32(gtid_30239, sizze_29476);\n    if (thread_active_31070) { }\n    \n    float res_30253;\n    float x_30256 = 0.0F;\n    int32_t chunk_sizze_30254;\n    int32_t chunk_offset_30255 = 0;\n    \n    while (slt32(chunk_offset_30255, sizze_29474)) {\n        if (slt32(sizze_29474 - chunk_offset_30255, tile_sizze_30924)) {\n            chunk_sizze_30254 = sizze_29474 - chunk_offset_30255;\n        } else {\n            chunk_sizze_30254 = tile_sizze_30924;\n        }\n        for (int32_t comb_iter_31071 = 0; comb_iter_31071 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31071++) {\n            int32_t cid_30944;\n            int32_t cid_30945;\n            int32_t flat_comb_id_31072 = comb_iter_31071 *\n                    tiled_group_sizze_30925 + local_tid_30249;\n            \n            cid_30944 = squot32(flat_comb_id_31072, tile_sizze_30924);\n            cid_30945 = flat_comb_id_31072 - squot32(flat_comb_id_31072,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30944, tile_sizze_30924) && slt32(cid_30945,\n                                                             chunk_sizze_30254)) &&\n                slt32(gtid_30238, sizze_29473)) {\n                float x_chunk_outer_elem_30943 = *(__global\n                                                   float *) &A_mem_30976[(gtid_30237 *\n                                                                          (sizze_29473 *\n                                                                           sizze_29474) +\n                                                                          gtid_30238 *\n                                                                          sizze_29474 +\n                                                              ",
            "            (chunk_offset_30255 +\n                                                                           ltid_30928)) *\n                                                                         4];\n                \n                *(__local float *) &mem_30988[(cid_30944 * tile_sizze_30924 +\n                                               cid_30945) * 4] =\n                    x_chunk_outer_elem_30943;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31070) { }\n        for (int32_t comb_iter_31073 = 0; comb_iter_31073 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31073++) {\n            int32_t cid_30949;\n            int32_t cid_30950;\n            int32_t flat_comb_id_31074 = comb_iter_31073 *\n                    tiled_group_sizze_30925 + local_tid_30249;\n            \n            cid_30949 = squot32(flat_comb_id_31074, tile_sizze_30924);\n            cid_30950 = flat_comb_id_31074 - squot32(flat_comb_id_31074,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30949, chunk_sizze_30254) && slt32(cid_30950,\n                                                              tile_sizze_30924)) &&\n                slt32(gtid_30239, sizze_29476)) {\n                float x_chunk_outer_elem_30948 = *(__global\n                                                   float *) &mem_30984[((chunk_offset_30255 +\n                                                                         ltid_30927) *\n                                                                        sizze_29476 +\n                                                                        gtid_30239) *\n                                                                       4];\n                \n                *(__local float *) &mem_30992[(cid_30949 * tile_sizze_30924 +\n                           ",
            "                    cid_30950) * 4] =\n                    x_chunk_outer_elem_30948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31070) { }\n        \n        float res_30259;\n        float sync_30952;\n        float acc_30262 = x_30256;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30260 = 1;\n        \n        if (thread_active_31070) {\n            if (chunk_sizze_30254 == tile_sizze_30924) {\n                for (int32_t i_30261 = 0; i_30261 < tile_sizze_30924;\n                     i_30261++) {\n                    float x_30265 = *(__local float *) &mem_30988[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30261) *\n                                                                  4];\n                    float x_30266 = *(__local float *) &mem_30992[(i_30261 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30268 = x_30265 * x_30266;\n                    float res_30270 = acc_30262 + res_30268;\n                    float acc_tmp_31075 = res_30270;\n                    \n                    acc_30262 = acc_tmp_31075;\n                }\n            } else {\n                for (int32_t i_30261 = 0; i_30261 < chunk_sizze_30254;\n                     i_30261++) {\n                    float x_30265 = *(__local float *) &mem_30988[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30261) *\n                                                                  4];\n                    float x_30266 = *(__local float *) &mem_30992[(i_30261 *\n                                                 ",
            "                  tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30268 = x_30265 * x_30266;\n                    float res_30270 = acc_30262 + res_30268;\n                    float acc_tmp_31075 = res_30270;\n                    \n                    acc_30262 = acc_tmp_31075;\n                }\n            }\n        }\n        res_30259 = acc_30262;\n        sync_30952 = res_30259;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30256 = sync_30952;\n        chunk_offset_30255 += tile_sizze_30924;\n    }\n    res_30253 = x_30256;\n    if (thread_active_31070) {\n        *(__global float *) &mem_30997[(gtid_30237 * (sizze_29473 *\n                                                      sizze_29476) +\n                                        gtid_30238 * sizze_29476 + gtid_30239) *\n                                       4] = res_30253;\n    }\n}\n__kernel void map_kernel_30292(int32_t sizze_29506, int32_t sizze_29507,\n                               int32_t sizze_29508, __global\n                               unsigned char *mem_30989, __global\n                               unsigned char *mem_30994, __global\n                               unsigned char *mem_30998)\n{\n    int32_t wave_sizze_31084;\n    int32_t group_sizze_31085;\n    bool thread_active_31086;\n    int32_t gtid_30283;\n    int32_t gtid_30284;\n    int32_t global_tid_30292;\n    int32_t local_tid_30293;\n    int32_t group_id_30294;\n    \n    global_tid_30292 = get_global_id(0);\n    local_tid_30293 = get_local_id(0);\n    group_sizze_31085 = get_local_size(0);\n    wave_sizze_31084 = LOCKSTEP_WIDTH;\n    group_id_30294 = get_group_id(0);\n    gtid_30283 = squot32(global_tid_30292, sizze_29507);\n    gtid_30284 = global_tid_30292 - squot32(global_tid_30292, sizze_29507) *\n        sizze_29507;\n    thread_active_31086 = slt32(gtid_30283, sizze_29506) && slt32(gtid_30284,\n                    ",
            "                                              sizze_29507);\n    \n    float res_30297;\n    \n    if (thread_active_31086) {\n        float x_30300 = 0.0F;\n        \n        for (int32_t chunk_offset_30299 = 0; chunk_offset_30299 < sizze_29508;\n             chunk_offset_30299++) {\n            float x_30309 = *(__global float *) &mem_30994[(chunk_offset_30299 *\n                                                            (sizze_29506 *\n                                                             sizze_29507) +\n                                                            gtid_30283 *\n                                                            sizze_29507 +\n                                                            gtid_30284) * 4];\n            float x_30310 = *(__global float *) &mem_30989[(gtid_30283 *\n                                                            sizze_29508 +\n                                                            chunk_offset_30299) *\n                                                           4];\n            float res_30312 = x_30309 * x_30310;\n            float res_30314 = x_30300 + res_30312;\n            float x_tmp_31087 = res_30314;\n            \n            x_30300 = x_tmp_31087;\n        }\n        res_30297 = x_30300;\n    }\n    if (thread_active_31086) {\n        *(__global float *) &mem_30998[(gtid_30283 * sizze_29507 + gtid_30284) *\n                                       4] = res_30297;\n    }\n}\n__kernel void map_kernel_30335(int32_t sizze_29506, int32_t sizze_29508,\n                               int32_t sizze_29509, int32_t sizze_29510,\n                               int32_t sizze_29511, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30985, __global\n                               unsigned char *mem_30989)\n{\n    int32_t wave_sizze_31080;\n    int32_t group_sizze_31081;\n    bool thread_active_31082;\n    int32_t gtid_30326;\n    int32_t gtid_30327;\n    int32_t global_tid_3",
            "0335;\n    int32_t local_tid_30336;\n    int32_t group_id_30337;\n    \n    global_tid_30335 = get_global_id(0);\n    local_tid_30336 = get_local_id(0);\n    group_sizze_31081 = get_local_size(0);\n    wave_sizze_31080 = LOCKSTEP_WIDTH;\n    group_id_30337 = get_group_id(0);\n    gtid_30326 = squot32(global_tid_30335, sizze_29508);\n    gtid_30327 = global_tid_30335 - squot32(global_tid_30335, sizze_29508) *\n        sizze_29508;\n    thread_active_31082 = slt32(gtid_30326, sizze_29506) && slt32(gtid_30327,\n                                                                  sizze_29508);\n    \n    float res_30339;\n    \n    if (thread_active_31082) {\n        float x_30342 = 0.0F;\n        \n        for (int32_t chunk_offset_30341 = 0; chunk_offset_30341 < sizze_29511;\n             chunk_offset_30341++) {\n            float x_30351 = *(__global float *) &mem_30985[(chunk_offset_30341 *\n                                                            (sizze_29509 *\n                                                             sizze_29510) +\n                                                            gtid_30326 *\n                                                            sizze_29510 +\n                                                            gtid_30327) * 4];\n            float x_30352 = *(__global\n                              float *) &C_mem_30980[chunk_offset_30341 * 4];\n            float res_30354 = x_30351 * x_30352;\n            float res_30356 = x_30342 + res_30354;\n            float x_tmp_31083 = res_30356;\n            \n            x_30342 = x_tmp_31083;\n        }\n        res_30339 = x_30342;\n    }\n    if (thread_active_31082) {\n        *(__global float *) &mem_30989[(gtid_30326 * sizze_29508 + gtid_30327) *\n                                       4] = res_30339;\n    }\n}\n__kernel void map_kernel_30378(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29559, int32_t sizze_29560,\n ",
            "                              int32_t sizze_29561, __global\n                               unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30993, __global\n                               unsigned char *mem_31003)\n{\n    __local volatile char *restrict mem_30996 = mem_aligned_0;\n    __local volatile char *restrict mem_30999 = mem_aligned_1;\n    int32_t wave_sizze_31096;\n    int32_t group_sizze_31097;\n    bool thread_active_31098;\n    int32_t gtid_30369;\n    int32_t gtid_30370;\n    int32_t global_tid_30378;\n    int32_t local_tid_30379;\n    int32_t group_id_30380;\n    \n    global_tid_30378 = get_global_id(0);\n    local_tid_30379 = get_local_id(0);\n    group_sizze_31097 = get_local_size(0);\n    wave_sizze_31096 = LOCKSTEP_WIDTH;\n    group_id_30380 = get_group_id(0);\n    gtid_30369 = squot32(global_tid_30378, sizze_29560);\n    gtid_30370 = global_tid_30378 - squot32(global_tid_30378, sizze_29560) *\n        sizze_29560;\n    thread_active_31098 = slt32(gtid_30369, sizze_29559) && slt32(gtid_30370,\n                                                                  sizze_29560);\n    \n    float x_30381;\n    float x_30383;\n    float x_30384;\n    \n    if (thread_active_31098) {\n        x_30381 = *(__global float *) &A_mem_30976[gtid_30369 * 4];\n        x_30383 = *(__global float *) &B_mem_30978[gtid_30370 * 4];\n        x_30384 = x_30381 * x_30383;\n    }\n    \n    float res_30385;\n    float x_30388 = 0.0F;\n    int32_t chunk_sizze_30386;\n    int32_t chunk_offset_30387 = 0;\n    \n    while (slt32(chunk_offset_30387, sizze_29561)) {\n        if (slt32(sizze_29561 - chunk_offset_30387, group_sizze_30373)) {\n            chunk_sizze_30386 = sizze_29561 - chunk_offset_30387;\n        } else {\n            chunk_sizze_30386 = group_sizze_30373;\n        }\n        \n        float res_30391;\n        float sync_30938;\n        \n",
            "        for (int32_t comb_iter_31099 = 0; comb_iter_31099 <\n             squot32(group_sizze_30373 + group_sizze_30373 - 1,\n                     group_sizze_30373); comb_iter_31099++) {\n            int32_t cid_30925;\n            int32_t flat_comb_id_31100 = comb_iter_31099 * group_sizze_30373 +\n                    local_tid_30379;\n            \n            cid_30925 = flat_comb_id_31100;\n            if (slt32(cid_30925, chunk_sizze_30386) && 1) {\n                float x_chunk_outer_elem_30924 = *(__global\n                                                   float *) &C_mem_30980[(chunk_offset_30387 +\n                                                                          local_tid_30379) *\n                                                                         4];\n                \n                *(__local float *) &mem_30996[cid_30925 * 4] =\n                    x_chunk_outer_elem_30924;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t comb_iter_31101 = 0; comb_iter_31101 <\n             squot32(group_sizze_30373 + group_sizze_30373 - 1,\n                     group_sizze_30373); comb_iter_31101++) {\n            int32_t cid_30928;\n            int32_t flat_comb_id_31102 = comb_iter_31101 * group_sizze_30373 +\n                    local_tid_30379;\n            \n            cid_30928 = flat_comb_id_31102;\n            if (slt32(cid_30928, chunk_sizze_30386) && 1) {\n                float x_chunk_outer_elem_30927 = *(__global\n                                                   float *) &mem_30993[(gtid_30369 *\n                                                                        sizze_29561 +\n                                                                        (chunk_offset_30387 +\n                                                                         local_tid_30379)) *\n                                                                       4];\n                \n                *(__local float *) &mem_30999[cid_30928 * 4] =\n              ",
            "      x_chunk_outer_elem_30927;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        float acc_30394 = x_30388;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30392 = 1;\n        \n        if (thread_active_31098) {\n            if (chunk_sizze_30386 == group_sizze_30373) {\n                for (int32_t i_30393 = 0; i_30393 < group_sizze_30373;\n                     i_30393++) {\n                    float x_30397 = *(__local float *) &mem_30996[i_30393 * 4];\n                    float x_30398 = *(__local float *) &mem_30999[i_30393 * 4];\n                    float res_30400 = x_30384 * x_30397;\n                    float res_30401 = x_30398 * res_30400;\n                    float res_30403 = acc_30394 + res_30401;\n                    float acc_tmp_31103 = res_30403;\n                    \n                    acc_30394 = acc_tmp_31103;\n                }\n            } else {\n                for (int32_t i_30393 = 0; i_30393 < chunk_sizze_30386;\n                     i_30393++) {\n                    float x_30397 = *(__local float *) &mem_30996[i_30393 * 4];\n                    float x_30398 = *(__local float *) &mem_30999[i_30393 * 4];\n                    float res_30400 = x_30384 * x_30397;\n                    float res_30401 = x_30398 * res_30400;\n                    float res_30403 = acc_30394 + res_30401;\n                    float acc_tmp_31103 = res_30403;\n                    \n                    acc_30394 = acc_tmp_31103;\n                }\n            }\n        }\n        res_30391 = acc_30394;\n        sync_30938 = res_30391;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30388 = sync_30938;\n        chunk_offset_30387 += group_sizze_30373;\n    }\n    res_30385 = x_30388;\n    if (thread_active_31098) {\n        *(__global float *) &mem_31003[(gtid_30369 * sizze_29560 + gtid_30370) *\n                                       4] = res_30385;\n    }\n}\n__kernel void map_kernel_30424(int32_t sizze_29559, int32_t sizze_29561,\n                            ",
            "   int32_t sizze_29562, int32_t sizze_29563,\n                               int32_t sizze_29564, __global\n                               unsigned char *E_mem_30984, __global\n                               unsigned char *mem_30989, __global\n                               unsigned char *mem_30993)\n{\n    int32_t wave_sizze_31092;\n    int32_t group_sizze_31093;\n    bool thread_active_31094;\n    int32_t gtid_30415;\n    int32_t gtid_30416;\n    int32_t global_tid_30424;\n    int32_t local_tid_30425;\n    int32_t group_id_30426;\n    \n    global_tid_30424 = get_global_id(0);\n    local_tid_30425 = get_local_id(0);\n    group_sizze_31093 = get_local_size(0);\n    wave_sizze_31092 = LOCKSTEP_WIDTH;\n    group_id_30426 = get_group_id(0);\n    gtid_30415 = squot32(global_tid_30424, sizze_29561);\n    gtid_30416 = global_tid_30424 - squot32(global_tid_30424, sizze_29561) *\n        sizze_29561;\n    thread_active_31094 = slt32(gtid_30415, sizze_29559) && slt32(gtid_30416,\n                                                                  sizze_29561);\n    \n    float res_30428;\n    \n    if (thread_active_31094) {\n        float x_30431 = 0.0F;\n        \n        for (int32_t chunk_offset_30430 = 0; chunk_offset_30430 < sizze_29564;\n             chunk_offset_30430++) {\n            float x_30440 = *(__global float *) &mem_30989[(chunk_offset_30430 *\n                                                            (sizze_29562 *\n                                                             sizze_29563) +\n                                                            gtid_30415 *\n                                                            sizze_29563 +\n                                                            gtid_30416) * 4];\n            float x_30441 = *(__global\n                              float *) &E_mem_30984[chunk_offset_30430 * 4];\n            float res_30443 = x_30440 * x_30441;\n            float res_30445 = x_30431 + res_30443;\n            float x_tmp_31095 = res_30445;\n            \n           ",
            " x_30431 = x_tmp_31095;\n        }\n        res_30428 = x_30431;\n    }\n    if (thread_active_31094) {\n        *(__global float *) &mem_30993[(gtid_30415 * sizze_29561 + gtid_30416) *\n                                       4] = res_30428;\n    }\n}\n__kernel void map_kernel_30470(int32_t sizze_29616, int32_t sizze_29617,\n                               int32_t sizze_29618, int32_t sizze_29620,\n                               __global unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *D_mem_30982, __global\n                               unsigned char *mem_30987)\n{\n    int32_t wave_sizze_31109;\n    int32_t group_sizze_31110;\n    bool thread_active_31111;\n    int32_t gtid_30459;\n    int32_t gtid_30460;\n    int32_t gtid_30461;\n    int32_t global_tid_30470;\n    int32_t local_tid_30471;\n    int32_t group_id_30472;\n    \n    global_tid_30470 = get_global_id(0);\n    local_tid_30471 = get_local_id(0);\n    group_sizze_31110 = get_local_size(0);\n    wave_sizze_31109 = LOCKSTEP_WIDTH;\n    group_id_30472 = get_group_id(0);\n    gtid_30459 = squot32(global_tid_30470, sizze_29617 * sizze_29620);\n    gtid_30460 = squot32(global_tid_30470 - squot32(global_tid_30470,\n                                                    sizze_29617 * sizze_29620) *\n                         (sizze_29617 * sizze_29620), sizze_29620);\n    gtid_30461 = global_tid_30470 - squot32(global_tid_30470, sizze_29617 *\n                                            sizze_29620) * (sizze_29617 *\n                                                            sizze_29620) -\n        squot32(global_tid_30470 - squot32(global_tid_30470, sizze_29617 *\n                                           sizze_29620) * (sizze_29617 *\n                                                           sizze_29620),\n                sizze_29620) * sizze_29620;\n    thread_active_31111 = (slt32(gt",
            "id_30459, sizze_29616) && slt32(gtid_30460,\n                                                                   sizze_29617)) &&\n        slt32(gtid_30461, sizze_29620);\n    \n    float x_30474;\n    float res_30476;\n    \n    if (thread_active_31111) {\n        x_30474 = *(__global float *) &D_mem_30982[gtid_30460 * 4];\n        \n        float x_30479 = 0.0F;\n        \n        for (int32_t chunk_offset_30478 = 0; chunk_offset_30478 < sizze_29618;\n             chunk_offset_30478++) {\n            float x_30490 = *(__global float *) &A_mem_30976[(gtid_30459 *\n                                                              (sizze_29617 *\n                                                               sizze_29618) +\n                                                              gtid_30460 *\n                                                              sizze_29618 +\n                                                              chunk_offset_30478) *\n                                                             4];\n            float x_30491 = *(__global\n                              float *) &B_mem_30978[(chunk_offset_30478 *\n                                                     sizze_29620 + gtid_30461) *\n                                                    4];\n            float x_30492 = *(__global\n                              float *) &C_mem_30980[chunk_offset_30478 * 4];\n            float x_30494 = x_30490 * x_30491;\n            float x_30495 = x_30492 * x_30494;\n            float res_30496 = x_30474 * x_30495;\n            float res_30498 = x_30479 + res_30496;\n            float x_tmp_31112 = res_30498;\n            \n            x_30479 = x_tmp_31112;\n        }\n        res_30476 = x_30479;\n    }\n    if (thread_active_31111) {\n        *(__global float *) &mem_30987[(gtid_30459 * (sizze_29617 *\n                                                      sizze_29620) +\n                                        gtid_30460 * sizze_29620 + gtid_30461) *\n                                       4] = ",
            "res_30476;\n    }\n}\n__kernel void map_kernel_30523(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29668, int32_t sizze_29669,\n                               int32_t sizze_29670, int32_t sizze_29675,\n                               __global unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30985, __global\n                               unsigned char *mem_30998)\n{\n    __local volatile char *restrict mem_30989 = mem_aligned_0;\n    __local volatile char *restrict mem_30993 = mem_aligned_1;\n    int32_t wave_sizze_31121;\n    int32_t group_sizze_31122;\n    bool thread_active_31123;\n    int32_t gtid_30512;\n    int32_t gtid_30513;\n    int32_t gtid_30514;\n    int32_t global_tid_30523;\n    int32_t local_tid_30524;\n    int32_t group_id_30525;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t ltid_30928;\n    \n    global_tid_30523 = get_global_id(0);\n    local_tid_30524 = get_local_id(0);\n    group_sizze_31122 = get_local_size(0);\n    wave_sizze_31121 = LOCKSTEP_WIDTH;\n    group_id_30525 = get_group_id(0);\n    gtid_30512 = squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924) + squot32(squot32(global_tid_30523,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29669 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29675 +\n                                                             tile_sizze_30924 -\n   ",
            "                                                          1,\n                                                             tile_sizze_30924));\n    gtid_30513 = squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924) + squot32(squot32(global_tid_30523,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924) -\n                                                     squot32(squot32(global_tid_30523,\n                                                                     tile_sizze_30924 *\n                                                                     tile_sizze_30924),\n                                                             squot32(sizze_29669 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924) *\n                                                             squot32(sizze_29675 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924)) *\n                                                     (squot32(sizze_29669 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                              ",
            "                tile_sizze_30924) *\n                                                      squot32(sizze_29675 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924)),\n                                                     squot32(sizze_29675 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n        tile_sizze_30924;\n    gtid_30514 = srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924 +\n        (squot32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924) -\n         squot32(squot32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924),\n                 squot32(sizze_29669 + tile_sizze_30924 - 1, tile_sizze_30924) *\n                 squot32(sizze_29675 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * (squot32(sizze_29669 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924) *\n                                               squot32(sizze_29675 +\n                                   ",
            "                    tile_sizze_30924 - 1,\n                                                       tile_sizze_30924)) -\n         squot32(squot32(global_tid_30523, tile_sizze_30924 *\n                         tile_sizze_30924) - squot32(squot32(global_tid_30523,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29669 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29675 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n                 (squot32(sizze_29669 + tile_sizze_30924 - 1,\n                          tile_sizze_30924) * squot32(sizze_29675 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)),\n                 squot32(sizze_29675 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * squot32(sizze_29675 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)) *\n        tile_sizze_30924;\n    ltid_30926 = squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924);\n    ltid_30927 = squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30523, tile_sizze_30924 *\n             ",
            "                           tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924);\n    ltid_30928 = srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30523, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30523, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924;\n    thread_active_31123 = (slt32(gtid_30512, sizze_29668) && slt32(gtid_30513,\n                                                                   sizze_29669)) &&\n        slt32(gtid_30514, sizze_29675);\n    if (thread_active_31123) { }\n    \n    float res_30528;\n    float x_30531 = 0.0F;\n    int32_t chunk_sizze_30529;\n    int32_t chunk_offset_30530 = 0;\n    \n    while (slt32(chunk_offset_30530, sizze_29670)) {\n        if (slt32(sizze_29670 - chunk_offset_30530, tile_sizze_30924)) {\n            chunk_sizze_30529 = sizze_29670 - chunk_offset_30530;\n        } else {\n            chunk_sizze_30529 = tile_sizze_30924;\n        }\n        for (int32_t comb_iter_31124 = 0; comb_iter_31124 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31124++) {\n            int32_t cid_30944;\n            int32_t cid_30945;\n            int32_t flat_comb_id_31125 = comb_iter_31124 *\n                    tiled_group_sizze_30925 + local_tid_",
            "30524;\n            \n            cid_30944 = squot32(flat_comb_id_31125, tile_sizze_30924);\n            cid_30945 = flat_comb_id_31125 - squot32(flat_comb_id_31125,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30944, tile_sizze_30924) && slt32(cid_30945,\n                                                             chunk_sizze_30529)) &&\n                slt32(gtid_30513, sizze_29669)) {\n                float x_chunk_outer_elem_30943 = *(__global\n                                                   float *) &mem_30985[(gtid_30512 *\n                                                                        (sizze_29669 *\n                                                                         sizze_29670) +\n                                                                        gtid_30513 *\n                                                                        sizze_29670 +\n                                                                        (chunk_offset_30530 +\n                                                                         ltid_30928)) *\n                                                                       4];\n                \n                *(__local float *) &mem_30989[(cid_30944 * tile_sizze_30924 +\n                                               cid_30945) * 4] =\n                    x_chunk_outer_elem_30943;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31123) { }\n        for (int32_t comb_iter_31126 = 0; comb_iter_31126 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31126++) {\n            int32_t cid_30949;\n            int32_t cid_30950;\n            int32_t flat_comb_id_31127 = comb_iter_31126 *\n                    tiled_group_sizze_30925 + local_tid_30524;\n            \n            cid_30949 = squot32(flat_comb_id",
            "_31127, tile_sizze_30924);\n            cid_30950 = flat_comb_id_31127 - squot32(flat_comb_id_31127,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30949, chunk_sizze_30529) && slt32(cid_30950,\n                                                              tile_sizze_30924)) &&\n                slt32(gtid_30514, sizze_29675)) {\n                float x_chunk_outer_elem_30948 = *(__global\n                                                   float *) &C_mem_30980[((chunk_offset_30530 +\n                                                                           ltid_30927) *\n                                                                          sizze_29675 +\n                                                                          gtid_30514) *\n                                                                         4];\n                \n                *(__local float *) &mem_30993[(cid_30949 * tile_sizze_30924 +\n                                               cid_30950) * 4] =\n                    x_chunk_outer_elem_30948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31123) { }\n        \n        float res_30534;\n        float sync_30952;\n        float acc_30537 = x_30531;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30535 = 1;\n        \n        if (thread_active_31123) {\n            if (chunk_sizze_30529 == tile_sizze_30924) {\n                for (int32_t i_30536 = 0; i_30536 < tile_sizze_30924;\n                     i_30536++) {\n                    float x_30540 = *(__local float *) &mem_30989[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30536) *\n                                                                  4];\n                    float x_30541 = *(__local float *) &mem_30993[(i_30536 *\n                          ",
            "                                         tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30543 = x_30540 * x_30541;\n                    float res_30545 = acc_30537 + res_30543;\n                    float acc_tmp_31128 = res_30545;\n                    \n                    acc_30537 = acc_tmp_31128;\n                }\n            } else {\n                for (int32_t i_30536 = 0; i_30536 < chunk_sizze_30529;\n                     i_30536++) {\n                    float x_30540 = *(__local float *) &mem_30989[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30536) *\n                                                                  4];\n                    float x_30541 = *(__local float *) &mem_30993[(i_30536 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30543 = x_30540 * x_30541;\n                    float res_30545 = acc_30537 + res_30543;\n                    float acc_tmp_31128 = res_30545;\n                    \n                    acc_30537 = acc_tmp_31128;\n                }\n            }\n        }\n        res_30534 = acc_30537;\n        sync_30952 = res_30534;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30531 = sync_30952;\n        chunk_offset_30530 += tile_sizze_30924;\n    }\n    res_30528 = x_30531;\n    if (thread_active_31123) {\n        *(__global float *) &mem_30998[(gtid_30512 * (sizze_29669 *\n                                                      sizze_29675) +\n                                        gtid_30513 * sizze_29675 + gtid_30514) *\n                                      ",
            " 4] = res_30528;\n    }\n}\n__kernel void map_kernel_30557(int32_t sizze_29668, int32_t sizze_29669,\n                               int32_t sizze_29670, int32_t sizze_29672,\n                               int32_t sizze_29673, __global\n                               unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *mem_30985)\n{\n    int32_t wave_sizze_31118;\n    int32_t group_sizze_31119;\n    bool thread_active_31120;\n    int32_t gtid_30546;\n    int32_t gtid_30547;\n    int32_t gtid_30548;\n    int32_t global_tid_30557;\n    int32_t local_tid_30558;\n    int32_t group_id_30559;\n    \n    global_tid_30557 = get_global_id(0);\n    local_tid_30558 = get_local_id(0);\n    group_sizze_31119 = get_local_size(0);\n    wave_sizze_31118 = LOCKSTEP_WIDTH;\n    group_id_30559 = get_group_id(0);\n    gtid_30546 = squot32(global_tid_30557, sizze_29669 * sizze_29670);\n    gtid_30547 = squot32(global_tid_30557 - squot32(global_tid_30557,\n                                                    sizze_29669 * sizze_29670) *\n                         (sizze_29669 * sizze_29670), sizze_29670);\n    gtid_30548 = global_tid_30557 - squot32(global_tid_30557, sizze_29669 *\n                                            sizze_29670) * (sizze_29669 *\n                                                            sizze_29670) -\n        squot32(global_tid_30557 - squot32(global_tid_30557, sizze_29669 *\n                                           sizze_29670) * (sizze_29669 *\n                                                           sizze_29670),\n                sizze_29670) * sizze_29670;\n    thread_active_31120 = (slt32(gtid_30546, sizze_29668) && slt32(gtid_30547,\n                                                                   sizze_29669)) &&\n        slt32(gtid_30548, sizze_29670);\n    \n    float x_30560;\n    float x_30561;\n    float res_30562;\n    \n    if (thread_active_31120) {\n        x_30560 = *(__global float ",
            "*) &A_mem_30976[(gtid_30546 * (sizze_29669 *\n                                                                  sizze_29670) +\n                                                    gtid_30547 * sizze_29670 +\n                                                    gtid_30548) * 4];\n        x_30561 = *(__global float *) &B_mem_30978[(gtid_30546 * (sizze_29672 *\n                                                                  sizze_29673) +\n                                                    gtid_30547 * sizze_29673 +\n                                                    gtid_30548) * 4];\n        res_30562 = x_30560 + x_30561;\n    }\n    if (thread_active_31120) {\n        *(__global float *) &mem_30985[(gtid_30546 * (sizze_29669 *\n                                                      sizze_29670) +\n                                        gtid_30547 * sizze_29670 + gtid_30548) *\n                                       4] = res_30562;\n    }\n}\n__kernel void map_kernel_30572(int32_t sizze_29729, int32_t sizze_29734,\n                               int32_t sizze_29736, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *D_mem_30982, __global\n                               unsigned char *mem_30986)\n{\n    int32_t wave_sizze_31134;\n    int32_t group_sizze_31135;\n    bool thread_active_31136;\n    int32_t gtid_30563;\n    int32_t gtid_30564;\n    int32_t global_tid_30572;\n    int32_t local_tid_30573;\n    int32_t group_id_30574;\n    \n    global_tid_30572 = get_global_id(0);\n    local_tid_30573 = get_local_id(0);\n    group_sizze_31135 = get_local_size(0);\n    wave_sizze_31134 = LOCKSTEP_WIDTH;\n    group_id_30574 = get_group_id(0);\n    gtid_30563 = squot32(global_tid_30572, sizze_29734);\n    gtid_30564 = global_tid_30572 - squot32(global_tid_30572, sizze_29734) *\n        sizze_29734;\n    thread_active_31136 = slt32(gtid_30563, sizze_29729) && slt32(gtid_30564,\n                                                                 ",
            " sizze_29734);\n    \n    float x_30575;\n    float x_30576;\n    float res_30577;\n    \n    if (thread_active_31136) {\n        x_30575 = *(__global float *) &C_mem_30980[(gtid_30563 * sizze_29734 +\n                                                    gtid_30564) * 4];\n        x_30576 = *(__global float *) &D_mem_30982[(gtid_30563 * sizze_29736 +\n                                                    gtid_30564) * 4];\n        res_30577 = x_30575 + x_30576;\n    }\n    if (thread_active_31136) {\n        *(__global float *) &mem_30986[(gtid_30563 * sizze_29734 + gtid_30564) *\n                                       4] = res_30577;\n    }\n}\n__kernel void map_kernel_30602(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29727, int32_t sizze_29728,\n                               int32_t sizze_29729, int32_t sizze_29734,\n                               __global unsigned char *mem_30986, __global\n                               unsigned char *mem_30991, __global\n                               unsigned char *mem_31004)\n{\n    __local volatile char *restrict mem_30995 = mem_aligned_0;\n    __local volatile char *restrict mem_30999 = mem_aligned_1;\n    int32_t wave_sizze_31140;\n    int32_t group_sizze_31141;\n    bool thread_active_31142;\n    int32_t gtid_30591;\n    int32_t gtid_30592;\n    int32_t gtid_30593;\n    int32_t global_tid_30602;\n    int32_t local_tid_30603;\n    int32_t group_id_30604;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t ltid_30928;\n    \n    global_tid_30602 = get_global_id(0);\n    local_tid_30603 = get_local_id(0);\n    group_sizze_31141 = get_local_size(0);\n    wave_sizze_31140 = LOCKSTEP_WIDTH;\n    group_id_30604 = get_group_id(0);\n    gtid_30591 = squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924) + squot32(squot32(global_tid_30602,\n      ",
            "                                                       tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29728 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29734 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924));\n    gtid_30592 = squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924) + squot32(squot32(global_tid_30602,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924) -\n                                                     squot32(squot32(global_tid_30602,\n                                                                     tile_sizze_30924 *\n                                                                     tile_sizze_30924),\n                                                             squot32(sizze_29728 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizz",
            "e_30924) *\n                                                             squot32(sizze_29734 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924)) *\n                                                     (squot32(sizze_29728 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924) *\n                                                      squot32(sizze_29734 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924)),\n                                                     squot32(sizze_29734 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n        tile_sizze_30924;\n    gtid_30593 = srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924 +\n        (s",
            "quot32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924) -\n         squot32(squot32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924),\n                 squot32(sizze_29728 + tile_sizze_30924 - 1, tile_sizze_30924) *\n                 squot32(sizze_29734 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * (squot32(sizze_29728 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924) *\n                                               squot32(sizze_29734 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924)) -\n         squot32(squot32(global_tid_30602, tile_sizze_30924 *\n                         tile_sizze_30924) - squot32(squot32(global_tid_30602,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29728 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29734 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n                 (squot32(sizze_29728 + tile_sizze_30924 - 1,\n                          tile_sizze_30924) * squot32(sizze_29734 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)),\n                 squot32(sizze_29734 + tile_sizze_30924 - 1,\n                         ti",
            "le_sizze_30924)) * squot32(sizze_29734 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)) *\n        tile_sizze_30924;\n    ltid_30926 = squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924);\n    ltid_30927 = squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924);\n    ltid_30928 = srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30602, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30602, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924;\n    thread_active_31142 = (slt32(gtid_30591, sizze_29727) && slt32(gtid_30592,\n                                                                   sizze_29728)) &&\n        slt32(gtid_30593, sizze_29734);\n    if (thread_active_31142) { }\n    \n    float res_30607;\n    float x_30610 = 0.0F;\n    int32_t chunk_sizze_30608;\n    int32_t chunk_offset_30609 = 0;\n    \n    while (slt32(chunk_offset_30609, sizze_29729)) {\n        if",
            " (slt32(sizze_29729 - chunk_offset_30609, tile_sizze_30924)) {\n            chunk_sizze_30608 = sizze_29729 - chunk_offset_30609;\n        } else {\n            chunk_sizze_30608 = tile_sizze_30924;\n        }\n        for (int32_t comb_iter_31143 = 0; comb_iter_31143 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31143++) {\n            int32_t cid_30944;\n            int32_t cid_30945;\n            int32_t flat_comb_id_31144 = comb_iter_31143 *\n                    tiled_group_sizze_30925 + local_tid_30603;\n            \n            cid_30944 = squot32(flat_comb_id_31144, tile_sizze_30924);\n            cid_30945 = flat_comb_id_31144 - squot32(flat_comb_id_31144,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30944, tile_sizze_30924) && slt32(cid_30945,\n                                                             chunk_sizze_30608)) &&\n                slt32(gtid_30592, sizze_29728)) {\n                float x_chunk_outer_elem_30943 = *(__global\n                                                   float *) &mem_30991[(gtid_30591 *\n                                                                        (sizze_29728 *\n                                                                         sizze_29729) +\n                                                                        gtid_30592 *\n                                                                        sizze_29729 +\n                                                                        (chunk_offset_30609 +\n                                                                         ltid_30928)) *\n                                                                       4];\n                \n                *(__local float *) &mem_30995[(cid_30944 * tile_sizze_30924 +\n                                               cid_30945) * 4] =\n   ",
            "                 x_chunk_outer_elem_30943;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31142) { }\n        for (int32_t comb_iter_31145 = 0; comb_iter_31145 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31145++) {\n            int32_t cid_30949;\n            int32_t cid_30950;\n            int32_t flat_comb_id_31146 = comb_iter_31145 *\n                    tiled_group_sizze_30925 + local_tid_30603;\n            \n            cid_30949 = squot32(flat_comb_id_31146, tile_sizze_30924);\n            cid_30950 = flat_comb_id_31146 - squot32(flat_comb_id_31146,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30949, chunk_sizze_30608) && slt32(cid_30950,\n                                                              tile_sizze_30924)) &&\n                slt32(gtid_30593, sizze_29734)) {\n                float x_chunk_outer_elem_30948 = *(__global\n                                                   float *) &mem_30986[((chunk_offset_30609 +\n                                                                         ltid_30927) *\n                                                                        sizze_29734 +\n                                                                        gtid_30593) *\n                                                                       4];\n                \n                *(__local float *) &mem_30999[(cid_30949 * tile_sizze_30924 +\n                                               cid_30950) * 4] =\n                    x_chunk_outer_elem_30948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31142) { }\n        \n        float res_30613;\n        float sync_30952;\n        float acc_30616 = x_30610;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30614 = 1;\n        \n        if ",
            "(thread_active_31142) {\n            if (chunk_sizze_30608 == tile_sizze_30924) {\n                for (int32_t i_30615 = 0; i_30615 < tile_sizze_30924;\n                     i_30615++) {\n                    float x_30619 = *(__local float *) &mem_30995[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30615) *\n                                                                  4];\n                    float x_30620 = *(__local float *) &mem_30999[(i_30615 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30622 = x_30619 * x_30620;\n                    float res_30624 = acc_30616 + res_30622;\n                    float acc_tmp_31147 = res_30624;\n                    \n                    acc_30616 = acc_tmp_31147;\n                }\n            } else {\n                for (int32_t i_30615 = 0; i_30615 < chunk_sizze_30608;\n                     i_30615++) {\n                    float x_30619 = *(__local float *) &mem_30995[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30615) *\n                                                                  4];\n                    float x_30620 = *(__local float *) &mem_30999[(i_30615 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30622 = x_30619 * x_30620;\n                    float res_30624 = acc_30616 + res_30622;\n                    float acc_tmp_31147 = res_30624;\n     ",
            "               \n                    acc_30616 = acc_tmp_31147;\n                }\n            }\n        }\n        res_30613 = acc_30616;\n        sync_30952 = res_30613;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30610 = sync_30952;\n        chunk_offset_30609 += tile_sizze_30924;\n    }\n    res_30607 = x_30610;\n    if (thread_active_31142) {\n        *(__global float *) &mem_31004[(gtid_30591 * (sizze_29728 *\n                                                      sizze_29734) +\n                                        gtid_30592 * sizze_29734 + gtid_30593) *\n                                       4] = res_30607;\n    }\n}\n__kernel void map_kernel_30636(int32_t sizze_29727, int32_t sizze_29728,\n                               int32_t sizze_29729, int32_t sizze_29731,\n                               int32_t sizze_29732, __global\n                               unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *mem_30991)\n{\n    int32_t wave_sizze_31137;\n    int32_t group_sizze_31138;\n    bool thread_active_31139;\n    int32_t gtid_30625;\n    int32_t gtid_30626;\n    int32_t gtid_30627;\n    int32_t global_tid_30636;\n    int32_t local_tid_30637;\n    int32_t group_id_30638;\n    \n    global_tid_30636 = get_global_id(0);\n    local_tid_30637 = get_local_id(0);\n    group_sizze_31138 = get_local_size(0);\n    wave_sizze_31137 = LOCKSTEP_WIDTH;\n    group_id_30638 = get_group_id(0);\n    gtid_30625 = squot32(global_tid_30636, sizze_29728 * sizze_29729);\n    gtid_30626 = squot32(global_tid_30636 - squot32(global_tid_30636,\n                                                    sizze_29728 * sizze_29729) *\n                         (sizze_29728 * sizze_29729), sizze_29729);\n    gtid_30627 = global_tid_30636 - squot32(global_tid_30636, sizze_29728 *\n                                            sizze_29729) * (sizze_29728 *\n                                                            sizze_29729) -\n ",
            "       squot32(global_tid_30636 - squot32(global_tid_30636, sizze_29728 *\n                                           sizze_29729) * (sizze_29728 *\n                                                           sizze_29729),\n                sizze_29729) * sizze_29729;\n    thread_active_31139 = (slt32(gtid_30625, sizze_29727) && slt32(gtid_30626,\n                                                                   sizze_29728)) &&\n        slt32(gtid_30627, sizze_29729);\n    \n    float x_30639;\n    float x_30640;\n    float res_30641;\n    \n    if (thread_active_31139) {\n        x_30639 = *(__global float *) &A_mem_30976[(gtid_30625 * (sizze_29728 *\n                                                                  sizze_29729) +\n                                                    gtid_30626 * sizze_29729 +\n                                                    gtid_30627) * 4];\n        x_30640 = *(__global float *) &B_mem_30978[(gtid_30625 * (sizze_29731 *\n                                                                  sizze_29732) +\n                                                    gtid_30626 * sizze_29732 +\n                                                    gtid_30627) * 4];\n        res_30641 = x_30639 + x_30640;\n    }\n    if (thread_active_31139) {\n        *(__global float *) &mem_30991[(gtid_30625 * (sizze_29728 *\n                                                      sizze_29729) +\n                                        gtid_30626 * sizze_29729 + gtid_30627) *\n                                       4] = res_30641;\n    }\n}\n__kernel void map_kernel_30651(int32_t sizze_29808, int32_t sizze_29813,\n                               int32_t sizze_29815, float c_29820,\n                               float d_29822, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *D_mem_30982, __global\n                               unsigned char *mem_30986)\n{\n    int32_t wave_sizze_31153;\n    int32_t group_sizze_31154;\n    b",
            "ool thread_active_31155;\n    int32_t gtid_30642;\n    int32_t gtid_30643;\n    int32_t global_tid_30651;\n    int32_t local_tid_30652;\n    int32_t group_id_30653;\n    \n    global_tid_30651 = get_global_id(0);\n    local_tid_30652 = get_local_id(0);\n    group_sizze_31154 = get_local_size(0);\n    wave_sizze_31153 = LOCKSTEP_WIDTH;\n    group_id_30653 = get_group_id(0);\n    gtid_30642 = squot32(global_tid_30651, sizze_29813);\n    gtid_30643 = global_tid_30651 - squot32(global_tid_30651, sizze_29813) *\n        sizze_29813;\n    thread_active_31155 = slt32(gtid_30642, sizze_29808) && slt32(gtid_30643,\n                                                                  sizze_29813);\n    \n    float x_30654;\n    float x_30655;\n    float res_30656;\n    float res_30657;\n    float res_30658;\n    \n    if (thread_active_31155) {\n        x_30654 = *(__global float *) &C_mem_30980[(gtid_30642 * sizze_29813 +\n                                                    gtid_30643) * 4];\n        x_30655 = *(__global float *) &D_mem_30982[(gtid_30642 * sizze_29815 +\n                                                    gtid_30643) * 4];\n        res_30656 = c_29820 * x_30654;\n        res_30657 = d_29822 * x_30655;\n        res_30658 = res_30656 + res_30657;\n    }\n    if (thread_active_31155) {\n        *(__global float *) &mem_30986[(gtid_30642 * sizze_29813 + gtid_30643) *\n                                       4] = res_30658;\n    }\n}\n__kernel void map_kernel_30683(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_29806, int32_t sizze_29807,\n                               int32_t sizze_29808, int32_t sizze_29813,\n                               __global unsigned char *mem_30986, __global\n                               unsigned char *mem_30991, __global\n                               unsigned char *mem_31004)\n{\n    __local volatile char *restrict mem_30995 = mem_aligned_0;\n    __local volatile char *",
            "restrict mem_30999 = mem_aligned_1;\n    int32_t wave_sizze_31159;\n    int32_t group_sizze_31160;\n    bool thread_active_31161;\n    int32_t gtid_30672;\n    int32_t gtid_30673;\n    int32_t gtid_30674;\n    int32_t global_tid_30683;\n    int32_t local_tid_30684;\n    int32_t group_id_30685;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t ltid_30928;\n    \n    global_tid_30683 = get_global_id(0);\n    local_tid_30684 = get_local_id(0);\n    group_sizze_31160 = get_local_size(0);\n    wave_sizze_31159 = LOCKSTEP_WIDTH;\n    group_id_30685 = get_group_id(0);\n    gtid_30672 = squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924) + squot32(squot32(global_tid_30683,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29807 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29813 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924));\n    gtid_30673 = squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_3",
            "0924) + squot32(squot32(global_tid_30683,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924) -\n                                                     squot32(squot32(global_tid_30683,\n                                                                     tile_sizze_30924 *\n                                                                     tile_sizze_30924),\n                                                             squot32(sizze_29807 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924) *\n                                                             squot32(sizze_29813 +\n                                                                     tile_sizze_30924 -\n                                                                     1,\n                                                                     tile_sizze_30924)) *\n                                                     (squot32(sizze_29807 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924) *\n                                                      squot32(sizze_29813 +\n                                                              tile_sizze_30924 -\n                                                              1,\n                                                              tile_sizze_30924)),\n                                                     squot32(sizze_29813 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                          ",
            "                   tile_sizze_30924)) *\n        tile_sizze_30924;\n    gtid_30674 = srem32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924 +\n        (squot32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924) -\n         squot32(squot32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924),\n                 squot32(sizze_29807 + tile_sizze_30924 - 1, tile_sizze_30924) *\n                 squot32(sizze_29813 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * (squot32(sizze_29807 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924) *\n                                               squot32(sizze_29813 +\n                                                       tile_sizze_30924 - 1,\n                                                       tile_sizze_30924)) -\n         squot32(squot32(global_tid_30683, tile_sizze_30924 *\n                         tile_sizze_30924) - squot32(squot32(global_tid_30683,\n                                                             tile_sizze_30924 *\n                                                             tile_sizze_30924),\n                                                     squot32(sizze_29807 +\n                                                             tile_sizze_30924 -\n                           ",
            "                                  1,\n                                                             tile_sizze_30924) *\n                                                     squot32(sizze_29813 +\n                                                             tile_sizze_30924 -\n                                                             1,\n                                                             tile_sizze_30924)) *\n                 (squot32(sizze_29807 + tile_sizze_30924 - 1,\n                          tile_sizze_30924) * squot32(sizze_29813 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)),\n                 squot32(sizze_29813 + tile_sizze_30924 - 1,\n                         tile_sizze_30924)) * squot32(sizze_29813 +\n                                                      tile_sizze_30924 - 1,\n                                                      tile_sizze_30924)) *\n        tile_sizze_30924;\n    ltid_30926 = squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                tile_sizze_30924), tile_sizze_30924 *\n                         tile_sizze_30924);\n    ltid_30927 = squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                tile_sizze_30924) -\n                         squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                                        tile_sizze_30924), tile_sizze_30924 *\n                                 tile_sizze_30924) * (tile_sizze_30924 *\n                                                      tile_sizze_30924),\n                         tile_sizze_30924);\n    ltid_30928 = srem32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924) -\n        squot32(srem32(global_tid_30683, tile_sizze_30924 * tile_sizze_30924),\n                tile_sizze_30924 * tile_sizze_30924) * (tile_sizze_30924 *\n                                                        tile_sizze_30924) -\n        squot32(srem32(global_tid_3",
            "0683, tile_sizze_30924 * tile_sizze_30924) -\n                squot32(srem32(global_tid_30683, tile_sizze_30924 *\n                               tile_sizze_30924), tile_sizze_30924 *\n                        tile_sizze_30924) * (tile_sizze_30924 *\n                                             tile_sizze_30924),\n                tile_sizze_30924) * tile_sizze_30924;\n    thread_active_31161 = (slt32(gtid_30672, sizze_29806) && slt32(gtid_30673,\n                                                                   sizze_29807)) &&\n        slt32(gtid_30674, sizze_29813);\n    if (thread_active_31161) { }\n    \n    float res_30688;\n    float x_30691 = 0.0F;\n    int32_t chunk_sizze_30689;\n    int32_t chunk_offset_30690 = 0;\n    \n    while (slt32(chunk_offset_30690, sizze_29808)) {\n        if (slt32(sizze_29808 - chunk_offset_30690, tile_sizze_30924)) {\n            chunk_sizze_30689 = sizze_29808 - chunk_offset_30690;\n        } else {\n            chunk_sizze_30689 = tile_sizze_30924;\n        }\n        for (int32_t comb_iter_31162 = 0; comb_iter_31162 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31162++) {\n            int32_t cid_30944;\n            int32_t cid_30945;\n            int32_t flat_comb_id_31163 = comb_iter_31162 *\n                    tiled_group_sizze_30925 + local_tid_30684;\n            \n            cid_30944 = squot32(flat_comb_id_31163, tile_sizze_30924);\n            cid_30945 = flat_comb_id_31163 - squot32(flat_comb_id_31163,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30944, tile_sizze_30924) && slt32(cid_30945,\n                                                             chunk_sizze_30689)) &&\n                slt32(gtid_30673, sizze_29807)) {\n                float x_chunk_outer_elem_30943 = *(__global\n                                                   float *) &mem",
            "_30991[(gtid_30672 *\n                                                                        (sizze_29807 *\n                                                                         sizze_29808) +\n                                                                        gtid_30673 *\n                                                                        sizze_29808 +\n                                                                        (chunk_offset_30690 +\n                                                                         ltid_30928)) *\n                                                                       4];\n                \n                *(__local float *) &mem_30995[(cid_30944 * tile_sizze_30924 +\n                                               cid_30945) * 4] =\n                    x_chunk_outer_elem_30943;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31161) { }\n        for (int32_t comb_iter_31164 = 0; comb_iter_31164 <\n             squot32(tile_sizze_30924 * tile_sizze_30924 +\n                     tiled_group_sizze_30925 - 1, tiled_group_sizze_30925);\n             comb_iter_31164++) {\n            int32_t cid_30949;\n            int32_t cid_30950;\n            int32_t flat_comb_id_31165 = comb_iter_31164 *\n                    tiled_group_sizze_30925 + local_tid_30684;\n            \n            cid_30949 = squot32(flat_comb_id_31165, tile_sizze_30924);\n            cid_30950 = flat_comb_id_31165 - squot32(flat_comb_id_31165,\n                                                     tile_sizze_30924) *\n                tile_sizze_30924;\n            if ((slt32(cid_30949, chunk_sizze_30689) && slt32(cid_30950,\n                                                              tile_sizze_30924)) &&\n                slt32(gtid_30674, sizze_29813)) {\n                float x_chunk_outer_elem_30948 = *(__global\n                                                   float *) &mem_30986[((chunk_offset_30690 +\n                                 ",
            "                                        ltid_30927) *\n                                                                        sizze_29813 +\n                                                                        gtid_30674) *\n                                                                       4];\n                \n                *(__local float *) &mem_30999[(cid_30949 * tile_sizze_30924 +\n                                               cid_30950) * 4] =\n                    x_chunk_outer_elem_30948;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_31161) { }\n        \n        float res_30694;\n        float sync_30952;\n        float acc_30697 = x_30691;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_30695 = 1;\n        \n        if (thread_active_31161) {\n            if (chunk_sizze_30689 == tile_sizze_30924) {\n                for (int32_t i_30696 = 0; i_30696 < tile_sizze_30924;\n                     i_30696++) {\n                    float x_30700 = *(__local float *) &mem_30995[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30696) *\n                                                                  4];\n                    float x_30701 = *(__local float *) &mem_30999[(i_30696 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30703 = x_30700 * x_30701;\n                    float res_30705 = acc_30697 + res_30703;\n                    float acc_tmp_31166 = res_30705;\n                    \n                    acc_30697 = acc_tmp_31166;\n                }\n            } else {\n                for (int32_t i_30696 = 0; i_30696 < chunk_sizze_30689;\n                     i_30696++) {\n                 ",
            "   float x_30700 = *(__local float *) &mem_30995[(ltid_30927 *\n                                                                   tile_sizze_30924 +\n                                                                   i_30696) *\n                                                                  4];\n                    float x_30701 = *(__local float *) &mem_30999[(i_30696 *\n                                                                   tile_sizze_30924 +\n                                                                   ltid_30928) *\n                                                                  4];\n                    float res_30703 = x_30700 * x_30701;\n                    float res_30705 = acc_30697 + res_30703;\n                    float acc_tmp_31166 = res_30705;\n                    \n                    acc_30697 = acc_tmp_31166;\n                }\n            }\n        }\n        res_30694 = acc_30697;\n        sync_30952 = res_30694;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_30691 = sync_30952;\n        chunk_offset_30690 += tile_sizze_30924;\n    }\n    res_30688 = x_30691;\n    if (thread_active_31161) {\n        *(__global float *) &mem_31004[(gtid_30672 * (sizze_29807 *\n                                                      sizze_29813) +\n                                        gtid_30673 * sizze_29813 + gtid_30674) *\n                                       4] = res_30688;\n    }\n}\n__kernel void map_kernel_30717(int32_t sizze_29806, int32_t sizze_29807,\n                               int32_t sizze_29808, int32_t sizze_29810,\n                               int32_t sizze_29811, float a_29816,\n                               float b_29818, __global\n                               unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *mem_30991)\n{\n    int32_t wave_sizze_31156;\n    int32_t group_sizze_31157;\n    bool thread_active_31158;\n    int32_t gtid_30706;\n    int3",
            "2_t gtid_30707;\n    int32_t gtid_30708;\n    int32_t global_tid_30717;\n    int32_t local_tid_30718;\n    int32_t group_id_30719;\n    \n    global_tid_30717 = get_global_id(0);\n    local_tid_30718 = get_local_id(0);\n    group_sizze_31157 = get_local_size(0);\n    wave_sizze_31156 = LOCKSTEP_WIDTH;\n    group_id_30719 = get_group_id(0);\n    gtid_30706 = squot32(global_tid_30717, sizze_29807 * sizze_29808);\n    gtid_30707 = squot32(global_tid_30717 - squot32(global_tid_30717,\n                                                    sizze_29807 * sizze_29808) *\n                         (sizze_29807 * sizze_29808), sizze_29808);\n    gtid_30708 = global_tid_30717 - squot32(global_tid_30717, sizze_29807 *\n                                            sizze_29808) * (sizze_29807 *\n                                                            sizze_29808) -\n        squot32(global_tid_30717 - squot32(global_tid_30717, sizze_29807 *\n                                           sizze_29808) * (sizze_29807 *\n                                                           sizze_29808),\n                sizze_29808) * sizze_29808;\n    thread_active_31158 = (slt32(gtid_30706, sizze_29806) && slt32(gtid_30707,\n                                                                   sizze_29807)) &&\n        slt32(gtid_30708, sizze_29808);\n    \n    float x_30720;\n    float x_30721;\n    float res_30722;\n    float res_30723;\n    float res_30724;\n    \n    if (thread_active_31158) {\n        x_30720 = *(__global float *) &A_mem_30976[(gtid_30706 * (sizze_29807 *\n                                                                  sizze_29808) +\n                                                    gtid_30707 * sizze_29808 +\n                                                    gtid_30708) * 4];\n        x_30721 = *(__global float *) &B_mem_30978[(gtid_30706 * (sizze_29810 *\n                                                                  sizze_29811) +\n                                                    gtid_30707 * sizze_2",
            "9811 +\n                                                    gtid_30708) * 4];\n        res_30722 = a_29816 * x_30720;\n        res_30723 = b_29818 * x_30721;\n        res_30724 = res_30722 + res_30723;\n    }\n    if (thread_active_31158) {\n        *(__global float *) &mem_30991[(gtid_30706 * (sizze_29807 *\n                                                      sizze_29808) +\n                                        gtid_30707 * sizze_29808 + gtid_30708) *\n                                       4] = res_30724;\n    }\n}\n__kernel void map_kernel_30732(int32_t sizze_29895, float e_29915,\n                               float f_29917, __global\n                               unsigned char *E_mem_30984, __global\n                               unsigned char *F_mem_30986, __global\n                               unsigned char *mem_30997)\n{\n    int32_t wave_sizze_31175;\n    int32_t group_sizze_31176;\n    bool thread_active_31177;\n    int32_t gtid_30725;\n    int32_t global_tid_30732;\n    int32_t local_tid_30733;\n    int32_t group_id_30734;\n    \n    global_tid_30732 = get_global_id(0);\n    local_tid_30733 = get_local_id(0);\n    group_sizze_31176 = get_local_size(0);\n    wave_sizze_31175 = LOCKSTEP_WIDTH;\n    group_id_30734 = get_group_id(0);\n    gtid_30725 = global_tid_30732;\n    thread_active_31177 = slt32(gtid_30725, sizze_29895);\n    \n    float x_30735;\n    float x_30736;\n    float res_30737;\n    float res_30738;\n    float res_30739;\n    \n    if (thread_active_31177) {\n        x_30735 = *(__global float *) &E_mem_30984[gtid_30725 * 4];\n        x_30736 = *(__global float *) &F_mem_30986[gtid_30725 * 4];\n        res_30737 = e_29915 * x_30735;\n        res_30738 = f_29917 * x_30736;\n        res_30739 = res_30737 + res_30738;\n    }\n    if (thread_active_31177) {\n        *(__global float *) &mem_30997[gtid_30725 * 4] = res_30739;\n    }\n}\n__kernel void map_kernel_30749(int32_t sizze_29895, int32_t sizze_29900,\n                               int32_t sizze_29902, float c_29911,\n               ",
            "                float d_29913, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *D_mem_30982, __global\n                               unsigned char *mem_30994)\n{\n    int32_t wave_sizze_31172;\n    int32_t group_sizze_31173;\n    bool thread_active_31174;\n    int32_t gtid_30740;\n    int32_t gtid_30741;\n    int32_t global_tid_30749;\n    int32_t local_tid_30750;\n    int32_t group_id_30751;\n    \n    global_tid_30749 = get_global_id(0);\n    local_tid_30750 = get_local_id(0);\n    group_sizze_31173 = get_local_size(0);\n    wave_sizze_31172 = LOCKSTEP_WIDTH;\n    group_id_30751 = get_group_id(0);\n    gtid_30740 = squot32(global_tid_30749, sizze_29900);\n    gtid_30741 = global_tid_30749 - squot32(global_tid_30749, sizze_29900) *\n        sizze_29900;\n    thread_active_31174 = slt32(gtid_30740, sizze_29895) && slt32(gtid_30741,\n                                                                  sizze_29900);\n    \n    float x_30752;\n    float x_30753;\n    float res_30754;\n    float res_30755;\n    float res_30756;\n    \n    if (thread_active_31174) {\n        x_30752 = *(__global float *) &C_mem_30980[(gtid_30740 * sizze_29900 +\n                                                    gtid_30741) * 4];\n        x_30753 = *(__global float *) &D_mem_30982[(gtid_30740 * sizze_29902 +\n                                                    gtid_30741) * 4];\n        res_30754 = c_29911 * x_30752;\n        res_30755 = d_29913 * x_30753;\n        res_30756 = res_30754 + res_30755;\n    }\n    if (thread_active_31174) {\n        *(__global float *) &mem_30994[(gtid_30740 * sizze_29900 + gtid_30741) *\n                                       4] = res_30756;\n    }\n}\n__kernel void map_kernel_30764(int32_t sizze_29894, float g_29919,\n                               float h_29921, __global\n                               unsigned char *G_mem_30988, __global\n                               unsigned char *H_mem_30990, __global\n                     ",
            "          unsigned char *mem_31000)\n{\n    int32_t wave_sizze_31178;\n    int32_t group_sizze_31179;\n    bool thread_active_31180;\n    int32_t gtid_30757;\n    int32_t global_tid_30764;\n    int32_t local_tid_30765;\n    int32_t group_id_30766;\n    \n    global_tid_30764 = get_global_id(0);\n    local_tid_30765 = get_local_id(0);\n    group_sizze_31179 = get_local_size(0);\n    wave_sizze_31178 = LOCKSTEP_WIDTH;\n    group_id_30766 = get_group_id(0);\n    gtid_30757 = global_tid_30764;\n    thread_active_31180 = slt32(gtid_30757, sizze_29894);\n    \n    float x_30767;\n    float x_30768;\n    float res_30769;\n    float res_30770;\n    float res_30771;\n    \n    if (thread_active_31180) {\n        x_30767 = *(__global float *) &G_mem_30988[gtid_30757 * 4];\n        x_30768 = *(__global float *) &H_mem_30990[gtid_30757 * 4];\n        res_30769 = g_29919 * x_30767;\n        res_30770 = h_29921 * x_30768;\n        res_30771 = res_30769 + res_30770;\n    }\n    if (thread_active_31180) {\n        *(__global float *) &mem_31000[gtid_30757 * 4] = res_30771;\n    }\n}\n__kernel void map_kernel_30798(int32_t sizze_29893, int32_t sizze_29894,\n                               int32_t sizze_29895, int32_t sizze_29900,\n                               __global unsigned char *mem_30994, __global\n                               unsigned char *mem_30997, __global\n                               unsigned char *mem_31000, __global\n                               unsigned char *mem_31005, __global\n                               unsigned char *mem_31010)\n{\n    int32_t wave_sizze_31184;\n    int32_t group_sizze_31185;\n    bool thread_active_31186;\n    int32_t gtid_30787;\n    int32_t gtid_30788;\n    int32_t gtid_30789;\n    int32_t global_tid_30798;\n    int32_t local_tid_30799;\n    int32_t group_id_30800;\n    \n    global_tid_30798 = get_global_id(0);\n    local_tid_30799 = get_local_id(0);\n    group_sizze_31185 = get_local_size(0);\n    wave_sizze_31184 = LOCKSTEP_WIDTH;\n    group_id_30800 = get_group_id(0);\n    gtid_30787 = ",
            "squot32(global_tid_30798, sizze_29894 * sizze_29900);\n    gtid_30788 = squot32(global_tid_30798 - squot32(global_tid_30798,\n                                                    sizze_29894 * sizze_29900) *\n                         (sizze_29894 * sizze_29900), sizze_29900);\n    gtid_30789 = global_tid_30798 - squot32(global_tid_30798, sizze_29894 *\n                                            sizze_29900) * (sizze_29894 *\n                                                            sizze_29900) -\n        squot32(global_tid_30798 - squot32(global_tid_30798, sizze_29894 *\n                                           sizze_29900) * (sizze_29894 *\n                                                           sizze_29900),\n                sizze_29900) * sizze_29900;\n    thread_active_31186 = (slt32(gtid_30787, sizze_29893) && slt32(gtid_30788,\n                                                                   sizze_29894)) &&\n        slt32(gtid_30789, sizze_29900);\n    \n    float x_30801;\n    float res_30804;\n    \n    if (thread_active_31186) {\n        x_30801 = *(__global float *) &mem_31000[gtid_30788 * 4];\n        \n        float x_30807 = 0.0F;\n        \n        for (int32_t chunk_offset_30806 = 0; chunk_offset_30806 < sizze_29895;\n             chunk_offset_30806++) {\n            float x_30818 = *(__global float *) &mem_31005[(gtid_30787 *\n                                                            (sizze_29894 *\n                                                             sizze_29895) +\n                                                            gtid_30788 *\n                                                            sizze_29895 +\n                                                            chunk_offset_30806) *\n                                                           4];\n            float x_30819 = *(__global float *) &mem_30994[(chunk_offset_30806 *\n                                                            sizze_29900 +\n                                                    ",
            "        gtid_30789) * 4];\n            float x_30820 = *(__global float *) &mem_30997[chunk_offset_30806 *\n                                                           4];\n            float x_30822 = x_30818 * x_30819;\n            float x_30823 = x_30820 * x_30822;\n            float res_30824 = x_30801 * x_30823;\n            float res_30826 = x_30807 + res_30824;\n            float x_tmp_31187 = res_30826;\n            \n            x_30807 = x_tmp_31187;\n        }\n        res_30804 = x_30807;\n    }\n    if (thread_active_31186) {\n        *(__global float *) &mem_31010[(gtid_30787 * (sizze_29894 *\n                                                      sizze_29900) +\n                                        gtid_30788 * sizze_29900 + gtid_30789) *\n                                       4] = res_30804;\n    }\n}\n__kernel void map_kernel_30838(int32_t sizze_29893, int32_t sizze_29894,\n                               int32_t sizze_29895, int32_t sizze_29897,\n                               int32_t sizze_29898, float a_29907,\n                               float b_29909, __global\n                               unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *mem_31005)\n{\n    int32_t wave_sizze_31181;\n    int32_t group_sizze_31182;\n    bool thread_active_31183;\n    int32_t gtid_30827;\n    int32_t gtid_30828;\n    int32_t gtid_30829;\n    int32_t global_tid_30838;\n    int32_t local_tid_30839;\n    int32_t group_id_30840;\n    \n    global_tid_30838 = get_global_id(0);\n    local_tid_30839 = get_local_id(0);\n    group_sizze_31182 = get_local_size(0);\n    wave_sizze_31181 = LOCKSTEP_WIDTH;\n    group_id_30840 = get_group_id(0);\n    gtid_30827 = squot32(global_tid_30838, sizze_29894 * sizze_29895);\n    gtid_30828 = squot32(global_tid_30838 - squot32(global_tid_30838,\n                                                    sizze_29894 * sizze_29895) *\n                         (sizze_29894 * sizze_298",
            "95), sizze_29895);\n    gtid_30829 = global_tid_30838 - squot32(global_tid_30838, sizze_29894 *\n                                            sizze_29895) * (sizze_29894 *\n                                                            sizze_29895) -\n        squot32(global_tid_30838 - squot32(global_tid_30838, sizze_29894 *\n                                           sizze_29895) * (sizze_29894 *\n                                                           sizze_29895),\n                sizze_29895) * sizze_29895;\n    thread_active_31183 = (slt32(gtid_30827, sizze_29893) && slt32(gtid_30828,\n                                                                   sizze_29894)) &&\n        slt32(gtid_30829, sizze_29895);\n    \n    float x_30841;\n    float x_30842;\n    float res_30843;\n    float res_30844;\n    float res_30845;\n    \n    if (thread_active_31183) {\n        x_30841 = *(__global float *) &A_mem_30976[(gtid_30827 * (sizze_29894 *\n                                                                  sizze_29895) +\n                                                    gtid_30828 * sizze_29895 +\n                                                    gtid_30829) * 4];\n        x_30842 = *(__global float *) &B_mem_30978[(gtid_30827 * (sizze_29897 *\n                                                                  sizze_29898) +\n                                                    gtid_30828 * sizze_29898 +\n                                                    gtid_30829) * 4];\n        res_30843 = a_29907 * x_30841;\n        res_30844 = b_29909 * x_30842;\n        res_30845 = res_30843 + res_30844;\n    }\n    if (thread_active_31183) {\n        *(__global float *) &mem_31005[(gtid_30827 * (sizze_29894 *\n                                                      sizze_29895) +\n                                        gtid_30828 * sizze_29895 + gtid_30829) *\n                                       4] = res_30845;\n    }\n}\n__kernel void map_kernel_30879(__local volatile int64_t *mem_aligned_0,\n               ",
            "                __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_30032, int32_t sizze_30033,\n                               int32_t sizze_30034, int32_t sizze_30036,\n                               int32_t sizze_30038, int32_t inner_ldim_30929,\n                               __global unsigned char *A_mem_30976, __global\n                               unsigned char *B_mem_30978, __global\n                               unsigned char *C_mem_30980, __global\n                               unsigned char *mem_30991)\n{\n    __local volatile char *restrict mem_30983 = mem_aligned_0;\n    __local volatile char *restrict mem_30986 = mem_aligned_1;\n    int32_t wave_sizze_31193;\n    int32_t group_sizze_31194;\n    bool thread_active_31195;\n    int32_t gtid_30868;\n    int32_t gtid_30869;\n    int32_t gtid_30870;\n    int32_t global_tid_30879;\n    int32_t local_tid_30880;\n    int32_t group_id_30881;\n    int32_t ltid_30926;\n    int32_t ltid_30927;\n    int32_t inner_ltid_30928;\n    \n    global_tid_30879 = get_global_id(0);\n    local_tid_30880 = get_local_id(0);\n    group_sizze_31194 = get_local_size(0);\n    wave_sizze_31193 = LOCKSTEP_WIDTH;\n    group_id_30881 = get_group_id(0);\n    gtid_30868 = squot32(srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                                inner_ldim_30929), sizze_30036 *\n                         inner_ldim_30929) + squot32(squot32(global_tid_30879,\n                                                             sizze_30032 *\n                                                             sizze_30036 *\n                                                             inner_ldim_30929),\n                                                     squot32(sizze_30036 +\n                                                             sizze_30036 - 1,\n                                                             sizze_30036) *\n                                                     squot32(sizze_30038 +\n                                ",
            "                             inner_ldim_30929 -\n                                                             1,\n                                                             inner_ldim_30929)) *\n        sizze_30032;\n    gtid_30869 = squot32(srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                                inner_ldim_30929) -\n                         squot32(srem32(global_tid_30879, sizze_30032 *\n                                        sizze_30036 * inner_ldim_30929),\n                                 sizze_30036 * inner_ldim_30929) *\n                         (sizze_30036 * inner_ldim_30929), inner_ldim_30929) +\n        squot32(squot32(global_tid_30879, sizze_30032 * sizze_30036 *\n                        inner_ldim_30929) - squot32(squot32(global_tid_30879,\n                                                            sizze_30032 *\n                                                            sizze_30036 *\n                                                            inner_ldim_30929),\n                                                    squot32(sizze_30036 +\n                                                            sizze_30036 - 1,\n                                                            sizze_30036) *\n                                                    squot32(sizze_30038 +\n                                                            inner_ldim_30929 -\n                                                            1,\n                                                            inner_ldim_30929)) *\n                (squot32(sizze_30036 + sizze_30036 - 1, sizze_30036) *\n                 squot32(sizze_30038 + inner_ldim_30929 - 1, inner_ldim_30929)),\n                squot32(sizze_30038 + inner_ldim_30929 - 1, inner_ldim_30929)) *\n        sizze_30036;\n    gtid_30870 = srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                        inner_ldim_30929) - squot32(srem32(global_tid_30879,\n                                                           sizze_",
            "30032 *\n                                                           sizze_30036 *\n                                                           inner_ldim_30929),\n                                                    sizze_30036 *\n                                                    inner_ldim_30929) *\n        (sizze_30036 * inner_ldim_30929) - squot32(srem32(global_tid_30879,\n                                                          sizze_30032 *\n                                                          sizze_30036 *\n                                                          inner_ldim_30929) -\n                                                   squot32(srem32(global_tid_30879,\n                                                                  sizze_30032 *\n                                                                  sizze_30036 *\n                                                                  inner_ldim_30929),\n                                                           sizze_30036 *\n                                                           inner_ldim_30929) *\n                                                   (sizze_30036 *\n                                                    inner_ldim_30929),\n                                                   inner_ldim_30929) *\n        inner_ldim_30929 + (squot32(global_tid_30879, sizze_30032 *\n                                    sizze_30036 * inner_ldim_30929) -\n                            squot32(squot32(global_tid_30879, sizze_30032 *\n                                            sizze_30036 * inner_ldim_30929),\n                                    squot32(sizze_30036 + sizze_30036 - 1,\n                                            sizze_30036) * squot32(sizze_30038 +\n                                                                   inner_ldim_30929 -\n                                                                   1,\n                                                                   inner_ldim_30929)) *\n                         ",
            "   (squot32(sizze_30036 + sizze_30036 - 1,\n                                     sizze_30036) * squot32(sizze_30038 +\n                                                            inner_ldim_30929 -\n                                                            1,\n                                                            inner_ldim_30929)) -\n                            squot32(squot32(global_tid_30879, sizze_30032 *\n                                            sizze_30036 * inner_ldim_30929) -\n                                    squot32(squot32(global_tid_30879,\n                                                    sizze_30032 * sizze_30036 *\n                                                    inner_ldim_30929),\n                                            squot32(sizze_30036 + sizze_30036 -\n                                                    1, sizze_30036) *\n                                            squot32(sizze_30038 +\n                                                    inner_ldim_30929 - 1,\n                                                    inner_ldim_30929)) *\n                                    (squot32(sizze_30036 + sizze_30036 - 1,\n                                             sizze_30036) *\n                                     squot32(sizze_30038 + inner_ldim_30929 - 1,\n                                             inner_ldim_30929)),\n                                    squot32(sizze_30038 + inner_ldim_30929 - 1,\n                                            inner_ldim_30929)) *\n                            squot32(sizze_30038 + inner_ldim_30929 - 1,\n                                    inner_ldim_30929)) * inner_ldim_30929;\n    ltid_30926 = squot32(srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                                inner_ldim_30929), sizze_30036 *\n                         inner_ldim_30929);\n    ltid_30927 = squot32(srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                                inner_ldim_30929) -\n                         squot3",
            "2(srem32(global_tid_30879, sizze_30032 *\n                                        sizze_30036 * inner_ldim_30929),\n                                 sizze_30036 * inner_ldim_30929) *\n                         (sizze_30036 * inner_ldim_30929), inner_ldim_30929);\n    inner_ltid_30928 = srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                              inner_ldim_30929) -\n        squot32(srem32(global_tid_30879, sizze_30032 * sizze_30036 *\n                       inner_ldim_30929), sizze_30036 * inner_ldim_30929) *\n        (sizze_30036 * inner_ldim_30929) - squot32(srem32(global_tid_30879,\n                                                          sizze_30032 *\n                                                          sizze_30036 *\n                                                          inner_ldim_30929) -\n                                                   squot32(srem32(global_tid_30879,\n                                                                  sizze_30032 *\n                                                                  sizze_30036 *\n                                                                  inner_ldim_30929),\n                                                           sizze_30036 *\n                                                           inner_ldim_30929) *\n                                                   (sizze_30036 *\n                                                    inner_ldim_30929),\n                                                   inner_ldim_30929) *\n        inner_ldim_30929;\n    thread_active_31195 = (slt32(gtid_30868, sizze_30032) && slt32(gtid_30869,\n                                                                   sizze_30036)) &&\n        slt32(gtid_30870, sizze_30038);\n    \n    float res_30885;\n    \n    if (thread_active_31195) {\n        float x_30888 = 0.0F;\n        \n        for (int32_t chunk_offset_30887 = 0; chunk_offset_30887 < sizze_30033;\n             chunk_offset_30887++) {\n            float x_30898 = *(__globa",
            "l\n                              float *) &C_mem_30980[(chunk_offset_30887 *\n                                                     sizze_30038 + gtid_30870) *\n                                                    4];\n            float res_30900;\n            float x_30903 = 0.0F;\n            int32_t chunk_sizze_30901;\n            int32_t chunk_offset_30902 = 0;\n            \n            while (slt32(chunk_offset_30902, sizze_30034)) {\n                if (slt32(sizze_30034 - chunk_offset_30902,\n                          group_sizze_30874)) {\n                    chunk_sizze_30901 = sizze_30034 - chunk_offset_30902;\n                } else {\n                    chunk_sizze_30901 = group_sizze_30874;\n                }\n                \n                float res_30906;\n                float sync_30951;\n                \n                for (int32_t comb_iter_31197 = 0; comb_iter_31197 <\n                     squot32(group_sizze_30874 + inner_ldim_30929 - 1,\n                             inner_ldim_30929); comb_iter_31197++) {\n                    int32_t cid_30925;\n                    int32_t flat_comb_id_31198 = comb_iter_31197 *\n                            inner_ldim_30929 + local_tid_30880;\n                    \n                    cid_30925 = flat_comb_id_31198;\n                    if (slt32(cid_30925, chunk_sizze_30901) && 1) {\n                        float x_chunk_outer_elem_30924 = *(__global\n                                                           float *) &A_mem_30976[(gtid_30868 *\n                                                                                  (sizze_30033 *\n                                                                                   sizze_30034) +\n                                                                                  chunk_offset_30887 *\n                                                                                  sizze_30034 +\n                                                                                  (chunk_offset_30902 +\n  ",
            "                                                                                 local_tid_30880)) *\n                                                                                 4];\n                        \n                        *(__local float *) &mem_30983[cid_30925 * 4] =\n                            x_chunk_outer_elem_30924;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                for (int32_t comb_iter_31199 = 0; comb_iter_31199 <\n                     squot32(group_sizze_30874 + inner_ldim_30929 - 1,\n                             inner_ldim_30929); comb_iter_31199++) {\n                    int32_t cid_30939;\n                    int32_t flat_comb_id_31200 = comb_iter_31199 *\n                            inner_ldim_30929 + local_tid_30880;\n                    \n                    cid_30939 = flat_comb_id_31200;\n                    if (slt32(cid_30939, chunk_sizze_30901) && 1) {\n                        float x_chunk_outer_elem_30938 = *(__global\n                                                           float *) &B_mem_30978[((chunk_offset_30902 +\n                                                                                   local_tid_30880) *\n                                                                                  sizze_30036 +\n                                                                                  gtid_30869) *\n                                                                                 4];\n                        \n                        *(__local float *) &mem_30986[cid_30939 * 4] =\n                            x_chunk_outer_elem_30938;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                \n                float acc_30909 = x_30903;\n                int32_t groupstream_mapaccum_dummy_chunk_sizze_30907 = 1;\n                \n                if (chunk_sizze_30901 == group_sizze_30874) {\n                    for (int32_t i_30908 = 0; i_30908 < gro",
            "up_sizze_30874;\n                         i_30908++) {\n                        float x_30912 = *(__local float *) &mem_30983[i_30908 *\n                                                                      4];\n                        float x_30913 = *(__local float *) &mem_30986[i_30908 *\n                                                                      4];\n                        float res_30915 = x_30912 * x_30913;\n                        float res_30917 = acc_30909 + res_30915;\n                        float acc_tmp_31201 = res_30917;\n                        \n                        acc_30909 = acc_tmp_31201;\n                    }\n                } else {\n                    for (int32_t i_30908 = 0; i_30908 < chunk_sizze_30901;\n                         i_30908++) {\n                        float x_30912 = *(__local float *) &mem_30983[i_30908 *\n                                                                      4];\n                        float x_30913 = *(__local float *) &mem_30986[i_30908 *\n                                                                      4];\n                        float res_30915 = x_30912 * x_30913;\n                        float res_30917 = acc_30909 + res_30915;\n                        float acc_tmp_31201 = res_30917;\n                        \n                        acc_30909 = acc_tmp_31201;\n                    }\n                }\n                res_30906 = acc_30909;\n                sync_30951 = res_30906;\n                barrier(CLK_LOCAL_MEM_FENCE);\n                x_30903 = sync_30951;\n                chunk_offset_30902 += group_sizze_30874;\n            }\n            res_30900 = x_30903;\n            \n            float res_30918 = x_30898 * res_30900;\n            float res_30920 = x_30888 + res_30918;\n            float x_tmp_31196 = res_30920;\n            \n            x_30888 = x_tmp_31196;\n        }\n        res_30885 = x_30888;\n    }\n    if (thread_active_31195) {\n        *(__global float *) &mem_30991[(gtid_30868 * (sizze_300",
            "36 *\n                                                      sizze_30038) +\n                                        gtid_30869 * sizze_30038 + gtid_30870) *\n                                       4] = res_30885;\n    }\n}\n",
            NULL};
struct memblock_device {
    int *references;
    cl_mem mem;
    int64_t size;
    const char *desc;
} ;
struct memblock_local {
    int *references;
    unsigned char mem;
    int64_t size;
    const char *desc;
} ;
struct memblock {
    int *references;
    char *mem;
    int64_t size;
    const char *desc;
} ;
static const char *size_names[] = {"group_size_30096", "group_size_30138",
                                   "group_size_30214", "group_size_30286",
                                   "group_size_30329", "group_size_30372",
                                   "group_size_30418", "group_size_30464",
                                   "group_size_30551", "group_size_30566",
                                   "group_size_30630", "group_size_30645",
                                   "group_size_30711", "group_size_30726",
                                   "group_size_30743", "group_size_30758",
                                   "group_size_30792", "group_size_30832",
                                   "group_size_30873", "tile_size_30923"};
static const char *size_classes[] = {"group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "tile_size"};
static const char *size_entry_points[] = {"t9", "t9", "t2", "t10", "t10", "t11",
                                          "t11", "t3", "t4", "t5", "t5", "t6",
                                          "t6", "t7", "t7", "t7", "t7", "t7",
                                          "t8", "t1"};
int futhark_get_num_sizes(void)
{
    return 20;
}
const char *futhark_get_size_name(int i)
{
    return size_names[i];
}
const char *futhark_get_size_class(int i)
{
    return size_classes[i];
}
const char *futhark_get_size_entry(int i)
{
    return size_entry_points[i];
}
struct sizes {
    size_t group_sizze_30096;
    size_t group_sizze_30138;
    size_t group_sizze_30214;
    size_t group_sizze_30286;
    size_t group_sizze_30329;
    size_t group_sizze_30372;
    size_t group_sizze_30418;
    size_t group_sizze_30464;
    size_t group_sizze_30551;
    size_t group_sizze_30566;
    size_t group_sizze_30630;
    size_t group_sizze_30645;
    size_t group_sizze_30711;
    size_t group_sizze_30726;
    size_t group_sizze_30743;
    size_t group_sizze_30758;
    size_t group_sizze_30792;
    size_t group_sizze_30832;
    size_t group_sizze_30873;
    size_t tile_sizze_30923;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[20];
} ;
struct futhark_context_config *futhark_context_config_new(void)
{
    struct futhark_context_config *cfg =
                                  malloc(sizeof(struct futhark_context_config));
    
    if (cfg == NULL)
        return NULL;
    cfg->sizes[0] = 0;
    cfg->sizes[1] = 0;
    cfg->sizes[2] = 0;
    cfg->sizes[3] = 0;
    cfg->sizes[4] = 0;
    cfg->sizes[5] = 0;
    cfg->sizes[6] = 0;
    cfg->sizes[7] = 0;
    cfg->sizes[8] = 0;
    cfg->sizes[9] = 0;
    cfg->sizes[10] = 0;
    cfg->sizes[11] = 0;
    cfg->sizes[12] = 0;
    cfg->sizes[13] = 0;
    cfg->sizes[14] = 0;
    cfg->sizes[15] = 0;
    cfg->sizes[16] = 0;
    cfg->sizes[17] = 0;
    cfg->sizes[18] = 0;
    cfg->sizes[19] = 0;
    opencl_config_init(&cfg->opencl, 20, size_names, cfg->sizes, size_classes,
                       size_entry_points);
    cfg->opencl.transpose_block_dim = 16;
    return cfg;
}
void futhark_context_config_free(struct futhark_context_config *cfg)
{
    free(cfg);
}
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag)
{
    cfg->opencl.logging = cfg->opencl.debugging = flag;
}
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag)
{
    cfg->opencl.logging = flag;
}
void futhark_context_config_set_device(struct futhark_context_config *cfg, const
                                       char *s)
{
    set_preferred_device(&cfg->opencl, s);
}
void futhark_context_config_set_platform(struct futhark_context_config *cfg,
                                         const char *s)
{
    set_preferred_platform(&cfg->opencl, s);
}
void futhark_context_config_dump_program_to(struct futhark_context_config *cfg,
                                            const char *path)
{
    cfg->opencl.dump_program_to = path;
}
void futhark_context_config_load_program_from(struct futhark_context_config *cfg,
                                              const char *path)
{
    cfg->opencl.load_program_from = path;
}
void futhark_context_config_set_default_group_size(struct futhark_context_config *cfg,
                                                   int size)
{
    cfg->opencl.default_group_size = size;
    cfg->opencl.default_group_size_changed = 1;
}
void futhark_context_config_set_default_num_groups(struct futhark_context_config *cfg,
                                                   int num)
{
    cfg->opencl.default_num_groups = num;
}
void futhark_context_config_set_default_tile_size(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_tile_size = size;
    cfg->opencl.default_tile_size_changed = 1;
}
void futhark_context_config_set_default_threshold(struct futhark_context_config *cfg,
                                                  int size)
{
    cfg->opencl.default_threshold = size;
}
int futhark_context_config_set_size(struct futhark_context_config *cfg, const
                                    char *size_name, size_t size_value)
{
    for (int i = 0; i < 20; i++) {
        if (strcmp(size_name, size_names[i]) == 0) {
            cfg->sizes[i] = size_value;
            return 0;
        }
    }
    return 1;
}
struct futhark_context {
    int detail_memory;
    int debugging;
    int logging;
    lock_t lock;
    char *error;
    int64_t peak_mem_usage_device;
    int64_t cur_mem_usage_device;
    int64_t peak_mem_usage_local;
    int64_t cur_mem_usage_local;
    int64_t peak_mem_usage_default;
    int64_t cur_mem_usage_default;
    int total_runs;
    long total_runtime;
    cl_kernel fut_kernel_map_transpose_f32;
    int fut_kernel_map_transpose_f32_total_runtime;
    int fut_kernel_map_transpose_f32_runs;
    cl_kernel fut_kernel_map_transpose_lowheight_f32;
    int fut_kernel_map_transpose_lowheight_f32_total_runtime;
    int fut_kernel_map_transpose_lowheight_f32_runs;
    cl_kernel fut_kernel_map_transpose_lowwidth_f32;
    int fut_kernel_map_transpose_lowwidth_f32_total_runtime;
    int fut_kernel_map_transpose_lowwidth_f32_runs;
    cl_kernel fut_kernel_map_transpose_small_f32;
    int fut_kernel_map_transpose_small_f32_total_runtime;
    int fut_kernel_map_transpose_small_f32_runs;
    cl_kernel map_kernel_30102;
    int map_kernel_30102_total_runtime;
    int map_kernel_30102_runs;
    cl_kernel map_kernel_30144;
    int map_kernel_30144_total_runtime;
    int map_kernel_30144_runs;
    cl_kernel map_kernel_30188;
    int map_kernel_30188_total_runtime;
    int map_kernel_30188_runs;
    cl_kernel map_kernel_30220;
    int map_kernel_30220_total_runtime;
    int map_kernel_30220_runs;
    cl_kernel map_kernel_30248;
    int map_kernel_30248_total_runtime;
    int map_kernel_30248_runs;
    cl_kernel map_kernel_30292;
    int map_kernel_30292_total_runtime;
    int map_kernel_30292_runs;
    cl_kernel map_kernel_30335;
    int map_kernel_30335_total_runtime;
    int map_kernel_30335_runs;
    cl_kernel map_kernel_30378;
    int map_kernel_30378_total_runtime;
    int map_kernel_30378_runs;
    cl_kernel map_kernel_30424;
    int map_kernel_30424_total_runtime;
    int map_kernel_30424_runs;
    cl_kernel map_kernel_30470;
    int map_kernel_30470_total_runtime;
    int map_kernel_30470_runs;
    cl_kernel map_kernel_30523;
    int map_kernel_30523_total_runtime;
    int map_kernel_30523_runs;
    cl_kernel map_kernel_30557;
    int map_kernel_30557_total_runtime;
    int map_kernel_30557_runs;
    cl_kernel map_kernel_30572;
    int map_kernel_30572_total_runtime;
    int map_kernel_30572_runs;
    cl_kernel map_kernel_30602;
    int map_kernel_30602_total_runtime;
    int map_kernel_30602_runs;
    cl_kernel map_kernel_30636;
    int map_kernel_30636_total_runtime;
    int map_kernel_30636_runs;
    cl_kernel map_kernel_30651;
    int map_kernel_30651_total_runtime;
    int map_kernel_30651_runs;
    cl_kernel map_kernel_30683;
    int map_kernel_30683_total_runtime;
    int map_kernel_30683_runs;
    cl_kernel map_kernel_30717;
    int map_kernel_30717_total_runtime;
    int map_kernel_30717_runs;
    cl_kernel map_kernel_30732;
    int map_kernel_30732_total_runtime;
    int map_kernel_30732_runs;
    cl_kernel map_kernel_30749;
    int map_kernel_30749_total_runtime;
    int map_kernel_30749_runs;
    cl_kernel map_kernel_30764;
    int map_kernel_30764_total_runtime;
    int map_kernel_30764_runs;
    cl_kernel map_kernel_30798;
    int map_kernel_30798_total_runtime;
    int map_kernel_30798_runs;
    cl_kernel map_kernel_30838;
    int map_kernel_30838_total_runtime;
    int map_kernel_30838_runs;
    cl_kernel map_kernel_30879;
    int map_kernel_30879_total_runtime;
    int map_kernel_30879_runs;
    struct opencl_context opencl;
    struct sizes sizes;
} ;
void post_opencl_setup(struct opencl_context *ctx,
                       struct opencl_device_option *option)
{
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "NVIDIA CUDA") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name,
                                            "AMD Accelerated Parallel Processing") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 64;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_num_groups = 128;
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_group_size = 256;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_GPU)
        ctx->cfg.default_tile_size = 32;
    if ((ctx->lockstep_width == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->lockstep_width = 1;
    if ((ctx->cfg.default_num_groups == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_CPU)
        clGetDeviceInfo(ctx->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(ctx->cfg.default_num_groups),
                        &ctx->cfg.default_num_groups, NULL);
    if ((ctx->cfg.default_group_size == 0 && strstr(option->platform_name,
                                                    "") != NULL) &&
        option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_group_size = 32;
    if ((ctx->cfg.default_tile_size == 0 && strstr(option->platform_name, "") !=
         NULL) && option->device_type == CL_DEVICE_TYPE_CPU)
        ctx->cfg.default_tile_size = 4;
}
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg)
{
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    
    if (ctx == NULL)
        return NULL;
    ctx->detail_memory = cfg->opencl.debugging;
    ctx->debugging = cfg->opencl.debugging;
    ctx->logging = cfg->opencl.logging;
    ctx->opencl.cfg = cfg->opencl;
    ctx->error = NULL;
    create_lock(&ctx->lock);
    ctx->peak_mem_usage_device = 0;
    ctx->cur_mem_usage_device = 0;
    ctx->peak_mem_usage_local = 0;
    ctx->cur_mem_usage_local = 0;
    ctx->peak_mem_usage_default = 0;
    ctx->cur_mem_usage_default = 0;
    ctx->total_runs = 0;
    ctx->total_runtime = 0;
    ctx->fut_kernel_map_transpose_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_runs = 0;
    ctx->fut_kernel_map_transpose_small_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_f32_runs = 0;
    ctx->map_kernel_30102_total_runtime = 0;
    ctx->map_kernel_30102_runs = 0;
    ctx->map_kernel_30144_total_runtime = 0;
    ctx->map_kernel_30144_runs = 0;
    ctx->map_kernel_30188_total_runtime = 0;
    ctx->map_kernel_30188_runs = 0;
    ctx->map_kernel_30220_total_runtime = 0;
    ctx->map_kernel_30220_runs = 0;
    ctx->map_kernel_30248_total_runtime = 0;
    ctx->map_kernel_30248_runs = 0;
    ctx->map_kernel_30292_total_runtime = 0;
    ctx->map_kernel_30292_runs = 0;
    ctx->map_kernel_30335_total_runtime = 0;
    ctx->map_kernel_30335_runs = 0;
    ctx->map_kernel_30378_total_runtime = 0;
    ctx->map_kernel_30378_runs = 0;
    ctx->map_kernel_30424_total_runtime = 0;
    ctx->map_kernel_30424_runs = 0;
    ctx->map_kernel_30470_total_runtime = 0;
    ctx->map_kernel_30470_runs = 0;
    ctx->map_kernel_30523_total_runtime = 0;
    ctx->map_kernel_30523_runs = 0;
    ctx->map_kernel_30557_total_runtime = 0;
    ctx->map_kernel_30557_runs = 0;
    ctx->map_kernel_30572_total_runtime = 0;
    ctx->map_kernel_30572_runs = 0;
    ctx->map_kernel_30602_total_runtime = 0;
    ctx->map_kernel_30602_runs = 0;
    ctx->map_kernel_30636_total_runtime = 0;
    ctx->map_kernel_30636_runs = 0;
    ctx->map_kernel_30651_total_runtime = 0;
    ctx->map_kernel_30651_runs = 0;
    ctx->map_kernel_30683_total_runtime = 0;
    ctx->map_kernel_30683_runs = 0;
    ctx->map_kernel_30717_total_runtime = 0;
    ctx->map_kernel_30717_runs = 0;
    ctx->map_kernel_30732_total_runtime = 0;
    ctx->map_kernel_30732_runs = 0;
    ctx->map_kernel_30749_total_runtime = 0;
    ctx->map_kernel_30749_runs = 0;
    ctx->map_kernel_30764_total_runtime = 0;
    ctx->map_kernel_30764_runs = 0;
    ctx->map_kernel_30798_total_runtime = 0;
    ctx->map_kernel_30798_runs = 0;
    ctx->map_kernel_30838_total_runtime = 0;
    ctx->map_kernel_30838_runs = 0;
    ctx->map_kernel_30879_total_runtime = 0;
    ctx->map_kernel_30879_runs = 0;
    
    int required_types = 0;
    cl_int error;
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    {
        ctx->fut_kernel_map_transpose_f32 = clCreateKernel(prog,
                                                           "fut_kernel_map_transpose_f32",
                                                           &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_f32");
    }
    {
        ctx->fut_kernel_map_transpose_lowheight_f32 = clCreateKernel(prog,
                                                                     "fut_kernel_map_transpose_lowheight_f32",
                                                                     &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowheight_f32");
    }
    {
        ctx->fut_kernel_map_transpose_lowwidth_f32 = clCreateKernel(prog,
                                                                    "fut_kernel_map_transpose_lowwidth_f32",
                                                                    &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_lowwidth_f32");
    }
    {
        ctx->fut_kernel_map_transpose_small_f32 = clCreateKernel(prog,
                                                                 "fut_kernel_map_transpose_small_f32",
                                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_small_f32");
    }
    {
        ctx->map_kernel_30102 = clCreateKernel(prog, "map_kernel_30102",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30102");
    }
    {
        ctx->map_kernel_30144 = clCreateKernel(prog, "map_kernel_30144",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30144");
    }
    {
        ctx->map_kernel_30188 = clCreateKernel(prog, "map_kernel_30188",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30188");
    }
    {
        ctx->map_kernel_30220 = clCreateKernel(prog, "map_kernel_30220",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30220");
    }
    {
        ctx->map_kernel_30248 = clCreateKernel(prog, "map_kernel_30248",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30248");
    }
    {
        ctx->map_kernel_30292 = clCreateKernel(prog, "map_kernel_30292",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30292");
    }
    {
        ctx->map_kernel_30335 = clCreateKernel(prog, "map_kernel_30335",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30335");
    }
    {
        ctx->map_kernel_30378 = clCreateKernel(prog, "map_kernel_30378",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30378");
    }
    {
        ctx->map_kernel_30424 = clCreateKernel(prog, "map_kernel_30424",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30424");
    }
    {
        ctx->map_kernel_30470 = clCreateKernel(prog, "map_kernel_30470",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30470");
    }
    {
        ctx->map_kernel_30523 = clCreateKernel(prog, "map_kernel_30523",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30523");
    }
    {
        ctx->map_kernel_30557 = clCreateKernel(prog, "map_kernel_30557",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30557");
    }
    {
        ctx->map_kernel_30572 = clCreateKernel(prog, "map_kernel_30572",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30572");
    }
    {
        ctx->map_kernel_30602 = clCreateKernel(prog, "map_kernel_30602",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30602");
    }
    {
        ctx->map_kernel_30636 = clCreateKernel(prog, "map_kernel_30636",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30636");
    }
    {
        ctx->map_kernel_30651 = clCreateKernel(prog, "map_kernel_30651",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30651");
    }
    {
        ctx->map_kernel_30683 = clCreateKernel(prog, "map_kernel_30683",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30683");
    }
    {
        ctx->map_kernel_30717 = clCreateKernel(prog, "map_kernel_30717",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30717");
    }
    {
        ctx->map_kernel_30732 = clCreateKernel(prog, "map_kernel_30732",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30732");
    }
    {
        ctx->map_kernel_30749 = clCreateKernel(prog, "map_kernel_30749",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30749");
    }
    {
        ctx->map_kernel_30764 = clCreateKernel(prog, "map_kernel_30764",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30764");
    }
    {
        ctx->map_kernel_30798 = clCreateKernel(prog, "map_kernel_30798",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30798");
    }
    {
        ctx->map_kernel_30838 = clCreateKernel(prog, "map_kernel_30838",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30838");
    }
    {
        ctx->map_kernel_30879 = clCreateKernel(prog, "map_kernel_30879",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_30879");
    }
    ctx->sizes.group_sizze_30096 = cfg->sizes[0];
    ctx->sizes.group_sizze_30138 = cfg->sizes[1];
    ctx->sizes.group_sizze_30214 = cfg->sizes[2];
    ctx->sizes.group_sizze_30286 = cfg->sizes[3];
    ctx->sizes.group_sizze_30329 = cfg->sizes[4];
    ctx->sizes.group_sizze_30372 = cfg->sizes[5];
    ctx->sizes.group_sizze_30418 = cfg->sizes[6];
    ctx->sizes.group_sizze_30464 = cfg->sizes[7];
    ctx->sizes.group_sizze_30551 = cfg->sizes[8];
    ctx->sizes.group_sizze_30566 = cfg->sizes[9];
    ctx->sizes.group_sizze_30630 = cfg->sizes[10];
    ctx->sizes.group_sizze_30645 = cfg->sizes[11];
    ctx->sizes.group_sizze_30711 = cfg->sizes[12];
    ctx->sizes.group_sizze_30726 = cfg->sizes[13];
    ctx->sizes.group_sizze_30743 = cfg->sizes[14];
    ctx->sizes.group_sizze_30758 = cfg->sizes[15];
    ctx->sizes.group_sizze_30792 = cfg->sizes[16];
    ctx->sizes.group_sizze_30832 = cfg->sizes[17];
    ctx->sizes.group_sizze_30873 = cfg->sizes[18];
    ctx->sizes.tile_sizze_30923 = cfg->sizes[19];
    return ctx;
}
void futhark_context_free(struct futhark_context *ctx)
{
    free_lock(&ctx->lock);
    free(ctx);
}
int futhark_context_sync(struct futhark_context *ctx)
{
    OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    return 0;
}
char *futhark_context_get_error(struct futhark_context *ctx)
{
    char *error = ctx->error;
    
    ctx->error = NULL;
    return error;
}
int futhark_context_clear_caches(struct futhark_context *ctx)
{
    OPENCL_SUCCEED(opencl_free_all(&ctx->opencl));
    return 0;
}
static void memblock_unref_device(struct futhark_context *ctx,
                                  struct memblock_device *block, const
                                  char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'device'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_device -= block->size;
            OPENCL_SUCCEED(opencl_free(&ctx->opencl, block->mem, block->desc));
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_device);
        }
        block->references = NULL;
    }
}
static void memblock_alloc_device(struct futhark_context *ctx,
                                  struct memblock_device *block, int64_t size,
                                  const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'device'",
              ctx->cur_mem_usage_device);
    memblock_unref_device(ctx, block, desc);
    OPENCL_SUCCEED(opencl_alloc(&ctx->opencl, size, desc, &block->mem));
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_device += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'device'",
                (long long) ctx->cur_mem_usage_device);
    if (ctx->cur_mem_usage_device > ctx->peak_mem_usage_device) {
        ctx->peak_mem_usage_device = ctx->cur_mem_usage_device;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set_device(struct futhark_context *ctx,
                                struct memblock_device *lhs,
                                struct memblock_device *rhs, const
                                char *lhs_desc)
{
    memblock_unref_device(ctx, lhs, lhs_desc);
    (*rhs->references)++;
    *lhs = *rhs;
}
static void memblock_unref_local(struct futhark_context *ctx,
                                 struct memblock_local *block, const char *desc)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (ctx->detail_memory)
            fprintf(stderr,
                    "Unreferencing block %s (allocated as %s) in %s: %d references remaining.\n",
                    desc, block->desc, "space 'local'", *block->references);
        if (*block->references == 0) {
            ctx->cur_mem_usage_local -= block->size;
            free(block->references);
            if (ctx->detail_memory)
                fprintf(stderr,
                        "%lld bytes freed (now allocated: %lld bytes)\n",
                        (long long) block->size,
                        (long long) ctx->cur_mem_usage_local);
        }
        block->references = NULL;
    }
}
static void memblock_alloc_local(struct futhark_context *ctx,
                                 struct memblock_local *block, int64_t size,
                                 const char *desc)
{
    if (size < 0)
        panic(1, "Negative allocation of %lld bytes attempted for %s in %s.\n",
              (long long) size, desc, "space 'local'",
              ctx->cur_mem_usage_local);
    memblock_unref_local(ctx, block, desc);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
    block->size = size;
    block->desc = desc;
    ctx->cur_mem_usage_local += size;
    if (ctx->detail_memory)
        fprintf(stderr,
                "Allocated %lld bytes for %s in %s (now allocated: %lld bytes)",
                (long long) size, desc, "space 'local'",
                (long long) ctx->cur_mem_usage_local);
    if (ctx->cur_mem_usage_local > ctx->peak_mem_usage_local) {
        ctx->peak_mem_usage_local = ctx->cur_mem_usage_local;
        if (ctx->detail_memory)
            fprintf(stderr, " (new peak).\n");
    } else if (ctx->detail_memory)
        fprintf(stderr, ".\n");
}
static void memblock_set_local(struct futhark_context *ctx,
                               struct memblock_local *lhs,
                               struct memblock_local *rhs, const char *lhs_desc)
{
    memblock_unref_local(ctx, lhs, lhs_desc);
    (*rhs->references)++;
    *lhs = *rhs;
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
        fprintf(stderr, "Peak memory usage for space 'device': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_device);
        fprintf(stderr, "Peak memory usage for space 'local': %lld bytes.\n",
                (long long) ctx->peak_mem_usage_local);
        fprintf(stderr, "Peak memory usage for default space: %lld bytes.\n",
                (long long) ctx->peak_mem_usage_default);
    }
    if (ctx->debugging) {
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_f32           executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_f32_runs,
                (long) ctx->fut_kernel_map_transpose_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_f32_total_runtime);
        ctx->total_runtime += ctx->fut_kernel_map_transpose_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_f32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowheight_f32 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowheight_f32_runs,
                (long) ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowheight_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowheight_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowheight_f32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_lowwidth_f32  executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_lowwidth_f32_runs,
                (long) ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_lowwidth_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_lowwidth_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_lowwidth_f32_runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_small_f32     executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->fut_kernel_map_transpose_small_f32_runs,
                (long) ctx->fut_kernel_map_transpose_small_f32_total_runtime /
                (ctx->fut_kernel_map_transpose_small_f32_runs !=
                 0 ? ctx->fut_kernel_map_transpose_small_f32_runs : 1),
                (long) ctx->fut_kernel_map_transpose_small_f32_total_runtime);
        ctx->total_runtime +=
            ctx->fut_kernel_map_transpose_small_f32_total_runtime;
        ctx->total_runs += ctx->fut_kernel_map_transpose_small_f32_runs;
        fprintf(stderr,
                "Kernel map_kernel_30102                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30102_runs,
                (long) ctx->map_kernel_30102_total_runtime /
                (ctx->map_kernel_30102_runs !=
                 0 ? ctx->map_kernel_30102_runs : 1),
                (long) ctx->map_kernel_30102_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30102_total_runtime;
        ctx->total_runs += ctx->map_kernel_30102_runs;
        fprintf(stderr,
                "Kernel map_kernel_30144                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30144_runs,
                (long) ctx->map_kernel_30144_total_runtime /
                (ctx->map_kernel_30144_runs !=
                 0 ? ctx->map_kernel_30144_runs : 1),
                (long) ctx->map_kernel_30144_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30144_total_runtime;
        ctx->total_runs += ctx->map_kernel_30144_runs;
        fprintf(stderr,
                "Kernel map_kernel_30188                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30188_runs,
                (long) ctx->map_kernel_30188_total_runtime /
                (ctx->map_kernel_30188_runs !=
                 0 ? ctx->map_kernel_30188_runs : 1),
                (long) ctx->map_kernel_30188_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30188_total_runtime;
        ctx->total_runs += ctx->map_kernel_30188_runs;
        fprintf(stderr,
                "Kernel map_kernel_30220                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30220_runs,
                (long) ctx->map_kernel_30220_total_runtime /
                (ctx->map_kernel_30220_runs !=
                 0 ? ctx->map_kernel_30220_runs : 1),
                (long) ctx->map_kernel_30220_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30220_total_runtime;
        ctx->total_runs += ctx->map_kernel_30220_runs;
        fprintf(stderr,
                "Kernel map_kernel_30248                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30248_runs,
                (long) ctx->map_kernel_30248_total_runtime /
                (ctx->map_kernel_30248_runs !=
                 0 ? ctx->map_kernel_30248_runs : 1),
                (long) ctx->map_kernel_30248_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30248_total_runtime;
        ctx->total_runs += ctx->map_kernel_30248_runs;
        fprintf(stderr,
                "Kernel map_kernel_30292                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30292_runs,
                (long) ctx->map_kernel_30292_total_runtime /
                (ctx->map_kernel_30292_runs !=
                 0 ? ctx->map_kernel_30292_runs : 1),
                (long) ctx->map_kernel_30292_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30292_total_runtime;
        ctx->total_runs += ctx->map_kernel_30292_runs;
        fprintf(stderr,
                "Kernel map_kernel_30335                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30335_runs,
                (long) ctx->map_kernel_30335_total_runtime /
                (ctx->map_kernel_30335_runs !=
                 0 ? ctx->map_kernel_30335_runs : 1),
                (long) ctx->map_kernel_30335_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30335_total_runtime;
        ctx->total_runs += ctx->map_kernel_30335_runs;
        fprintf(stderr,
                "Kernel map_kernel_30378                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30378_runs,
                (long) ctx->map_kernel_30378_total_runtime /
                (ctx->map_kernel_30378_runs !=
                 0 ? ctx->map_kernel_30378_runs : 1),
                (long) ctx->map_kernel_30378_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30378_total_runtime;
        ctx->total_runs += ctx->map_kernel_30378_runs;
        fprintf(stderr,
                "Kernel map_kernel_30424                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30424_runs,
                (long) ctx->map_kernel_30424_total_runtime /
                (ctx->map_kernel_30424_runs !=
                 0 ? ctx->map_kernel_30424_runs : 1),
                (long) ctx->map_kernel_30424_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30424_total_runtime;
        ctx->total_runs += ctx->map_kernel_30424_runs;
        fprintf(stderr,
                "Kernel map_kernel_30470                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30470_runs,
                (long) ctx->map_kernel_30470_total_runtime /
                (ctx->map_kernel_30470_runs !=
                 0 ? ctx->map_kernel_30470_runs : 1),
                (long) ctx->map_kernel_30470_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30470_total_runtime;
        ctx->total_runs += ctx->map_kernel_30470_runs;
        fprintf(stderr,
                "Kernel map_kernel_30523                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30523_runs,
                (long) ctx->map_kernel_30523_total_runtime /
                (ctx->map_kernel_30523_runs !=
                 0 ? ctx->map_kernel_30523_runs : 1),
                (long) ctx->map_kernel_30523_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30523_total_runtime;
        ctx->total_runs += ctx->map_kernel_30523_runs;
        fprintf(stderr,
                "Kernel map_kernel_30557                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30557_runs,
                (long) ctx->map_kernel_30557_total_runtime /
                (ctx->map_kernel_30557_runs !=
                 0 ? ctx->map_kernel_30557_runs : 1),
                (long) ctx->map_kernel_30557_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30557_total_runtime;
        ctx->total_runs += ctx->map_kernel_30557_runs;
        fprintf(stderr,
                "Kernel map_kernel_30572                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30572_runs,
                (long) ctx->map_kernel_30572_total_runtime /
                (ctx->map_kernel_30572_runs !=
                 0 ? ctx->map_kernel_30572_runs : 1),
                (long) ctx->map_kernel_30572_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30572_total_runtime;
        ctx->total_runs += ctx->map_kernel_30572_runs;
        fprintf(stderr,
                "Kernel map_kernel_30602                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30602_runs,
                (long) ctx->map_kernel_30602_total_runtime /
                (ctx->map_kernel_30602_runs !=
                 0 ? ctx->map_kernel_30602_runs : 1),
                (long) ctx->map_kernel_30602_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30602_total_runtime;
        ctx->total_runs += ctx->map_kernel_30602_runs;
        fprintf(stderr,
                "Kernel map_kernel_30636                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30636_runs,
                (long) ctx->map_kernel_30636_total_runtime /
                (ctx->map_kernel_30636_runs !=
                 0 ? ctx->map_kernel_30636_runs : 1),
                (long) ctx->map_kernel_30636_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30636_total_runtime;
        ctx->total_runs += ctx->map_kernel_30636_runs;
        fprintf(stderr,
                "Kernel map_kernel_30651                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30651_runs,
                (long) ctx->map_kernel_30651_total_runtime /
                (ctx->map_kernel_30651_runs !=
                 0 ? ctx->map_kernel_30651_runs : 1),
                (long) ctx->map_kernel_30651_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30651_total_runtime;
        ctx->total_runs += ctx->map_kernel_30651_runs;
        fprintf(stderr,
                "Kernel map_kernel_30683                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30683_runs,
                (long) ctx->map_kernel_30683_total_runtime /
                (ctx->map_kernel_30683_runs !=
                 0 ? ctx->map_kernel_30683_runs : 1),
                (long) ctx->map_kernel_30683_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30683_total_runtime;
        ctx->total_runs += ctx->map_kernel_30683_runs;
        fprintf(stderr,
                "Kernel map_kernel_30717                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30717_runs,
                (long) ctx->map_kernel_30717_total_runtime /
                (ctx->map_kernel_30717_runs !=
                 0 ? ctx->map_kernel_30717_runs : 1),
                (long) ctx->map_kernel_30717_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30717_total_runtime;
        ctx->total_runs += ctx->map_kernel_30717_runs;
        fprintf(stderr,
                "Kernel map_kernel_30732                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30732_runs,
                (long) ctx->map_kernel_30732_total_runtime /
                (ctx->map_kernel_30732_runs !=
                 0 ? ctx->map_kernel_30732_runs : 1),
                (long) ctx->map_kernel_30732_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30732_total_runtime;
        ctx->total_runs += ctx->map_kernel_30732_runs;
        fprintf(stderr,
                "Kernel map_kernel_30749                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30749_runs,
                (long) ctx->map_kernel_30749_total_runtime /
                (ctx->map_kernel_30749_runs !=
                 0 ? ctx->map_kernel_30749_runs : 1),
                (long) ctx->map_kernel_30749_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30749_total_runtime;
        ctx->total_runs += ctx->map_kernel_30749_runs;
        fprintf(stderr,
                "Kernel map_kernel_30764                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30764_runs,
                (long) ctx->map_kernel_30764_total_runtime /
                (ctx->map_kernel_30764_runs !=
                 0 ? ctx->map_kernel_30764_runs : 1),
                (long) ctx->map_kernel_30764_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30764_total_runtime;
        ctx->total_runs += ctx->map_kernel_30764_runs;
        fprintf(stderr,
                "Kernel map_kernel_30798                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30798_runs,
                (long) ctx->map_kernel_30798_total_runtime /
                (ctx->map_kernel_30798_runs !=
                 0 ? ctx->map_kernel_30798_runs : 1),
                (long) ctx->map_kernel_30798_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30798_total_runtime;
        ctx->total_runs += ctx->map_kernel_30798_runs;
        fprintf(stderr,
                "Kernel map_kernel_30838                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30838_runs,
                (long) ctx->map_kernel_30838_total_runtime /
                (ctx->map_kernel_30838_runs !=
                 0 ? ctx->map_kernel_30838_runs : 1),
                (long) ctx->map_kernel_30838_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30838_total_runtime;
        ctx->total_runs += ctx->map_kernel_30838_runs;
        fprintf(stderr,
                "Kernel map_kernel_30879                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_30879_runs,
                (long) ctx->map_kernel_30879_total_runtime /
                (ctx->map_kernel_30879_runs !=
                 0 ? ctx->map_kernel_30879_runs : 1),
                (long) ctx->map_kernel_30879_total_runtime);
        ctx->total_runtime += ctx->map_kernel_30879_total_runtime;
        ctx->total_runs += ctx->map_kernel_30879_runs;
        if (ctx->debugging)
            fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_map_transpose_opencl_f32(struct futhark_context *ctx,
                                           struct memblock_device destmem_0,
                                           int32_t destoffset_1,
                                           struct memblock_device srcmem_2,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8);
static int futrts_t9(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31222,
                     struct memblock_device *out_mem_p_31223,
                     int32_t *out_out_arrsizze_31224,
                     int32_t *out_out_arrsizze_31225, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29396,
                     int32_t sizze_29397, int32_t sizze_29398,
                     int32_t sizze_29399, int32_t sizze_29400,
                     int32_t sizze_29401);
static int futrts_t1(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31238,
                     struct memblock_device *out_mem_p_31239,
                     int32_t *out_out_arrsizze_31240,
                     int32_t *out_out_arrsizze_31241,
                     int32_t *out_out_arrsizze_31242, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978, int32_t sizze_29441,
                     int32_t sizze_29442, int32_t sizze_29443,
                     int32_t sizze_29444, int32_t sizze_29445);
static int futrts_t2(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31248,
                     struct memblock_device *out_mem_p_31249,
                     int32_t *out_out_arrsizze_31250,
                     int32_t *out_out_arrsizze_31251,
                     int32_t *out_out_arrsizze_31252, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29472,
                     int32_t sizze_29473, int32_t sizze_29474,
                     int32_t sizze_29475, int32_t sizze_29476);
static int futrts_t10(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_31263,
                      struct memblock_device *out_mem_p_31264,
                      int32_t *out_out_arrsizze_31265,
                      int32_t *out_out_arrsizze_31266,
                      int64_t A_mem_sizze_30975,
                      struct memblock_device A_mem_30976,
                      int64_t B_mem_sizze_30977,
                      struct memblock_device B_mem_30978,
                      int64_t C_mem_sizze_30979,
                      struct memblock_device C_mem_30980, int32_t sizze_29506,
                      int32_t sizze_29507, int32_t sizze_29508,
                      int32_t sizze_29509, int32_t sizze_29510,
                      int32_t sizze_29511, int32_t sizze_29512);
static int futrts_t11(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_31279,
                      struct memblock_device *out_mem_p_31280,
                      int32_t *out_out_arrsizze_31281,
                      int32_t *out_out_arrsizze_31282,
                      int64_t A_mem_sizze_30975,
                      struct memblock_device A_mem_30976,
                      int64_t B_mem_sizze_30977,
                      struct memblock_device B_mem_30978,
                      int64_t C_mem_sizze_30979,
                      struct memblock_device C_mem_30980,
                      int64_t D_mem_sizze_30981,
                      struct memblock_device D_mem_30982,
                      int64_t E_mem_sizze_30983,
                      struct memblock_device E_mem_30984, int32_t sizze_29559,
                      int32_t sizze_29560, int32_t sizze_29561,
                      int32_t sizze_29562, int32_t sizze_29563,
                      int32_t sizze_29564, int32_t sizze_29565);
static int futrts_t3(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31294,
                     struct memblock_device *out_mem_p_31295,
                     int32_t *out_out_arrsizze_31296,
                     int32_t *out_out_arrsizze_31297,
                     int32_t *out_out_arrsizze_31298, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29616,
                     int32_t sizze_29617, int32_t sizze_29618,
                     int32_t sizze_29619, int32_t sizze_29620,
                     int32_t sizze_29621, int32_t sizze_29622);
static int futrts_t4(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31304,
                     struct memblock_device *out_mem_p_31305,
                     int32_t *out_out_arrsizze_31306,
                     int32_t *out_out_arrsizze_31307,
                     int32_t *out_out_arrsizze_31308, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29668,
                     int32_t sizze_29669, int32_t sizze_29670,
                     int32_t sizze_29671, int32_t sizze_29672,
                     int32_t sizze_29673, int32_t sizze_29674,
                     int32_t sizze_29675);
static int futrts_t5(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31319,
                     struct memblock_device *out_mem_p_31320,
                     int32_t *out_out_arrsizze_31321,
                     int32_t *out_out_arrsizze_31322,
                     int32_t *out_out_arrsizze_31323, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29727,
                     int32_t sizze_29728, int32_t sizze_29729,
                     int32_t sizze_29730, int32_t sizze_29731,
                     int32_t sizze_29732, int32_t sizze_29733,
                     int32_t sizze_29734, int32_t sizze_29735,
                     int32_t sizze_29736);
static int futrts_t6(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31339,
                     struct memblock_device *out_mem_p_31340,
                     int32_t *out_out_arrsizze_31341,
                     int32_t *out_out_arrsizze_31342,
                     int32_t *out_out_arrsizze_31343, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29806,
                     int32_t sizze_29807, int32_t sizze_29808,
                     int32_t sizze_29809, int32_t sizze_29810,
                     int32_t sizze_29811, int32_t sizze_29812,
                     int32_t sizze_29813, int32_t sizze_29814,
                     int32_t sizze_29815, float a_29816, float b_29818,
                     float c_29820, float d_29822);
static int futrts_t7(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31359,
                     struct memblock_device *out_mem_p_31360,
                     int32_t *out_out_arrsizze_31361,
                     int32_t *out_out_arrsizze_31362,
                     int32_t *out_out_arrsizze_31363, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982,
                     int64_t E_mem_sizze_30983,
                     struct memblock_device E_mem_30984,
                     int64_t F_mem_sizze_30985,
                     struct memblock_device F_mem_30986,
                     int64_t G_mem_sizze_30987,
                     struct memblock_device G_mem_30988,
                     int64_t H_mem_sizze_30989,
                     struct memblock_device H_mem_30990, int32_t sizze_29893,
                     int32_t sizze_29894, int32_t sizze_29895,
                     int32_t sizze_29896, int32_t sizze_29897,
                     int32_t sizze_29898, int32_t sizze_29899,
                     int32_t sizze_29900, int32_t sizze_29901,
                     int32_t sizze_29902, int32_t sizze_29903,
                     int32_t sizze_29904, int32_t sizze_29905,
                     int32_t sizze_29906, float a_29907, float b_29909,
                     float c_29911, float d_29913, float e_29915, float f_29917,
                     float g_29919, float h_29921);
static int futrts_t8(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31389,
                     struct memblock_device *out_mem_p_31390,
                     int32_t *out_out_arrsizze_31391,
                     int32_t *out_out_arrsizze_31392,
                     int32_t *out_out_arrsizze_31393, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_30032,
                     int32_t sizze_30033, int32_t sizze_30034,
                     int32_t sizze_30035, int32_t sizze_30036,
                     int32_t sizze_30037, int32_t sizze_30038);
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
static int futrts_map_transpose_opencl_f32(struct futhark_context *ctx,
                                           struct memblock_device destmem_0,
                                           int32_t destoffset_1,
                                           struct memblock_device srcmem_2,
                                           int32_t srcoffset_3,
                                           int32_t num_arrays_4,
                                           int32_t x_elems_5, int32_t y_elems_6,
                                           int32_t in_elems_7,
                                           int32_t out_elems_8)
{
    if (!(num_arrays_4 * x_elems_5 * y_elems_6 == 0)) {
        if (in_elems_7 == out_elems_8 && ((num_arrays_4 == 1 || x_elems_5 *
                                           y_elems_6 == in_elems_7) &&
                                          (x_elems_5 == 1 || y_elems_6 == 1))) {
            if (in_elems_7 * sizeof(float) > 0) {
                OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue,
                                                   srcmem_2.mem, destmem_0.mem,
                                                   srcoffset_3, destoffset_1,
                                                   in_elems_7 * sizeof(float),
                                                   0, NULL, NULL));
                if (ctx->debugging)
                    OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            }
        } else {
            if (sle32(x_elems_5, squot32(16, 2)) && slt32(16, y_elems_6)) {
                int32_t muly_9 = squot32(16, x_elems_5);
                int32_t new_height_10;
                
                new_height_10 = squot32(y_elems_6 + muly_9 - 1, muly_9);
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              0, sizeof(destmem_0.mem),
                                              &destmem_0.mem));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              1, sizeof(destoffset_1),
                                              &destoffset_1));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              2, sizeof(srcmem_2.mem),
                                              &srcmem_2.mem));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              3, sizeof(srcoffset_3),
                                              &srcoffset_3));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              4, sizeof(x_elems_5),
                                              &x_elems_5));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              5, sizeof(y_elems_6),
                                              &y_elems_6));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              6, sizeof(in_elems_7),
                                              &in_elems_7));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              7, sizeof(out_elems_8),
                                              &out_elems_8));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              8, sizeof(muly_9), &muly_9));
                OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowwidth_f32,
                                              9, 272 * sizeof(float), NULL));
                if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16), 16)) *
                    (new_height_10 + srem32(16 - srem32(new_height_10, 16),
                                            16)) * num_arrays_4 != 0) {
                    const size_t global_work_sizze_31202[3] = {x_elems_5 +
                                                               srem32(16 -
                                                                      srem32(x_elems_5,
                                                                             16),
                                                                      16),
                                                               new_height_10 +
                                                               srem32(16 -
                                                                      srem32(new_height_10,
                                                                             16),
                                                                      16),
                                                               num_arrays_4};
                    const size_t local_work_sizze_31206[3] = {16, 16, 1};
                    int64_t time_start_31203 = 0, time_end_31204 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_f32");
                        fprintf(stderr, "%zu", global_work_sizze_31202[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_31202[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_31202[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_31206[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_31206[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_31206[2]);
                        fprintf(stderr, "].\n");
                        time_start_31203 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                          ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                          3, NULL,
                                                          global_work_sizze_31202,
                                                          local_work_sizze_31206,
                                                          0, NULL, NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                        time_end_31204 = get_wall_time();
                        
                        long time_diff_31205 = time_end_31204 -
                             time_start_31203;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime +=
                            time_diff_31205;
                        ctx->fut_kernel_map_transpose_lowwidth_f32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_f32",
                                time_diff_31205);
                    }
                }
            } else {
                if (sle32(y_elems_6, squot32(16, 2)) && slt32(16, x_elems_5)) {
                    int32_t mulx_11 = squot32(16, y_elems_6);
                    int32_t new_width_12;
                    
                    new_width_12 = squot32(x_elems_5 + mulx_11 - 1, mulx_11);
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  0, sizeof(destmem_0.mem),
                                                  &destmem_0.mem));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  1, sizeof(destoffset_1),
                                                  &destoffset_1));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  2, sizeof(srcmem_2.mem),
                                                  &srcmem_2.mem));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  3, sizeof(srcoffset_3),
                                                  &srcoffset_3));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  4, sizeof(x_elems_5),
                                                  &x_elems_5));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  5, sizeof(y_elems_6),
                                                  &y_elems_6));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  6, sizeof(in_elems_7),
                                                  &in_elems_7));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  7, sizeof(out_elems_8),
                                                  &out_elems_8));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  8, sizeof(mulx_11),
                                                  &mulx_11));
                    OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_lowheight_f32,
                                                  9, 272 * sizeof(float),
                                                  NULL));
                    if (1 * (new_width_12 + srem32(16 - srem32(new_width_12,
                                                               16), 16)) *
                        (y_elems_6 + srem32(16 - srem32(y_elems_6, 16), 16)) *
                        num_arrays_4 != 0) {
                        const size_t global_work_sizze_31207[3] =
                                     {new_width_12 + srem32(16 -
                                                            srem32(new_width_12,
                                                                   16), 16),
                                      y_elems_6 + srem32(16 - srem32(y_elems_6,
                                                                     16), 16),
                                      num_arrays_4};
                        const size_t local_work_sizze_31211[3] = {16, 16, 1};
                        int64_t time_start_31208 = 0, time_end_31209 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_f32");
                            fprintf(stderr, "%zu", global_work_sizze_31207[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_31207[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_31207[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_31211[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_31211[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_31211[2]);
                            fprintf(stderr, "].\n");
                            time_start_31208 = get_wall_time();
                        }
                        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                              ctx->fut_kernel_map_transpose_lowheight_f32,
                                                              3, NULL,
                                                              global_work_sizze_31207,
                                                              local_work_sizze_31211,
                                                              0, NULL, NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                            time_end_31209 = get_wall_time();
                            
                            long time_diff_31210 = time_end_31209 -
                                 time_start_31208;
                            
                            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime +=
                                time_diff_31210;
                            ctx->fut_kernel_map_transpose_lowheight_f32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_f32",
                                    time_diff_31210);
                        }
                    }
                } else {
                    if (sle32(x_elems_5, squot32(16, 2)) && sle32(y_elems_6,
                                                                  squot32(16,
                                                                          2))) {
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      0, sizeof(destmem_0.mem),
                                                      &destmem_0.mem));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      1, sizeof(destoffset_1),
                                                      &destoffset_1));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      2, sizeof(srcmem_2.mem),
                                                      &srcmem_2.mem));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      3, sizeof(srcoffset_3),
                                                      &srcoffset_3));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      4, sizeof(num_arrays_4),
                                                      &num_arrays_4));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      5, sizeof(x_elems_5),
                                                      &x_elems_5));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      6, sizeof(y_elems_6),
                                                      &y_elems_6));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      7, sizeof(in_elems_7),
                                                      &in_elems_7));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_small_f32,
                                                      8, sizeof(out_elems_8),
                                                      &out_elems_8));
                        if (1 * (num_arrays_4 * x_elems_5 * y_elems_6 +
                                 srem32(256 - srem32(num_arrays_4 * x_elems_5 *
                                                     y_elems_6, 256), 256)) !=
                            0) {
                            const size_t global_work_sizze_31212[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_31216[1] = {256};
                            int64_t time_start_31213 = 0, time_end_31214 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_31212[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_31216[0]);
                                fprintf(stderr, "].\n");
                                time_start_31213 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_small_f32,
                                                                  1, NULL,
                                                                  global_work_sizze_31212,
                                                                  local_work_sizze_31216,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_31214 = get_wall_time();
                                
                                long time_diff_31215 = time_end_31214 -
                                     time_start_31213;
                                
                                ctx->fut_kernel_map_transpose_small_f32_total_runtime +=
                                    time_diff_31215;
                                ctx->fut_kernel_map_transpose_small_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_f32",
                                        time_diff_31215);
                            }
                        }
                    } else {
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      0, sizeof(destmem_0.mem),
                                                      &destmem_0.mem));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      1, sizeof(destoffset_1),
                                                      &destoffset_1));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      2, sizeof(srcmem_2.mem),
                                                      &srcmem_2.mem));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      3, sizeof(srcoffset_3),
                                                      &srcoffset_3));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      4, sizeof(x_elems_5),
                                                      &x_elems_5));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      5, sizeof(y_elems_6),
                                                      &y_elems_6));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      6, sizeof(in_elems_7),
                                                      &in_elems_7));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      7, sizeof(out_elems_8),
                                                      &out_elems_8));
                        OPENCL_SUCCEED(clSetKernelArg(ctx->fut_kernel_map_transpose_f32,
                                                      8, 272 * sizeof(float),
                                                      NULL));
                        if (1 * (x_elems_5 + srem32(16 - srem32(x_elems_5, 16),
                                                    16)) * (y_elems_6 +
                                                            srem32(16 -
                                                                   srem32(y_elems_6,
                                                                          16),
                                                                   16)) *
                            num_arrays_4 != 0) {
                            const size_t global_work_sizze_31217[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_31221[3] = {16, 16,
                                                                      1};
                            int64_t time_start_31218 = 0, time_end_31219 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_31217[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_31217[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_31217[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_31221[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_31221[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_31221[2]);
                                fprintf(stderr, "].\n");
                                time_start_31218 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_f32,
                                                                  3, NULL,
                                                                  global_work_sizze_31217,
                                                                  local_work_sizze_31221,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_31219 = get_wall_time();
                                
                                long time_diff_31220 = time_end_31219 -
                                     time_start_31218;
                                
                                ctx->fut_kernel_map_transpose_f32_total_runtime +=
                                    time_diff_31220;
                                ctx->fut_kernel_map_transpose_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_f32",
                                        time_diff_31220);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts_t9(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31222,
                     struct memblock_device *out_mem_p_31223,
                     int32_t *out_out_arrsizze_31224,
                     int32_t *out_out_arrsizze_31225, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29396,
                     int32_t sizze_29397, int32_t sizze_29398,
                     int32_t sizze_29399, int32_t sizze_29400,
                     int32_t sizze_29401)
{
    int64_t out_memsizze_31036;
    struct memblock_device out_mem_31035;
    
    out_mem_31035.references = NULL;
    
    int32_t out_arrsizze_31037;
    int32_t out_arrsizze_31038;
    bool dim_zzero_29405 = 0 == sizze_29399;
    bool dim_zzero_29406 = 0 == sizze_29400;
    bool old_empty_29407 = dim_zzero_29405 || dim_zzero_29406;
    bool dim_zzero_29408 = 0 == sizze_29398;
    bool new_empty_29409 = dim_zzero_29406 || dim_zzero_29408;
    bool both_empty_29410 = old_empty_29407 && new_empty_29409;
    bool dim_match_29411 = sizze_29398 == sizze_29399;
    bool empty_or_match_29412 = both_empty_29410 || dim_match_29411;
    bool empty_or_match_cert_29413;
    
    if (!empty_or_match_29412) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:49:1-51:65",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31035, "out_mem_31035");
        return 1;
    }
    
    bool dim_zzero_29415 = 0 == sizze_29401;
    bool both_empty_29416 = dim_zzero_29406 && dim_zzero_29415;
    bool dim_match_29417 = sizze_29400 == sizze_29401;
    bool empty_or_match_29418 = both_empty_29416 || dim_match_29417;
    bool empty_or_match_cert_29419;
    
    if (!empty_or_match_29418) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:49:1-51:65",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31035, "out_mem_31035");
        return 1;
    }
    
    int32_t group_sizze_30097;
    
    group_sizze_30097 = ctx->sizes.group_sizze_30096;
    
    int32_t y_30098 = group_sizze_30097 - 1;
    int32_t x_30099 = sizze_29398 + y_30098;
    int32_t num_groups_30100 = squot32(x_30099, group_sizze_30097);
    int32_t num_threads_30101 = group_sizze_30097 * num_groups_30100;
    int32_t convop_x_30982 = sizze_29399 * sizze_29400;
    int64_t binop_x_30983 = sext_i32_i64(convop_x_30982);
    int64_t bytes_30981 = 4 * binop_x_30983;
    struct memblock_device mem_30984;
    
    mem_30984.references = NULL;
    memblock_alloc_device(ctx, &mem_30984, bytes_30981, "mem_30984");
    
    int call_ret_31226 = futrts_map_transpose_opencl_f32(ctx, mem_30984, 0,
                                                         B_mem_30978, 0, 1,
                                                         sizze_29400,
                                                         sizze_29399,
                                                         sizze_29399 *
                                                         sizze_29400,
                                                         sizze_29399 *
                                                         sizze_29400);
    
    assert(call_ret_31226 == 0);
    
    int64_t binop_x_30986 = sext_i32_i64(sizze_29398);
    int64_t bytes_30985 = 4 * binop_x_30986;
    struct memblock_device mem_30987;
    
    mem_30987.references = NULL;
    memblock_alloc_device(ctx, &mem_30987, bytes_30985, "mem_30987");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 0, sizeof(sizze_29398),
                                  &sizze_29398));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 1, sizeof(sizze_29399),
                                  &sizze_29399));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 2, sizeof(sizze_29400),
                                  &sizze_29400));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 3,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 4,
                                  sizeof(mem_30984.mem), &mem_30984.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30102, 5,
                                  sizeof(mem_30987.mem), &mem_30987.mem));
    if (1 * (num_groups_30100 * group_sizze_30097) != 0) {
        const size_t global_work_sizze_31227[1] = {num_groups_30100 *
                     group_sizze_30097};
        const size_t local_work_sizze_31231[1] = {group_sizze_30097};
        int64_t time_start_31228 = 0, time_end_31229 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30102");
            fprintf(stderr, "%zu", global_work_sizze_31227[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31231[0]);
            fprintf(stderr, "].\n");
            time_start_31228 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30102, 1, NULL,
                                              global_work_sizze_31227,
                                              local_work_sizze_31231, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31229 = get_wall_time();
            
            long time_diff_31230 = time_end_31229 - time_start_31228;
            
            ctx->map_kernel_30102_total_runtime += time_diff_31230;
            ctx->map_kernel_30102_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30102",
                    time_diff_31230);
        }
    }
    memblock_unref_device(ctx, &mem_30984, "mem_30984");
    
    int32_t nesting_sizze_30137 = sizze_29396 * sizze_29397;
    int32_t group_sizze_30139;
    
    group_sizze_30139 = ctx->sizes.group_sizze_30138;
    
    int32_t y_30140 = group_sizze_30139 - 1;
    int32_t x_30141 = nesting_sizze_30137 + y_30140;
    int32_t num_groups_30142 = squot32(x_30141, group_sizze_30139);
    int32_t num_threads_30143 = group_sizze_30139 * num_groups_30142;
    int32_t binop_x_30989 = sizze_29396 * sizze_29398;
    int32_t convop_x_30990 = sizze_29397 * binop_x_30989;
    int64_t binop_x_30991 = sext_i32_i64(convop_x_30990);
    int64_t bytes_30988 = 4 * binop_x_30991;
    struct memblock_device mem_30992;
    
    mem_30992.references = NULL;
    memblock_alloc_device(ctx, &mem_30992, bytes_30988, "mem_30992");
    
    int call_ret_31232 = futrts_map_transpose_opencl_f32(ctx, mem_30992, 0,
                                                         A_mem_30976, 0, 1,
                                                         sizze_29398,
                                                         sizze_29396 *
                                                         sizze_29397,
                                                         sizze_29396 *
                                                         sizze_29397 *
                                                         sizze_29398,
                                                         sizze_29396 *
                                                         sizze_29397 *
                                                         sizze_29398);
    
    assert(call_ret_31232 == 0);
    
    int64_t binop_x_30995 = sext_i32_i64(nesting_sizze_30137);
    int64_t bytes_30993 = 4 * binop_x_30995;
    struct memblock_device mem_30996;
    
    mem_30996.references = NULL;
    memblock_alloc_device(ctx, &mem_30996, bytes_30993, "mem_30996");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 0, sizeof(sizze_29396),
                                  &sizze_29396));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 1, sizeof(sizze_29397),
                                  &sizze_29397));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 2, sizeof(sizze_29398),
                                  &sizze_29398));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 3,
                                  sizeof(mem_30987.mem), &mem_30987.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 4,
                                  sizeof(mem_30992.mem), &mem_30992.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30144, 5,
                                  sizeof(mem_30996.mem), &mem_30996.mem));
    if (1 * (num_groups_30142 * group_sizze_30139) != 0) {
        const size_t global_work_sizze_31233[1] = {num_groups_30142 *
                     group_sizze_30139};
        const size_t local_work_sizze_31237[1] = {group_sizze_30139};
        int64_t time_start_31234 = 0, time_end_31235 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30144");
            fprintf(stderr, "%zu", global_work_sizze_31233[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31237[0]);
            fprintf(stderr, "].\n");
            time_start_31234 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30144, 1, NULL,
                                              global_work_sizze_31233,
                                              local_work_sizze_31237, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31235 = get_wall_time();
            
            long time_diff_31236 = time_end_31235 - time_start_31234;
            
            ctx->map_kernel_30144_total_runtime += time_diff_31236;
            ctx->map_kernel_30144_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30144",
                    time_diff_31236);
        }
    }
    memblock_unref_device(ctx, &mem_30987, "mem_30987");
    memblock_unref_device(ctx, &mem_30992, "mem_30992");
    out_arrsizze_31037 = sizze_29396;
    out_arrsizze_31038 = sizze_29397;
    out_memsizze_31036 = bytes_30993;
    memblock_set_device(ctx, &out_mem_31035, &mem_30996, "mem_30996");
    *out_out_memsizze_31222 = out_memsizze_31036;
    (*out_mem_p_31223).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31223, &out_mem_31035,
                        "out_mem_31035");
    *out_out_arrsizze_31224 = out_arrsizze_31037;
    *out_out_arrsizze_31225 = out_arrsizze_31038;
    memblock_unref_device(ctx, &mem_30996, "mem_30996");
    memblock_unref_device(ctx, &mem_30992, "mem_30992");
    memblock_unref_device(ctx, &mem_30987, "mem_30987");
    memblock_unref_device(ctx, &mem_30984, "mem_30984");
    memblock_unref_device(ctx, &out_mem_31035, "out_mem_31035");
    return 0;
}
static int futrts_t1(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31238,
                     struct memblock_device *out_mem_p_31239,
                     int32_t *out_out_arrsizze_31240,
                     int32_t *out_out_arrsizze_31241,
                     int32_t *out_out_arrsizze_31242, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978, int32_t sizze_29441,
                     int32_t sizze_29442, int32_t sizze_29443,
                     int32_t sizze_29444, int32_t sizze_29445)
{
    int64_t out_memsizze_31048;
    struct memblock_device out_mem_31047;
    
    out_mem_31047.references = NULL;
    
    int32_t out_arrsizze_31049;
    int32_t out_arrsizze_31050;
    int32_t out_arrsizze_31051;
    bool dim_zzero_29448 = 0 == sizze_29444;
    bool dim_zzero_29449 = 0 == sizze_29445;
    bool old_empty_29450 = dim_zzero_29448 || dim_zzero_29449;
    bool dim_zzero_29451 = 0 == sizze_29443;
    bool new_empty_29452 = dim_zzero_29449 || dim_zzero_29451;
    bool both_empty_29453 = old_empty_29450 && new_empty_29452;
    bool dim_match_29454 = sizze_29443 == sizze_29444;
    bool empty_or_match_29455 = both_empty_29453 || dim_match_29454;
    bool empty_or_match_cert_29456;
    
    if (!empty_or_match_29455) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:16:1-17:75",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31047, "out_mem_31047");
        return 1;
    }
    
    int32_t tile_sizze_30924;
    
    tile_sizze_30924 = ctx->sizes.tile_sizze_30923;
    
    int32_t tiled_group_sizze_30925 = tile_sizze_30924 * tile_sizze_30924;
    int32_t y_30932 = tile_sizze_30924 - 1;
    int32_t x_30933 = sizze_29442 + y_30932;
    int32_t groups_in_dim_30934 = squot32(x_30933, tile_sizze_30924);
    int32_t x_30936 = sizze_29445 + y_30932;
    int32_t groups_in_dim_30937 = squot32(x_30936, tile_sizze_30924);
    int32_t y_30939 = groups_in_dim_30934 * groups_in_dim_30937;
    int32_t num_groups_30940 = sizze_29441 * y_30939;
    int32_t num_threads_30941 = tiled_group_sizze_30925 * num_groups_30940;
    int32_t binop_x_30988 = sizze_29441 * sizze_29442;
    int32_t convop_x_30989 = sizze_29445 * binop_x_30988;
    int64_t binop_x_30990 = sext_i32_i64(convop_x_30989);
    int64_t bytes_30987 = 4 * binop_x_30990;
    struct memblock_device mem_30991;
    
    mem_30991.references = NULL;
    memblock_alloc_device(ctx, &mem_30991, bytes_30987, "mem_30991");
    
    int64_t binop_x_30981 = sext_i32_i64(tiled_group_sizze_30925);
    int64_t bytes_30979 = 4 * binop_x_30981;
    struct memblock_local mem_30982;
    
    mem_30982.references = NULL;
    
    struct memblock_local mem_30986;
    
    mem_30986.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 0, bytes_30979, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 1, bytes_30979, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 2, sizeof(sizze_29441),
                                  &sizze_29441));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 3, sizeof(sizze_29442),
                                  &sizze_29442));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 4, sizeof(sizze_29443),
                                  &sizze_29443));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 5, sizeof(sizze_29445),
                                  &sizze_29445));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 6,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 7,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30188, 8,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    if (1 * (num_groups_30940 * tiled_group_sizze_30925) != 0) {
        const size_t global_work_sizze_31243[1] = {num_groups_30940 *
                     tiled_group_sizze_30925};
        const size_t local_work_sizze_31247[1] = {tiled_group_sizze_30925};
        int64_t time_start_31244 = 0, time_end_31245 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30188");
            fprintf(stderr, "%zu", global_work_sizze_31243[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31247[0]);
            fprintf(stderr, "].\n");
            time_start_31244 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30188, 1, NULL,
                                              global_work_sizze_31243,
                                              local_work_sizze_31247, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31245 = get_wall_time();
            
            long time_diff_31246 = time_end_31245 - time_start_31244;
            
            ctx->map_kernel_30188_total_runtime += time_diff_31246;
            ctx->map_kernel_30188_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30188",
                    time_diff_31246);
        }
    }
    memblock_unref_local(ctx, &mem_30982, "mem_30982");
    memblock_unref_local(ctx, &mem_30986, "mem_30986");
    out_arrsizze_31049 = sizze_29441;
    out_arrsizze_31050 = sizze_29442;
    out_arrsizze_31051 = sizze_29445;
    out_memsizze_31048 = bytes_30987;
    memblock_set_device(ctx, &out_mem_31047, &mem_30991, "mem_30991");
    *out_out_memsizze_31238 = out_memsizze_31048;
    (*out_mem_p_31239).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31239, &out_mem_31047,
                        "out_mem_31047");
    *out_out_arrsizze_31240 = out_arrsizze_31049;
    *out_out_arrsizze_31241 = out_arrsizze_31050;
    *out_out_arrsizze_31242 = out_arrsizze_31051;
    memblock_unref_local(ctx, &mem_30986, "mem_30986");
    memblock_unref_local(ctx, &mem_30982, "mem_30982");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_device(ctx, &out_mem_31047, "out_mem_31047");
    return 0;
}
static int futrts_t2(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31248,
                     struct memblock_device *out_mem_p_31249,
                     int32_t *out_out_arrsizze_31250,
                     int32_t *out_out_arrsizze_31251,
                     int32_t *out_out_arrsizze_31252, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29472,
                     int32_t sizze_29473, int32_t sizze_29474,
                     int32_t sizze_29475, int32_t sizze_29476)
{
    int64_t out_memsizze_31061;
    struct memblock_device out_mem_31060;
    
    out_mem_31060.references = NULL;
    
    int32_t out_arrsizze_31062;
    int32_t out_arrsizze_31063;
    int32_t out_arrsizze_31064;
    bool dim_zzero_29480 = 0 == sizze_29475;
    bool dim_zzero_29481 = 0 == sizze_29474;
    bool both_empty_29482 = dim_zzero_29480 && dim_zzero_29481;
    bool dim_match_29483 = sizze_29474 == sizze_29475;
    bool empty_or_match_29484 = both_empty_29482 || dim_match_29483;
    bool empty_or_match_cert_29485;
    
    if (!empty_or_match_29484) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:19:1-21:79",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31060, "out_mem_31060");
        return 1;
    }
    
    int32_t nesting_sizze_30213 = sizze_29474 * sizze_29476;
    int32_t group_sizze_30215;
    
    group_sizze_30215 = ctx->sizes.group_sizze_30214;
    
    int32_t y_30216 = group_sizze_30215 - 1;
    int32_t x_30217 = nesting_sizze_30213 + y_30216;
    int32_t num_groups_30218 = squot32(x_30217, group_sizze_30215);
    int32_t num_threads_30219 = group_sizze_30215 * num_groups_30218;
    int64_t binop_x_30983 = sext_i32_i64(nesting_sizze_30213);
    int64_t bytes_30981 = 4 * binop_x_30983;
    struct memblock_device mem_30984;
    
    mem_30984.references = NULL;
    memblock_alloc_device(ctx, &mem_30984, bytes_30981, "mem_30984");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30220, 0, sizeof(sizze_29474),
                                  &sizze_29474));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30220, 1, sizeof(sizze_29476),
                                  &sizze_29476));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30220, 2,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30220, 3,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30220, 4,
                                  sizeof(mem_30984.mem), &mem_30984.mem));
    if (1 * (num_groups_30218 * group_sizze_30215) != 0) {
        const size_t global_work_sizze_31253[1] = {num_groups_30218 *
                     group_sizze_30215};
        const size_t local_work_sizze_31257[1] = {group_sizze_30215};
        int64_t time_start_31254 = 0, time_end_31255 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30220");
            fprintf(stderr, "%zu", global_work_sizze_31253[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31257[0]);
            fprintf(stderr, "].\n");
            time_start_31254 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30220, 1, NULL,
                                              global_work_sizze_31253,
                                              local_work_sizze_31257, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31255 = get_wall_time();
            
            long time_diff_31256 = time_end_31255 - time_start_31254;
            
            ctx->map_kernel_30220_total_runtime += time_diff_31256;
            ctx->map_kernel_30220_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30220",
                    time_diff_31256);
        }
    }
    
    int32_t tile_sizze_30924;
    
    tile_sizze_30924 = ctx->sizes.tile_sizze_30923;
    
    int32_t tiled_group_sizze_30925 = tile_sizze_30924 * tile_sizze_30924;
    int32_t y_30932 = tile_sizze_30924 - 1;
    int32_t x_30933 = sizze_29473 + y_30932;
    int32_t groups_in_dim_30934 = squot32(x_30933, tile_sizze_30924);
    int32_t x_30936 = sizze_29476 + y_30932;
    int32_t groups_in_dim_30937 = squot32(x_30936, tile_sizze_30924);
    int32_t y_30939 = groups_in_dim_30934 * groups_in_dim_30937;
    int32_t num_groups_30940 = sizze_29472 * y_30939;
    int32_t num_threads_30941 = tiled_group_sizze_30925 * num_groups_30940;
    int32_t binop_x_30994 = sizze_29472 * sizze_29473;
    int32_t convop_x_30995 = sizze_29476 * binop_x_30994;
    int64_t binop_x_30996 = sext_i32_i64(convop_x_30995);
    int64_t bytes_30993 = 4 * binop_x_30996;
    struct memblock_device mem_30997;
    
    mem_30997.references = NULL;
    memblock_alloc_device(ctx, &mem_30997, bytes_30993, "mem_30997");
    
    int64_t binop_x_30987 = sext_i32_i64(tiled_group_sizze_30925);
    int64_t bytes_30985 = 4 * binop_x_30987;
    struct memblock_local mem_30988;
    
    mem_30988.references = NULL;
    
    struct memblock_local mem_30992;
    
    mem_30992.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 0, bytes_30985, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 1, bytes_30985, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 2, sizeof(sizze_29472),
                                  &sizze_29472));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 3, sizeof(sizze_29473),
                                  &sizze_29473));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 4, sizeof(sizze_29474),
                                  &sizze_29474));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 5, sizeof(sizze_29476),
                                  &sizze_29476));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 6,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 7,
                                  sizeof(mem_30984.mem), &mem_30984.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30248, 8,
                                  sizeof(mem_30997.mem), &mem_30997.mem));
    if (1 * (num_groups_30940 * tiled_group_sizze_30925) != 0) {
        const size_t global_work_sizze_31258[1] = {num_groups_30940 *
                     tiled_group_sizze_30925};
        const size_t local_work_sizze_31262[1] = {tiled_group_sizze_30925};
        int64_t time_start_31259 = 0, time_end_31260 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30248");
            fprintf(stderr, "%zu", global_work_sizze_31258[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31262[0]);
            fprintf(stderr, "].\n");
            time_start_31259 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30248, 1, NULL,
                                              global_work_sizze_31258,
                                              local_work_sizze_31262, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31260 = get_wall_time();
            
            long time_diff_31261 = time_end_31260 - time_start_31259;
            
            ctx->map_kernel_30248_total_runtime += time_diff_31261;
            ctx->map_kernel_30248_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30248",
                    time_diff_31261);
        }
    }
    memblock_unref_device(ctx, &mem_30984, "mem_30984");
    memblock_unref_local(ctx, &mem_30988, "mem_30988");
    memblock_unref_local(ctx, &mem_30992, "mem_30992");
    out_arrsizze_31062 = sizze_29472;
    out_arrsizze_31063 = sizze_29473;
    out_arrsizze_31064 = sizze_29476;
    out_memsizze_31061 = bytes_30993;
    memblock_set_device(ctx, &out_mem_31060, &mem_30997, "mem_30997");
    *out_out_memsizze_31248 = out_memsizze_31061;
    (*out_mem_p_31249).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31249, &out_mem_31060,
                        "out_mem_31060");
    *out_out_arrsizze_31250 = out_arrsizze_31062;
    *out_out_arrsizze_31251 = out_arrsizze_31063;
    *out_out_arrsizze_31252 = out_arrsizze_31064;
    memblock_unref_local(ctx, &mem_30992, "mem_30992");
    memblock_unref_local(ctx, &mem_30988, "mem_30988");
    memblock_unref_device(ctx, &mem_30997, "mem_30997");
    memblock_unref_device(ctx, &mem_30984, "mem_30984");
    memblock_unref_device(ctx, &out_mem_31060, "out_mem_31060");
    return 0;
}
static int futrts_t10(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_31263,
                      struct memblock_device *out_mem_p_31264,
                      int32_t *out_out_arrsizze_31265,
                      int32_t *out_out_arrsizze_31266,
                      int64_t A_mem_sizze_30975,
                      struct memblock_device A_mem_30976,
                      int64_t B_mem_sizze_30977,
                      struct memblock_device B_mem_30978,
                      int64_t C_mem_sizze_30979,
                      struct memblock_device C_mem_30980, int32_t sizze_29506,
                      int32_t sizze_29507, int32_t sizze_29508,
                      int32_t sizze_29509, int32_t sizze_29510,
                      int32_t sizze_29511, int32_t sizze_29512)
{
    int64_t out_memsizze_31077;
    struct memblock_device out_mem_31076;
    
    out_mem_31076.references = NULL;
    
    int32_t out_arrsizze_31078;
    int32_t out_arrsizze_31079;
    bool dim_zzero_29516 = 0 == sizze_29509;
    bool dim_zzero_29517 = 0 == sizze_29510;
    bool dim_zzero_29518 = 0 == sizze_29511;
    bool y_29519 = dim_zzero_29517 || dim_zzero_29518;
    bool old_empty_29520 = dim_zzero_29516 || y_29519;
    bool dim_zzero_29521 = 0 == sizze_29506;
    bool dim_zzero_29522 = 0 == sizze_29508;
    bool y_29523 = dim_zzero_29518 || dim_zzero_29522;
    bool new_empty_29524 = dim_zzero_29521 || y_29523;
    bool both_empty_29525 = old_empty_29520 && new_empty_29524;
    bool dim_match_29526 = sizze_29506 == sizze_29509;
    bool dim_match_29527 = sizze_29508 == sizze_29510;
    bool match_29528 = dim_match_29526 && dim_match_29527;
    bool empty_or_match_29529 = both_empty_29525 || match_29528;
    bool empty_or_match_cert_29530;
    
    if (!empty_or_match_29529) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:53:1-55:72",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31076, "out_mem_31076");
        return 1;
    }
    
    bool dim_zzero_29531 = 0 == sizze_29512;
    bool both_empty_29532 = dim_zzero_29518 && dim_zzero_29531;
    bool dim_match_29533 = sizze_29511 == sizze_29512;
    bool empty_or_match_29534 = both_empty_29532 || dim_match_29533;
    bool empty_or_match_cert_29535;
    
    if (!empty_or_match_29534) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:53:1-55:72",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31076, "out_mem_31076");
        return 1;
    }
    
    int32_t nesting_sizze_30328 = sizze_29506 * sizze_29508;
    int32_t group_sizze_30330;
    
    group_sizze_30330 = ctx->sizes.group_sizze_30329;
    
    int32_t y_30331 = group_sizze_30330 - 1;
    int32_t x_30332 = nesting_sizze_30328 + y_30331;
    int32_t num_groups_30333 = squot32(x_30332, group_sizze_30330);
    int32_t num_threads_30334 = group_sizze_30330 * num_groups_30333;
    int32_t binop_x_30982 = sizze_29509 * sizze_29511;
    int32_t convop_x_30983 = sizze_29510 * binop_x_30982;
    int64_t binop_x_30984 = sext_i32_i64(convop_x_30983);
    int64_t bytes_30981 = 4 * binop_x_30984;
    struct memblock_device mem_30985;
    
    mem_30985.references = NULL;
    memblock_alloc_device(ctx, &mem_30985, bytes_30981, "mem_30985");
    
    int call_ret_31267 = futrts_map_transpose_opencl_f32(ctx, mem_30985, 0,
                                                         B_mem_30978, 0, 1,
                                                         sizze_29511,
                                                         sizze_29509 *
                                                         sizze_29510,
                                                         sizze_29509 *
                                                         sizze_29510 *
                                                         sizze_29511,
                                                         sizze_29509 *
                                                         sizze_29510 *
                                                         sizze_29511);
    
    assert(call_ret_31267 == 0);
    
    int64_t binop_x_30988 = sext_i32_i64(nesting_sizze_30328);
    int64_t bytes_30986 = 4 * binop_x_30988;
    struct memblock_device mem_30989;
    
    mem_30989.references = NULL;
    memblock_alloc_device(ctx, &mem_30989, bytes_30986, "mem_30989");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 0, sizeof(sizze_29506),
                                  &sizze_29506));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 1, sizeof(sizze_29508),
                                  &sizze_29508));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 2, sizeof(sizze_29509),
                                  &sizze_29509));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 3, sizeof(sizze_29510),
                                  &sizze_29510));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 4, sizeof(sizze_29511),
                                  &sizze_29511));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 5,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 6,
                                  sizeof(mem_30985.mem), &mem_30985.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30335, 7,
                                  sizeof(mem_30989.mem), &mem_30989.mem));
    if (1 * (num_groups_30333 * group_sizze_30330) != 0) {
        const size_t global_work_sizze_31268[1] = {num_groups_30333 *
                     group_sizze_30330};
        const size_t local_work_sizze_31272[1] = {group_sizze_30330};
        int64_t time_start_31269 = 0, time_end_31270 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30335");
            fprintf(stderr, "%zu", global_work_sizze_31268[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31272[0]);
            fprintf(stderr, "].\n");
            time_start_31269 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30335, 1, NULL,
                                              global_work_sizze_31268,
                                              local_work_sizze_31272, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31270 = get_wall_time();
            
            long time_diff_31271 = time_end_31270 - time_start_31269;
            
            ctx->map_kernel_30335_total_runtime += time_diff_31271;
            ctx->map_kernel_30335_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30335",
                    time_diff_31271);
        }
    }
    memblock_unref_device(ctx, &mem_30985, "mem_30985");
    
    int32_t nesting_sizze_30285 = sizze_29506 * sizze_29507;
    int32_t group_sizze_30287;
    
    group_sizze_30287 = ctx->sizes.group_sizze_30286;
    
    int32_t y_30288 = group_sizze_30287 - 1;
    int32_t x_30289 = nesting_sizze_30285 + y_30288;
    int32_t num_groups_30290 = squot32(x_30289, group_sizze_30287);
    int32_t num_threads_30291 = group_sizze_30287 * num_groups_30290;
    int32_t convop_x_30992 = sizze_29507 * nesting_sizze_30328;
    int64_t binop_x_30993 = sext_i32_i64(convop_x_30992);
    int64_t bytes_30990 = 4 * binop_x_30993;
    struct memblock_device mem_30994;
    
    mem_30994.references = NULL;
    memblock_alloc_device(ctx, &mem_30994, bytes_30990, "mem_30994");
    
    int call_ret_31273 = futrts_map_transpose_opencl_f32(ctx, mem_30994, 0,
                                                         A_mem_30976, 0, 1,
                                                         sizze_29508,
                                                         sizze_29506 *
                                                         sizze_29507,
                                                         sizze_29506 *
                                                         sizze_29507 *
                                                         sizze_29508,
                                                         sizze_29506 *
                                                         sizze_29507 *
                                                         sizze_29508);
    
    assert(call_ret_31273 == 0);
    
    int64_t binop_x_30997 = sext_i32_i64(nesting_sizze_30285);
    int64_t bytes_30995 = 4 * binop_x_30997;
    struct memblock_device mem_30998;
    
    mem_30998.references = NULL;
    memblock_alloc_device(ctx, &mem_30998, bytes_30995, "mem_30998");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 0, sizeof(sizze_29506),
                                  &sizze_29506));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 1, sizeof(sizze_29507),
                                  &sizze_29507));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 2, sizeof(sizze_29508),
                                  &sizze_29508));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 3,
                                  sizeof(mem_30989.mem), &mem_30989.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 4,
                                  sizeof(mem_30994.mem), &mem_30994.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30292, 5,
                                  sizeof(mem_30998.mem), &mem_30998.mem));
    if (1 * (num_groups_30290 * group_sizze_30287) != 0) {
        const size_t global_work_sizze_31274[1] = {num_groups_30290 *
                     group_sizze_30287};
        const size_t local_work_sizze_31278[1] = {group_sizze_30287};
        int64_t time_start_31275 = 0, time_end_31276 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30292");
            fprintf(stderr, "%zu", global_work_sizze_31274[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31278[0]);
            fprintf(stderr, "].\n");
            time_start_31275 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30292, 1, NULL,
                                              global_work_sizze_31274,
                                              local_work_sizze_31278, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31276 = get_wall_time();
            
            long time_diff_31277 = time_end_31276 - time_start_31275;
            
            ctx->map_kernel_30292_total_runtime += time_diff_31277;
            ctx->map_kernel_30292_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30292",
                    time_diff_31277);
        }
    }
    memblock_unref_device(ctx, &mem_30989, "mem_30989");
    memblock_unref_device(ctx, &mem_30994, "mem_30994");
    out_arrsizze_31078 = sizze_29506;
    out_arrsizze_31079 = sizze_29507;
    out_memsizze_31077 = bytes_30995;
    memblock_set_device(ctx, &out_mem_31076, &mem_30998, "mem_30998");
    *out_out_memsizze_31263 = out_memsizze_31077;
    (*out_mem_p_31264).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31264, &out_mem_31076,
                        "out_mem_31076");
    *out_out_arrsizze_31265 = out_arrsizze_31078;
    *out_out_arrsizze_31266 = out_arrsizze_31079;
    memblock_unref_device(ctx, &mem_30998, "mem_30998");
    memblock_unref_device(ctx, &mem_30994, "mem_30994");
    memblock_unref_device(ctx, &mem_30989, "mem_30989");
    memblock_unref_device(ctx, &mem_30985, "mem_30985");
    memblock_unref_device(ctx, &out_mem_31076, "out_mem_31076");
    return 0;
}
static int futrts_t11(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_31279,
                      struct memblock_device *out_mem_p_31280,
                      int32_t *out_out_arrsizze_31281,
                      int32_t *out_out_arrsizze_31282,
                      int64_t A_mem_sizze_30975,
                      struct memblock_device A_mem_30976,
                      int64_t B_mem_sizze_30977,
                      struct memblock_device B_mem_30978,
                      int64_t C_mem_sizze_30979,
                      struct memblock_device C_mem_30980,
                      int64_t D_mem_sizze_30981,
                      struct memblock_device D_mem_30982,
                      int64_t E_mem_sizze_30983,
                      struct memblock_device E_mem_30984, int32_t sizze_29559,
                      int32_t sizze_29560, int32_t sizze_29561,
                      int32_t sizze_29562, int32_t sizze_29563,
                      int32_t sizze_29564, int32_t sizze_29565)
{
    int64_t out_memsizze_31089;
    struct memblock_device out_mem_31088;
    
    out_mem_31088.references = NULL;
    
    int32_t out_arrsizze_31090;
    int32_t out_arrsizze_31091;
    bool dim_zzero_29571 = 0 == sizze_29562;
    bool dim_zzero_29572 = 0 == sizze_29563;
    bool dim_zzero_29573 = 0 == sizze_29564;
    bool y_29574 = dim_zzero_29572 || dim_zzero_29573;
    bool old_empty_29575 = dim_zzero_29571 || y_29574;
    bool dim_zzero_29576 = 0 == sizze_29559;
    bool dim_zzero_29577 = 0 == sizze_29561;
    bool y_29578 = dim_zzero_29573 || dim_zzero_29577;
    bool new_empty_29579 = dim_zzero_29576 || y_29578;
    bool both_empty_29580 = old_empty_29575 && new_empty_29579;
    bool dim_match_29581 = sizze_29559 == sizze_29562;
    bool dim_match_29582 = sizze_29561 == sizze_29563;
    bool match_29583 = dim_match_29581 && dim_match_29582;
    bool empty_or_match_29584 = both_empty_29580 || match_29583;
    bool empty_or_match_cert_29585;
    
    if (!empty_or_match_29584) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:57:1-60:76",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31088, "out_mem_31088");
        return 1;
    }
    
    bool dim_zzero_29586 = 0 == sizze_29565;
    bool both_empty_29587 = dim_zzero_29573 && dim_zzero_29586;
    bool dim_match_29588 = sizze_29564 == sizze_29565;
    bool empty_or_match_29589 = both_empty_29587 || dim_match_29588;
    bool empty_or_match_cert_29590;
    
    if (!empty_or_match_29589) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:57:1-60:76",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31088, "out_mem_31088");
        return 1;
    }
    
    int32_t nesting_sizze_30417 = sizze_29559 * sizze_29561;
    int32_t group_sizze_30419;
    
    group_sizze_30419 = ctx->sizes.group_sizze_30418;
    
    int32_t y_30420 = group_sizze_30419 - 1;
    int32_t x_30421 = nesting_sizze_30417 + y_30420;
    int32_t num_groups_30422 = squot32(x_30421, group_sizze_30419);
    int32_t num_threads_30423 = group_sizze_30419 * num_groups_30422;
    int32_t binop_x_30986 = sizze_29562 * sizze_29564;
    int32_t convop_x_30987 = sizze_29563 * binop_x_30986;
    int64_t binop_x_30988 = sext_i32_i64(convop_x_30987);
    int64_t bytes_30985 = 4 * binop_x_30988;
    struct memblock_device mem_30989;
    
    mem_30989.references = NULL;
    memblock_alloc_device(ctx, &mem_30989, bytes_30985, "mem_30989");
    
    int call_ret_31283 = futrts_map_transpose_opencl_f32(ctx, mem_30989, 0,
                                                         D_mem_30982, 0, 1,
                                                         sizze_29564,
                                                         sizze_29562 *
                                                         sizze_29563,
                                                         sizze_29562 *
                                                         sizze_29563 *
                                                         sizze_29564,
                                                         sizze_29562 *
                                                         sizze_29563 *
                                                         sizze_29564);
    
    assert(call_ret_31283 == 0);
    
    int64_t binop_x_30992 = sext_i32_i64(nesting_sizze_30417);
    int64_t bytes_30990 = 4 * binop_x_30992;
    struct memblock_device mem_30993;
    
    mem_30993.references = NULL;
    memblock_alloc_device(ctx, &mem_30993, bytes_30990, "mem_30993");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 0, sizeof(sizze_29559),
                                  &sizze_29559));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 1, sizeof(sizze_29561),
                                  &sizze_29561));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 2, sizeof(sizze_29562),
                                  &sizze_29562));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 3, sizeof(sizze_29563),
                                  &sizze_29563));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 4, sizeof(sizze_29564),
                                  &sizze_29564));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 5,
                                  sizeof(E_mem_30984.mem), &E_mem_30984.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 6,
                                  sizeof(mem_30989.mem), &mem_30989.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30424, 7,
                                  sizeof(mem_30993.mem), &mem_30993.mem));
    if (1 * (num_groups_30422 * group_sizze_30419) != 0) {
        const size_t global_work_sizze_31284[1] = {num_groups_30422 *
                     group_sizze_30419};
        const size_t local_work_sizze_31288[1] = {group_sizze_30419};
        int64_t time_start_31285 = 0, time_end_31286 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30424");
            fprintf(stderr, "%zu", global_work_sizze_31284[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31288[0]);
            fprintf(stderr, "].\n");
            time_start_31285 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30424, 1, NULL,
                                              global_work_sizze_31284,
                                              local_work_sizze_31288, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31286 = get_wall_time();
            
            long time_diff_31287 = time_end_31286 - time_start_31285;
            
            ctx->map_kernel_30424_total_runtime += time_diff_31287;
            ctx->map_kernel_30424_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30424",
                    time_diff_31287);
        }
    }
    memblock_unref_device(ctx, &mem_30989, "mem_30989");
    
    int32_t nesting_sizze_30371 = sizze_29559 * sizze_29560;
    int32_t group_sizze_30373;
    
    group_sizze_30373 = ctx->sizes.group_sizze_30372;
    
    int32_t y_30374 = group_sizze_30373 - 1;
    int32_t x_30375 = nesting_sizze_30371 + y_30374;
    int32_t num_groups_30376 = squot32(x_30375, group_sizze_30373);
    int32_t num_threads_30377 = group_sizze_30373 * num_groups_30376;
    int64_t binop_x_31002 = sext_i32_i64(nesting_sizze_30371);
    int64_t bytes_31000 = 4 * binop_x_31002;
    struct memblock_device mem_31003;
    
    mem_31003.references = NULL;
    memblock_alloc_device(ctx, &mem_31003, bytes_31000, "mem_31003");
    
    int64_t binop_x_30995 = sext_i32_i64(group_sizze_30373);
    int64_t bytes_30994 = 4 * binop_x_30995;
    struct memblock_local mem_30996;
    
    mem_30996.references = NULL;
    
    struct memblock_local mem_30999;
    
    mem_30999.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 0, bytes_30994, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 1, bytes_30994, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 2, sizeof(sizze_29559),
                                  &sizze_29559));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 3, sizeof(sizze_29560),
                                  &sizze_29560));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 4, sizeof(sizze_29561),
                                  &sizze_29561));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 5,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 6,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 7,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 8,
                                  sizeof(mem_30993.mem), &mem_30993.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30378, 9,
                                  sizeof(mem_31003.mem), &mem_31003.mem));
    if (1 * (num_groups_30376 * group_sizze_30373) != 0) {
        const size_t global_work_sizze_31289[1] = {num_groups_30376 *
                     group_sizze_30373};
        const size_t local_work_sizze_31293[1] = {group_sizze_30373};
        int64_t time_start_31290 = 0, time_end_31291 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30378");
            fprintf(stderr, "%zu", global_work_sizze_31289[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31293[0]);
            fprintf(stderr, "].\n");
            time_start_31290 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30378, 1, NULL,
                                              global_work_sizze_31289,
                                              local_work_sizze_31293, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31291 = get_wall_time();
            
            long time_diff_31292 = time_end_31291 - time_start_31290;
            
            ctx->map_kernel_30378_total_runtime += time_diff_31292;
            ctx->map_kernel_30378_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30378",
                    time_diff_31292);
        }
    }
    memblock_unref_device(ctx, &mem_30993, "mem_30993");
    memblock_unref_local(ctx, &mem_30996, "mem_30996");
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    out_arrsizze_31090 = sizze_29559;
    out_arrsizze_31091 = sizze_29560;
    out_memsizze_31089 = bytes_31000;
    memblock_set_device(ctx, &out_mem_31088, &mem_31003, "mem_31003");
    *out_out_memsizze_31279 = out_memsizze_31089;
    (*out_mem_p_31280).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31280, &out_mem_31088,
                        "out_mem_31088");
    *out_out_arrsizze_31281 = out_arrsizze_31090;
    *out_out_arrsizze_31282 = out_arrsizze_31091;
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    memblock_unref_local(ctx, &mem_30996, "mem_30996");
    memblock_unref_device(ctx, &mem_31003, "mem_31003");
    memblock_unref_device(ctx, &mem_30993, "mem_30993");
    memblock_unref_device(ctx, &mem_30989, "mem_30989");
    memblock_unref_device(ctx, &out_mem_31088, "out_mem_31088");
    return 0;
}
static int futrts_t3(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31294,
                     struct memblock_device *out_mem_p_31295,
                     int32_t *out_out_arrsizze_31296,
                     int32_t *out_out_arrsizze_31297,
                     int32_t *out_out_arrsizze_31298, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29616,
                     int32_t sizze_29617, int32_t sizze_29618,
                     int32_t sizze_29619, int32_t sizze_29620,
                     int32_t sizze_29621, int32_t sizze_29622)
{
    int64_t out_memsizze_31105;
    struct memblock_device out_mem_31104;
    
    out_mem_31104.references = NULL;
    
    int32_t out_arrsizze_31106;
    int32_t out_arrsizze_31107;
    int32_t out_arrsizze_31108;
    bool dim_zzero_29627 = 0 == sizze_29619;
    bool dim_zzero_29628 = 0 == sizze_29620;
    bool old_empty_29629 = dim_zzero_29627 || dim_zzero_29628;
    bool dim_zzero_29630 = 0 == sizze_29618;
    bool new_empty_29631 = dim_zzero_29628 || dim_zzero_29630;
    bool both_empty_29632 = old_empty_29629 && new_empty_29631;
    bool dim_match_29633 = sizze_29618 == sizze_29619;
    bool empty_or_match_29634 = both_empty_29632 || dim_match_29633;
    bool empty_or_match_cert_29635;
    
    if (!empty_or_match_29634) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:23:1-24:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31104, "out_mem_31104");
        return 1;
    }
    
    bool dim_zzero_29637 = 0 == sizze_29621;
    bool both_empty_29638 = dim_zzero_29630 && dim_zzero_29637;
    bool dim_match_29639 = sizze_29618 == sizze_29621;
    bool empty_or_match_29640 = both_empty_29638 || dim_match_29639;
    bool empty_or_match_cert_29641;
    
    if (!empty_or_match_29640) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:23:1-24:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31104, "out_mem_31104");
        return 1;
    }
    
    bool dim_zzero_29642 = 0 == sizze_29622;
    bool dim_zzero_29643 = 0 == sizze_29617;
    bool both_empty_29644 = dim_zzero_29642 && dim_zzero_29643;
    bool dim_match_29645 = sizze_29617 == sizze_29622;
    bool empty_or_match_29646 = both_empty_29644 || dim_match_29645;
    bool empty_or_match_cert_29647;
    
    if (!empty_or_match_29646) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:23:1-24:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31104, "out_mem_31104");
        return 1;
    }
    
    int32_t nesting_sizze_30462 = sizze_29617 * sizze_29620;
    int32_t nesting_sizze_30463 = sizze_29616 * nesting_sizze_30462;
    int32_t group_sizze_30465;
    
    group_sizze_30465 = ctx->sizes.group_sizze_30464;
    
    int32_t y_30466 = group_sizze_30465 - 1;
    int32_t x_30467 = nesting_sizze_30463 + y_30466;
    int32_t num_groups_30468 = squot32(x_30467, group_sizze_30465);
    int32_t num_threads_30469 = group_sizze_30465 * num_groups_30468;
    int32_t binop_x_30984 = sizze_29616 * sizze_29617;
    int32_t convop_x_30985 = sizze_29620 * binop_x_30984;
    int64_t binop_x_30986 = sext_i32_i64(convop_x_30985);
    int64_t bytes_30983 = 4 * binop_x_30986;
    struct memblock_device mem_30987;
    
    mem_30987.references = NULL;
    memblock_alloc_device(ctx, &mem_30987, bytes_30983, "mem_30987");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 0, sizeof(sizze_29616),
                                  &sizze_29616));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 1, sizeof(sizze_29617),
                                  &sizze_29617));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 2, sizeof(sizze_29618),
                                  &sizze_29618));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 3, sizeof(sizze_29620),
                                  &sizze_29620));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 4,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 5,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 6,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 7,
                                  sizeof(D_mem_30982.mem), &D_mem_30982.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30470, 8,
                                  sizeof(mem_30987.mem), &mem_30987.mem));
    if (1 * (num_groups_30468 * group_sizze_30465) != 0) {
        const size_t global_work_sizze_31299[1] = {num_groups_30468 *
                     group_sizze_30465};
        const size_t local_work_sizze_31303[1] = {group_sizze_30465};
        int64_t time_start_31300 = 0, time_end_31301 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30470");
            fprintf(stderr, "%zu", global_work_sizze_31299[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31303[0]);
            fprintf(stderr, "].\n");
            time_start_31300 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30470, 1, NULL,
                                              global_work_sizze_31299,
                                              local_work_sizze_31303, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31301 = get_wall_time();
            
            long time_diff_31302 = time_end_31301 - time_start_31300;
            
            ctx->map_kernel_30470_total_runtime += time_diff_31302;
            ctx->map_kernel_30470_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30470",
                    time_diff_31302);
        }
    }
    out_arrsizze_31106 = sizze_29616;
    out_arrsizze_31107 = sizze_29617;
    out_arrsizze_31108 = sizze_29620;
    out_memsizze_31105 = bytes_30983;
    memblock_set_device(ctx, &out_mem_31104, &mem_30987, "mem_30987");
    *out_out_memsizze_31294 = out_memsizze_31105;
    (*out_mem_p_31295).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31295, &out_mem_31104,
                        "out_mem_31104");
    *out_out_arrsizze_31296 = out_arrsizze_31106;
    *out_out_arrsizze_31297 = out_arrsizze_31107;
    *out_out_arrsizze_31298 = out_arrsizze_31108;
    memblock_unref_device(ctx, &mem_30987, "mem_30987");
    memblock_unref_device(ctx, &out_mem_31104, "out_mem_31104");
    return 0;
}
static int futrts_t4(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31304,
                     struct memblock_device *out_mem_p_31305,
                     int32_t *out_out_arrsizze_31306,
                     int32_t *out_out_arrsizze_31307,
                     int32_t *out_out_arrsizze_31308, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_29668,
                     int32_t sizze_29669, int32_t sizze_29670,
                     int32_t sizze_29671, int32_t sizze_29672,
                     int32_t sizze_29673, int32_t sizze_29674,
                     int32_t sizze_29675)
{
    int64_t out_memsizze_31114;
    struct memblock_device out_mem_31113;
    
    out_mem_31113.references = NULL;
    
    int32_t out_arrsizze_31115;
    int32_t out_arrsizze_31116;
    int32_t out_arrsizze_31117;
    bool dim_zzero_29679 = 0 == sizze_29671;
    bool dim_zzero_29680 = 0 == sizze_29672;
    bool dim_zzero_29681 = 0 == sizze_29673;
    bool y_29682 = dim_zzero_29680 || dim_zzero_29681;
    bool old_empty_29683 = dim_zzero_29679 || y_29682;
    bool dim_zzero_29684 = 0 == sizze_29668;
    bool dim_zzero_29685 = 0 == sizze_29669;
    bool dim_zzero_29686 = 0 == sizze_29670;
    bool y_29687 = dim_zzero_29685 || dim_zzero_29686;
    bool new_empty_29688 = dim_zzero_29684 || y_29687;
    bool both_empty_29689 = old_empty_29683 && new_empty_29688;
    bool dim_match_29690 = sizze_29668 == sizze_29671;
    bool dim_match_29691 = sizze_29669 == sizze_29672;
    bool dim_match_29692 = sizze_29670 == sizze_29673;
    bool y_29693 = dim_match_29691 && dim_match_29692;
    bool match_29694 = dim_match_29690 && y_29693;
    bool empty_or_match_29695 = both_empty_29689 || match_29694;
    bool empty_or_match_cert_29696;
    
    if (!empty_or_match_29695) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:26:1-27:85",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31113, "out_mem_31113");
        return 1;
    }
    
    bool dim_zzero_29697 = 0 == sizze_29674;
    bool dim_zzero_29698 = 0 == sizze_29675;
    bool old_empty_29699 = dim_zzero_29697 || dim_zzero_29698;
    bool new_empty_29700 = dim_zzero_29686 || dim_zzero_29698;
    bool both_empty_29701 = old_empty_29699 && new_empty_29700;
    bool dim_match_29702 = sizze_29670 == sizze_29674;
    bool empty_or_match_29703 = both_empty_29701 || dim_match_29702;
    bool empty_or_match_cert_29704;
    
    if (!empty_or_match_29703) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:26:1-27:85",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31113, "out_mem_31113");
        return 1;
    }
    
    int32_t nesting_sizze_30549 = sizze_29669 * sizze_29670;
    int32_t nesting_sizze_30550 = sizze_29668 * nesting_sizze_30549;
    int32_t group_sizze_30552;
    
    group_sizze_30552 = ctx->sizes.group_sizze_30551;
    
    int32_t y_30553 = group_sizze_30552 - 1;
    int32_t x_30554 = nesting_sizze_30550 + y_30553;
    int32_t num_groups_30555 = squot32(x_30554, group_sizze_30552);
    int32_t num_threads_30556 = group_sizze_30552 * num_groups_30555;
    int32_t binop_x_30982 = sizze_29668 * sizze_29669;
    int32_t convop_x_30983 = sizze_29670 * binop_x_30982;
    int64_t binop_x_30984 = sext_i32_i64(convop_x_30983);
    int64_t bytes_30981 = 4 * binop_x_30984;
    struct memblock_device mem_30985;
    
    mem_30985.references = NULL;
    memblock_alloc_device(ctx, &mem_30985, bytes_30981, "mem_30985");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 0, sizeof(sizze_29668),
                                  &sizze_29668));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 1, sizeof(sizze_29669),
                                  &sizze_29669));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 2, sizeof(sizze_29670),
                                  &sizze_29670));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 3, sizeof(sizze_29672),
                                  &sizze_29672));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 4, sizeof(sizze_29673),
                                  &sizze_29673));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 5,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 6,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30557, 7,
                                  sizeof(mem_30985.mem), &mem_30985.mem));
    if (1 * (num_groups_30555 * group_sizze_30552) != 0) {
        const size_t global_work_sizze_31309[1] = {num_groups_30555 *
                     group_sizze_30552};
        const size_t local_work_sizze_31313[1] = {group_sizze_30552};
        int64_t time_start_31310 = 0, time_end_31311 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30557");
            fprintf(stderr, "%zu", global_work_sizze_31309[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31313[0]);
            fprintf(stderr, "].\n");
            time_start_31310 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30557, 1, NULL,
                                              global_work_sizze_31309,
                                              local_work_sizze_31313, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31311 = get_wall_time();
            
            long time_diff_31312 = time_end_31311 - time_start_31310;
            
            ctx->map_kernel_30557_total_runtime += time_diff_31312;
            ctx->map_kernel_30557_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30557",
                    time_diff_31312);
        }
    }
    
    int32_t tile_sizze_30924;
    
    tile_sizze_30924 = ctx->sizes.tile_sizze_30923;
    
    int32_t tiled_group_sizze_30925 = tile_sizze_30924 * tile_sizze_30924;
    int32_t y_30932 = tile_sizze_30924 - 1;
    int32_t x_30933 = sizze_29669 + y_30932;
    int32_t groups_in_dim_30934 = squot32(x_30933, tile_sizze_30924);
    int32_t x_30936 = sizze_29675 + y_30932;
    int32_t groups_in_dim_30937 = squot32(x_30936, tile_sizze_30924);
    int32_t y_30939 = groups_in_dim_30934 * groups_in_dim_30937;
    int32_t num_groups_30940 = sizze_29668 * y_30939;
    int32_t num_threads_30941 = tiled_group_sizze_30925 * num_groups_30940;
    int32_t convop_x_30996 = sizze_29675 * binop_x_30982;
    int64_t binop_x_30997 = sext_i32_i64(convop_x_30996);
    int64_t bytes_30994 = 4 * binop_x_30997;
    struct memblock_device mem_30998;
    
    mem_30998.references = NULL;
    memblock_alloc_device(ctx, &mem_30998, bytes_30994, "mem_30998");
    
    int64_t binop_x_30988 = sext_i32_i64(tiled_group_sizze_30925);
    int64_t bytes_30986 = 4 * binop_x_30988;
    struct memblock_local mem_30989;
    
    mem_30989.references = NULL;
    
    struct memblock_local mem_30993;
    
    mem_30993.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 0, bytes_30986, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 1, bytes_30986, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 2, sizeof(sizze_29668),
                                  &sizze_29668));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 3, sizeof(sizze_29669),
                                  &sizze_29669));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 4, sizeof(sizze_29670),
                                  &sizze_29670));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 5, sizeof(sizze_29675),
                                  &sizze_29675));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 6,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 7,
                                  sizeof(mem_30985.mem), &mem_30985.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30523, 8,
                                  sizeof(mem_30998.mem), &mem_30998.mem));
    if (1 * (num_groups_30940 * tiled_group_sizze_30925) != 0) {
        const size_t global_work_sizze_31314[1] = {num_groups_30940 *
                     tiled_group_sizze_30925};
        const size_t local_work_sizze_31318[1] = {tiled_group_sizze_30925};
        int64_t time_start_31315 = 0, time_end_31316 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30523");
            fprintf(stderr, "%zu", global_work_sizze_31314[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31318[0]);
            fprintf(stderr, "].\n");
            time_start_31315 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30523, 1, NULL,
                                              global_work_sizze_31314,
                                              local_work_sizze_31318, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31316 = get_wall_time();
            
            long time_diff_31317 = time_end_31316 - time_start_31315;
            
            ctx->map_kernel_30523_total_runtime += time_diff_31317;
            ctx->map_kernel_30523_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30523",
                    time_diff_31317);
        }
    }
    memblock_unref_device(ctx, &mem_30985, "mem_30985");
    memblock_unref_local(ctx, &mem_30989, "mem_30989");
    memblock_unref_local(ctx, &mem_30993, "mem_30993");
    out_arrsizze_31115 = sizze_29668;
    out_arrsizze_31116 = sizze_29669;
    out_arrsizze_31117 = sizze_29675;
    out_memsizze_31114 = bytes_30994;
    memblock_set_device(ctx, &out_mem_31113, &mem_30998, "mem_30998");
    *out_out_memsizze_31304 = out_memsizze_31114;
    (*out_mem_p_31305).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31305, &out_mem_31113,
                        "out_mem_31113");
    *out_out_arrsizze_31306 = out_arrsizze_31115;
    *out_out_arrsizze_31307 = out_arrsizze_31116;
    *out_out_arrsizze_31308 = out_arrsizze_31117;
    memblock_unref_local(ctx, &mem_30993, "mem_30993");
    memblock_unref_local(ctx, &mem_30989, "mem_30989");
    memblock_unref_device(ctx, &mem_30998, "mem_30998");
    memblock_unref_device(ctx, &mem_30985, "mem_30985");
    memblock_unref_device(ctx, &out_mem_31113, "out_mem_31113");
    return 0;
}
static int futrts_t5(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31319,
                     struct memblock_device *out_mem_p_31320,
                     int32_t *out_out_arrsizze_31321,
                     int32_t *out_out_arrsizze_31322,
                     int32_t *out_out_arrsizze_31323, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29727,
                     int32_t sizze_29728, int32_t sizze_29729,
                     int32_t sizze_29730, int32_t sizze_29731,
                     int32_t sizze_29732, int32_t sizze_29733,
                     int32_t sizze_29734, int32_t sizze_29735,
                     int32_t sizze_29736)
{
    int64_t out_memsizze_31130;
    struct memblock_device out_mem_31129;
    
    out_mem_31129.references = NULL;
    
    int32_t out_arrsizze_31131;
    int32_t out_arrsizze_31132;
    int32_t out_arrsizze_31133;
    bool dim_zzero_29741 = 0 == sizze_29730;
    bool dim_zzero_29742 = 0 == sizze_29731;
    bool dim_zzero_29743 = 0 == sizze_29732;
    bool y_29744 = dim_zzero_29742 || dim_zzero_29743;
    bool old_empty_29745 = dim_zzero_29741 || y_29744;
    bool dim_zzero_29746 = 0 == sizze_29727;
    bool dim_zzero_29747 = 0 == sizze_29728;
    bool dim_zzero_29748 = 0 == sizze_29729;
    bool y_29749 = dim_zzero_29747 || dim_zzero_29748;
    bool new_empty_29750 = dim_zzero_29746 || y_29749;
    bool both_empty_29751 = old_empty_29745 && new_empty_29750;
    bool dim_match_29752 = sizze_29727 == sizze_29730;
    bool dim_match_29753 = sizze_29728 == sizze_29731;
    bool dim_match_29754 = sizze_29729 == sizze_29732;
    bool y_29755 = dim_match_29753 && dim_match_29754;
    bool match_29756 = dim_match_29752 && y_29755;
    bool empty_or_match_29757 = both_empty_29751 || match_29756;
    bool empty_or_match_cert_29758;
    
    if (!empty_or_match_29757) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:29:1-30:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31129, "out_mem_31129");
        return 1;
    }
    
    bool dim_zzero_29759 = 0 == sizze_29733;
    bool dim_zzero_29760 = 0 == sizze_29734;
    bool old_empty_29761 = dim_zzero_29759 || dim_zzero_29760;
    bool new_empty_29762 = dim_zzero_29748 || dim_zzero_29760;
    bool both_empty_29763 = old_empty_29761 && new_empty_29762;
    bool dim_match_29764 = sizze_29729 == sizze_29733;
    bool empty_or_match_29765 = both_empty_29763 || dim_match_29764;
    bool empty_or_match_cert_29766;
    
    if (!empty_or_match_29765) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:29:1-30:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31129, "out_mem_31129");
        return 1;
    }
    
    bool dim_zzero_29768 = 0 == sizze_29735;
    bool dim_zzero_29769 = 0 == sizze_29736;
    bool old_empty_29770 = dim_zzero_29768 || dim_zzero_29769;
    bool both_empty_29771 = new_empty_29762 && old_empty_29770;
    bool dim_match_29772 = sizze_29729 == sizze_29735;
    bool dim_match_29773 = sizze_29734 == sizze_29736;
    bool match_29774 = dim_match_29772 && dim_match_29773;
    bool empty_or_match_29775 = both_empty_29771 || match_29774;
    bool empty_or_match_cert_29776;
    
    if (!empty_or_match_29775) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:29:1-30:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31129, "out_mem_31129");
        return 1;
    }
    
    int32_t nesting_sizze_30565 = sizze_29729 * sizze_29734;
    int32_t group_sizze_30567;
    
    group_sizze_30567 = ctx->sizes.group_sizze_30566;
    
    int32_t y_30568 = group_sizze_30567 - 1;
    int32_t x_30569 = nesting_sizze_30565 + y_30568;
    int32_t num_groups_30570 = squot32(x_30569, group_sizze_30567);
    int32_t num_threads_30571 = group_sizze_30567 * num_groups_30570;
    int64_t binop_x_30985 = sext_i32_i64(nesting_sizze_30565);
    int64_t bytes_30983 = 4 * binop_x_30985;
    struct memblock_device mem_30986;
    
    mem_30986.references = NULL;
    memblock_alloc_device(ctx, &mem_30986, bytes_30983, "mem_30986");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 0, sizeof(sizze_29729),
                                  &sizze_29729));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 1, sizeof(sizze_29734),
                                  &sizze_29734));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 2, sizeof(sizze_29736),
                                  &sizze_29736));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 3,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 4,
                                  sizeof(D_mem_30982.mem), &D_mem_30982.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30572, 5,
                                  sizeof(mem_30986.mem), &mem_30986.mem));
    if (1 * (num_groups_30570 * group_sizze_30567) != 0) {
        const size_t global_work_sizze_31324[1] = {num_groups_30570 *
                     group_sizze_30567};
        const size_t local_work_sizze_31328[1] = {group_sizze_30567};
        int64_t time_start_31325 = 0, time_end_31326 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30572");
            fprintf(stderr, "%zu", global_work_sizze_31324[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31328[0]);
            fprintf(stderr, "].\n");
            time_start_31325 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30572, 1, NULL,
                                              global_work_sizze_31324,
                                              local_work_sizze_31328, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31326 = get_wall_time();
            
            long time_diff_31327 = time_end_31326 - time_start_31325;
            
            ctx->map_kernel_30572_total_runtime += time_diff_31327;
            ctx->map_kernel_30572_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30572",
                    time_diff_31327);
        }
    }
    
    int32_t nesting_sizze_30628 = sizze_29728 * sizze_29729;
    int32_t nesting_sizze_30629 = sizze_29727 * nesting_sizze_30628;
    int32_t group_sizze_30631;
    
    group_sizze_30631 = ctx->sizes.group_sizze_30630;
    
    int32_t y_30632 = group_sizze_30631 - 1;
    int32_t x_30633 = nesting_sizze_30629 + y_30632;
    int32_t num_groups_30634 = squot32(x_30633, group_sizze_30631);
    int32_t num_threads_30635 = group_sizze_30631 * num_groups_30634;
    int32_t binop_x_30988 = sizze_29727 * sizze_29728;
    int32_t convop_x_30989 = sizze_29729 * binop_x_30988;
    int64_t binop_x_30990 = sext_i32_i64(convop_x_30989);
    int64_t bytes_30987 = 4 * binop_x_30990;
    struct memblock_device mem_30991;
    
    mem_30991.references = NULL;
    memblock_alloc_device(ctx, &mem_30991, bytes_30987, "mem_30991");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 0, sizeof(sizze_29727),
                                  &sizze_29727));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 1, sizeof(sizze_29728),
                                  &sizze_29728));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 2, sizeof(sizze_29729),
                                  &sizze_29729));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 3, sizeof(sizze_29731),
                                  &sizze_29731));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 4, sizeof(sizze_29732),
                                  &sizze_29732));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 5,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 6,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30636, 7,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    if (1 * (num_groups_30634 * group_sizze_30631) != 0) {
        const size_t global_work_sizze_31329[1] = {num_groups_30634 *
                     group_sizze_30631};
        const size_t local_work_sizze_31333[1] = {group_sizze_30631};
        int64_t time_start_31330 = 0, time_end_31331 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30636");
            fprintf(stderr, "%zu", global_work_sizze_31329[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31333[0]);
            fprintf(stderr, "].\n");
            time_start_31330 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30636, 1, NULL,
                                              global_work_sizze_31329,
                                              local_work_sizze_31333, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31331 = get_wall_time();
            
            long time_diff_31332 = time_end_31331 - time_start_31330;
            
            ctx->map_kernel_30636_total_runtime += time_diff_31332;
            ctx->map_kernel_30636_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30636",
                    time_diff_31332);
        }
    }
    
    int32_t tile_sizze_30924;
    
    tile_sizze_30924 = ctx->sizes.tile_sizze_30923;
    
    int32_t tiled_group_sizze_30925 = tile_sizze_30924 * tile_sizze_30924;
    int32_t y_30932 = tile_sizze_30924 - 1;
    int32_t x_30933 = sizze_29728 + y_30932;
    int32_t groups_in_dim_30934 = squot32(x_30933, tile_sizze_30924);
    int32_t x_30936 = sizze_29734 + y_30932;
    int32_t groups_in_dim_30937 = squot32(x_30936, tile_sizze_30924);
    int32_t y_30939 = groups_in_dim_30934 * groups_in_dim_30937;
    int32_t num_groups_30940 = sizze_29727 * y_30939;
    int32_t num_threads_30941 = tiled_group_sizze_30925 * num_groups_30940;
    int32_t convop_x_31002 = sizze_29734 * binop_x_30988;
    int64_t binop_x_31003 = sext_i32_i64(convop_x_31002);
    int64_t bytes_31000 = 4 * binop_x_31003;
    struct memblock_device mem_31004;
    
    mem_31004.references = NULL;
    memblock_alloc_device(ctx, &mem_31004, bytes_31000, "mem_31004");
    
    int64_t binop_x_30994 = sext_i32_i64(tiled_group_sizze_30925);
    int64_t bytes_30992 = 4 * binop_x_30994;
    struct memblock_local mem_30995;
    
    mem_30995.references = NULL;
    
    struct memblock_local mem_30999;
    
    mem_30999.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 0, bytes_30992, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 1, bytes_30992, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 2, sizeof(sizze_29727),
                                  &sizze_29727));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 3, sizeof(sizze_29728),
                                  &sizze_29728));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 4, sizeof(sizze_29729),
                                  &sizze_29729));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 5, sizeof(sizze_29734),
                                  &sizze_29734));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 6,
                                  sizeof(mem_30986.mem), &mem_30986.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 7,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30602, 8,
                                  sizeof(mem_31004.mem), &mem_31004.mem));
    if (1 * (num_groups_30940 * tiled_group_sizze_30925) != 0) {
        const size_t global_work_sizze_31334[1] = {num_groups_30940 *
                     tiled_group_sizze_30925};
        const size_t local_work_sizze_31338[1] = {tiled_group_sizze_30925};
        int64_t time_start_31335 = 0, time_end_31336 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30602");
            fprintf(stderr, "%zu", global_work_sizze_31334[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31338[0]);
            fprintf(stderr, "].\n");
            time_start_31335 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30602, 1, NULL,
                                              global_work_sizze_31334,
                                              local_work_sizze_31338, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31336 = get_wall_time();
            
            long time_diff_31337 = time_end_31336 - time_start_31335;
            
            ctx->map_kernel_30602_total_runtime += time_diff_31337;
            ctx->map_kernel_30602_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30602",
                    time_diff_31337);
        }
    }
    memblock_unref_device(ctx, &mem_30986, "mem_30986");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_local(ctx, &mem_30995, "mem_30995");
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    out_arrsizze_31131 = sizze_29727;
    out_arrsizze_31132 = sizze_29728;
    out_arrsizze_31133 = sizze_29734;
    out_memsizze_31130 = bytes_31000;
    memblock_set_device(ctx, &out_mem_31129, &mem_31004, "mem_31004");
    *out_out_memsizze_31319 = out_memsizze_31130;
    (*out_mem_p_31320).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31320, &out_mem_31129,
                        "out_mem_31129");
    *out_out_arrsizze_31321 = out_arrsizze_31131;
    *out_out_arrsizze_31322 = out_arrsizze_31132;
    *out_out_arrsizze_31323 = out_arrsizze_31133;
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    memblock_unref_local(ctx, &mem_30995, "mem_30995");
    memblock_unref_device(ctx, &mem_31004, "mem_31004");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_device(ctx, &mem_30986, "mem_30986");
    memblock_unref_device(ctx, &out_mem_31129, "out_mem_31129");
    return 0;
}
static int futrts_t6(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31339,
                     struct memblock_device *out_mem_p_31340,
                     int32_t *out_out_arrsizze_31341,
                     int32_t *out_out_arrsizze_31342,
                     int32_t *out_out_arrsizze_31343, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982, int32_t sizze_29806,
                     int32_t sizze_29807, int32_t sizze_29808,
                     int32_t sizze_29809, int32_t sizze_29810,
                     int32_t sizze_29811, int32_t sizze_29812,
                     int32_t sizze_29813, int32_t sizze_29814,
                     int32_t sizze_29815, float a_29816, float b_29818,
                     float c_29820, float d_29822)
{
    int64_t out_memsizze_31149;
    struct memblock_device out_mem_31148;
    
    out_mem_31148.references = NULL;
    
    int32_t out_arrsizze_31150;
    int32_t out_arrsizze_31151;
    int32_t out_arrsizze_31152;
    bool dim_zzero_29824 = 0 == sizze_29809;
    bool dim_zzero_29825 = 0 == sizze_29810;
    bool dim_zzero_29826 = 0 == sizze_29811;
    bool y_29827 = dim_zzero_29825 || dim_zzero_29826;
    bool old_empty_29828 = dim_zzero_29824 || y_29827;
    bool dim_zzero_29829 = 0 == sizze_29806;
    bool dim_zzero_29830 = 0 == sizze_29807;
    bool dim_zzero_29831 = 0 == sizze_29808;
    bool y_29832 = dim_zzero_29830 || dim_zzero_29831;
    bool new_empty_29833 = dim_zzero_29829 || y_29832;
    bool both_empty_29834 = old_empty_29828 && new_empty_29833;
    bool dim_match_29835 = sizze_29806 == sizze_29809;
    bool dim_match_29836 = sizze_29807 == sizze_29810;
    bool dim_match_29837 = sizze_29808 == sizze_29811;
    bool y_29838 = dim_match_29836 && dim_match_29837;
    bool match_29839 = dim_match_29835 && y_29838;
    bool empty_or_match_29840 = both_empty_29834 || match_29839;
    bool empty_or_match_cert_29841;
    
    if (!empty_or_match_29840) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:32:1-35:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31148, "out_mem_31148");
        return 1;
    }
    
    bool dim_zzero_29842 = 0 == sizze_29812;
    bool dim_zzero_29843 = 0 == sizze_29813;
    bool old_empty_29844 = dim_zzero_29842 || dim_zzero_29843;
    bool new_empty_29845 = dim_zzero_29831 || dim_zzero_29843;
    bool both_empty_29846 = old_empty_29844 && new_empty_29845;
    bool dim_match_29847 = sizze_29808 == sizze_29812;
    bool empty_or_match_29848 = both_empty_29846 || dim_match_29847;
    bool empty_or_match_cert_29849;
    
    if (!empty_or_match_29848) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:32:1-35:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31148, "out_mem_31148");
        return 1;
    }
    
    bool dim_zzero_29850 = 0 == sizze_29814;
    bool dim_zzero_29851 = 0 == sizze_29815;
    bool old_empty_29852 = dim_zzero_29850 || dim_zzero_29851;
    bool both_empty_29853 = new_empty_29845 && old_empty_29852;
    bool dim_match_29854 = sizze_29808 == sizze_29814;
    bool dim_match_29855 = sizze_29813 == sizze_29815;
    bool match_29856 = dim_match_29854 && dim_match_29855;
    bool empty_or_match_29857 = both_empty_29853 || match_29856;
    bool empty_or_match_cert_29858;
    
    if (!empty_or_match_29857) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:32:1-35:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31148, "out_mem_31148");
        return 1;
    }
    
    int32_t nesting_sizze_30644 = sizze_29808 * sizze_29813;
    int32_t group_sizze_30646;
    
    group_sizze_30646 = ctx->sizes.group_sizze_30645;
    
    int32_t y_30647 = group_sizze_30646 - 1;
    int32_t x_30648 = nesting_sizze_30644 + y_30647;
    int32_t num_groups_30649 = squot32(x_30648, group_sizze_30646);
    int32_t num_threads_30650 = group_sizze_30646 * num_groups_30649;
    int64_t binop_x_30985 = sext_i32_i64(nesting_sizze_30644);
    int64_t bytes_30983 = 4 * binop_x_30985;
    struct memblock_device mem_30986;
    
    mem_30986.references = NULL;
    memblock_alloc_device(ctx, &mem_30986, bytes_30983, "mem_30986");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 0, sizeof(sizze_29808),
                                  &sizze_29808));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 1, sizeof(sizze_29813),
                                  &sizze_29813));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 2, sizeof(sizze_29815),
                                  &sizze_29815));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 3, sizeof(c_29820),
                                  &c_29820));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 4, sizeof(d_29822),
                                  &d_29822));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 5,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 6,
                                  sizeof(D_mem_30982.mem), &D_mem_30982.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30651, 7,
                                  sizeof(mem_30986.mem), &mem_30986.mem));
    if (1 * (num_groups_30649 * group_sizze_30646) != 0) {
        const size_t global_work_sizze_31344[1] = {num_groups_30649 *
                     group_sizze_30646};
        const size_t local_work_sizze_31348[1] = {group_sizze_30646};
        int64_t time_start_31345 = 0, time_end_31346 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30651");
            fprintf(stderr, "%zu", global_work_sizze_31344[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31348[0]);
            fprintf(stderr, "].\n");
            time_start_31345 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30651, 1, NULL,
                                              global_work_sizze_31344,
                                              local_work_sizze_31348, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31346 = get_wall_time();
            
            long time_diff_31347 = time_end_31346 - time_start_31345;
            
            ctx->map_kernel_30651_total_runtime += time_diff_31347;
            ctx->map_kernel_30651_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30651",
                    time_diff_31347);
        }
    }
    
    int32_t nesting_sizze_30709 = sizze_29807 * sizze_29808;
    int32_t nesting_sizze_30710 = sizze_29806 * nesting_sizze_30709;
    int32_t group_sizze_30712;
    
    group_sizze_30712 = ctx->sizes.group_sizze_30711;
    
    int32_t y_30713 = group_sizze_30712 - 1;
    int32_t x_30714 = nesting_sizze_30710 + y_30713;
    int32_t num_groups_30715 = squot32(x_30714, group_sizze_30712);
    int32_t num_threads_30716 = group_sizze_30712 * num_groups_30715;
    int32_t binop_x_30988 = sizze_29806 * sizze_29807;
    int32_t convop_x_30989 = sizze_29808 * binop_x_30988;
    int64_t binop_x_30990 = sext_i32_i64(convop_x_30989);
    int64_t bytes_30987 = 4 * binop_x_30990;
    struct memblock_device mem_30991;
    
    mem_30991.references = NULL;
    memblock_alloc_device(ctx, &mem_30991, bytes_30987, "mem_30991");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 0, sizeof(sizze_29806),
                                  &sizze_29806));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 1, sizeof(sizze_29807),
                                  &sizze_29807));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 2, sizeof(sizze_29808),
                                  &sizze_29808));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 3, sizeof(sizze_29810),
                                  &sizze_29810));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 4, sizeof(sizze_29811),
                                  &sizze_29811));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 5, sizeof(a_29816),
                                  &a_29816));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 6, sizeof(b_29818),
                                  &b_29818));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 7,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 8,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30717, 9,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    if (1 * (num_groups_30715 * group_sizze_30712) != 0) {
        const size_t global_work_sizze_31349[1] = {num_groups_30715 *
                     group_sizze_30712};
        const size_t local_work_sizze_31353[1] = {group_sizze_30712};
        int64_t time_start_31350 = 0, time_end_31351 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30717");
            fprintf(stderr, "%zu", global_work_sizze_31349[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31353[0]);
            fprintf(stderr, "].\n");
            time_start_31350 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30717, 1, NULL,
                                              global_work_sizze_31349,
                                              local_work_sizze_31353, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31351 = get_wall_time();
            
            long time_diff_31352 = time_end_31351 - time_start_31350;
            
            ctx->map_kernel_30717_total_runtime += time_diff_31352;
            ctx->map_kernel_30717_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30717",
                    time_diff_31352);
        }
    }
    
    int32_t tile_sizze_30924;
    
    tile_sizze_30924 = ctx->sizes.tile_sizze_30923;
    
    int32_t tiled_group_sizze_30925 = tile_sizze_30924 * tile_sizze_30924;
    int32_t y_30932 = tile_sizze_30924 - 1;
    int32_t x_30933 = sizze_29807 + y_30932;
    int32_t groups_in_dim_30934 = squot32(x_30933, tile_sizze_30924);
    int32_t x_30936 = sizze_29813 + y_30932;
    int32_t groups_in_dim_30937 = squot32(x_30936, tile_sizze_30924);
    int32_t y_30939 = groups_in_dim_30934 * groups_in_dim_30937;
    int32_t num_groups_30940 = sizze_29806 * y_30939;
    int32_t num_threads_30941 = tiled_group_sizze_30925 * num_groups_30940;
    int32_t convop_x_31002 = sizze_29813 * binop_x_30988;
    int64_t binop_x_31003 = sext_i32_i64(convop_x_31002);
    int64_t bytes_31000 = 4 * binop_x_31003;
    struct memblock_device mem_31004;
    
    mem_31004.references = NULL;
    memblock_alloc_device(ctx, &mem_31004, bytes_31000, "mem_31004");
    
    int64_t binop_x_30994 = sext_i32_i64(tiled_group_sizze_30925);
    int64_t bytes_30992 = 4 * binop_x_30994;
    struct memblock_local mem_30995;
    
    mem_30995.references = NULL;
    
    struct memblock_local mem_30999;
    
    mem_30999.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 0, bytes_30992, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 1, bytes_30992, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 2, sizeof(sizze_29806),
                                  &sizze_29806));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 3, sizeof(sizze_29807),
                                  &sizze_29807));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 4, sizeof(sizze_29808),
                                  &sizze_29808));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 5, sizeof(sizze_29813),
                                  &sizze_29813));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 6,
                                  sizeof(mem_30986.mem), &mem_30986.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 7,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30683, 8,
                                  sizeof(mem_31004.mem), &mem_31004.mem));
    if (1 * (num_groups_30940 * tiled_group_sizze_30925) != 0) {
        const size_t global_work_sizze_31354[1] = {num_groups_30940 *
                     tiled_group_sizze_30925};
        const size_t local_work_sizze_31358[1] = {tiled_group_sizze_30925};
        int64_t time_start_31355 = 0, time_end_31356 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30683");
            fprintf(stderr, "%zu", global_work_sizze_31354[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31358[0]);
            fprintf(stderr, "].\n");
            time_start_31355 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30683, 1, NULL,
                                              global_work_sizze_31354,
                                              local_work_sizze_31358, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31356 = get_wall_time();
            
            long time_diff_31357 = time_end_31356 - time_start_31355;
            
            ctx->map_kernel_30683_total_runtime += time_diff_31357;
            ctx->map_kernel_30683_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30683",
                    time_diff_31357);
        }
    }
    memblock_unref_device(ctx, &mem_30986, "mem_30986");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_local(ctx, &mem_30995, "mem_30995");
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    out_arrsizze_31150 = sizze_29806;
    out_arrsizze_31151 = sizze_29807;
    out_arrsizze_31152 = sizze_29813;
    out_memsizze_31149 = bytes_31000;
    memblock_set_device(ctx, &out_mem_31148, &mem_31004, "mem_31004");
    *out_out_memsizze_31339 = out_memsizze_31149;
    (*out_mem_p_31340).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31340, &out_mem_31148,
                        "out_mem_31148");
    *out_out_arrsizze_31341 = out_arrsizze_31150;
    *out_out_arrsizze_31342 = out_arrsizze_31151;
    *out_out_arrsizze_31343 = out_arrsizze_31152;
    memblock_unref_local(ctx, &mem_30999, "mem_30999");
    memblock_unref_local(ctx, &mem_30995, "mem_30995");
    memblock_unref_device(ctx, &mem_31004, "mem_31004");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_device(ctx, &mem_30986, "mem_30986");
    memblock_unref_device(ctx, &out_mem_31148, "out_mem_31148");
    return 0;
}
static int futrts_t7(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31359,
                     struct memblock_device *out_mem_p_31360,
                     int32_t *out_out_arrsizze_31361,
                     int32_t *out_out_arrsizze_31362,
                     int32_t *out_out_arrsizze_31363, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980,
                     int64_t D_mem_sizze_30981,
                     struct memblock_device D_mem_30982,
                     int64_t E_mem_sizze_30983,
                     struct memblock_device E_mem_30984,
                     int64_t F_mem_sizze_30985,
                     struct memblock_device F_mem_30986,
                     int64_t G_mem_sizze_30987,
                     struct memblock_device G_mem_30988,
                     int64_t H_mem_sizze_30989,
                     struct memblock_device H_mem_30990, int32_t sizze_29893,
                     int32_t sizze_29894, int32_t sizze_29895,
                     int32_t sizze_29896, int32_t sizze_29897,
                     int32_t sizze_29898, int32_t sizze_29899,
                     int32_t sizze_29900, int32_t sizze_29901,
                     int32_t sizze_29902, int32_t sizze_29903,
                     int32_t sizze_29904, int32_t sizze_29905,
                     int32_t sizze_29906, float a_29907, float b_29909,
                     float c_29911, float d_29913, float e_29915, float f_29917,
                     float g_29919, float h_29921)
{
    int64_t out_memsizze_31168;
    struct memblock_device out_mem_31167;
    
    out_mem_31167.references = NULL;
    
    int32_t out_arrsizze_31169;
    int32_t out_arrsizze_31170;
    int32_t out_arrsizze_31171;
    bool dim_zzero_29923 = 0 == sizze_29896;
    bool dim_zzero_29924 = 0 == sizze_29897;
    bool dim_zzero_29925 = 0 == sizze_29898;
    bool y_29926 = dim_zzero_29924 || dim_zzero_29925;
    bool old_empty_29927 = dim_zzero_29923 || y_29926;
    bool dim_zzero_29928 = 0 == sizze_29893;
    bool dim_zzero_29929 = 0 == sizze_29894;
    bool dim_zzero_29930 = 0 == sizze_29895;
    bool y_29931 = dim_zzero_29929 || dim_zzero_29930;
    bool new_empty_29932 = dim_zzero_29928 || y_29931;
    bool both_empty_29933 = old_empty_29927 && new_empty_29932;
    bool dim_match_29934 = sizze_29893 == sizze_29896;
    bool dim_match_29935 = sizze_29894 == sizze_29897;
    bool dim_match_29936 = sizze_29895 == sizze_29898;
    bool y_29937 = dim_match_29935 && dim_match_29936;
    bool match_29938 = dim_match_29934 && y_29937;
    bool empty_or_match_29939 = both_empty_29933 || match_29938;
    bool empty_or_match_cert_29940;
    
    if (!empty_or_match_29939) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29941 = 0 == sizze_29899;
    bool dim_zzero_29942 = 0 == sizze_29900;
    bool old_empty_29943 = dim_zzero_29941 || dim_zzero_29942;
    bool new_empty_29944 = dim_zzero_29930 || dim_zzero_29942;
    bool both_empty_29945 = old_empty_29943 && new_empty_29944;
    bool dim_match_29946 = sizze_29895 == sizze_29899;
    bool empty_or_match_29947 = both_empty_29945 || dim_match_29946;
    bool empty_or_match_cert_29948;
    
    if (!empty_or_match_29947) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29949 = 0 == sizze_29901;
    bool dim_zzero_29950 = 0 == sizze_29902;
    bool old_empty_29951 = dim_zzero_29949 || dim_zzero_29950;
    bool both_empty_29952 = new_empty_29944 && old_empty_29951;
    bool dim_match_29953 = sizze_29895 == sizze_29901;
    bool dim_match_29954 = sizze_29900 == sizze_29902;
    bool match_29955 = dim_match_29953 && dim_match_29954;
    bool empty_or_match_29956 = both_empty_29952 || match_29955;
    bool empty_or_match_cert_29957;
    
    if (!empty_or_match_29956) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29958 = 0 == sizze_29903;
    bool both_empty_29959 = dim_zzero_29930 && dim_zzero_29958;
    bool dim_match_29960 = sizze_29895 == sizze_29903;
    bool empty_or_match_29961 = both_empty_29959 || dim_match_29960;
    bool empty_or_match_cert_29962;
    
    if (!empty_or_match_29961) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29963 = 0 == sizze_29904;
    bool both_empty_29964 = dim_zzero_29930 && dim_zzero_29963;
    bool dim_match_29965 = sizze_29895 == sizze_29904;
    bool empty_or_match_29966 = both_empty_29964 || dim_match_29965;
    bool empty_or_match_cert_29967;
    
    if (!empty_or_match_29966) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29968 = 0 == sizze_29905;
    bool both_empty_29969 = dim_zzero_29929 && dim_zzero_29968;
    bool dim_match_29970 = sizze_29894 == sizze_29905;
    bool empty_or_match_29971 = both_empty_29969 || dim_match_29970;
    bool empty_or_match_cert_29972;
    
    if (!empty_or_match_29971) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    bool dim_zzero_29973 = 0 == sizze_29906;
    bool both_empty_29974 = dim_zzero_29929 && dim_zzero_29973;
    bool dim_match_29975 = sizze_29894 == sizze_29906;
    bool empty_or_match_29976 = both_empty_29974 || dim_match_29975;
    bool empty_or_match_cert_29977;
    
    if (!empty_or_match_29976) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:37:1-42:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
        return 1;
    }
    
    int32_t nesting_sizze_30742 = sizze_29895 * sizze_29900;
    int32_t group_sizze_30744;
    
    group_sizze_30744 = ctx->sizes.group_sizze_30743;
    
    int32_t y_30745 = group_sizze_30744 - 1;
    int32_t x_30746 = nesting_sizze_30742 + y_30745;
    int32_t num_groups_30747 = squot32(x_30746, group_sizze_30744);
    int32_t num_threads_30748 = group_sizze_30744 * num_groups_30747;
    int64_t binop_x_30993 = sext_i32_i64(nesting_sizze_30742);
    int64_t bytes_30991 = 4 * binop_x_30993;
    struct memblock_device mem_30994;
    
    mem_30994.references = NULL;
    memblock_alloc_device(ctx, &mem_30994, bytes_30991, "mem_30994");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 0, sizeof(sizze_29895),
                                  &sizze_29895));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 1, sizeof(sizze_29900),
                                  &sizze_29900));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 2, sizeof(sizze_29902),
                                  &sizze_29902));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 3, sizeof(c_29911),
                                  &c_29911));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 4, sizeof(d_29913),
                                  &d_29913));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 5,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 6,
                                  sizeof(D_mem_30982.mem), &D_mem_30982.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30749, 7,
                                  sizeof(mem_30994.mem), &mem_30994.mem));
    if (1 * (num_groups_30747 * group_sizze_30744) != 0) {
        const size_t global_work_sizze_31364[1] = {num_groups_30747 *
                     group_sizze_30744};
        const size_t local_work_sizze_31368[1] = {group_sizze_30744};
        int64_t time_start_31365 = 0, time_end_31366 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30749");
            fprintf(stderr, "%zu", global_work_sizze_31364[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31368[0]);
            fprintf(stderr, "].\n");
            time_start_31365 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30749, 1, NULL,
                                              global_work_sizze_31364,
                                              local_work_sizze_31368, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31366 = get_wall_time();
            
            long time_diff_31367 = time_end_31366 - time_start_31365;
            
            ctx->map_kernel_30749_total_runtime += time_diff_31367;
            ctx->map_kernel_30749_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30749",
                    time_diff_31367);
        }
    }
    
    int32_t group_sizze_30727;
    
    group_sizze_30727 = ctx->sizes.group_sizze_30726;
    
    int32_t y_30728 = group_sizze_30727 - 1;
    int32_t x_30729 = sizze_29895 + y_30728;
    int32_t num_groups_30730 = squot32(x_30729, group_sizze_30727);
    int32_t num_threads_30731 = group_sizze_30727 * num_groups_30730;
    int64_t binop_x_30996 = sext_i32_i64(sizze_29895);
    int64_t bytes_30995 = 4 * binop_x_30996;
    struct memblock_device mem_30997;
    
    mem_30997.references = NULL;
    memblock_alloc_device(ctx, &mem_30997, bytes_30995, "mem_30997");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 0, sizeof(sizze_29895),
                                  &sizze_29895));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 1, sizeof(e_29915),
                                  &e_29915));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 2, sizeof(f_29917),
                                  &f_29917));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 3,
                                  sizeof(E_mem_30984.mem), &E_mem_30984.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 4,
                                  sizeof(F_mem_30986.mem), &F_mem_30986.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30732, 5,
                                  sizeof(mem_30997.mem), &mem_30997.mem));
    if (1 * (num_groups_30730 * group_sizze_30727) != 0) {
        const size_t global_work_sizze_31369[1] = {num_groups_30730 *
                     group_sizze_30727};
        const size_t local_work_sizze_31373[1] = {group_sizze_30727};
        int64_t time_start_31370 = 0, time_end_31371 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30732");
            fprintf(stderr, "%zu", global_work_sizze_31369[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31373[0]);
            fprintf(stderr, "].\n");
            time_start_31370 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30732, 1, NULL,
                                              global_work_sizze_31369,
                                              local_work_sizze_31373, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31371 = get_wall_time();
            
            long time_diff_31372 = time_end_31371 - time_start_31370;
            
            ctx->map_kernel_30732_total_runtime += time_diff_31372;
            ctx->map_kernel_30732_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30732",
                    time_diff_31372);
        }
    }
    
    int32_t group_sizze_30759;
    
    group_sizze_30759 = ctx->sizes.group_sizze_30758;
    
    int32_t y_30760 = group_sizze_30759 - 1;
    int32_t x_30761 = sizze_29894 + y_30760;
    int32_t num_groups_30762 = squot32(x_30761, group_sizze_30759);
    int32_t num_threads_30763 = group_sizze_30759 * num_groups_30762;
    int64_t binop_x_30999 = sext_i32_i64(sizze_29894);
    int64_t bytes_30998 = 4 * binop_x_30999;
    struct memblock_device mem_31000;
    
    mem_31000.references = NULL;
    memblock_alloc_device(ctx, &mem_31000, bytes_30998, "mem_31000");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 0, sizeof(sizze_29894),
                                  &sizze_29894));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 1, sizeof(g_29919),
                                  &g_29919));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 2, sizeof(h_29921),
                                  &h_29921));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 3,
                                  sizeof(G_mem_30988.mem), &G_mem_30988.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 4,
                                  sizeof(H_mem_30990.mem), &H_mem_30990.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30764, 5,
                                  sizeof(mem_31000.mem), &mem_31000.mem));
    if (1 * (num_groups_30762 * group_sizze_30759) != 0) {
        const size_t global_work_sizze_31374[1] = {num_groups_30762 *
                     group_sizze_30759};
        const size_t local_work_sizze_31378[1] = {group_sizze_30759};
        int64_t time_start_31375 = 0, time_end_31376 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30764");
            fprintf(stderr, "%zu", global_work_sizze_31374[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31378[0]);
            fprintf(stderr, "].\n");
            time_start_31375 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30764, 1, NULL,
                                              global_work_sizze_31374,
                                              local_work_sizze_31378, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31376 = get_wall_time();
            
            long time_diff_31377 = time_end_31376 - time_start_31375;
            
            ctx->map_kernel_30764_total_runtime += time_diff_31377;
            ctx->map_kernel_30764_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30764",
                    time_diff_31377);
        }
    }
    
    int32_t nesting_sizze_30830 = sizze_29894 * sizze_29895;
    int32_t nesting_sizze_30831 = sizze_29893 * nesting_sizze_30830;
    int32_t group_sizze_30833;
    
    group_sizze_30833 = ctx->sizes.group_sizze_30832;
    
    int32_t y_30834 = group_sizze_30833 - 1;
    int32_t x_30835 = nesting_sizze_30831 + y_30834;
    int32_t num_groups_30836 = squot32(x_30835, group_sizze_30833);
    int32_t num_threads_30837 = group_sizze_30833 * num_groups_30836;
    int32_t binop_x_31002 = sizze_29893 * sizze_29894;
    int32_t convop_x_31003 = sizze_29895 * binop_x_31002;
    int64_t binop_x_31004 = sext_i32_i64(convop_x_31003);
    int64_t bytes_31001 = 4 * binop_x_31004;
    struct memblock_device mem_31005;
    
    mem_31005.references = NULL;
    memblock_alloc_device(ctx, &mem_31005, bytes_31001, "mem_31005");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 0, sizeof(sizze_29893),
                                  &sizze_29893));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 1, sizeof(sizze_29894),
                                  &sizze_29894));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 2, sizeof(sizze_29895),
                                  &sizze_29895));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 3, sizeof(sizze_29897),
                                  &sizze_29897));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 4, sizeof(sizze_29898),
                                  &sizze_29898));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 5, sizeof(a_29907),
                                  &a_29907));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 6, sizeof(b_29909),
                                  &b_29909));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 7,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 8,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30838, 9,
                                  sizeof(mem_31005.mem), &mem_31005.mem));
    if (1 * (num_groups_30836 * group_sizze_30833) != 0) {
        const size_t global_work_sizze_31379[1] = {num_groups_30836 *
                     group_sizze_30833};
        const size_t local_work_sizze_31383[1] = {group_sizze_30833};
        int64_t time_start_31380 = 0, time_end_31381 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30838");
            fprintf(stderr, "%zu", global_work_sizze_31379[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31383[0]);
            fprintf(stderr, "].\n");
            time_start_31380 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30838, 1, NULL,
                                              global_work_sizze_31379,
                                              local_work_sizze_31383, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31381 = get_wall_time();
            
            long time_diff_31382 = time_end_31381 - time_start_31380;
            
            ctx->map_kernel_30838_total_runtime += time_diff_31382;
            ctx->map_kernel_30838_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30838",
                    time_diff_31382);
        }
    }
    
    int32_t nesting_sizze_30790 = sizze_29894 * sizze_29900;
    int32_t nesting_sizze_30791 = sizze_29893 * nesting_sizze_30790;
    int32_t group_sizze_30793;
    
    group_sizze_30793 = ctx->sizes.group_sizze_30792;
    
    int32_t y_30794 = group_sizze_30793 - 1;
    int32_t x_30795 = nesting_sizze_30791 + y_30794;
    int32_t num_groups_30796 = squot32(x_30795, group_sizze_30793);
    int32_t num_threads_30797 = group_sizze_30793 * num_groups_30796;
    int32_t convop_x_31008 = sizze_29900 * binop_x_31002;
    int64_t binop_x_31009 = sext_i32_i64(convop_x_31008);
    int64_t bytes_31006 = 4 * binop_x_31009;
    struct memblock_device mem_31010;
    
    mem_31010.references = NULL;
    memblock_alloc_device(ctx, &mem_31010, bytes_31006, "mem_31010");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 0, sizeof(sizze_29893),
                                  &sizze_29893));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 1, sizeof(sizze_29894),
                                  &sizze_29894));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 2, sizeof(sizze_29895),
                                  &sizze_29895));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 3, sizeof(sizze_29900),
                                  &sizze_29900));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 4,
                                  sizeof(mem_30994.mem), &mem_30994.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 5,
                                  sizeof(mem_30997.mem), &mem_30997.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 6,
                                  sizeof(mem_31000.mem), &mem_31000.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 7,
                                  sizeof(mem_31005.mem), &mem_31005.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30798, 8,
                                  sizeof(mem_31010.mem), &mem_31010.mem));
    if (1 * (num_groups_30796 * group_sizze_30793) != 0) {
        const size_t global_work_sizze_31384[1] = {num_groups_30796 *
                     group_sizze_30793};
        const size_t local_work_sizze_31388[1] = {group_sizze_30793};
        int64_t time_start_31385 = 0, time_end_31386 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30798");
            fprintf(stderr, "%zu", global_work_sizze_31384[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31388[0]);
            fprintf(stderr, "].\n");
            time_start_31385 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30798, 1, NULL,
                                              global_work_sizze_31384,
                                              local_work_sizze_31388, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31386 = get_wall_time();
            
            long time_diff_31387 = time_end_31386 - time_start_31385;
            
            ctx->map_kernel_30798_total_runtime += time_diff_31387;
            ctx->map_kernel_30798_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30798",
                    time_diff_31387);
        }
    }
    memblock_unref_device(ctx, &mem_30994, "mem_30994");
    memblock_unref_device(ctx, &mem_30997, "mem_30997");
    memblock_unref_device(ctx, &mem_31000, "mem_31000");
    memblock_unref_device(ctx, &mem_31005, "mem_31005");
    out_arrsizze_31169 = sizze_29893;
    out_arrsizze_31170 = sizze_29894;
    out_arrsizze_31171 = sizze_29900;
    out_memsizze_31168 = bytes_31006;
    memblock_set_device(ctx, &out_mem_31167, &mem_31010, "mem_31010");
    *out_out_memsizze_31359 = out_memsizze_31168;
    (*out_mem_p_31360).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31360, &out_mem_31167,
                        "out_mem_31167");
    *out_out_arrsizze_31361 = out_arrsizze_31169;
    *out_out_arrsizze_31362 = out_arrsizze_31170;
    *out_out_arrsizze_31363 = out_arrsizze_31171;
    memblock_unref_device(ctx, &mem_31010, "mem_31010");
    memblock_unref_device(ctx, &mem_31005, "mem_31005");
    memblock_unref_device(ctx, &mem_31000, "mem_31000");
    memblock_unref_device(ctx, &mem_30997, "mem_30997");
    memblock_unref_device(ctx, &mem_30994, "mem_30994");
    memblock_unref_device(ctx, &out_mem_31167, "out_mem_31167");
    return 0;
}
static int futrts_t8(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_31389,
                     struct memblock_device *out_mem_p_31390,
                     int32_t *out_out_arrsizze_31391,
                     int32_t *out_out_arrsizze_31392,
                     int32_t *out_out_arrsizze_31393, int64_t A_mem_sizze_30975,
                     struct memblock_device A_mem_30976,
                     int64_t B_mem_sizze_30977,
                     struct memblock_device B_mem_30978,
                     int64_t C_mem_sizze_30979,
                     struct memblock_device C_mem_30980, int32_t sizze_30032,
                     int32_t sizze_30033, int32_t sizze_30034,
                     int32_t sizze_30035, int32_t sizze_30036,
                     int32_t sizze_30037, int32_t sizze_30038)
{
    int64_t out_memsizze_31189;
    struct memblock_device out_mem_31188;
    
    out_mem_31188.references = NULL;
    
    int32_t out_arrsizze_31190;
    int32_t out_arrsizze_31191;
    int32_t out_arrsizze_31192;
    bool dim_zzero_30042 = 0 == sizze_30035;
    bool dim_zzero_30043 = 0 == sizze_30036;
    bool old_empty_30044 = dim_zzero_30042 || dim_zzero_30043;
    bool dim_zzero_30045 = 0 == sizze_30034;
    bool new_empty_30046 = dim_zzero_30043 || dim_zzero_30045;
    bool both_empty_30047 = old_empty_30044 && new_empty_30046;
    bool dim_match_30048 = sizze_30034 == sizze_30035;
    bool empty_or_match_30049 = both_empty_30047 || dim_match_30048;
    bool empty_or_match_cert_30050;
    
    if (!empty_or_match_30049) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:44:1-47:41",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31188, "out_mem_31188");
        return 1;
    }
    
    bool dim_zzero_30052 = 0 == sizze_30037;
    bool dim_zzero_30053 = 0 == sizze_30038;
    bool old_empty_30054 = dim_zzero_30052 || dim_zzero_30053;
    bool dim_zzero_30055 = 0 == sizze_30033;
    bool new_empty_30056 = dim_zzero_30053 || dim_zzero_30055;
    bool both_empty_30057 = old_empty_30054 && new_empty_30056;
    bool dim_match_30058 = sizze_30033 == sizze_30037;
    bool empty_or_match_30059 = both_empty_30057 || dim_match_30058;
    bool empty_or_match_cert_30060;
    
    if (!empty_or_match_30059) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "3d.fut:44:1-47:41",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_31188, "out_mem_31188");
        return 1;
    }
    
    int32_t group_sizze_30874;
    
    group_sizze_30874 = ctx->sizes.group_sizze_30873;
    
    int32_t inner_ldim_30929 = smin32(sizze_30038, group_sizze_30874);
    int32_t y_30931 = sizze_30036 * sizze_30038;
    int32_t threads_necessary_30932 = sizze_30032 * y_30931;
    int32_t y_30933 = inner_ldim_30929 - 1;
    int32_t x_30934 = threads_necessary_30932 + y_30933;
    int32_t groups_necessary_30935 = squot32(x_30934, inner_ldim_30929);
    int32_t num_threads_30936 = inner_ldim_30929 * groups_necessary_30935;
    int32_t binop_x_30988 = sizze_30032 * sizze_30036;
    int32_t convop_x_30989 = sizze_30038 * binop_x_30988;
    int64_t binop_x_30990 = sext_i32_i64(convop_x_30989);
    int64_t bytes_30987 = 4 * binop_x_30990;
    struct memblock_device mem_30991;
    
    mem_30991.references = NULL;
    memblock_alloc_device(ctx, &mem_30991, bytes_30987, "mem_30991");
    
    int64_t binop_x_30982 = sext_i32_i64(group_sizze_30874);
    int64_t bytes_30981 = 4 * binop_x_30982;
    struct memblock_local mem_30983;
    
    mem_30983.references = NULL;
    
    struct memblock_local mem_30986;
    
    mem_30986.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 0, bytes_30981, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 1, bytes_30981, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 2, sizeof(sizze_30032),
                                  &sizze_30032));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 3, sizeof(sizze_30033),
                                  &sizze_30033));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 4, sizeof(sizze_30034),
                                  &sizze_30034));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 5, sizeof(sizze_30036),
                                  &sizze_30036));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 6, sizeof(sizze_30038),
                                  &sizze_30038));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 7,
                                  sizeof(inner_ldim_30929), &inner_ldim_30929));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 8,
                                  sizeof(A_mem_30976.mem), &A_mem_30976.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 9,
                                  sizeof(B_mem_30978.mem), &B_mem_30978.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 10,
                                  sizeof(C_mem_30980.mem), &C_mem_30980.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_30879, 11,
                                  sizeof(mem_30991.mem), &mem_30991.mem));
    if (1 * (groups_necessary_30935 * inner_ldim_30929) != 0) {
        const size_t global_work_sizze_31394[1] = {groups_necessary_30935 *
                     inner_ldim_30929};
        const size_t local_work_sizze_31398[1] = {inner_ldim_30929};
        int64_t time_start_31395 = 0, time_end_31396 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_30879");
            fprintf(stderr, "%zu", global_work_sizze_31394[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_31398[0]);
            fprintf(stderr, "].\n");
            time_start_31395 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_30879, 1, NULL,
                                              global_work_sizze_31394,
                                              local_work_sizze_31398, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_31396 = get_wall_time();
            
            long time_diff_31397 = time_end_31396 - time_start_31395;
            
            ctx->map_kernel_30879_total_runtime += time_diff_31397;
            ctx->map_kernel_30879_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_30879",
                    time_diff_31397);
        }
    }
    memblock_unref_local(ctx, &mem_30983, "mem_30983");
    memblock_unref_local(ctx, &mem_30986, "mem_30986");
    out_arrsizze_31190 = sizze_30032;
    out_arrsizze_31191 = sizze_30036;
    out_arrsizze_31192 = sizze_30038;
    out_memsizze_31189 = bytes_30987;
    memblock_set_device(ctx, &out_mem_31188, &mem_30991, "mem_30991");
    *out_out_memsizze_31389 = out_memsizze_31189;
    (*out_mem_p_31390).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_31390, &out_mem_31188,
                        "out_mem_31188");
    *out_out_arrsizze_31391 = out_arrsizze_31190;
    *out_out_arrsizze_31392 = out_arrsizze_31191;
    *out_out_arrsizze_31393 = out_arrsizze_31192;
    memblock_unref_local(ctx, &mem_30986, "mem_30986");
    memblock_unref_local(ctx, &mem_30983, "mem_30983");
    memblock_unref_device(ctx, &mem_30991, "mem_30991");
    memblock_unref_device(ctx, &out_mem_31188, "out_mem_31188");
    return 0;
}
struct futhark_f32_1d {
    struct memblock_device mem;
    int64_t shape[1];
} ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0)
{
    struct futhark_f32_1d *arr = malloc(sizeof(struct futhark_f32_1d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    memblock_alloc_device(ctx, &arr->mem, dim0 * sizeof(float), "arr->mem");
    if (dim0 * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(ctx->opencl.queue, arr->mem.mem,
                                            CL_TRUE, 0, dim0 * sizeof(float),
                                            data + 0, 0, NULL, NULL));
    arr->shape[0] = dim0;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr)
{
    lock_lock(&ctx->lock);
    memblock_unref_device(ctx, &arr->mem, "arr->mem");
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, arr->mem.mem,
                                           CL_TRUE, 0, arr->shape[0] *
                                           sizeof(float), data + 0, 0, NULL,
                                           NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr)
{
    return arr->shape;
}
struct futhark_f32_2d {
    struct memblock_device mem;
    int64_t shape[2];
} ;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1)
{
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * sizeof(float),
                          "arr->mem");
    if (dim0 * dim1 * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(ctx->opencl.queue, arr->mem.mem,
                                            CL_TRUE, 0, dim0 * dim1 *
                                            sizeof(float), data + 0, 0, NULL,
                                            NULL));
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr)
{
    lock_lock(&ctx->lock);
    memblock_unref_device(ctx, &arr->mem, "arr->mem");
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_2d(struct futhark_context *ctx,
                          struct futhark_f32_2d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, arr->mem.mem,
                                           CL_TRUE, 0, arr->shape[0] *
                                           arr->shape[1] * sizeof(float), data +
                                           0, 0, NULL, NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
int64_t *futhark_shape_f32_2d(struct futhark_context *ctx,
                              struct futhark_f32_2d *arr)
{
    return arr->shape;
}
struct futhark_f32_3d {
    struct memblock_device mem;
    int64_t shape[3];
} ;
struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx,
                                          float *data, int dim0, int dim1,
                                          int dim2)
{
    struct futhark_f32_3d *arr = malloc(sizeof(struct futhark_f32_3d));
    
    if (arr == NULL)
        return NULL;
    lock_lock(&ctx->lock);
    arr->mem.references = NULL;
    memblock_alloc_device(ctx, &arr->mem, dim0 * dim1 * dim2 * sizeof(float),
                          "arr->mem");
    if (dim0 * dim1 * dim2 * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(ctx->opencl.queue, arr->mem.mem,
                                            CL_TRUE, 0, dim0 * dim1 * dim2 *
                                            sizeof(float), data + 0, 0, NULL,
                                            NULL));
    arr->shape[0] = dim0;
    arr->shape[1] = dim1;
    arr->shape[2] = dim2;
    lock_unlock(&ctx->lock);
    return arr;
}
int futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr)
{
    lock_lock(&ctx->lock);
    memblock_unref_device(ctx, &arr->mem, "arr->mem");
    lock_unlock(&ctx->lock);
    free(arr);
    return 0;
}
int futhark_values_f32_3d(struct futhark_context *ctx,
                          struct futhark_f32_3d *arr, float *data)
{
    lock_lock(&ctx->lock);
    if (arr->shape[0] * arr->shape[1] * arr->shape[2] * sizeof(float) > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, arr->mem.mem,
                                           CL_TRUE, 0, arr->shape[0] *
                                           arr->shape[1] * arr->shape[2] *
                                           sizeof(float), data + 0, 0, NULL,
                                           NULL));
    lock_unlock(&ctx->lock);
    return 0;
}
int64_t *futhark_shape_f32_3d(struct futhark_context *ctx,
                              struct futhark_f32_3d *arr)
{
    return arr->shape;
}
int futhark_entry_t9(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int32_t sizze_29396;
    int32_t sizze_29397;
    int32_t sizze_29398;
    int32_t sizze_29399;
    int32_t sizze_29400;
    int32_t sizze_29401;
    int64_t out_memsizze_31036;
    struct memblock_device out_mem_31035;
    
    out_mem_31035.references = NULL;
    
    int32_t out_arrsizze_31037;
    int32_t out_arrsizze_31038;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29396 = in0->shape[0];
    sizze_29397 = in0->shape[1];
    sizze_29398 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29399 = in1->shape[0];
    sizze_29400 = in1->shape[1];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29401 = in2->shape[0];
    
    int ret = futrts_t9(ctx, &out_memsizze_31036, &out_mem_31035,
                        &out_arrsizze_31037, &out_arrsizze_31038,
                        A_mem_sizze_30975, A_mem_30976, B_mem_sizze_30977,
                        B_mem_30978, C_mem_sizze_30979, C_mem_30980,
                        sizze_29396, sizze_29397, sizze_29398, sizze_29399,
                        sizze_29400, sizze_29401);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_31035;
        (*out0)->shape[0] = out_arrsizze_31037;
        (*out0)->shape[1] = out_arrsizze_31038;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t1(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int32_t sizze_29441;
    int32_t sizze_29442;
    int32_t sizze_29443;
    int32_t sizze_29444;
    int32_t sizze_29445;
    int64_t out_memsizze_31048;
    struct memblock_device out_mem_31047;
    
    out_mem_31047.references = NULL;
    
    int32_t out_arrsizze_31049;
    int32_t out_arrsizze_31050;
    int32_t out_arrsizze_31051;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29441 = in0->shape[0];
    sizze_29442 = in0->shape[1];
    sizze_29443 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29444 = in1->shape[0];
    sizze_29445 = in1->shape[1];
    
    int ret = futrts_t1(ctx, &out_memsizze_31048, &out_mem_31047,
                        &out_arrsizze_31049, &out_arrsizze_31050,
                        &out_arrsizze_31051, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, sizze_29441,
                        sizze_29442, sizze_29443, sizze_29444, sizze_29445);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31047;
        (*out0)->shape[0] = out_arrsizze_31049;
        (*out0)->shape[1] = out_arrsizze_31050;
        (*out0)->shape[2] = out_arrsizze_31051;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t2(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int32_t sizze_29472;
    int32_t sizze_29473;
    int32_t sizze_29474;
    int32_t sizze_29475;
    int32_t sizze_29476;
    int64_t out_memsizze_31061;
    struct memblock_device out_mem_31060;
    
    out_mem_31060.references = NULL;
    
    int32_t out_arrsizze_31062;
    int32_t out_arrsizze_31063;
    int32_t out_arrsizze_31064;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29472 = in0->shape[0];
    sizze_29473 = in0->shape[1];
    sizze_29474 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29475 = in1->shape[0];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29476 = in2->shape[0];
    
    int ret = futrts_t2(ctx, &out_memsizze_31061, &out_mem_31060,
                        &out_arrsizze_31062, &out_arrsizze_31063,
                        &out_arrsizze_31064, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, sizze_29472, sizze_29473, sizze_29474,
                        sizze_29475, sizze_29476);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31060;
        (*out0)->shape[0] = out_arrsizze_31062;
        (*out0)->shape[1] = out_arrsizze_31063;
        (*out0)->shape[2] = out_arrsizze_31064;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t10(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_3d *in0, const
                      struct futhark_f32_3d *in1, const
                      struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int32_t sizze_29506;
    int32_t sizze_29507;
    int32_t sizze_29508;
    int32_t sizze_29509;
    int32_t sizze_29510;
    int32_t sizze_29511;
    int32_t sizze_29512;
    int64_t out_memsizze_31077;
    struct memblock_device out_mem_31076;
    
    out_mem_31076.references = NULL;
    
    int32_t out_arrsizze_31078;
    int32_t out_arrsizze_31079;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29506 = in0->shape[0];
    sizze_29507 = in0->shape[1];
    sizze_29508 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29509 = in1->shape[0];
    sizze_29510 = in1->shape[1];
    sizze_29511 = in1->shape[2];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29512 = in2->shape[0];
    
    int ret = futrts_t10(ctx, &out_memsizze_31077, &out_mem_31076,
                         &out_arrsizze_31078, &out_arrsizze_31079,
                         A_mem_sizze_30975, A_mem_30976, B_mem_sizze_30977,
                         B_mem_30978, C_mem_sizze_30979, C_mem_30980,
                         sizze_29506, sizze_29507, sizze_29508, sizze_29509,
                         sizze_29510, sizze_29511, sizze_29512);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_31076;
        (*out0)->shape[0] = out_arrsizze_31078;
        (*out0)->shape[1] = out_arrsizze_31079;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t11(struct futhark_context *ctx, struct futhark_f32_2d **out0,
                      const struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1, const
                      struct futhark_f32_1d *in2, const
                      struct futhark_f32_3d *in3, const
                      struct futhark_f32_1d *in4)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int64_t D_mem_sizze_30981;
    struct memblock_device D_mem_30982;
    
    D_mem_30982.references = NULL;
    
    int64_t E_mem_sizze_30983;
    struct memblock_device E_mem_30984;
    
    E_mem_30984.references = NULL;
    
    int32_t sizze_29559;
    int32_t sizze_29560;
    int32_t sizze_29561;
    int32_t sizze_29562;
    int32_t sizze_29563;
    int32_t sizze_29564;
    int32_t sizze_29565;
    int64_t out_memsizze_31089;
    struct memblock_device out_mem_31088;
    
    out_mem_31088.references = NULL;
    
    int32_t out_arrsizze_31090;
    int32_t out_arrsizze_31091;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29559 = in0->shape[0];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29560 = in1->shape[0];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29561 = in2->shape[0];
    D_mem_30982 = in3->mem;
    D_mem_sizze_30981 = in3->mem.size;
    sizze_29562 = in3->shape[0];
    sizze_29563 = in3->shape[1];
    sizze_29564 = in3->shape[2];
    E_mem_30984 = in4->mem;
    E_mem_sizze_30983 = in4->mem.size;
    sizze_29565 = in4->shape[0];
    
    int ret = futrts_t11(ctx, &out_memsizze_31089, &out_mem_31088,
                         &out_arrsizze_31090, &out_arrsizze_31091,
                         A_mem_sizze_30975, A_mem_30976, B_mem_sizze_30977,
                         B_mem_30978, C_mem_sizze_30979, C_mem_30980,
                         D_mem_sizze_30981, D_mem_30982, E_mem_sizze_30983,
                         E_mem_30984, sizze_29559, sizze_29560, sizze_29561,
                         sizze_29562, sizze_29563, sizze_29564, sizze_29565);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_31088;
        (*out0)->shape[0] = out_arrsizze_31090;
        (*out0)->shape[1] = out_arrsizze_31091;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t3(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int64_t D_mem_sizze_30981;
    struct memblock_device D_mem_30982;
    
    D_mem_30982.references = NULL;
    
    int32_t sizze_29616;
    int32_t sizze_29617;
    int32_t sizze_29618;
    int32_t sizze_29619;
    int32_t sizze_29620;
    int32_t sizze_29621;
    int32_t sizze_29622;
    int64_t out_memsizze_31105;
    struct memblock_device out_mem_31104;
    
    out_mem_31104.references = NULL;
    
    int32_t out_arrsizze_31106;
    int32_t out_arrsizze_31107;
    int32_t out_arrsizze_31108;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29616 = in0->shape[0];
    sizze_29617 = in0->shape[1];
    sizze_29618 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29619 = in1->shape[0];
    sizze_29620 = in1->shape[1];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29621 = in2->shape[0];
    D_mem_30982 = in3->mem;
    D_mem_sizze_30981 = in3->mem.size;
    sizze_29622 = in3->shape[0];
    
    int ret = futrts_t3(ctx, &out_memsizze_31105, &out_mem_31104,
                        &out_arrsizze_31106, &out_arrsizze_31107,
                        &out_arrsizze_31108, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, D_mem_sizze_30981, D_mem_30982,
                        sizze_29616, sizze_29617, sizze_29618, sizze_29619,
                        sizze_29620, sizze_29621, sizze_29622);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31104;
        (*out0)->shape[0] = out_arrsizze_31106;
        (*out0)->shape[1] = out_arrsizze_31107;
        (*out0)->shape[2] = out_arrsizze_31108;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t4(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_3d *in1, const
                     struct futhark_f32_2d *in2)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int32_t sizze_29668;
    int32_t sizze_29669;
    int32_t sizze_29670;
    int32_t sizze_29671;
    int32_t sizze_29672;
    int32_t sizze_29673;
    int32_t sizze_29674;
    int32_t sizze_29675;
    int64_t out_memsizze_31114;
    struct memblock_device out_mem_31113;
    
    out_mem_31113.references = NULL;
    
    int32_t out_arrsizze_31115;
    int32_t out_arrsizze_31116;
    int32_t out_arrsizze_31117;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29668 = in0->shape[0];
    sizze_29669 = in0->shape[1];
    sizze_29670 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29671 = in1->shape[0];
    sizze_29672 = in1->shape[1];
    sizze_29673 = in1->shape[2];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29674 = in2->shape[0];
    sizze_29675 = in2->shape[1];
    
    int ret = futrts_t4(ctx, &out_memsizze_31114, &out_mem_31113,
                        &out_arrsizze_31115, &out_arrsizze_31116,
                        &out_arrsizze_31117, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, sizze_29668, sizze_29669, sizze_29670,
                        sizze_29671, sizze_29672, sizze_29673, sizze_29674,
                        sizze_29675);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31113;
        (*out0)->shape[0] = out_arrsizze_31115;
        (*out0)->shape[1] = out_arrsizze_31116;
        (*out0)->shape[2] = out_arrsizze_31117;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t5(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_3d *in1, const
                     struct futhark_f32_2d *in2, const
                     struct futhark_f32_2d *in3)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int64_t D_mem_sizze_30981;
    struct memblock_device D_mem_30982;
    
    D_mem_30982.references = NULL;
    
    int32_t sizze_29727;
    int32_t sizze_29728;
    int32_t sizze_29729;
    int32_t sizze_29730;
    int32_t sizze_29731;
    int32_t sizze_29732;
    int32_t sizze_29733;
    int32_t sizze_29734;
    int32_t sizze_29735;
    int32_t sizze_29736;
    int64_t out_memsizze_31130;
    struct memblock_device out_mem_31129;
    
    out_mem_31129.references = NULL;
    
    int32_t out_arrsizze_31131;
    int32_t out_arrsizze_31132;
    int32_t out_arrsizze_31133;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_29727 = in0->shape[0];
    sizze_29728 = in0->shape[1];
    sizze_29729 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_29730 = in1->shape[0];
    sizze_29731 = in1->shape[1];
    sizze_29732 = in1->shape[2];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_29733 = in2->shape[0];
    sizze_29734 = in2->shape[1];
    D_mem_30982 = in3->mem;
    D_mem_sizze_30981 = in3->mem.size;
    sizze_29735 = in3->shape[0];
    sizze_29736 = in3->shape[1];
    
    int ret = futrts_t5(ctx, &out_memsizze_31130, &out_mem_31129,
                        &out_arrsizze_31131, &out_arrsizze_31132,
                        &out_arrsizze_31133, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, D_mem_sizze_30981, D_mem_30982,
                        sizze_29727, sizze_29728, sizze_29729, sizze_29730,
                        sizze_29731, sizze_29732, sizze_29733, sizze_29734,
                        sizze_29735, sizze_29736);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31129;
        (*out0)->shape[0] = out_arrsizze_31131;
        (*out0)->shape[1] = out_arrsizze_31132;
        (*out0)->shape[2] = out_arrsizze_31133;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t6(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const float in0, const struct futhark_f32_3d *in1, const
                     float in2, const struct futhark_f32_3d *in3, const
                     float in4, const struct futhark_f32_2d *in5, const
                     float in6, const struct futhark_f32_2d *in7)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int64_t D_mem_sizze_30981;
    struct memblock_device D_mem_30982;
    
    D_mem_30982.references = NULL;
    
    int32_t sizze_29806;
    int32_t sizze_29807;
    int32_t sizze_29808;
    int32_t sizze_29809;
    int32_t sizze_29810;
    int32_t sizze_29811;
    int32_t sizze_29812;
    int32_t sizze_29813;
    int32_t sizze_29814;
    int32_t sizze_29815;
    float a_29816;
    float b_29818;
    float c_29820;
    float d_29822;
    int64_t out_memsizze_31149;
    struct memblock_device out_mem_31148;
    
    out_mem_31148.references = NULL;
    
    int32_t out_arrsizze_31150;
    int32_t out_arrsizze_31151;
    int32_t out_arrsizze_31152;
    
    lock_lock(&ctx->lock);
    a_29816 = in0;
    A_mem_30976 = in1->mem;
    A_mem_sizze_30975 = in1->mem.size;
    sizze_29806 = in1->shape[0];
    sizze_29807 = in1->shape[1];
    sizze_29808 = in1->shape[2];
    b_29818 = in2;
    B_mem_30978 = in3->mem;
    B_mem_sizze_30977 = in3->mem.size;
    sizze_29809 = in3->shape[0];
    sizze_29810 = in3->shape[1];
    sizze_29811 = in3->shape[2];
    c_29820 = in4;
    C_mem_30980 = in5->mem;
    C_mem_sizze_30979 = in5->mem.size;
    sizze_29812 = in5->shape[0];
    sizze_29813 = in5->shape[1];
    d_29822 = in6;
    D_mem_30982 = in7->mem;
    D_mem_sizze_30981 = in7->mem.size;
    sizze_29814 = in7->shape[0];
    sizze_29815 = in7->shape[1];
    
    int ret = futrts_t6(ctx, &out_memsizze_31149, &out_mem_31148,
                        &out_arrsizze_31150, &out_arrsizze_31151,
                        &out_arrsizze_31152, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, D_mem_sizze_30981, D_mem_30982,
                        sizze_29806, sizze_29807, sizze_29808, sizze_29809,
                        sizze_29810, sizze_29811, sizze_29812, sizze_29813,
                        sizze_29814, sizze_29815, a_29816, b_29818, c_29820,
                        d_29822);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31148;
        (*out0)->shape[0] = out_arrsizze_31150;
        (*out0)->shape[1] = out_arrsizze_31151;
        (*out0)->shape[2] = out_arrsizze_31152;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t7(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const float in0, const struct futhark_f32_3d *in1, const
                     float in2, const struct futhark_f32_3d *in3, const
                     float in4, const struct futhark_f32_2d *in5, const
                     float in6, const struct futhark_f32_2d *in7, const
                     float in8, const struct futhark_f32_1d *in9, const
                     float in10, const struct futhark_f32_1d *in11, const
                     float in12, const struct futhark_f32_1d *in13, const
                     float in14, const struct futhark_f32_1d *in15)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int64_t D_mem_sizze_30981;
    struct memblock_device D_mem_30982;
    
    D_mem_30982.references = NULL;
    
    int64_t E_mem_sizze_30983;
    struct memblock_device E_mem_30984;
    
    E_mem_30984.references = NULL;
    
    int64_t F_mem_sizze_30985;
    struct memblock_device F_mem_30986;
    
    F_mem_30986.references = NULL;
    
    int64_t G_mem_sizze_30987;
    struct memblock_device G_mem_30988;
    
    G_mem_30988.references = NULL;
    
    int64_t H_mem_sizze_30989;
    struct memblock_device H_mem_30990;
    
    H_mem_30990.references = NULL;
    
    int32_t sizze_29893;
    int32_t sizze_29894;
    int32_t sizze_29895;
    int32_t sizze_29896;
    int32_t sizze_29897;
    int32_t sizze_29898;
    int32_t sizze_29899;
    int32_t sizze_29900;
    int32_t sizze_29901;
    int32_t sizze_29902;
    int32_t sizze_29903;
    int32_t sizze_29904;
    int32_t sizze_29905;
    int32_t sizze_29906;
    float a_29907;
    float b_29909;
    float c_29911;
    float d_29913;
    float e_29915;
    float f_29917;
    float g_29919;
    float h_29921;
    int64_t out_memsizze_31168;
    struct memblock_device out_mem_31167;
    
    out_mem_31167.references = NULL;
    
    int32_t out_arrsizze_31169;
    int32_t out_arrsizze_31170;
    int32_t out_arrsizze_31171;
    
    lock_lock(&ctx->lock);
    a_29907 = in0;
    A_mem_30976 = in1->mem;
    A_mem_sizze_30975 = in1->mem.size;
    sizze_29893 = in1->shape[0];
    sizze_29894 = in1->shape[1];
    sizze_29895 = in1->shape[2];
    b_29909 = in2;
    B_mem_30978 = in3->mem;
    B_mem_sizze_30977 = in3->mem.size;
    sizze_29896 = in3->shape[0];
    sizze_29897 = in3->shape[1];
    sizze_29898 = in3->shape[2];
    c_29911 = in4;
    C_mem_30980 = in5->mem;
    C_mem_sizze_30979 = in5->mem.size;
    sizze_29899 = in5->shape[0];
    sizze_29900 = in5->shape[1];
    d_29913 = in6;
    D_mem_30982 = in7->mem;
    D_mem_sizze_30981 = in7->mem.size;
    sizze_29901 = in7->shape[0];
    sizze_29902 = in7->shape[1];
    e_29915 = in8;
    E_mem_30984 = in9->mem;
    E_mem_sizze_30983 = in9->mem.size;
    sizze_29903 = in9->shape[0];
    f_29917 = in10;
    F_mem_30986 = in11->mem;
    F_mem_sizze_30985 = in11->mem.size;
    sizze_29904 = in11->shape[0];
    g_29919 = in12;
    G_mem_30988 = in13->mem;
    G_mem_sizze_30987 = in13->mem.size;
    sizze_29905 = in13->shape[0];
    h_29921 = in14;
    H_mem_30990 = in15->mem;
    H_mem_sizze_30989 = in15->mem.size;
    sizze_29906 = in15->shape[0];
    
    int ret = futrts_t7(ctx, &out_memsizze_31168, &out_mem_31167,
                        &out_arrsizze_31169, &out_arrsizze_31170,
                        &out_arrsizze_31171, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, D_mem_sizze_30981, D_mem_30982,
                        E_mem_sizze_30983, E_mem_30984, F_mem_sizze_30985,
                        F_mem_30986, G_mem_sizze_30987, G_mem_30988,
                        H_mem_sizze_30989, H_mem_30990, sizze_29893,
                        sizze_29894, sizze_29895, sizze_29896, sizze_29897,
                        sizze_29898, sizze_29899, sizze_29900, sizze_29901,
                        sizze_29902, sizze_29903, sizze_29904, sizze_29905,
                        sizze_29906, a_29907, b_29909, c_29911, d_29913,
                        e_29915, f_29917, g_29919, h_29921);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31167;
        (*out0)->shape[0] = out_arrsizze_31169;
        (*out0)->shape[1] = out_arrsizze_31170;
        (*out0)->shape[2] = out_arrsizze_31171;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t8(struct futhark_context *ctx, struct futhark_f32_3d **out0,
                     const struct futhark_f32_3d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_2d *in2)
{
    int64_t A_mem_sizze_30975;
    struct memblock_device A_mem_30976;
    
    A_mem_30976.references = NULL;
    
    int64_t B_mem_sizze_30977;
    struct memblock_device B_mem_30978;
    
    B_mem_30978.references = NULL;
    
    int64_t C_mem_sizze_30979;
    struct memblock_device C_mem_30980;
    
    C_mem_30980.references = NULL;
    
    int32_t sizze_30032;
    int32_t sizze_30033;
    int32_t sizze_30034;
    int32_t sizze_30035;
    int32_t sizze_30036;
    int32_t sizze_30037;
    int32_t sizze_30038;
    int64_t out_memsizze_31189;
    struct memblock_device out_mem_31188;
    
    out_mem_31188.references = NULL;
    
    int32_t out_arrsizze_31190;
    int32_t out_arrsizze_31191;
    int32_t out_arrsizze_31192;
    
    lock_lock(&ctx->lock);
    A_mem_30976 = in0->mem;
    A_mem_sizze_30975 = in0->mem.size;
    sizze_30032 = in0->shape[0];
    sizze_30033 = in0->shape[1];
    sizze_30034 = in0->shape[2];
    B_mem_30978 = in1->mem;
    B_mem_sizze_30977 = in1->mem.size;
    sizze_30035 = in1->shape[0];
    sizze_30036 = in1->shape[1];
    C_mem_30980 = in2->mem;
    C_mem_sizze_30979 = in2->mem.size;
    sizze_30037 = in2->shape[0];
    sizze_30038 = in2->shape[1];
    
    int ret = futrts_t8(ctx, &out_memsizze_31189, &out_mem_31188,
                        &out_arrsizze_31190, &out_arrsizze_31191,
                        &out_arrsizze_31192, A_mem_sizze_30975, A_mem_30976,
                        B_mem_sizze_30977, B_mem_30978, C_mem_sizze_30979,
                        C_mem_30980, sizze_30032, sizze_30033, sizze_30034,
                        sizze_30035, sizze_30036, sizze_30037, sizze_30038);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_31188;
        (*out0)->shape[0] = out_arrsizze_31190;
        (*out0)->shape[1] = out_arrsizze_31191;
        (*out0)->shape[2] = out_arrsizze_31192;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
