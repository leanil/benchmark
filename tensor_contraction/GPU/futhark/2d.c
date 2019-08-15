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
struct futhark_f32_1d ;
struct futhark_f32_2d ;

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_t1_(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_1d *in1);
int futhark_entry_t1(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_1d *in1);
int futhark_entry_t2(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2);
int futhark_entry_t3(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2);
int futhark_entry_t4(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3);
int futhark_entry_t5(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const float in0, const struct futhark_f32_2d *in1, const
                     float in2, const struct futhark_f32_2d *in3, const
                     float in4, const struct futhark_f32_1d *in5, const
                     float in6, const struct futhark_f32_1d *in7);
int futhark_entry_t6(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_1d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3);
int futhark_entry_t7(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2);
int futhark_entry_t9(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3);
int futhark_entry_t10(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_2d *in1, const
                      struct futhark_f32_2d *in2, const
                      struct futhark_f32_1d *in3);
int futhark_entry_t8(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_2d *in2, const
                     struct futhark_f32_1d *in3);

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
static void futrts_cli_entry_t1_(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18007;
    int64_t read_shape_18008[2];
    float *read_arr_18009 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18009, read_shape_18008, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18010;
    int64_t read_shape_18011[1];
    float *read_arr_18012 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18012, read_shape_18011, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18013;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18007 = futhark_new_f32_2d(ctx, read_arr_18009,
                                                      read_shape_18008[0],
                                                      read_shape_18008[1])) !=
            0);
        assert((read_value_18010 = futhark_new_f32_1d(ctx, read_arr_18012,
                                                      read_shape_18011[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1_(ctx, &result_18013, read_value_18007,
                              read_value_18010);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18007) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18010) == 0);
        assert(futhark_free_f32_1d(ctx, result_18013) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18007 = futhark_new_f32_2d(ctx, read_arr_18009,
                                                      read_shape_18008[0],
                                                      read_shape_18008[1])) !=
            0);
        assert((read_value_18010 = futhark_new_f32_1d(ctx, read_arr_18012,
                                                      read_shape_18011[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1_(ctx, &result_18013, read_value_18007,
                              read_value_18010);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18007) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18010) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18013) == 0);
        }
    }
    free(read_arr_18009);
    free(read_arr_18012);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18013)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18013, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18013), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18013) == 0);
}
static void futrts_cli_entry_t1(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18014;
    int64_t read_shape_18015[2];
    float *read_arr_18016 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18016, read_shape_18015, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18017;
    int64_t read_shape_18018[1];
    float *read_arr_18019 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18019, read_shape_18018, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18020;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18014 = futhark_new_f32_2d(ctx, read_arr_18016,
                                                      read_shape_18015[0],
                                                      read_shape_18015[1])) !=
            0);
        assert((read_value_18017 = futhark_new_f32_1d(ctx, read_arr_18019,
                                                      read_shape_18018[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1(ctx, &result_18020, read_value_18014,
                             read_value_18017);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18014) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18017) == 0);
        assert(futhark_free_f32_1d(ctx, result_18020) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18014 = futhark_new_f32_2d(ctx, read_arr_18016,
                                                      read_shape_18015[0],
                                                      read_shape_18015[1])) !=
            0);
        assert((read_value_18017 = futhark_new_f32_1d(ctx, read_arr_18019,
                                                      read_shape_18018[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t1(ctx, &result_18020, read_value_18014,
                             read_value_18017);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18014) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18017) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18020) == 0);
        }
    }
    free(read_arr_18016);
    free(read_arr_18019);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18020)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18020, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18020), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18020) == 0);
}
static void futrts_cli_entry_t2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18021;
    int64_t read_shape_18022[2];
    float *read_arr_18023 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18023, read_shape_18022, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18024;
    int64_t read_shape_18025[1];
    float *read_arr_18026 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18026, read_shape_18025, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18027;
    int64_t read_shape_18028[1];
    float *read_arr_18029 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18029, read_shape_18028, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18030;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18021 = futhark_new_f32_2d(ctx, read_arr_18023,
                                                      read_shape_18022[0],
                                                      read_shape_18022[1])) !=
            0);
        assert((read_value_18024 = futhark_new_f32_1d(ctx, read_arr_18026,
                                                      read_shape_18025[0])) !=
            0);
        assert((read_value_18027 = futhark_new_f32_1d(ctx, read_arr_18029,
                                                      read_shape_18028[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t2(ctx, &result_18030, read_value_18021,
                             read_value_18024, read_value_18027);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18021) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18024) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18027) == 0);
        assert(futhark_free_f32_1d(ctx, result_18030) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18021 = futhark_new_f32_2d(ctx, read_arr_18023,
                                                      read_shape_18022[0],
                                                      read_shape_18022[1])) !=
            0);
        assert((read_value_18024 = futhark_new_f32_1d(ctx, read_arr_18026,
                                                      read_shape_18025[0])) !=
            0);
        assert((read_value_18027 = futhark_new_f32_1d(ctx, read_arr_18029,
                                                      read_shape_18028[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t2(ctx, &result_18030, read_value_18021,
                             read_value_18024, read_value_18027);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18021) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18024) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18027) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18030) == 0);
        }
    }
    free(read_arr_18023);
    free(read_arr_18026);
    free(read_arr_18029);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18030)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18030, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18030), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18030) == 0);
}
static void futrts_cli_entry_t3(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18031;
    int64_t read_shape_18032[2];
    float *read_arr_18033 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18033, read_shape_18032, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18034;
    int64_t read_shape_18035[2];
    float *read_arr_18036 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18036, read_shape_18035, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18037;
    int64_t read_shape_18038[1];
    float *read_arr_18039 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18039, read_shape_18038, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18040;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18031 = futhark_new_f32_2d(ctx, read_arr_18033,
                                                      read_shape_18032[0],
                                                      read_shape_18032[1])) !=
            0);
        assert((read_value_18034 = futhark_new_f32_2d(ctx, read_arr_18036,
                                                      read_shape_18035[0],
                                                      read_shape_18035[1])) !=
            0);
        assert((read_value_18037 = futhark_new_f32_1d(ctx, read_arr_18039,
                                                      read_shape_18038[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t3(ctx, &result_18040, read_value_18031,
                             read_value_18034, read_value_18037);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18031) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18034) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18037) == 0);
        assert(futhark_free_f32_1d(ctx, result_18040) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18031 = futhark_new_f32_2d(ctx, read_arr_18033,
                                                      read_shape_18032[0],
                                                      read_shape_18032[1])) !=
            0);
        assert((read_value_18034 = futhark_new_f32_2d(ctx, read_arr_18036,
                                                      read_shape_18035[0],
                                                      read_shape_18035[1])) !=
            0);
        assert((read_value_18037 = futhark_new_f32_1d(ctx, read_arr_18039,
                                                      read_shape_18038[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t3(ctx, &result_18040, read_value_18031,
                             read_value_18034, read_value_18037);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18031) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18034) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18037) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18040) == 0);
        }
    }
    free(read_arr_18033);
    free(read_arr_18036);
    free(read_arr_18039);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18040)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18040, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18040), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18040) == 0);
}
static void futrts_cli_entry_t4(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18041;
    int64_t read_shape_18042[2];
    float *read_arr_18043 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18043, read_shape_18042, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18044;
    int64_t read_shape_18045[2];
    float *read_arr_18046 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18046, read_shape_18045, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18047;
    int64_t read_shape_18048[1];
    float *read_arr_18049 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18049, read_shape_18048, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18050;
    int64_t read_shape_18051[1];
    float *read_arr_18052 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18052, read_shape_18051, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18053;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18041 = futhark_new_f32_2d(ctx, read_arr_18043,
                                                      read_shape_18042[0],
                                                      read_shape_18042[1])) !=
            0);
        assert((read_value_18044 = futhark_new_f32_2d(ctx, read_arr_18046,
                                                      read_shape_18045[0],
                                                      read_shape_18045[1])) !=
            0);
        assert((read_value_18047 = futhark_new_f32_1d(ctx, read_arr_18049,
                                                      read_shape_18048[0])) !=
            0);
        assert((read_value_18050 = futhark_new_f32_1d(ctx, read_arr_18052,
                                                      read_shape_18051[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t4(ctx, &result_18053, read_value_18041,
                             read_value_18044, read_value_18047,
                             read_value_18050);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18041) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18044) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18047) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18050) == 0);
        assert(futhark_free_f32_1d(ctx, result_18053) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18041 = futhark_new_f32_2d(ctx, read_arr_18043,
                                                      read_shape_18042[0],
                                                      read_shape_18042[1])) !=
            0);
        assert((read_value_18044 = futhark_new_f32_2d(ctx, read_arr_18046,
                                                      read_shape_18045[0],
                                                      read_shape_18045[1])) !=
            0);
        assert((read_value_18047 = futhark_new_f32_1d(ctx, read_arr_18049,
                                                      read_shape_18048[0])) !=
            0);
        assert((read_value_18050 = futhark_new_f32_1d(ctx, read_arr_18052,
                                                      read_shape_18051[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t4(ctx, &result_18053, read_value_18041,
                             read_value_18044, read_value_18047,
                             read_value_18050);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18041) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18044) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18047) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18050) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18053) == 0);
        }
    }
    free(read_arr_18043);
    free(read_arr_18046);
    free(read_arr_18049);
    free(read_arr_18052);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18053)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18053, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18053), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18053) == 0);
}
static void futrts_cli_entry_t5(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    float read_value_18054;
    
    if (read_scalar(&f32_info, &read_value_18054) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18055;
    int64_t read_shape_18056[2];
    float *read_arr_18057 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18057, read_shape_18056, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    float read_value_18058;
    
    if (read_scalar(&f32_info, &read_value_18058) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18059;
    int64_t read_shape_18060[2];
    float *read_arr_18061 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18061, read_shape_18060, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[][]",
              f32_info.type_name, strerror(errno));
    
    float read_value_18062;
    
    if (read_scalar(&f32_info, &read_value_18062) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18063;
    int64_t read_shape_18064[1];
    float *read_arr_18065 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18065, read_shape_18064, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_18066;
    
    if (read_scalar(&f32_info, &read_value_18066) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 6,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18067;
    int64_t read_shape_18068[1];
    float *read_arr_18069 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18069, read_shape_18068, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18070;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        assert((read_value_18055 = futhark_new_f32_2d(ctx, read_arr_18057,
                                                      read_shape_18056[0],
                                                      read_shape_18056[1])) !=
            0);
        ;
        assert((read_value_18059 = futhark_new_f32_2d(ctx, read_arr_18061,
                                                      read_shape_18060[0],
                                                      read_shape_18060[1])) !=
            0);
        ;
        assert((read_value_18063 = futhark_new_f32_1d(ctx, read_arr_18065,
                                                      read_shape_18064[0])) !=
            0);
        ;
        assert((read_value_18067 = futhark_new_f32_1d(ctx, read_arr_18069,
                                                      read_shape_18068[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t5(ctx, &result_18070, read_value_18054,
                             read_value_18055, read_value_18058,
                             read_value_18059, read_value_18062,
                             read_value_18063, read_value_18066,
                             read_value_18067);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_18055) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_18059) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_18063) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_18067) == 0);
        assert(futhark_free_f32_1d(ctx, result_18070) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        assert((read_value_18055 = futhark_new_f32_2d(ctx, read_arr_18057,
                                                      read_shape_18056[0],
                                                      read_shape_18056[1])) !=
            0);
        ;
        assert((read_value_18059 = futhark_new_f32_2d(ctx, read_arr_18061,
                                                      read_shape_18060[0],
                                                      read_shape_18060[1])) !=
            0);
        ;
        assert((read_value_18063 = futhark_new_f32_1d(ctx, read_arr_18065,
                                                      read_shape_18064[0])) !=
            0);
        ;
        assert((read_value_18067 = futhark_new_f32_1d(ctx, read_arr_18069,
                                                      read_shape_18068[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t5(ctx, &result_18070, read_value_18054,
                             read_value_18055, read_value_18058,
                             read_value_18059, read_value_18062,
                             read_value_18063, read_value_18066,
                             read_value_18067);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_18055) == 0);
        ;
        assert(futhark_free_f32_2d(ctx, read_value_18059) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_18063) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_18067) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18070) == 0);
        }
    }
    ;
    free(read_arr_18057);
    ;
    free(read_arr_18061);
    ;
    free(read_arr_18065);
    ;
    free(read_arr_18069);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18070)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18070, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18070), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18070) == 0);
}
static void futrts_cli_entry_t6(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_18071;
    int64_t read_shape_18072[1];
    float *read_arr_18073 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18073, read_shape_18072, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18074;
    int64_t read_shape_18075[1];
    float *read_arr_18076 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18076, read_shape_18075, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18077;
    int64_t read_shape_18078[1];
    float *read_arr_18079 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18079, read_shape_18078, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18080;
    int64_t read_shape_18081[1];
    float *read_arr_18082 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18082, read_shape_18081, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18083;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18071 = futhark_new_f32_1d(ctx, read_arr_18073,
                                                      read_shape_18072[0])) !=
            0);
        assert((read_value_18074 = futhark_new_f32_1d(ctx, read_arr_18076,
                                                      read_shape_18075[0])) !=
            0);
        assert((read_value_18077 = futhark_new_f32_1d(ctx, read_arr_18079,
                                                      read_shape_18078[0])) !=
            0);
        assert((read_value_18080 = futhark_new_f32_1d(ctx, read_arr_18082,
                                                      read_shape_18081[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t6(ctx, &result_18083, read_value_18071,
                             read_value_18074, read_value_18077,
                             read_value_18080);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_18071) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18074) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18077) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18080) == 0);
        assert(futhark_free_f32_1d(ctx, result_18083) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18071 = futhark_new_f32_1d(ctx, read_arr_18073,
                                                      read_shape_18072[0])) !=
            0);
        assert((read_value_18074 = futhark_new_f32_1d(ctx, read_arr_18076,
                                                      read_shape_18075[0])) !=
            0);
        assert((read_value_18077 = futhark_new_f32_1d(ctx, read_arr_18079,
                                                      read_shape_18078[0])) !=
            0);
        assert((read_value_18080 = futhark_new_f32_1d(ctx, read_arr_18082,
                                                      read_shape_18081[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t6(ctx, &result_18083, read_value_18071,
                             read_value_18074, read_value_18077,
                             read_value_18080);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_18071) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18074) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18077) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18080) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18083) == 0);
        }
    }
    free(read_arr_18073);
    free(read_arr_18076);
    free(read_arr_18079);
    free(read_arr_18082);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18083)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18083, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18083), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18083) == 0);
}
static void futrts_cli_entry_t7(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18084;
    int64_t read_shape_18085[2];
    float *read_arr_18086 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18086, read_shape_18085, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18087;
    int64_t read_shape_18088[2];
    float *read_arr_18089 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18089, read_shape_18088, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18090;
    int64_t read_shape_18091[1];
    float *read_arr_18092 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18092, read_shape_18091, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18093;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18084 = futhark_new_f32_2d(ctx, read_arr_18086,
                                                      read_shape_18085[0],
                                                      read_shape_18085[1])) !=
            0);
        assert((read_value_18087 = futhark_new_f32_2d(ctx, read_arr_18089,
                                                      read_shape_18088[0],
                                                      read_shape_18088[1])) !=
            0);
        assert((read_value_18090 = futhark_new_f32_1d(ctx, read_arr_18092,
                                                      read_shape_18091[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t7(ctx, &result_18093, read_value_18084,
                             read_value_18087, read_value_18090);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18084) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18087) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18090) == 0);
        assert(futhark_free_f32_1d(ctx, result_18093) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18084 = futhark_new_f32_2d(ctx, read_arr_18086,
                                                      read_shape_18085[0],
                                                      read_shape_18085[1])) !=
            0);
        assert((read_value_18087 = futhark_new_f32_2d(ctx, read_arr_18089,
                                                      read_shape_18088[0],
                                                      read_shape_18088[1])) !=
            0);
        assert((read_value_18090 = futhark_new_f32_1d(ctx, read_arr_18092,
                                                      read_shape_18091[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t7(ctx, &result_18093, read_value_18084,
                             read_value_18087, read_value_18090);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18084) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18087) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18090) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18093) == 0);
        }
    }
    free(read_arr_18086);
    free(read_arr_18089);
    free(read_arr_18092);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18093)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18093, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18093), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18093) == 0);
}
static void futrts_cli_entry_t9(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18094;
    int64_t read_shape_18095[2];
    float *read_arr_18096 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18096, read_shape_18095, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18097;
    int64_t read_shape_18098[2];
    float *read_arr_18099 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18099, read_shape_18098, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18100;
    int64_t read_shape_18101[1];
    float *read_arr_18102 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18102, read_shape_18101, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18103;
    int64_t read_shape_18104[1];
    float *read_arr_18105 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18105, read_shape_18104, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18106;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18094 = futhark_new_f32_2d(ctx, read_arr_18096,
                                                      read_shape_18095[0],
                                                      read_shape_18095[1])) !=
            0);
        assert((read_value_18097 = futhark_new_f32_2d(ctx, read_arr_18099,
                                                      read_shape_18098[0],
                                                      read_shape_18098[1])) !=
            0);
        assert((read_value_18100 = futhark_new_f32_1d(ctx, read_arr_18102,
                                                      read_shape_18101[0])) !=
            0);
        assert((read_value_18103 = futhark_new_f32_1d(ctx, read_arr_18105,
                                                      read_shape_18104[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t9(ctx, &result_18106, read_value_18094,
                             read_value_18097, read_value_18100,
                             read_value_18103);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18094) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18097) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18100) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18103) == 0);
        assert(futhark_free_f32_1d(ctx, result_18106) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18094 = futhark_new_f32_2d(ctx, read_arr_18096,
                                                      read_shape_18095[0],
                                                      read_shape_18095[1])) !=
            0);
        assert((read_value_18097 = futhark_new_f32_2d(ctx, read_arr_18099,
                                                      read_shape_18098[0],
                                                      read_shape_18098[1])) !=
            0);
        assert((read_value_18100 = futhark_new_f32_1d(ctx, read_arr_18102,
                                                      read_shape_18101[0])) !=
            0);
        assert((read_value_18103 = futhark_new_f32_1d(ctx, read_arr_18105,
                                                      read_shape_18104[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t9(ctx, &result_18106, read_value_18094,
                             read_value_18097, read_value_18100,
                             read_value_18103);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18094) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18097) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18100) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18103) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18106) == 0);
        }
    }
    free(read_arr_18096);
    free(read_arr_18099);
    free(read_arr_18102);
    free(read_arr_18105);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18106)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18106, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18106), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18106) == 0);
}
static void futrts_cli_entry_t10(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18107;
    int64_t read_shape_18108[2];
    float *read_arr_18109 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18109, read_shape_18108, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18110;
    int64_t read_shape_18111[2];
    float *read_arr_18112 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18112, read_shape_18111, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18113;
    int64_t read_shape_18114[2];
    float *read_arr_18115 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18115, read_shape_18114, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18116;
    int64_t read_shape_18117[1];
    float *read_arr_18118 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18118, read_shape_18117, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18119;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18107 = futhark_new_f32_2d(ctx, read_arr_18109,
                                                      read_shape_18108[0],
                                                      read_shape_18108[1])) !=
            0);
        assert((read_value_18110 = futhark_new_f32_2d(ctx, read_arr_18112,
                                                      read_shape_18111[0],
                                                      read_shape_18111[1])) !=
            0);
        assert((read_value_18113 = futhark_new_f32_2d(ctx, read_arr_18115,
                                                      read_shape_18114[0],
                                                      read_shape_18114[1])) !=
            0);
        assert((read_value_18116 = futhark_new_f32_1d(ctx, read_arr_18118,
                                                      read_shape_18117[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t10(ctx, &result_18119, read_value_18107,
                              read_value_18110, read_value_18113,
                              read_value_18116);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18107) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18110) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18113) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18116) == 0);
        assert(futhark_free_f32_1d(ctx, result_18119) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18107 = futhark_new_f32_2d(ctx, read_arr_18109,
                                                      read_shape_18108[0],
                                                      read_shape_18108[1])) !=
            0);
        assert((read_value_18110 = futhark_new_f32_2d(ctx, read_arr_18112,
                                                      read_shape_18111[0],
                                                      read_shape_18111[1])) !=
            0);
        assert((read_value_18113 = futhark_new_f32_2d(ctx, read_arr_18115,
                                                      read_shape_18114[0],
                                                      read_shape_18114[1])) !=
            0);
        assert((read_value_18116 = futhark_new_f32_1d(ctx, read_arr_18118,
                                                      read_shape_18117[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t10(ctx, &result_18119, read_value_18107,
                              read_value_18110, read_value_18113,
                              read_value_18116);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18107) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18110) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18113) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18116) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18119) == 0);
        }
    }
    free(read_arr_18109);
    free(read_arr_18112);
    free(read_arr_18115);
    free(read_arr_18118);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18119)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18119, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18119), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18119) == 0);
}
static void futrts_cli_entry_t8(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_2d *read_value_18120;
    int64_t read_shape_18121[2];
    float *read_arr_18122 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18122, read_shape_18121, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18123;
    int64_t read_shape_18124[2];
    float *read_arr_18125 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18125, read_shape_18124, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_2d *read_value_18126;
    int64_t read_shape_18127[2];
    float *read_arr_18128 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18128, read_shape_18127, 2) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[][]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_18129;
    int64_t read_shape_18130[1];
    float *read_arr_18131 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_18131, read_shape_18130, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *result_18132;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_18120 = futhark_new_f32_2d(ctx, read_arr_18122,
                                                      read_shape_18121[0],
                                                      read_shape_18121[1])) !=
            0);
        assert((read_value_18123 = futhark_new_f32_2d(ctx, read_arr_18125,
                                                      read_shape_18124[0],
                                                      read_shape_18124[1])) !=
            0);
        assert((read_value_18126 = futhark_new_f32_2d(ctx, read_arr_18128,
                                                      read_shape_18127[0],
                                                      read_shape_18127[1])) !=
            0);
        assert((read_value_18129 = futhark_new_f32_1d(ctx, read_arr_18131,
                                                      read_shape_18130[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t8(ctx, &result_18132, read_value_18120,
                             read_value_18123, read_value_18126,
                             read_value_18129);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18120) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18123) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18126) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18129) == 0);
        assert(futhark_free_f32_1d(ctx, result_18132) == 0);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_18120 = futhark_new_f32_2d(ctx, read_arr_18122,
                                                      read_shape_18121[0],
                                                      read_shape_18121[1])) !=
            0);
        assert((read_value_18123 = futhark_new_f32_2d(ctx, read_arr_18125,
                                                      read_shape_18124[0],
                                                      read_shape_18124[1])) !=
            0);
        assert((read_value_18126 = futhark_new_f32_2d(ctx, read_arr_18128,
                                                      read_shape_18127[0],
                                                      read_shape_18127[1])) !=
            0);
        assert((read_value_18129 = futhark_new_f32_1d(ctx, read_arr_18131,
                                                      read_shape_18130[0])) !=
            0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_t8(ctx, &result_18132, read_value_18120,
                             read_value_18123, read_value_18126,
                             read_value_18129);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_2d(ctx, read_value_18120) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18123) == 0);
        assert(futhark_free_f32_2d(ctx, read_value_18126) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_18129) == 0);
        if (run < num_runs - 1) {
            assert(futhark_free_f32_1d(ctx, result_18132) == 0);
        }
    }
    free(read_arr_18122);
    free(read_arr_18125);
    free(read_arr_18128);
    free(read_arr_18131);
    {
        float *arr = calloc(sizeof(float), futhark_shape_f32_1d(ctx,
                                                                result_18132)[0]);
        
        assert(arr != NULL);
        assert(futhark_values_f32_1d(ctx, result_18132, arr) == 0);
        write_array(stdout, binary_output, &f32_info, arr,
                    futhark_shape_f32_1d(ctx, result_18132), 1);
        free(arr);
    }
    printf("\n");
    assert(futhark_free_f32_1d(ctx, result_18132) == 0);
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="t1_", .fun =
                                                futrts_cli_entry_t1_}, {.name =
                                                                        "t1",
                                                                        .fun =
                                                                        futrts_cli_entry_t1},
                                               {.name ="t2", .fun =
                                                futrts_cli_entry_t2}, {.name =
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
                                               {.name ="t9", .fun =
                                                futrts_cli_entry_t9}, {.name =
                                                                       "t10",
                                                                       .fun =
                                                                       futrts_cli_entry_t10},
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
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define group_sizze_16758 (group_size_16757)\n#define max_num_groups_16760 (max_num_groups_16759)\n#define group_sizze_17104 (group_size_17103)\n__kernel void chunked_reduce_ker",
            "nel_16774(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_16260,\n                                          int32_t sizze_16261,\n                                          int32_t num_threads_16766,\n                                          int32_t per_thread_elements_16769,\n                                          int32_t per_chunk_17513, __global\n                                          unsigned char *mem_17590,\n                                          int32_t binop_x_17600, __global\n                                          unsigned char *mem_17603, __global\n                                          unsigned char *mem_17613, __global\n                                          unsigned char *mem_17621, __global\n                                          unsigned char *mem_17624, __global\n                                          unsigned char *mem_17633, __global\n                                          unsigned char *mem_17636, __global\n                                          unsigned char *mem_17640, __global\n                                          unsigned char *double_buffer_mem_17669)\n{\n    __local volatile char *restrict mem_17630 = mem_aligned_0;\n    int32_t wave_sizze_17707;\n    int32_t group_sizze_17708;\n    bool thread_active_17709;\n    int32_t global_tid_16774;\n    int32_t local_tid_16775;\n    int32_t group_id_16776;\n    \n    global_tid_16774 = get_global_id(0);\n    local_tid_16775 = get_local_id(0);\n    group_sizze_17708 = get_local_size(0);\n    wave_sizze_17707 = LOCKSTEP_WIDTH;\n    group_id_16776 = get_group_id(0);\n    thread_active_17709 = 1;\n    \n    int32_t chunk_sizze_16782;\n    int32_t starting_point_17710 = global_tid_16774 * per_thread_elements_16769;\n    int32_t remaining_elements_17711 = sizze_16260 - starting_point_17710;\n    \n    if (sle32(remaining_elements_17711, 0) || sle32(sizze_16260,\n                                                    start",
            "ing_point_17710)) {\n        chunk_sizze_16782 = 0;\n    } else {\n        if (slt32(sizze_16260, (global_tid_16774 + 1) *\n                  per_thread_elements_16769)) {\n            chunk_sizze_16782 = sizze_16260 - global_tid_16774 *\n                per_thread_elements_16769;\n        } else {\n            chunk_sizze_16782 = per_thread_elements_16769;\n        }\n    }\n    \n    int32_t slice_offset_16783;\n    \n    if (thread_active_17709) {\n        slice_offset_16783 = per_thread_elements_16769 * global_tid_16774;\n        for (int32_t i_17712 = 0; i_17712 < sizze_16261; i_17712++) {\n            *(__global float *) &double_buffer_mem_17669[(group_id_16776 *\n                                                          (sizze_16261 *\n                                                           group_sizze_16758) +\n                                                          i_17712 *\n                                                          group_sizze_16758 +\n                                                          local_tid_16775) *\n                                                         4] = *(__global\n                                                                float *) &mem_17590[i_17712 *\n                                                                                    4];\n        }\n        for (int32_t i_16790 = 0; i_16790 < chunk_sizze_16782; i_16790++) {\n            int32_t j_p_i_t_s_17544 = slice_offset_16783 + i_16790;\n            int32_t new_index_17545 = squot32(j_p_i_t_s_17544, per_chunk_17513);\n            int32_t binop_y_17547 = per_chunk_17513 * new_index_17545;\n            int32_t new_index_17548 = j_p_i_t_s_17544 - binop_y_17547;\n            float x_16795 = *(__global float *) &mem_17613[(new_index_17548 *\n                                                            num_threads_16766 +\n                                                            new_index_17545) *\n                                                           4];\n            int32_t binop_x_175",
            "49 = sizze_16261 * j_p_i_t_s_17544;\n            \n            for (int32_t i_16800 = 0; i_16800 < sizze_16261; i_16800++) {\n                int32_t binop_x_17550 = i_16800 + binop_x_17549;\n                int32_t new_index_17552 = squot32(binop_x_17550, binop_x_17600);\n                int32_t binop_y_17560 = new_index_17552 * binop_x_17600;\n                int32_t binop_x_17561 = binop_x_17550 - binop_y_17560;\n                int32_t new_index_17562 = squot32(binop_x_17561, sizze_16261);\n                int32_t binop_y_17582 = sizze_16261 * new_index_17562;\n                int32_t new_index_17583 = binop_x_17561 - binop_y_17582;\n                float x_16801 = *(__global\n                                  float *) &mem_17603[(new_index_17562 *\n                                                       (sizze_16261 *\n                                                        num_threads_16766) +\n                                                       new_index_17583 *\n                                                       num_threads_16766 +\n                                                       new_index_17552) * 4];\n                float res_16802 = x_16795 * x_16801;\n                \n                *(__global float *) &mem_17621[(group_id_16776 * (sizze_16261 *\n                                                                  group_sizze_16758) +\n                                                i_16800 * group_sizze_16758 +\n                                                local_tid_16775) * 4] =\n                    res_16802;\n            }\n            for (int32_t i_16808 = 0; i_16808 < sizze_16261; i_16808++) {\n                float x_16809 = *(__global\n                                  float *) &double_buffer_mem_17669[(group_id_16776 *\n                                                                     (sizze_16261 *\n                                                                      group_sizze_16758) +\n                                                                 ",
            "    i_16808 *\n                                                                     group_sizze_16758 +\n                                                                     local_tid_16775) *\n                                                                    4];\n                float x_16810 = *(__global float *) &mem_17621[(group_id_16776 *\n                                                                (sizze_16261 *\n                                                                 group_sizze_16758) +\n                                                                i_16808 *\n                                                                group_sizze_16758 +\n                                                                local_tid_16775) *\n                                                               4];\n                float res_16811 = x_16809 + x_16810;\n                \n                *(__global float *) &mem_17624[(group_id_16776 * (sizze_16261 *\n                                                                  group_sizze_16758) +\n                                                i_16808 * group_sizze_16758 +\n                                                local_tid_16775) * 4] =\n                    res_16811;\n            }\n            for (int32_t i_17716 = 0; i_17716 < sizze_16261; i_17716++) {\n                *(__global float *) &double_buffer_mem_17669[(group_id_16776 *\n                                                              (sizze_16261 *\n                                                               group_sizze_16758) +\n                                                              i_17716 *\n                                                              group_sizze_16758 +\n                                                              local_tid_16775) *\n                                                             4] = *(__global\n                                                                    float *) &mem_17624[(group_id_16776 *\n                   ",
            "                                                                      (sizze_16261 *\n                                                                                          group_sizze_16758) +\n                                                                                         i_17716 *\n                                                                                         group_sizze_16758 +\n                                                                                         local_tid_16775) *\n                                                                                        4];\n            }\n        }\n    }\n    for (int32_t comb_iter_17717 = 0; comb_iter_17717 <\n         squot32(group_sizze_16758 + group_sizze_16758 - 1, group_sizze_16758);\n         comb_iter_17717++) {\n        int32_t combine_id_16780;\n        int32_t flat_comb_id_17718 = comb_iter_17717 * group_sizze_16758 +\n                local_tid_16775;\n        \n        combine_id_16780 = flat_comb_id_17718;\n        if (slt32(combine_id_16780, group_sizze_16758) && 1) {\n            for (int32_t i_17719 = 0; i_17719 < sizze_16261; i_17719++) {\n                *(__local float *) &mem_17630[(combine_id_16780 * sizze_16261 +\n                                               i_17719) * 4] = *(__global\n                                                                 float *) &double_buffer_mem_17669[(group_id_16776 *\n                                                                                                    (sizze_16261 *\n                                                                                                     group_sizze_16758) +\n                                                                                                    i_17719 *\n                                                                                                    group_sizze_16758 +\n                                                                                                    local_tid_16775) *\n        ",
            "                                                                                           4];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_17720;\n    int32_t my_index_16816;\n    int32_t other_offset_16817;\n    \n    my_index_16816 = local_tid_16775;\n    other_offset_16817 = 0;\n    if (slt32(local_tid_16775, group_sizze_16758)) { }\n    other_offset_16817 = 1;\n    while (slt32(other_offset_16817, wave_sizze_17707)) {\n        if (slt32(local_tid_16775 + other_offset_16817, group_sizze_16758) &&\n            ((local_tid_16775 - squot32(local_tid_16775, wave_sizze_17707) *\n              wave_sizze_17707) & (2 * other_offset_16817 - 1)) == 0) {\n            // read array element\n            { }\n            if (thread_active_17709) {\n                for (int32_t i_16823 = 0; i_16823 < sizze_16261; i_16823++) {\n                    float x_16824 = *(volatile __local\n                                      float *) &mem_17630[(my_index_16816 *\n                                                           sizze_16261 +\n                                                           i_16823) * 4];\n                    float x_16825 = *(volatile __local\n                                      float *) &mem_17630[((my_index_16816 +\n                                                            other_offset_16817) *\n                                                           sizze_16261 +\n                                                           i_16823) * 4];\n                    float res_16826 = x_16824 + x_16825;\n                    \n                    *(volatile __global float *) &mem_17633[(group_id_16776 *\n                                                             (sizze_16261 *\n                                                              group_sizze_16758) +\n                                                             i_16823 *\n                                                             group_sizze_16758 +\n                                  ",
            "                           local_tid_16775) *\n                                                            4] = res_16826;\n                }\n            }\n            for (int32_t i_17722 = 0; i_17722 < sizze_16261; i_17722++) {\n                *(volatile __local float *) &mem_17630[(my_index_16816 *\n                                                        sizze_16261 + i_17722) *\n                                                       4] = *(volatile __global\n                                                              float *) &mem_17633[(group_id_16776 *\n                                                                                   (sizze_16261 *\n                                                                                    group_sizze_16758) +\n                                                                                   i_17722 *\n                                                                                   group_sizze_16758 +\n                                                                                   local_tid_16775) *\n                                                                                  4];\n            }\n        }\n        other_offset_16817 *= 2;\n    }\n    skip_waves_17720 = 1;\n    while (slt32(skip_waves_17720, squot32(group_sizze_16758 +\n                                           wave_sizze_17707 - 1,\n                                           wave_sizze_17707))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_16817 = skip_waves_17720 * wave_sizze_17707;\n        if (slt32(local_tid_16775 + other_offset_16817, group_sizze_16758) &&\n            ((local_tid_16775 - squot32(local_tid_16775, wave_sizze_17707) *\n              wave_sizze_17707) == 0 && (squot32(local_tid_16775,\n                                                 wave_sizze_17707) & (2 *\n                                                                      skip_waves_17720 -\n                                                                      1)) ==\n    ",
            "         0)) {\n            // read array element\n            { }\n            if (thread_active_17709) {\n                for (int32_t i_16823 = 0; i_16823 < sizze_16261; i_16823++) {\n                    float x_16824 = *(__local\n                                      float *) &mem_17630[(my_index_16816 *\n                                                           sizze_16261 +\n                                                           i_16823) * 4];\n                    float x_16825 = *(__local\n                                      float *) &mem_17630[((my_index_16816 +\n                                                            other_offset_16817) *\n                                                           sizze_16261 +\n                                                           i_16823) * 4];\n                    float res_16826 = x_16824 + x_16825;\n                    \n                    *(__global float *) &mem_17633[(group_id_16776 *\n                                                    (sizze_16261 *\n                                                     group_sizze_16758) +\n                                                    i_16823 *\n                                                    group_sizze_16758 +\n                                                    local_tid_16775) * 4] =\n                        res_16826;\n                }\n            }\n            for (int32_t i_17724 = 0; i_17724 < sizze_16261; i_17724++) {\n                *(__local float *) &mem_17630[(my_index_16816 * sizze_16261 +\n                                               i_17724) * 4] = *(__global\n                                                                 float *) &mem_17633[(group_id_16776 *\n                                                                                      (sizze_16261 *\n                                                                                       group_sizze_16758) +\n                                                                                      i_17724 ",
            "*\n                                                                                      group_sizze_16758 +\n                                                                                      local_tid_16775) *\n                                                                                     4];\n            }\n        }\n        skip_waves_17720 *= 2;\n    }\n    for (int32_t i_17725 = 0; i_17725 < sizze_16261; i_17725++) {\n        *(__global float *) &mem_17636[(group_id_16776 * (sizze_16261 *\n                                                          group_sizze_16758) +\n                                        i_17725 * group_sizze_16758 +\n                                        local_tid_16775) * 4] = *(__local\n                                                                  float *) &mem_17630[(my_index_16816 *\n                                                                                       sizze_16261 +\n                                                                                       i_17725) *\n                                                                                      4];\n    }\n    if (local_tid_16775 == 0) {\n        for (int32_t i_17727 = 0; i_17727 < sizze_16261; i_17727++) {\n            *(__global float *) &mem_17640[(group_id_16776 * sizze_16261 +\n                                            i_17727) * 4] = *(__global\n                                                              float *) &mem_17636[(group_id_16776 *\n                                                                                   (sizze_16261 *\n                                                                                    group_sizze_16758) +\n                                                                                   i_17727 *\n                                                                                   group_sizze_16758 +\n                                                                                   local_tid_16775) *\n                            ",
            "                                                      4];\n        }\n    }\n}\n__kernel void fut_kernel_map_transpose_f32(__global float *odata,\n                                           uint odata_offset, __global\n                                           float *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_f32(__global float *odata,\n                                                     uint odata_offset, __global\n                                                     float *idata,\n                                           ",
            "          uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_f32(__global float *odata,\n                                                    uint odata_offset, __global\n                                 ",
            "                   float *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    uint index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_small_f32(__global float *odata,\n                                                 uint odata_offset, __global\n                     ",
            "                            float *idata,\n                                                 uint idata_offset,\n                                                 uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n                                                                   width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void kernel_replicate_16271(int32_t sizze_16261, __global\n                                     unsigned char *mem_17590)\n{\n    const uint replicate_gtid_16271 = get_global_id(0);\n    \n    if (replicate_gtid_16271 >= sizze_16261)\n        return;\n    *(__global float *) &mem_17590[replicate_gtid_16271 * 4] = 0.0F;\n}\n__kernel void map_kernel_16856(int32_t sizze_16285, int32_t sizze_16286,\n                               __global unsigned char *B_mem_17587, __global\n                               unsigned char *mem_17591, __global\n                               unsigned char *mem_17594)\n{\n    int32_t wave_sizze_17747;\n    int32_t group_sizze_17748;\n    bool thread_active_17749;\n    int32_t gtid_16849;\n    int32_t global_tid_16856;\n    int32_t local_tid_16857;\n    int32_t group_id_16858;\n    \n    global_tid_16856 = get_global_id(0);\n    local_tid_16857 = get_local_id(0);\n    group_sizze_17748 = get_local_size(0);\n    wave_sizze_17747",
            " = LOCKSTEP_WIDTH;\n    group_id_16858 = get_group_id(0);\n    gtid_16849 = global_tid_16856;\n    thread_active_17749 = slt32(gtid_16849, sizze_16285);\n    \n    float res_16860;\n    \n    if (thread_active_17749) {\n        float x_16863 = 0.0F;\n        \n        for (int32_t chunk_offset_16862 = 0; chunk_offset_16862 < sizze_16286;\n             chunk_offset_16862++) {\n            float x_16872 = *(__global float *) &mem_17591[(chunk_offset_16862 *\n                                                            sizze_16285 +\n                                                            gtid_16849) * 4];\n            float x_16873 = *(__global\n                              float *) &B_mem_17587[chunk_offset_16862 * 4];\n            float res_16875 = x_16872 * x_16873;\n            float res_16877 = x_16863 + res_16875;\n            float x_tmp_17750 = res_16877;\n            \n            x_16863 = x_tmp_17750;\n        }\n        res_16860 = x_16863;\n    }\n    if (thread_active_17749) {\n        *(__global float *) &mem_17594[gtid_16849 * 4] = res_16860;\n    }\n}\n__kernel void map_kernel_16896(int32_t sizze_16306, int32_t sizze_16307,\n                               __global unsigned char *B_mem_17587, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *mem_17593, __global\n                               unsigned char *mem_17596)\n{\n    int32_t wave_sizze_17754;\n    int32_t group_sizze_17755;\n    bool thread_active_17756;\n    int32_t gtid_16889;\n    int32_t global_tid_16896;\n    int32_t local_tid_16897;\n    int32_t group_id_16898;\n    \n    global_tid_16896 = get_global_id(0);\n    local_tid_16897 = get_local_id(0);\n    group_sizze_17755 = get_local_size(0);\n    wave_sizze_17754 = LOCKSTEP_WIDTH;\n    group_id_16898 = get_group_id(0);\n    gtid_16889 = global_tid_16896;\n    thread_active_17756 = slt32(gtid_16889, sizze_16306);\n    \n    float x_16900;\n    float res_16901;\n    \n    if (thread_active_17756) {\n        x_16900 =",
            " *(__global float *) &C_mem_17589[gtid_16889 * 4];\n        \n        float x_16904 = 0.0F;\n        \n        for (int32_t chunk_offset_16903 = 0; chunk_offset_16903 < sizze_16307;\n             chunk_offset_16903++) {\n            float x_16913 = *(__global float *) &mem_17593[(chunk_offset_16903 *\n                                                            sizze_16306 +\n                                                            gtid_16889) * 4];\n            float x_16914 = *(__global\n                              float *) &B_mem_17587[chunk_offset_16903 * 4];\n            float x_16916 = x_16913 * x_16914;\n            float res_16917 = x_16900 * x_16916;\n            float res_16919 = x_16904 + res_16917;\n            float x_tmp_17757 = res_16919;\n            \n            x_16904 = x_tmp_17757;\n        }\n        res_16901 = x_16904;\n    }\n    if (thread_active_17756) {\n        *(__global float *) &mem_17596[gtid_16889 * 4] = res_16901;\n    }\n}\n__kernel void map_kernel_16940(int32_t sizze_16338, int32_t sizze_16339,\n                               int32_t sizze_16340, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *mem_17593, __global\n                               unsigned char *mem_17597, __global\n                               unsigned char *mem_17600)\n{\n    int32_t wave_sizze_17761;\n    int32_t group_sizze_17762;\n    bool thread_active_17763;\n    int32_t gtid_16933;\n    int32_t global_tid_16940;\n    int32_t local_tid_16941;\n    int32_t group_id_16942;\n    \n    global_tid_16940 = get_global_id(0);\n    local_tid_16941 = get_local_id(0);\n    group_sizze_17762 = get_local_size(0);\n    wave_sizze_17761 = LOCKSTEP_WIDTH;\n    group_id_16942 = get_group_id(0);\n    gtid_16933 = global_tid_16940;\n    thread_active_17763 = slt32(gtid_16933, sizze_16338);\n    \n    float res_16945;\n    \n    if (thread_active_17763) {\n        float x_16948 = 0.0F;\n        \n        for (int32_t chunk_offset_16947 = 0; chunk_o",
            "ffset_16947 < sizze_16339;\n             chunk_offset_16947++) {\n            float x_16959 = *(__global float *) &mem_17593[(chunk_offset_16947 *\n                                                            sizze_16338 +\n                                                            gtid_16933) * 4];\n            float x_16960 = *(__global float *) &mem_17597[(chunk_offset_16947 *\n                                                            sizze_16340 +\n                                                            gtid_16933) * 4];\n            float x_16961 = *(__global\n                              float *) &C_mem_17589[chunk_offset_16947 * 4];\n            float res_16963 = x_16959 + x_16960;\n            float res_16964 = x_16961 * res_16963;\n            float res_16966 = x_16948 + res_16964;\n            float x_tmp_17764 = res_16966;\n            \n            x_16948 = x_tmp_17764;\n        }\n        res_16945 = x_16948;\n    }\n    if (thread_active_17763) {\n        *(__global float *) &mem_17600[gtid_16933 * 4] = res_16945;\n    }\n}\n__kernel void map_kernel_16974(int32_t sizze_16378, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17594)\n{\n    int32_t wave_sizze_17768;\n    int32_t group_sizze_17769;\n    bool thread_active_17770;\n    int32_t gtid_16967;\n    int32_t global_tid_16974;\n    int32_t local_tid_16975;\n    int32_t group_id_16976;\n    \n    global_tid_16974 = get_global_id(0);\n    local_tid_16975 = get_local_id(0);\n    group_sizze_17769 = get_local_size(0);\n    wave_sizze_17768 = LOCKSTEP_WIDTH;\n    group_id_16976 = get_group_id(0);\n    gtid_16967 = global_tid_16974;\n    thread_active_17770 = slt32(gtid_16967, sizze_16378);\n    \n    float x_16977;\n    float x_16978;\n    float res_16979;\n    \n    if (thread_active_17770) {\n        x_16977 = *(__global float *) &C_mem_17589[gtid_16967 * 4];\n        x_16978 = *(__global float *",
            ") &D_mem_17591[gtid_16967 * 4];\n        res_16979 = x_16977 + x_16978;\n    }\n    if (thread_active_17770) {\n        *(__global float *) &mem_17594[gtid_16967 * 4] = res_16979;\n    }\n}\n__kernel void map_kernel_17000(int32_t sizze_16377, int32_t sizze_16378,\n                               int32_t sizze_16379, __global\n                               unsigned char *mem_17594, __global\n                               unsigned char *mem_17598, __global\n                               unsigned char *mem_17602, __global\n                               unsigned char *mem_17605)\n{\n    int32_t wave_sizze_17771;\n    int32_t group_sizze_17772;\n    bool thread_active_17773;\n    int32_t gtid_16993;\n    int32_t global_tid_17000;\n    int32_t local_tid_17001;\n    int32_t group_id_17002;\n    \n    global_tid_17000 = get_global_id(0);\n    local_tid_17001 = get_local_id(0);\n    group_sizze_17772 = get_local_size(0);\n    wave_sizze_17771 = LOCKSTEP_WIDTH;\n    group_id_17002 = get_group_id(0);\n    gtid_16993 = global_tid_17000;\n    thread_active_17773 = slt32(gtid_16993, sizze_16377);\n    \n    float res_17005;\n    \n    if (thread_active_17773) {\n        float x_17008 = 0.0F;\n        \n        for (int32_t chunk_offset_17007 = 0; chunk_offset_17007 < sizze_16378;\n             chunk_offset_17007++) {\n            float x_17019 = *(__global float *) &mem_17598[(chunk_offset_17007 *\n                                                            sizze_16377 +\n                                                            gtid_16993) * 4];\n            float x_17020 = *(__global float *) &mem_17602[(chunk_offset_17007 *\n                                                            sizze_16379 +\n                                                            gtid_16993) * 4];\n            float x_17021 = *(__global float *) &mem_17594[chunk_offset_17007 *\n                                                           4];\n            float res_17023 = x_17019 + x_17020;\n            float res_17024 = x_17021 * res_17023;",
            "\n            float res_17026 = x_17008 + res_17024;\n            float x_tmp_17774 = res_17026;\n            \n            x_17008 = x_tmp_17774;\n        }\n        res_17005 = x_17008;\n    }\n    if (thread_active_17773) {\n        *(__global float *) &mem_17605[gtid_16993 * 4] = res_17005;\n    }\n}\n__kernel void map_kernel_17034(int32_t sizze_16429, float c_16438,\n                               float d_16440, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17594)\n{\n    int32_t wave_sizze_17778;\n    int32_t group_sizze_17779;\n    bool thread_active_17780;\n    int32_t gtid_17027;\n    int32_t global_tid_17034;\n    int32_t local_tid_17035;\n    int32_t group_id_17036;\n    \n    global_tid_17034 = get_global_id(0);\n    local_tid_17035 = get_local_id(0);\n    group_sizze_17779 = get_local_size(0);\n    wave_sizze_17778 = LOCKSTEP_WIDTH;\n    group_id_17036 = get_group_id(0);\n    gtid_17027 = global_tid_17034;\n    thread_active_17780 = slt32(gtid_17027, sizze_16429);\n    \n    float x_17037;\n    float x_17038;\n    float res_17039;\n    float res_17040;\n    float res_17041;\n    \n    if (thread_active_17780) {\n        x_17037 = *(__global float *) &C_mem_17589[gtid_17027 * 4];\n        x_17038 = *(__global float *) &D_mem_17591[gtid_17027 * 4];\n        res_17039 = c_16438 * x_17037;\n        res_17040 = d_16440 * x_17038;\n        res_17041 = res_17039 + res_17040;\n    }\n    if (thread_active_17780) {\n        *(__global float *) &mem_17594[gtid_17027 * 4] = res_17041;\n    }\n}\n__kernel void map_kernel_17062(int32_t sizze_16428, int32_t sizze_16429,\n                               int32_t sizze_16430, float a_16434,\n                               float b_16436, __global unsigned char *mem_17594,\n                               __global unsigned char *mem_17598, __global\n                               unsigned char *mem_17602, __global\n              ",
            "                 unsigned char *mem_17605)\n{\n    int32_t wave_sizze_17781;\n    int32_t group_sizze_17782;\n    bool thread_active_17783;\n    int32_t gtid_17055;\n    int32_t global_tid_17062;\n    int32_t local_tid_17063;\n    int32_t group_id_17064;\n    \n    global_tid_17062 = get_global_id(0);\n    local_tid_17063 = get_local_id(0);\n    group_sizze_17782 = get_local_size(0);\n    wave_sizze_17781 = LOCKSTEP_WIDTH;\n    group_id_17064 = get_group_id(0);\n    gtid_17055 = global_tid_17062;\n    thread_active_17783 = slt32(gtid_17055, sizze_16428);\n    \n    float res_17067;\n    \n    if (thread_active_17783) {\n        float x_17070 = 0.0F;\n        \n        for (int32_t chunk_offset_17069 = 0; chunk_offset_17069 < sizze_16429;\n             chunk_offset_17069++) {\n            float x_17081 = *(__global float *) &mem_17598[(chunk_offset_17069 *\n                                                            sizze_16428 +\n                                                            gtid_17055) * 4];\n            float x_17082 = *(__global float *) &mem_17602[(chunk_offset_17069 *\n                                                            sizze_16430 +\n                                                            gtid_17055) * 4];\n            float x_17083 = *(__global float *) &mem_17594[chunk_offset_17069 *\n                                                           4];\n            float res_17085 = a_16434 * x_17081;\n            float res_17086 = b_16436 * x_17082;\n            float res_17087 = res_17085 + res_17086;\n            float res_17088 = x_17083 * res_17087;\n            float res_17090 = x_17070 + res_17088;\n            float x_tmp_17784 = res_17090;\n            \n            x_17070 = x_tmp_17784;\n        }\n        res_17067 = x_17070;\n    }\n    if (thread_active_17783) {\n        *(__global float *) &mem_17605[gtid_17055 * 4] = res_17067;\n    }\n}\n__kernel void map_kernel_17109(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_",
            "aligned_1,\n                               int32_t sizze_16487, int32_t sizze_16488,\n                               __global unsigned char *A_mem_17585, __global\n                               unsigned char *B_mem_17587, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17600)\n{\n    __local volatile char *restrict mem_17594 = mem_aligned_0;\n    __local volatile char *restrict mem_17597 = mem_aligned_1;\n    int32_t wave_sizze_17788;\n    int32_t group_sizze_17789;\n    bool thread_active_17790;\n    int32_t gtid_17102;\n    int32_t global_tid_17109;\n    int32_t local_tid_17110;\n    int32_t group_id_17111;\n    \n    global_tid_17109 = get_global_id(0);\n    local_tid_17110 = get_local_id(0);\n    group_sizze_17789 = get_local_size(0);\n    wave_sizze_17788 = LOCKSTEP_WIDTH;\n    group_id_17111 = get_group_id(0);\n    gtid_17102 = global_tid_17109;\n    thread_active_17790 = slt32(gtid_17102, sizze_16487);\n    \n    float x_17112;\n    float x_17113;\n    \n    if (thread_active_17790) {\n        x_17112 = *(__global float *) &A_mem_17585[gtid_17102 * 4];\n        x_17113 = *(__global float *) &C_mem_17589[gtid_17102 * 4];\n    }\n    \n    float res_17114;\n    float x_17117 = 0.0F;\n    int32_t chunk_sizze_17115;\n    int32_t chunk_offset_17116 = 0;\n    \n    while (slt32(chunk_offset_17116, sizze_16488)) {\n        if (slt32(sizze_16488 - chunk_offset_17116, group_sizze_17104)) {\n            chunk_sizze_17115 = sizze_16488 - chunk_offset_17116;\n        } else {\n            chunk_sizze_17115 = group_sizze_17104;\n        }\n        \n        float res_17120;\n        float sync_17532;\n        \n        for (int32_t comb_iter_17791 = 0; comb_iter_17791 <\n             squot32(group_sizze_17104 + group_sizze_17104 - 1,\n                     group_sizze_17104); comb_iter_17791++) {\n            int32_t cid_17528;\n            int32_t flat_comb_id_17792 = comb_it",
            "er_17791 * group_sizze_17104 +\n                    local_tid_17110;\n            \n            cid_17528 = flat_comb_id_17792;\n            if (slt32(cid_17528, chunk_sizze_17115) && 1) {\n                float x_chunk_outer_elem_17527 = *(__global\n                                                   float *) &B_mem_17587[(chunk_offset_17116 +\n                                                                          local_tid_17110) *\n                                                                         4];\n                \n                *(__local float *) &mem_17594[cid_17528 * 4] =\n                    x_chunk_outer_elem_17527;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t comb_iter_17793 = 0; comb_iter_17793 <\n             squot32(group_sizze_17104 + group_sizze_17104 - 1,\n                     group_sizze_17104); comb_iter_17793++) {\n            int32_t cid_17531;\n            int32_t flat_comb_id_17794 = comb_iter_17793 * group_sizze_17104 +\n                    local_tid_17110;\n            \n            cid_17531 = flat_comb_id_17794;\n            if (slt32(cid_17531, chunk_sizze_17115) && 1) {\n                float x_chunk_outer_elem_17530 = *(__global\n                                                   float *) &D_mem_17591[(chunk_offset_17116 +\n                                                                          local_tid_17110) *\n                                                                         4];\n                \n                *(__local float *) &mem_17597[cid_17531 * 4] =\n                    x_chunk_outer_elem_17530;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        float acc_17123 = x_17117;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_17121 = 1;\n        \n        if (thread_active_17790) {\n            if (chunk_sizze_17115 == group_sizze_17104) {\n                for (int32_t i_17122 = 0; i_17122 < group_sizze_17104;\n                     i_17122++) {\n                    fl",
            "oat x_17126 = *(__local float *) &mem_17594[i_17122 * 4];\n                    float x_17127 = *(__local float *) &mem_17597[i_17122 * 4];\n                    float x_17129 = x_17112 * x_17126;\n                    float x_17130 = x_17113 * x_17129;\n                    float res_17131 = x_17127 * x_17130;\n                    float res_17133 = acc_17123 + res_17131;\n                    float acc_tmp_17795 = res_17133;\n                    \n                    acc_17123 = acc_tmp_17795;\n                }\n            } else {\n                for (int32_t i_17122 = 0; i_17122 < chunk_sizze_17115;\n                     i_17122++) {\n                    float x_17126 = *(__local float *) &mem_17594[i_17122 * 4];\n                    float x_17127 = *(__local float *) &mem_17597[i_17122 * 4];\n                    float x_17129 = x_17112 * x_17126;\n                    float x_17130 = x_17113 * x_17129;\n                    float res_17131 = x_17127 * x_17130;\n                    float res_17133 = acc_17123 + res_17131;\n                    float acc_tmp_17795 = res_17133;\n                    \n                    acc_17123 = acc_tmp_17795;\n                }\n            }\n        }\n        res_17120 = acc_17123;\n        sync_17532 = res_17120;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_17117 = sync_17532;\n        chunk_offset_17116 += group_sizze_17104;\n    }\n    res_17114 = x_17117;\n    if (thread_active_17790) {\n        *(__global float *) &mem_17600[gtid_17102 * 4] = res_17114;\n    }\n}\n__kernel void map_kernel_17152(int32_t sizze_16522, int32_t sizze_16523,\n                               int32_t sizze_16524, __global\n                               unsigned char *D_mem_17589, __global\n                               unsigned char *mem_17593, __global\n                               unsigned char *mem_17596)\n{\n    int32_t wave_sizze_17799;\n    int32_t group_sizze_17800;\n    bool thread_active_17801;\n    int32_t gtid_17145;\n    int32_t global_tid_17152;\n    int32_t local_tid_17153;",
            "\n    int32_t group_id_17154;\n    \n    global_tid_17152 = get_global_id(0);\n    local_tid_17153 = get_local_id(0);\n    group_sizze_17800 = get_local_size(0);\n    wave_sizze_17799 = LOCKSTEP_WIDTH;\n    group_id_17154 = get_group_id(0);\n    gtid_17145 = global_tid_17152;\n    thread_active_17801 = slt32(gtid_17145, sizze_16522);\n    \n    float res_17156;\n    \n    if (thread_active_17801) {\n        float x_17159 = 0.0F;\n        \n        for (int32_t chunk_offset_17158 = 0; chunk_offset_17158 < sizze_16524;\n             chunk_offset_17158++) {\n            float x_17168 = *(__global float *) &mem_17593[(chunk_offset_17158 *\n                                                            sizze_16523 +\n                                                            gtid_17145) * 4];\n            float x_17169 = *(__global\n                              float *) &D_mem_17589[chunk_offset_17158 * 4];\n            float res_17171 = x_17168 * x_17169;\n            float res_17173 = x_17159 + res_17171;\n            float x_tmp_17802 = res_17173;\n            \n            x_17159 = x_tmp_17802;\n        }\n        res_17156 = x_17159;\n    }\n    if (thread_active_17801) {\n        *(__global float *) &mem_17596[gtid_17145 * 4] = res_17156;\n    }\n}\n__kernel void map_kernel_17192(int32_t sizze_16521, int32_t sizze_16522,\n                               __global unsigned char *mem_17596, __global\n                               unsigned char *mem_17600, __global\n                               unsigned char *mem_17603)\n{\n    int32_t wave_sizze_17803;\n    int32_t group_sizze_17804;\n    bool thread_active_17805;\n    int32_t gtid_17185;\n    int32_t global_tid_17192;\n    int32_t local_tid_17193;\n    int32_t group_id_17194;\n    \n    global_tid_17192 = get_global_id(0);\n    local_tid_17193 = get_local_id(0);\n    group_sizze_17804 = get_local_size(0);\n    wave_sizze_17803 = LOCKSTEP_WIDTH;\n    group_id_17194 = get_group_id(0);\n    gtid_17185 = global_tid_17192;\n    thread_active_17805 = slt32(gtid_17185, sizze",
            "_16521);\n    \n    float res_17196;\n    \n    if (thread_active_17805) {\n        float x_17199 = 0.0F;\n        \n        for (int32_t chunk_offset_17198 = 0; chunk_offset_17198 < sizze_16522;\n             chunk_offset_17198++) {\n            float x_17208 = *(__global float *) &mem_17600[(chunk_offset_17198 *\n                                                            sizze_16521 +\n                                                            gtid_17185) * 4];\n            float x_17209 = *(__global float *) &mem_17596[chunk_offset_17198 *\n                                                           4];\n            float res_17211 = x_17208 * x_17209;\n            float res_17213 = x_17199 + res_17211;\n            float x_tmp_17806 = res_17213;\n            \n            x_17199 = x_tmp_17806;\n        }\n        res_17196 = x_17199;\n    }\n    if (thread_active_17805) {\n        *(__global float *) &mem_17603[gtid_17185 * 4] = res_17196;\n    }\n}\n__kernel void map_kernel_17221(int32_t sizze_16566, __global\n                               unsigned char *C_mem_17589, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17594)\n{\n    int32_t wave_sizze_17810;\n    int32_t group_sizze_17811;\n    bool thread_active_17812;\n    int32_t gtid_17214;\n    int32_t global_tid_17221;\n    int32_t local_tid_17222;\n    int32_t group_id_17223;\n    \n    global_tid_17221 = get_global_id(0);\n    local_tid_17222 = get_local_id(0);\n    group_sizze_17811 = get_local_size(0);\n    wave_sizze_17810 = LOCKSTEP_WIDTH;\n    group_id_17223 = get_group_id(0);\n    gtid_17214 = global_tid_17221;\n    thread_active_17812 = slt32(gtid_17214, sizze_16566);\n    \n    float x_17224;\n    float x_17225;\n    float res_17226;\n    \n    if (thread_active_17812) {\n        x_17224 = *(__global float *) &C_mem_17589[gtid_17214 * 4];\n        x_17225 = *(__global float *) &D_mem_17591[gtid_17214 * 4];\n        res_17226 = x_17224 + x_17225;\n    }\n    if (thread_act",
            "ive_17812) {\n        *(__global float *) &mem_17594[gtid_17214 * 4] = res_17226;\n    }\n}\n__kernel void map_kernel_17256(int32_t sizze_16563, int32_t sizze_16564,\n                               int32_t sizze_16566, __global\n                               unsigned char *B_mem_17587, __global\n                               unsigned char *mem_17594, __global\n                               unsigned char *mem_17598, __global\n                               unsigned char *mem_17601)\n{\n    int32_t wave_sizze_17813;\n    int32_t group_sizze_17814;\n    bool thread_active_17815;\n    int32_t gtid_17249;\n    int32_t global_tid_17256;\n    int32_t local_tid_17257;\n    int32_t group_id_17258;\n    \n    global_tid_17256 = get_global_id(0);\n    local_tid_17257 = get_local_id(0);\n    group_sizze_17814 = get_local_size(0);\n    wave_sizze_17813 = LOCKSTEP_WIDTH;\n    group_id_17258 = get_group_id(0);\n    gtid_17249 = global_tid_17256;\n    thread_active_17815 = slt32(gtid_17249, sizze_16563);\n    \n    float res_17260;\n    \n    if (thread_active_17815) {\n        float x_17263 = 0.0F;\n        \n        for (int32_t chunk_offset_17262 = 0; chunk_offset_17262 < sizze_16566;\n             chunk_offset_17262++) {\n            float x_17273 = *(__global float *) &mem_17594[chunk_offset_17262 *\n                                                           4];\n            float res_17275;\n            float x_17278 = 0.0F;\n            \n            for (int32_t chunk_offset_17277 = 0; chunk_offset_17277 <\n                 sizze_16564; chunk_offset_17277++) {\n                float x_17287 = *(__global\n                                  float *) &mem_17598[(chunk_offset_17277 *\n                                                       sizze_16563 +\n                                                       gtid_17249) * 4];\n                float x_17288 = *(__global\n                                  float *) &B_mem_17587[(chunk_offset_17277 *\n                                                         sizze_16566 +\n     ",
            "                                                    chunk_offset_17262) *\n                                                        4];\n                float res_17290 = x_17287 * x_17288;\n                float res_17292 = x_17278 + res_17290;\n                float x_tmp_17817 = res_17292;\n                \n                x_17278 = x_tmp_17817;\n            }\n            res_17275 = x_17278;\n            \n            float res_17293 = x_17273 * res_17275;\n            float res_17295 = x_17263 + res_17293;\n            float x_tmp_17816 = res_17295;\n            \n            x_17263 = x_tmp_17816;\n        }\n        res_17260 = x_17263;\n    }\n    if (thread_active_17815) {\n        *(__global float *) &mem_17601[gtid_17249 * 4] = res_17260;\n    }\n}\n__kernel void map_kernel_17314(int32_t sizze_16617, int32_t sizze_16619,\n                               int32_t sizze_16620, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17595, __global\n                               unsigned char *mem_17598)\n{\n    int32_t wave_sizze_17821;\n    int32_t group_sizze_17822;\n    bool thread_active_17823;\n    int32_t gtid_17307;\n    int32_t global_tid_17314;\n    int32_t local_tid_17315;\n    int32_t group_id_17316;\n    \n    global_tid_17314 = get_global_id(0);\n    local_tid_17315 = get_local_id(0);\n    group_sizze_17822 = get_local_size(0);\n    wave_sizze_17821 = LOCKSTEP_WIDTH;\n    group_id_17316 = get_group_id(0);\n    gtid_17307 = global_tid_17314;\n    thread_active_17823 = slt32(gtid_17307, sizze_16619);\n    \n    float res_17318;\n    \n    if (thread_active_17823) {\n        float x_17321 = 0.0F;\n        \n        for (int32_t chunk_offset_17320 = 0; chunk_offset_17320 < sizze_16617;\n             chunk_offset_17320++) {\n            float x_17330 = *(__global float *) &mem_17595[(chunk_offset_17320 *\n                                                            sizze_16620 +\n                                                   ",
            "         gtid_17307) * 4];\n            float x_17331 = *(__global\n                              float *) &D_mem_17591[chunk_offset_17320 * 4];\n            float res_17333 = x_17330 * x_17331;\n            float res_17335 = x_17321 + res_17333;\n            float x_tmp_17824 = res_17335;\n            \n            x_17321 = x_tmp_17824;\n        }\n        res_17318 = x_17321;\n    }\n    if (thread_active_17823) {\n        *(__global float *) &mem_17598[gtid_17307 * 4] = res_17318;\n    }\n}\n__kernel void map_kernel_17365(int32_t sizze_16616, int32_t sizze_16617,\n                               int32_t sizze_16619, __global\n                               unsigned char *B_mem_17587, __global\n                               unsigned char *mem_17598, __global\n                               unsigned char *mem_17602, __global\n                               unsigned char *mem_17605)\n{\n    int32_t wave_sizze_17825;\n    int32_t group_sizze_17826;\n    bool thread_active_17827;\n    int32_t gtid_17358;\n    int32_t global_tid_17365;\n    int32_t local_tid_17366;\n    int32_t group_id_17367;\n    \n    global_tid_17365 = get_global_id(0);\n    local_tid_17366 = get_local_id(0);\n    group_sizze_17826 = get_local_size(0);\n    wave_sizze_17825 = LOCKSTEP_WIDTH;\n    group_id_17367 = get_group_id(0);\n    gtid_17358 = global_tid_17365;\n    thread_active_17827 = slt32(gtid_17358, sizze_16616);\n    \n    float res_17369;\n    \n    if (thread_active_17827) {\n        float x_17372 = 0.0F;\n        \n        for (int32_t chunk_offset_17371 = 0; chunk_offset_17371 < sizze_16619;\n             chunk_offset_17371++) {\n            float x_17382 = *(__global float *) &mem_17598[chunk_offset_17371 *\n                                                           4];\n            float res_17384;\n            float x_17387 = 0.0F;\n            \n            for (int32_t chunk_offset_17386 = 0; chunk_offset_17386 <\n                 sizze_16617; chunk_offset_17386++) {\n                float x_17396 = *(__global\n                  ",
            "                float *) &mem_17602[(chunk_offset_17386 *\n                                                       sizze_16616 +\n                                                       gtid_17358) * 4];\n                float x_17397 = *(__global\n                                  float *) &B_mem_17587[(chunk_offset_17386 *\n                                                         sizze_16619 +\n                                                         chunk_offset_17371) *\n                                                        4];\n                float res_17399 = x_17396 * x_17397;\n                float res_17401 = x_17387 + res_17399;\n                float x_tmp_17829 = res_17401;\n                \n                x_17387 = x_tmp_17829;\n            }\n            res_17384 = x_17387;\n            \n            float res_17402 = x_17382 * res_17384;\n            float res_17404 = x_17372 + res_17402;\n            float x_tmp_17828 = res_17404;\n            \n            x_17372 = x_tmp_17828;\n        }\n        res_17369 = x_17372;\n    }\n    if (thread_active_17827) {\n        *(__global float *) &mem_17605[gtid_17358 * 4] = res_17369;\n    }\n}\n__kernel void map_kernel_17423(int32_t sizze_16680, int32_t sizze_16683,\n                               int32_t sizze_16684, __global\n                               unsigned char *D_mem_17591, __global\n                               unsigned char *mem_17595, __global\n                               unsigned char *mem_17598)\n{\n    int32_t wave_sizze_17833;\n    int32_t group_sizze_17834;\n    bool thread_active_17835;\n    int32_t gtid_17416;\n    int32_t global_tid_17423;\n    int32_t local_tid_17424;\n    int32_t group_id_17425;\n    \n    global_tid_17423 = get_global_id(0);\n    local_tid_17424 = get_local_id(0);\n    group_sizze_17834 = get_local_size(0);\n    wave_sizze_17833 = LOCKSTEP_WIDTH;\n    group_id_17425 = get_group_id(0);\n    gtid_17416 = global_tid_17423;\n    thread_active_17835 = slt32(gtid_17416, sizze_16680);\n    \n    float res_17427;\n",
            "    \n    if (thread_active_17835) {\n        float x_17430 = 0.0F;\n        \n        for (int32_t chunk_offset_17429 = 0; chunk_offset_17429 < sizze_16684;\n             chunk_offset_17429++) {\n            float x_17439 = *(__global float *) &mem_17595[(chunk_offset_17429 *\n                                                            sizze_16683 +\n                                                            gtid_17416) * 4];\n            float x_17440 = *(__global\n                              float *) &D_mem_17591[chunk_offset_17429 * 4];\n            float res_17442 = x_17439 * x_17440;\n            float res_17444 = x_17430 + res_17442;\n            float x_tmp_17836 = res_17444;\n            \n            x_17430 = x_tmp_17836;\n        }\n        res_17427 = x_17430;\n    }\n    if (thread_active_17835) {\n        *(__global float *) &mem_17598[gtid_17416 * 4] = res_17427;\n    }\n}\n__kernel void map_kernel_17465(int32_t sizze_16679, int32_t sizze_16680,\n                               int32_t sizze_16681, __global\n                               unsigned char *mem_17598, __global\n                               unsigned char *mem_17602, __global\n                               unsigned char *mem_17606, __global\n                               unsigned char *mem_17609)\n{\n    int32_t wave_sizze_17837;\n    int32_t group_sizze_17838;\n    bool thread_active_17839;\n    int32_t gtid_17458;\n    int32_t global_tid_17465;\n    int32_t local_tid_17466;\n    int32_t group_id_17467;\n    \n    global_tid_17465 = get_global_id(0);\n    local_tid_17466 = get_local_id(0);\n    group_sizze_17838 = get_local_size(0);\n    wave_sizze_17837 = LOCKSTEP_WIDTH;\n    group_id_17467 = get_group_id(0);\n    gtid_17458 = global_tid_17465;\n    thread_active_17839 = slt32(gtid_17458, sizze_16679);\n    \n    float res_17470;\n    \n    if (thread_active_17839) {\n        float x_17473 = 0.0F;\n        \n        for (int32_t chunk_offset_17472 = 0; chunk_offset_17472 < sizze_16680;\n             chunk_offset_17472++) {\n         ",
            "   float x_17484 = *(__global float *) &mem_17602[(chunk_offset_17472 *\n                                                            sizze_16679 +\n                                                            gtid_17458) * 4];\n            float x_17485 = *(__global float *) &mem_17606[(chunk_offset_17472 *\n                                                            sizze_16681 +\n                                                            gtid_17458) * 4];\n            float x_17486 = *(__global float *) &mem_17598[chunk_offset_17472 *\n                                                           4];\n            float res_17488 = x_17484 + x_17485;\n            float res_17489 = x_17486 * res_17488;\n            float res_17491 = x_17473 + res_17489;\n            float x_tmp_17840 = res_17491;\n            \n            x_17473 = x_tmp_17840;\n        }\n        res_17470 = x_17473;\n    }\n    if (thread_active_17839) {\n        *(__global float *) &mem_17609[gtid_17458 * 4] = res_17470;\n    }\n}\n__kernel void reduce_kernel_16829(__local volatile int64_t *mem_aligned_0,\n                                  int32_t sizze_16261, int32_t num_groups_16765,\n                                  __global unsigned char *mem_17590, __global\n                                  unsigned char *mem_17640, __global\n                                  unsigned char *mem_17653, __global\n                                  unsigned char *mem_17656, __global\n                                  unsigned char *mem_17659, __global\n                                  unsigned char *mem_17666)\n{\n    __local volatile char *restrict mem_17650 = mem_aligned_0;\n    int32_t wave_sizze_17728;\n    int32_t group_sizze_17729;\n    bool thread_active_17730;\n    int32_t global_tid_16829;\n    int32_t local_tid_16830;\n    int32_t group_id_16831;\n    \n    global_tid_16829 = get_global_id(0);\n    local_tid_16830 = get_local_id(0);\n    group_sizze_17729 = get_local_size(0);\n    wave_sizze_17728 = LOCKSTEP_WIDTH;\n    group_id_16831 = get_g",
            "roup_id(0);\n    thread_active_17730 = 1;\n    \n    bool in_bounds_16832;\n    \n    if (thread_active_17730) {\n        in_bounds_16832 = slt32(local_tid_16830, num_groups_16765);\n        if (in_bounds_16832) {\n            for (int32_t i_17731 = 0; i_17731 < sizze_16261; i_17731++) {\n                *(__global float *) &mem_17666[(group_id_16831 * (sizze_16261 *\n                                                                  max_num_groups_16760) +\n                                                i_17731 * max_num_groups_16760 +\n                                                local_tid_16830) * 4] =\n                    *(__global float *) &mem_17640[(global_tid_16829 *\n                                                    sizze_16261 + i_17731) * 4];\n            }\n        } else {\n            for (int32_t i_17732 = 0; i_17732 < sizze_16261; i_17732++) {\n                *(__global float *) &mem_17666[(group_id_16831 * (sizze_16261 *\n                                                                  max_num_groups_16760) +\n                                                i_17732 * max_num_groups_16760 +\n                                                local_tid_16830) * 4] =\n                    *(__global float *) &mem_17590[i_17732 * 4];\n            }\n        }\n    }\n    for (int32_t comb_iter_17733 = 0; comb_iter_17733 <\n         squot32(max_num_groups_16760 + max_num_groups_16760 - 1,\n                 max_num_groups_16760); comb_iter_17733++) {\n        int32_t combine_id_16836;\n        int32_t flat_comb_id_17734 = comb_iter_17733 * max_num_groups_16760 +\n                local_tid_16830;\n        \n        combine_id_16836 = flat_comb_id_17734;\n        if (slt32(combine_id_16836, max_num_groups_16760) && 1) {\n            for (int32_t i_17735 = 0; i_17735 < sizze_16261; i_17735++) {\n                *(__local float *) &mem_17650[(combine_id_16836 * sizze_16261 +\n                                               i_17735) * 4] = *(__global\n                                           ",
            "                      float *) &mem_17666[(group_id_16831 *\n                                                                                      (sizze_16261 *\n                                                                                       max_num_groups_16760) +\n                                                                                      i_17735 *\n                                                                                      max_num_groups_16760 +\n                                                                                      local_tid_16830) *\n                                                                                     4];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_17736;\n    int32_t my_index_16772;\n    int32_t other_offset_16773;\n    \n    my_index_16772 = local_tid_16830;\n    other_offset_16773 = 0;\n    if (slt32(local_tid_16830, max_num_groups_16760)) { }\n    other_offset_16773 = 1;\n    while (slt32(other_offset_16773, wave_sizze_17728)) {\n        if (slt32(local_tid_16830 + other_offset_16773, max_num_groups_16760) &&\n            ((local_tid_16830 - squot32(local_tid_16830, wave_sizze_17728) *\n              wave_sizze_17728) & (2 * other_offset_16773 - 1)) == 0) {\n            // read array element\n            { }\n            if (thread_active_17730) {\n                for (int32_t i_16741 = 0; i_16741 < sizze_16261; i_16741++) {\n                    float x_16277 = *(volatile __local\n                                      float *) &mem_17650[(my_index_16772 *\n                                                           sizze_16261 +\n                                                           i_16741) * 4];\n                    float x_16278 = *(volatile __local\n                                      float *) &mem_17650[((my_index_16772 +\n                                                            other_offset_16773) *\n                                                           sizze_1",
            "6261 +\n                                                           i_16741) * 4];\n                    float res_16279 = x_16277 + x_16278;\n                    \n                    *(volatile __global float *) &mem_17653[(group_id_16831 *\n                                                             (sizze_16261 *\n                                                              max_num_groups_16760) +\n                                                             i_16741 *\n                                                             max_num_groups_16760 +\n                                                             local_tid_16830) *\n                                                            4] = res_16279;\n                }\n            }\n            for (int32_t i_17738 = 0; i_17738 < sizze_16261; i_17738++) {\n                *(volatile __local float *) &mem_17650[(my_index_16772 *\n                                                        sizze_16261 + i_17738) *\n                                                       4] = *(volatile __global\n                                                              float *) &mem_17653[(group_id_16831 *\n                                                                                   (sizze_16261 *\n                                                                                    max_num_groups_16760) +\n                                                                                   i_17738 *\n                                                                                   max_num_groups_16760 +\n                                                                                   local_tid_16830) *\n                                                                                  4];\n            }\n        }\n        other_offset_16773 *= 2;\n    }\n    skip_waves_17736 = 1;\n    while (slt32(skip_waves_17736, squot32(max_num_groups_16760 +\n                                           wave_sizze_17728 - 1,\n                                     ",
            "      wave_sizze_17728))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_16773 = skip_waves_17736 * wave_sizze_17728;\n        if (slt32(local_tid_16830 + other_offset_16773, max_num_groups_16760) &&\n            ((local_tid_16830 - squot32(local_tid_16830, wave_sizze_17728) *\n              wave_sizze_17728) == 0 && (squot32(local_tid_16830,\n                                                 wave_sizze_17728) & (2 *\n                                                                      skip_waves_17736 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            { }\n            if (thread_active_17730) {\n                for (int32_t i_16741 = 0; i_16741 < sizze_16261; i_16741++) {\n                    float x_16277 = *(__local\n                                      float *) &mem_17650[(my_index_16772 *\n                                                           sizze_16261 +\n                                                           i_16741) * 4];\n                    float x_16278 = *(__local\n                                      float *) &mem_17650[((my_index_16772 +\n                                                            other_offset_16773) *\n                                                           sizze_16261 +\n                                                           i_16741) * 4];\n                    float res_16279 = x_16277 + x_16278;\n                    \n                    *(__global float *) &mem_17653[(group_id_16831 *\n                                                    (sizze_16261 *\n                                                     max_num_groups_16760) +\n                                                    i_16741 *\n                                                    max_num_groups_16760 +\n                                                    local_tid_16830) * 4] =\n                        res_16279;\n                }\n            }\n            for (int32_t i",
            "_17740 = 0; i_17740 < sizze_16261; i_17740++) {\n                *(__local float *) &mem_17650[(my_index_16772 * sizze_16261 +\n                                               i_17740) * 4] = *(__global\n                                                                 float *) &mem_17653[(group_id_16831 *\n                                                                                      (sizze_16261 *\n                                                                                       max_num_groups_16760) +\n                                                                                      i_17740 *\n                                                                                      max_num_groups_16760 +\n                                                                                      local_tid_16830) *\n                                                                                     4];\n            }\n        }\n        skip_waves_17736 *= 2;\n    }\n    for (int32_t i_17741 = 0; i_17741 < sizze_16261; i_17741++) {\n        *(__global float *) &mem_17656[(group_id_16831 * (sizze_16261 *\n                                                          max_num_groups_16760) +\n                                        i_17741 * max_num_groups_16760 +\n                                        local_tid_16830) * 4] = *(__local\n                                                                  float *) &mem_17650[(my_index_16772 *\n                                                                                       sizze_16261 +\n                                                                                       i_17741) *\n                                                                                      4];\n    }\n    if (local_tid_16830 == 0) {\n        for (int32_t i_17743 = 0; i_17743 < sizze_16261; i_17743++) {\n            *(__global float *) &mem_17659[(group_id_16831 * sizze_16261 +\n                                            i_17743) * 4] = *(__global\n           ",
            "                                                   float *) &mem_17656[(group_id_16831 *\n                                                                                   (sizze_16261 *\n                                                                                    max_num_groups_16760) +\n                                                                                   i_17743 *\n                                                                                   max_num_groups_16760 +\n                                                                                   local_tid_16830) *\n                                                                                  4];\n        }\n    }\n}\n",
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
static const char *size_names[] = {"group_size_16757", "max_num_groups_16759",
                                   "group_size_16850", "group_size_16890",
                                   "group_size_16934", "group_size_16968",
                                   "group_size_16994", "group_size_17028",
                                   "group_size_17056", "group_size_17103",
                                   "group_size_17146", "group_size_17186",
                                   "group_size_17215", "group_size_17250",
                                   "group_size_17308", "group_size_17359",
                                   "group_size_17417", "group_size_17459",
                                   "group_size_17703"};
static const char *size_classes[] = {"group_size", "num_groups", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size"};
static const char *size_entry_points[] = {"t1_", "t1_", "t1", "t2", "t3", "t4",
                                          "t4", "t5", "t5", "t6", "t7", "t7",
                                          "t9", "t9", "t10", "t10", "t8", "t8",
                                          "t1_"};
int futhark_get_num_sizes(void)
{
    return 19;
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
    size_t group_sizze_16757;
    size_t max_num_groups_16759;
    size_t group_sizze_16850;
    size_t group_sizze_16890;
    size_t group_sizze_16934;
    size_t group_sizze_16968;
    size_t group_sizze_16994;
    size_t group_sizze_17028;
    size_t group_sizze_17056;
    size_t group_sizze_17103;
    size_t group_sizze_17146;
    size_t group_sizze_17186;
    size_t group_sizze_17215;
    size_t group_sizze_17250;
    size_t group_sizze_17308;
    size_t group_sizze_17359;
    size_t group_sizze_17417;
    size_t group_sizze_17459;
    size_t group_sizze_17703;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[19];
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
    opencl_config_init(&cfg->opencl, 19, size_names, cfg->sizes, size_classes,
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
    for (int i = 0; i < 19; i++) {
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
    cl_kernel chunked_reduce_kernel_16774;
    int chunked_reduce_kernel_16774_total_runtime;
    int chunked_reduce_kernel_16774_runs;
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
    cl_kernel kernel_replicate_16271;
    int kernel_replicate_16271_total_runtime;
    int kernel_replicate_16271_runs;
    cl_kernel map_kernel_16856;
    int map_kernel_16856_total_runtime;
    int map_kernel_16856_runs;
    cl_kernel map_kernel_16896;
    int map_kernel_16896_total_runtime;
    int map_kernel_16896_runs;
    cl_kernel map_kernel_16940;
    int map_kernel_16940_total_runtime;
    int map_kernel_16940_runs;
    cl_kernel map_kernel_16974;
    int map_kernel_16974_total_runtime;
    int map_kernel_16974_runs;
    cl_kernel map_kernel_17000;
    int map_kernel_17000_total_runtime;
    int map_kernel_17000_runs;
    cl_kernel map_kernel_17034;
    int map_kernel_17034_total_runtime;
    int map_kernel_17034_runs;
    cl_kernel map_kernel_17062;
    int map_kernel_17062_total_runtime;
    int map_kernel_17062_runs;
    cl_kernel map_kernel_17109;
    int map_kernel_17109_total_runtime;
    int map_kernel_17109_runs;
    cl_kernel map_kernel_17152;
    int map_kernel_17152_total_runtime;
    int map_kernel_17152_runs;
    cl_kernel map_kernel_17192;
    int map_kernel_17192_total_runtime;
    int map_kernel_17192_runs;
    cl_kernel map_kernel_17221;
    int map_kernel_17221_total_runtime;
    int map_kernel_17221_runs;
    cl_kernel map_kernel_17256;
    int map_kernel_17256_total_runtime;
    int map_kernel_17256_runs;
    cl_kernel map_kernel_17314;
    int map_kernel_17314_total_runtime;
    int map_kernel_17314_runs;
    cl_kernel map_kernel_17365;
    int map_kernel_17365_total_runtime;
    int map_kernel_17365_runs;
    cl_kernel map_kernel_17423;
    int map_kernel_17423_total_runtime;
    int map_kernel_17423_runs;
    cl_kernel map_kernel_17465;
    int map_kernel_17465_total_runtime;
    int map_kernel_17465_runs;
    cl_kernel reduce_kernel_16829;
    int reduce_kernel_16829_total_runtime;
    int reduce_kernel_16829_runs;
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
    ctx->chunked_reduce_kernel_16774_total_runtime = 0;
    ctx->chunked_reduce_kernel_16774_runs = 0;
    ctx->fut_kernel_map_transpose_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_runs = 0;
    ctx->fut_kernel_map_transpose_small_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_f32_runs = 0;
    ctx->kernel_replicate_16271_total_runtime = 0;
    ctx->kernel_replicate_16271_runs = 0;
    ctx->map_kernel_16856_total_runtime = 0;
    ctx->map_kernel_16856_runs = 0;
    ctx->map_kernel_16896_total_runtime = 0;
    ctx->map_kernel_16896_runs = 0;
    ctx->map_kernel_16940_total_runtime = 0;
    ctx->map_kernel_16940_runs = 0;
    ctx->map_kernel_16974_total_runtime = 0;
    ctx->map_kernel_16974_runs = 0;
    ctx->map_kernel_17000_total_runtime = 0;
    ctx->map_kernel_17000_runs = 0;
    ctx->map_kernel_17034_total_runtime = 0;
    ctx->map_kernel_17034_runs = 0;
    ctx->map_kernel_17062_total_runtime = 0;
    ctx->map_kernel_17062_runs = 0;
    ctx->map_kernel_17109_total_runtime = 0;
    ctx->map_kernel_17109_runs = 0;
    ctx->map_kernel_17152_total_runtime = 0;
    ctx->map_kernel_17152_runs = 0;
    ctx->map_kernel_17192_total_runtime = 0;
    ctx->map_kernel_17192_runs = 0;
    ctx->map_kernel_17221_total_runtime = 0;
    ctx->map_kernel_17221_runs = 0;
    ctx->map_kernel_17256_total_runtime = 0;
    ctx->map_kernel_17256_runs = 0;
    ctx->map_kernel_17314_total_runtime = 0;
    ctx->map_kernel_17314_runs = 0;
    ctx->map_kernel_17365_total_runtime = 0;
    ctx->map_kernel_17365_runs = 0;
    ctx->map_kernel_17423_total_runtime = 0;
    ctx->map_kernel_17423_runs = 0;
    ctx->map_kernel_17465_total_runtime = 0;
    ctx->map_kernel_17465_runs = 0;
    ctx->reduce_kernel_16829_total_runtime = 0;
    ctx->reduce_kernel_16829_runs = 0;
    
    int required_types = 0;
    cl_int error;
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    {
        ctx->chunked_reduce_kernel_16774 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_16774",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_16774");
    }
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
        ctx->kernel_replicate_16271 = clCreateKernel(prog,
                                                     "kernel_replicate_16271",
                                                     &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "kernel_replicate_16271");
    }
    {
        ctx->map_kernel_16856 = clCreateKernel(prog, "map_kernel_16856",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_16856");
    }
    {
        ctx->map_kernel_16896 = clCreateKernel(prog, "map_kernel_16896",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_16896");
    }
    {
        ctx->map_kernel_16940 = clCreateKernel(prog, "map_kernel_16940",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_16940");
    }
    {
        ctx->map_kernel_16974 = clCreateKernel(prog, "map_kernel_16974",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_16974");
    }
    {
        ctx->map_kernel_17000 = clCreateKernel(prog, "map_kernel_17000",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17000");
    }
    {
        ctx->map_kernel_17034 = clCreateKernel(prog, "map_kernel_17034",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17034");
    }
    {
        ctx->map_kernel_17062 = clCreateKernel(prog, "map_kernel_17062",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17062");
    }
    {
        ctx->map_kernel_17109 = clCreateKernel(prog, "map_kernel_17109",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17109");
    }
    {
        ctx->map_kernel_17152 = clCreateKernel(prog, "map_kernel_17152",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17152");
    }
    {
        ctx->map_kernel_17192 = clCreateKernel(prog, "map_kernel_17192",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17192");
    }
    {
        ctx->map_kernel_17221 = clCreateKernel(prog, "map_kernel_17221",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17221");
    }
    {
        ctx->map_kernel_17256 = clCreateKernel(prog, "map_kernel_17256",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17256");
    }
    {
        ctx->map_kernel_17314 = clCreateKernel(prog, "map_kernel_17314",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17314");
    }
    {
        ctx->map_kernel_17365 = clCreateKernel(prog, "map_kernel_17365",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17365");
    }
    {
        ctx->map_kernel_17423 = clCreateKernel(prog, "map_kernel_17423",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17423");
    }
    {
        ctx->map_kernel_17465 = clCreateKernel(prog, "map_kernel_17465",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_17465");
    }
    {
        ctx->reduce_kernel_16829 = clCreateKernel(prog, "reduce_kernel_16829",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_16829");
    }
    ctx->sizes.group_sizze_16757 = cfg->sizes[0];
    ctx->sizes.max_num_groups_16759 = cfg->sizes[1];
    ctx->sizes.group_sizze_16850 = cfg->sizes[2];
    ctx->sizes.group_sizze_16890 = cfg->sizes[3];
    ctx->sizes.group_sizze_16934 = cfg->sizes[4];
    ctx->sizes.group_sizze_16968 = cfg->sizes[5];
    ctx->sizes.group_sizze_16994 = cfg->sizes[6];
    ctx->sizes.group_sizze_17028 = cfg->sizes[7];
    ctx->sizes.group_sizze_17056 = cfg->sizes[8];
    ctx->sizes.group_sizze_17103 = cfg->sizes[9];
    ctx->sizes.group_sizze_17146 = cfg->sizes[10];
    ctx->sizes.group_sizze_17186 = cfg->sizes[11];
    ctx->sizes.group_sizze_17215 = cfg->sizes[12];
    ctx->sizes.group_sizze_17250 = cfg->sizes[13];
    ctx->sizes.group_sizze_17308 = cfg->sizes[14];
    ctx->sizes.group_sizze_17359 = cfg->sizes[15];
    ctx->sizes.group_sizze_17417 = cfg->sizes[16];
    ctx->sizes.group_sizze_17459 = cfg->sizes[17];
    ctx->sizes.group_sizze_17703 = cfg->sizes[18];
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
                "Kernel chunked_reduce_kernel_16774            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_16774_runs,
                (long) ctx->chunked_reduce_kernel_16774_total_runtime /
                (ctx->chunked_reduce_kernel_16774_runs !=
                 0 ? ctx->chunked_reduce_kernel_16774_runs : 1),
                (long) ctx->chunked_reduce_kernel_16774_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_16774_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_16774_runs;
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
                "Kernel kernel_replicate_16271                 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->kernel_replicate_16271_runs,
                (long) ctx->kernel_replicate_16271_total_runtime /
                (ctx->kernel_replicate_16271_runs !=
                 0 ? ctx->kernel_replicate_16271_runs : 1),
                (long) ctx->kernel_replicate_16271_total_runtime);
        ctx->total_runtime += ctx->kernel_replicate_16271_total_runtime;
        ctx->total_runs += ctx->kernel_replicate_16271_runs;
        fprintf(stderr,
                "Kernel map_kernel_16856                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_16856_runs,
                (long) ctx->map_kernel_16856_total_runtime /
                (ctx->map_kernel_16856_runs !=
                 0 ? ctx->map_kernel_16856_runs : 1),
                (long) ctx->map_kernel_16856_total_runtime);
        ctx->total_runtime += ctx->map_kernel_16856_total_runtime;
        ctx->total_runs += ctx->map_kernel_16856_runs;
        fprintf(stderr,
                "Kernel map_kernel_16896                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_16896_runs,
                (long) ctx->map_kernel_16896_total_runtime /
                (ctx->map_kernel_16896_runs !=
                 0 ? ctx->map_kernel_16896_runs : 1),
                (long) ctx->map_kernel_16896_total_runtime);
        ctx->total_runtime += ctx->map_kernel_16896_total_runtime;
        ctx->total_runs += ctx->map_kernel_16896_runs;
        fprintf(stderr,
                "Kernel map_kernel_16940                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_16940_runs,
                (long) ctx->map_kernel_16940_total_runtime /
                (ctx->map_kernel_16940_runs !=
                 0 ? ctx->map_kernel_16940_runs : 1),
                (long) ctx->map_kernel_16940_total_runtime);
        ctx->total_runtime += ctx->map_kernel_16940_total_runtime;
        ctx->total_runs += ctx->map_kernel_16940_runs;
        fprintf(stderr,
                "Kernel map_kernel_16974                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_16974_runs,
                (long) ctx->map_kernel_16974_total_runtime /
                (ctx->map_kernel_16974_runs !=
                 0 ? ctx->map_kernel_16974_runs : 1),
                (long) ctx->map_kernel_16974_total_runtime);
        ctx->total_runtime += ctx->map_kernel_16974_total_runtime;
        ctx->total_runs += ctx->map_kernel_16974_runs;
        fprintf(stderr,
                "Kernel map_kernel_17000                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17000_runs,
                (long) ctx->map_kernel_17000_total_runtime /
                (ctx->map_kernel_17000_runs !=
                 0 ? ctx->map_kernel_17000_runs : 1),
                (long) ctx->map_kernel_17000_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17000_total_runtime;
        ctx->total_runs += ctx->map_kernel_17000_runs;
        fprintf(stderr,
                "Kernel map_kernel_17034                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17034_runs,
                (long) ctx->map_kernel_17034_total_runtime /
                (ctx->map_kernel_17034_runs !=
                 0 ? ctx->map_kernel_17034_runs : 1),
                (long) ctx->map_kernel_17034_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17034_total_runtime;
        ctx->total_runs += ctx->map_kernel_17034_runs;
        fprintf(stderr,
                "Kernel map_kernel_17062                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17062_runs,
                (long) ctx->map_kernel_17062_total_runtime /
                (ctx->map_kernel_17062_runs !=
                 0 ? ctx->map_kernel_17062_runs : 1),
                (long) ctx->map_kernel_17062_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17062_total_runtime;
        ctx->total_runs += ctx->map_kernel_17062_runs;
        fprintf(stderr,
                "Kernel map_kernel_17109                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17109_runs,
                (long) ctx->map_kernel_17109_total_runtime /
                (ctx->map_kernel_17109_runs !=
                 0 ? ctx->map_kernel_17109_runs : 1),
                (long) ctx->map_kernel_17109_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17109_total_runtime;
        ctx->total_runs += ctx->map_kernel_17109_runs;
        fprintf(stderr,
                "Kernel map_kernel_17152                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17152_runs,
                (long) ctx->map_kernel_17152_total_runtime /
                (ctx->map_kernel_17152_runs !=
                 0 ? ctx->map_kernel_17152_runs : 1),
                (long) ctx->map_kernel_17152_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17152_total_runtime;
        ctx->total_runs += ctx->map_kernel_17152_runs;
        fprintf(stderr,
                "Kernel map_kernel_17192                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17192_runs,
                (long) ctx->map_kernel_17192_total_runtime /
                (ctx->map_kernel_17192_runs !=
                 0 ? ctx->map_kernel_17192_runs : 1),
                (long) ctx->map_kernel_17192_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17192_total_runtime;
        ctx->total_runs += ctx->map_kernel_17192_runs;
        fprintf(stderr,
                "Kernel map_kernel_17221                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17221_runs,
                (long) ctx->map_kernel_17221_total_runtime /
                (ctx->map_kernel_17221_runs !=
                 0 ? ctx->map_kernel_17221_runs : 1),
                (long) ctx->map_kernel_17221_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17221_total_runtime;
        ctx->total_runs += ctx->map_kernel_17221_runs;
        fprintf(stderr,
                "Kernel map_kernel_17256                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17256_runs,
                (long) ctx->map_kernel_17256_total_runtime /
                (ctx->map_kernel_17256_runs !=
                 0 ? ctx->map_kernel_17256_runs : 1),
                (long) ctx->map_kernel_17256_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17256_total_runtime;
        ctx->total_runs += ctx->map_kernel_17256_runs;
        fprintf(stderr,
                "Kernel map_kernel_17314                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17314_runs,
                (long) ctx->map_kernel_17314_total_runtime /
                (ctx->map_kernel_17314_runs !=
                 0 ? ctx->map_kernel_17314_runs : 1),
                (long) ctx->map_kernel_17314_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17314_total_runtime;
        ctx->total_runs += ctx->map_kernel_17314_runs;
        fprintf(stderr,
                "Kernel map_kernel_17365                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17365_runs,
                (long) ctx->map_kernel_17365_total_runtime /
                (ctx->map_kernel_17365_runs !=
                 0 ? ctx->map_kernel_17365_runs : 1),
                (long) ctx->map_kernel_17365_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17365_total_runtime;
        ctx->total_runs += ctx->map_kernel_17365_runs;
        fprintf(stderr,
                "Kernel map_kernel_17423                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17423_runs,
                (long) ctx->map_kernel_17423_total_runtime /
                (ctx->map_kernel_17423_runs !=
                 0 ? ctx->map_kernel_17423_runs : 1),
                (long) ctx->map_kernel_17423_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17423_total_runtime;
        ctx->total_runs += ctx->map_kernel_17423_runs;
        fprintf(stderr,
                "Kernel map_kernel_17465                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_17465_runs,
                (long) ctx->map_kernel_17465_total_runtime /
                (ctx->map_kernel_17465_runs !=
                 0 ? ctx->map_kernel_17465_runs : 1),
                (long) ctx->map_kernel_17465_total_runtime);
        ctx->total_runtime += ctx->map_kernel_17465_total_runtime;
        ctx->total_runs += ctx->map_kernel_17465_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_16829                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_16829_runs,
                (long) ctx->reduce_kernel_16829_total_runtime /
                (ctx->reduce_kernel_16829_runs !=
                 0 ? ctx->reduce_kernel_16829_runs : 1),
                (long) ctx->reduce_kernel_16829_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_16829_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_16829_runs;
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
static int futrts_t1_(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_17861,
                      struct memblock_device *out_mem_p_17862,
                      int32_t *out_out_arrsizze_17863,
                      int64_t A_mem_sizze_17584,
                      struct memblock_device A_mem_17585,
                      int64_t B_mem_sizze_17586,
                      struct memblock_device B_mem_17587, int32_t sizze_16260,
                      int32_t sizze_16261, int32_t sizze_16262);
static int futrts_t1(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17881,
                     struct memblock_device *out_mem_p_17882,
                     int32_t *out_out_arrsizze_17883, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587, int32_t sizze_16285,
                     int32_t sizze_16286, int32_t sizze_16287);
static int futrts_t2(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17890,
                     struct memblock_device *out_mem_p_17891,
                     int32_t *out_out_arrsizze_17892, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589, int32_t sizze_16306,
                     int32_t sizze_16307, int32_t sizze_16308,
                     int32_t sizze_16309);
static int futrts_t3(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17899,
                     struct memblock_device *out_mem_p_17900,
                     int32_t *out_out_arrsizze_17901, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589, int32_t sizze_16338,
                     int32_t sizze_16339, int32_t sizze_16340,
                     int32_t sizze_16341, int32_t sizze_16342);
static int futrts_t4(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17909,
                     struct memblock_device *out_mem_p_17910,
                     int32_t *out_out_arrsizze_17911, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16377,
                     int32_t sizze_16378, int32_t sizze_16379,
                     int32_t sizze_16380, int32_t sizze_16381,
                     int32_t sizze_16382);
static int futrts_t5(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17924,
                     struct memblock_device *out_mem_p_17925,
                     int32_t *out_out_arrsizze_17926, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16428,
                     int32_t sizze_16429, int32_t sizze_16430,
                     int32_t sizze_16431, int32_t sizze_16432,
                     int32_t sizze_16433, float a_16434, float b_16436,
                     float c_16438, float d_16440);
static int futrts_t6(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17939,
                     struct memblock_device *out_mem_p_17940,
                     int32_t *out_out_arrsizze_17941, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16487,
                     int32_t sizze_16488, int32_t sizze_16489,
                     int32_t sizze_16490);
static int futrts_t7(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17947,
                     struct memblock_device *out_mem_p_17948,
                     int32_t *out_out_arrsizze_17949, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t C_mem_sizze_17586,
                     struct memblock_device C_mem_17587,
                     int64_t D_mem_sizze_17588,
                     struct memblock_device D_mem_17589, int32_t sizze_16521,
                     int32_t sizze_16522, int32_t sizze_16523,
                     int32_t sizze_16524, int32_t sizze_16525);
static int futrts_t9(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17962,
                     struct memblock_device *out_mem_p_17963,
                     int32_t *out_out_arrsizze_17964, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16563,
                     int32_t sizze_16564, int32_t sizze_16565,
                     int32_t sizze_16566, int32_t sizze_16567,
                     int32_t sizze_16568);
static int futrts_t10(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_17976,
                      struct memblock_device *out_mem_p_17977,
                      int32_t *out_out_arrsizze_17978,
                      int64_t A_mem_sizze_17584,
                      struct memblock_device A_mem_17585,
                      int64_t B_mem_sizze_17586,
                      struct memblock_device B_mem_17587,
                      int64_t C_mem_sizze_17588,
                      struct memblock_device C_mem_17589,
                      int64_t D_mem_sizze_17590,
                      struct memblock_device D_mem_17591, int32_t sizze_16616,
                      int32_t sizze_16617, int32_t sizze_16618,
                      int32_t sizze_16619, int32_t sizze_16620,
                      int32_t sizze_16621, int32_t sizze_16622);
static int futrts_t8(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17991,
                     struct memblock_device *out_mem_p_17992,
                     int32_t *out_out_arrsizze_17993, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16679,
                     int32_t sizze_16680, int32_t sizze_16681,
                     int32_t sizze_16682, int32_t sizze_16683,
                     int32_t sizze_16684, int32_t sizze_16685);
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
                    const size_t global_work_sizze_17841[3] = {x_elems_5 +
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
                    const size_t local_work_sizze_17845[3] = {16, 16, 1};
                    int64_t time_start_17842 = 0, time_end_17843 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_f32");
                        fprintf(stderr, "%zu", global_work_sizze_17841[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_17841[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_17841[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_17845[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_17845[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_17845[2]);
                        fprintf(stderr, "].\n");
                        time_start_17842 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                          ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                          3, NULL,
                                                          global_work_sizze_17841,
                                                          local_work_sizze_17845,
                                                          0, NULL, NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                        time_end_17843 = get_wall_time();
                        
                        long time_diff_17844 = time_end_17843 -
                             time_start_17842;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime +=
                            time_diff_17844;
                        ctx->fut_kernel_map_transpose_lowwidth_f32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_f32",
                                time_diff_17844);
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
                        const size_t global_work_sizze_17846[3] =
                                     {new_width_12 + srem32(16 -
                                                            srem32(new_width_12,
                                                                   16), 16),
                                      y_elems_6 + srem32(16 - srem32(y_elems_6,
                                                                     16), 16),
                                      num_arrays_4};
                        const size_t local_work_sizze_17850[3] = {16, 16, 1};
                        int64_t time_start_17847 = 0, time_end_17848 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_f32");
                            fprintf(stderr, "%zu", global_work_sizze_17846[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_17846[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_17846[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_17850[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_17850[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_17850[2]);
                            fprintf(stderr, "].\n");
                            time_start_17847 = get_wall_time();
                        }
                        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                              ctx->fut_kernel_map_transpose_lowheight_f32,
                                                              3, NULL,
                                                              global_work_sizze_17846,
                                                              local_work_sizze_17850,
                                                              0, NULL, NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                            time_end_17848 = get_wall_time();
                            
                            long time_diff_17849 = time_end_17848 -
                                 time_start_17847;
                            
                            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime +=
                                time_diff_17849;
                            ctx->fut_kernel_map_transpose_lowheight_f32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_f32",
                                    time_diff_17849);
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
                            const size_t global_work_sizze_17851[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_17855[1] = {256};
                            int64_t time_start_17852 = 0, time_end_17853 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_17851[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_17855[0]);
                                fprintf(stderr, "].\n");
                                time_start_17852 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_small_f32,
                                                                  1, NULL,
                                                                  global_work_sizze_17851,
                                                                  local_work_sizze_17855,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_17853 = get_wall_time();
                                
                                long time_diff_17854 = time_end_17853 -
                                     time_start_17852;
                                
                                ctx->fut_kernel_map_transpose_small_f32_total_runtime +=
                                    time_diff_17854;
                                ctx->fut_kernel_map_transpose_small_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_f32",
                                        time_diff_17854);
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
                            const size_t global_work_sizze_17856[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_17860[3] = {16, 16,
                                                                      1};
                            int64_t time_start_17857 = 0, time_end_17858 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_17856[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_17856[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_17856[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_17860[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_17860[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_17860[2]);
                                fprintf(stderr, "].\n");
                                time_start_17857 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_f32,
                                                                  3, NULL,
                                                                  global_work_sizze_17856,
                                                                  local_work_sizze_17860,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_17858 = get_wall_time();
                                
                                long time_diff_17859 = time_end_17858 -
                                     time_start_17857;
                                
                                ctx->fut_kernel_map_transpose_f32_total_runtime +=
                                    time_diff_17859;
                                ctx->fut_kernel_map_transpose_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_f32",
                                        time_diff_17859);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts_t1_(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_17861,
                      struct memblock_device *out_mem_p_17862,
                      int32_t *out_out_arrsizze_17863,
                      int64_t A_mem_sizze_17584,
                      struct memblock_device A_mem_17585,
                      int64_t B_mem_sizze_17586,
                      struct memblock_device B_mem_17587, int32_t sizze_16260,
                      int32_t sizze_16261, int32_t sizze_16262)
{
    int64_t out_memsizze_17699;
    struct memblock_device out_mem_17698;
    
    out_mem_17698.references = NULL;
    
    int32_t out_arrsizze_17700;
    bool dim_zzero_16265 = 0 == sizze_16262;
    bool dim_zzero_16266 = 0 == sizze_16260;
    bool both_empty_16267 = dim_zzero_16265 && dim_zzero_16266;
    bool dim_match_16268 = sizze_16260 == sizze_16262;
    bool empty_or_match_16269 = both_empty_16267 || dim_match_16268;
    bool empty_or_match_cert_16270;
    
    if (!empty_or_match_16269) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:1:1-4:44",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17698, "out_mem_17698");
        return 1;
    }
    
    int64_t binop_x_17589 = sext_i32_i64(sizze_16261);
    int64_t bytes_17588 = 4 * binop_x_17589;
    struct memblock_device mem_17590;
    
    mem_17590.references = NULL;
    memblock_alloc_device(ctx, &mem_17590, bytes_17588, "mem_17590");
    
    int32_t group_sizze_17703;
    int32_t num_groups_17704;
    
    group_sizze_17703 = ctx->sizes.group_sizze_17703;
    num_groups_17704 = squot32(sizze_16261 + sext_i32_i32(group_sizze_17703) -
                               1, sext_i32_i32(group_sizze_17703));
    OPENCL_SUCCEED(clSetKernelArg(ctx->kernel_replicate_16271, 0,
                                  sizeof(sizze_16261), &sizze_16261));
    OPENCL_SUCCEED(clSetKernelArg(ctx->kernel_replicate_16271, 1,
                                  sizeof(mem_17590.mem), &mem_17590.mem));
    if (1 * (num_groups_17704 * group_sizze_17703) != 0) {
        const size_t global_work_sizze_17864[1] = {num_groups_17704 *
                     group_sizze_17703};
        const size_t local_work_sizze_17868[1] = {group_sizze_17703};
        int64_t time_start_17865 = 0, time_end_17866 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "kernel_replicate_16271");
            fprintf(stderr, "%zu", global_work_sizze_17864[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17868[0]);
            fprintf(stderr, "].\n");
            time_start_17865 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->kernel_replicate_16271, 1,
                                              NULL, global_work_sizze_17864,
                                              local_work_sizze_17868, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17866 = get_wall_time();
            
            long time_diff_17867 = time_end_17866 - time_start_17865;
            
            ctx->kernel_replicate_16271_total_runtime += time_diff_17867;
            ctx->kernel_replicate_16271_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "kernel_replicate_16271", time_diff_17867);
        }
    }
    
    int32_t group_sizze_16758;
    
    group_sizze_16758 = ctx->sizes.group_sizze_16757;
    
    int32_t max_num_groups_16760;
    
    max_num_groups_16760 = ctx->sizes.max_num_groups_16759;
    
    int32_t y_16761 = group_sizze_16758 - 1;
    int32_t x_16762 = sizze_16260 + y_16761;
    int32_t w_div_group_sizze_16763 = squot32(x_16762, group_sizze_16758);
    int32_t num_groups_maybe_zzero_16764 = smin32(max_num_groups_16760,
                                                  w_div_group_sizze_16763);
    int32_t num_groups_16765 = smax32(1, num_groups_maybe_zzero_16764);
    int32_t num_threads_16766 = group_sizze_16758 * num_groups_16765;
    int32_t y_16767 = num_threads_16766 - 1;
    int32_t x_16768 = sizze_16260 + y_16767;
    int32_t per_thread_elements_16769 = squot32(x_16768, num_threads_16766);
    int32_t y_17496 = smod32(sizze_16260, num_threads_16766);
    int32_t x_17497 = num_threads_16766 - y_17496;
    int32_t y_17498 = smod32(x_17497, num_threads_16766);
    int32_t padded_sizze_17499 = sizze_16260 + y_17498;
    int32_t per_chunk_17501 = squot32(padded_sizze_17499, num_threads_16766);
    int32_t convop_x_17592 = sizze_16261 * y_17498;
    int64_t binop_x_17593 = sext_i32_i64(convop_x_17592);
    int64_t bytes_17591 = 4 * binop_x_17593;
    struct memblock_device mem_17594;
    
    mem_17594.references = NULL;
    memblock_alloc_device(ctx, &mem_17594, bytes_17591, "mem_17594");
    
    int32_t convop_x_17596 = sizze_16261 * padded_sizze_17499;
    int64_t binop_x_17597 = sext_i32_i64(convop_x_17596);
    int64_t bytes_17595 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17595, "mem_17598");
    
    int32_t tmp_offs_17705 = 0;
    
    if (sizze_16260 * sizze_16261 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, A_mem_17585.mem,
                                           mem_17598.mem, 0, sizze_16261 *
                                           tmp_offs_17705 * 4, sizze_16260 *
                                           sizze_16261 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_17705 += sizze_16260;
    if (y_17498 * sizze_16261 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_17594.mem,
                                           mem_17598.mem, 0, sizze_16261 *
                                           tmp_offs_17705 * 4, y_17498 *
                                           sizze_16261 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_17705 += y_17498;
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    
    int32_t binop_x_17600 = sizze_16261 * per_chunk_17501;
    int32_t convop_x_17601 = num_threads_16766 * binop_x_17600;
    int64_t binop_x_17602 = sext_i32_i64(convop_x_17601);
    int64_t bytes_17599 = 4 * binop_x_17602;
    struct memblock_device mem_17603;
    
    mem_17603.references = NULL;
    memblock_alloc_device(ctx, &mem_17603, bytes_17599, "mem_17603");
    
    int call_ret_17869 = futrts_map_transpose_opencl_f32(ctx, mem_17603, 0,
                                                         mem_17598, 0, 1,
                                                         per_chunk_17501 *
                                                         sizze_16261,
                                                         num_threads_16766,
                                                         num_threads_16766 *
                                                         per_chunk_17501 *
                                                         sizze_16261,
                                                         num_threads_16766 *
                                                         per_chunk_17501 *
                                                         sizze_16261);
    
    assert(call_ret_17869 == 0);
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    
    int32_t y_17508 = smod32(sizze_16262, num_threads_16766);
    int32_t x_17509 = num_threads_16766 - y_17508;
    int32_t y_17510 = smod32(x_17509, num_threads_16766);
    int32_t padded_sizze_17511 = sizze_16262 + y_17510;
    int32_t per_chunk_17513 = squot32(padded_sizze_17511, num_threads_16766);
    int64_t binop_x_17605 = sext_i32_i64(y_17510);
    int64_t bytes_17604 = 4 * binop_x_17605;
    struct memblock_device mem_17606;
    
    mem_17606.references = NULL;
    memblock_alloc_device(ctx, &mem_17606, bytes_17604, "mem_17606");
    
    int64_t binop_x_17608 = sext_i32_i64(padded_sizze_17511);
    int64_t bytes_17607 = 4 * binop_x_17608;
    struct memblock_device mem_17609;
    
    mem_17609.references = NULL;
    memblock_alloc_device(ctx, &mem_17609, bytes_17607, "mem_17609");
    
    int32_t tmp_offs_17706 = 0;
    
    if (sizze_16262 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, B_mem_17587.mem,
                                           mem_17609.mem, 0, tmp_offs_17706 * 4,
                                           sizze_16262 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_17706 += sizze_16262;
    if (y_17510 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_17606.mem,
                                           mem_17609.mem, 0, tmp_offs_17706 * 4,
                                           y_17510 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_17706 += y_17510;
    memblock_unref_device(ctx, &mem_17606, "mem_17606");
    
    int32_t convop_x_17611 = num_threads_16766 * per_chunk_17513;
    int64_t binop_x_17612 = sext_i32_i64(convop_x_17611);
    int64_t bytes_17610 = 4 * binop_x_17612;
    struct memblock_device mem_17613;
    
    mem_17613.references = NULL;
    memblock_alloc_device(ctx, &mem_17613, bytes_17610, "mem_17613");
    
    int call_ret_17870 = futrts_map_transpose_opencl_f32(ctx, mem_17613, 0,
                                                         mem_17609, 0, 1,
                                                         per_chunk_17513,
                                                         num_threads_16766,
                                                         num_threads_16766 *
                                                         per_chunk_17513,
                                                         num_threads_16766 *
                                                         per_chunk_17513);
    
    assert(call_ret_17870 == 0);
    memblock_unref_device(ctx, &mem_17609, "mem_17609");
    
    int32_t convop_x_17638 = sizze_16261 * num_groups_16765;
    int64_t binop_x_17639 = sext_i32_i64(convop_x_17638);
    int64_t bytes_17637 = 4 * binop_x_17639;
    struct memblock_device mem_17640;
    
    mem_17640.references = NULL;
    memblock_alloc_device(ctx, &mem_17640, bytes_17637, "mem_17640");
    
    int32_t convop_x_17628 = sizze_16261 * group_sizze_16758;
    int64_t binop_x_17629 = sext_i32_i64(convop_x_17628);
    int64_t bytes_17627 = 4 * binop_x_17629;
    int64_t num_threads64_17672 = sext_i32_i64(num_threads_16766);
    int64_t total_sizze_17673 = bytes_17588 * num_threads64_17672;
    struct memblock_device mem_17621;
    
    mem_17621.references = NULL;
    memblock_alloc_device(ctx, &mem_17621, total_sizze_17673, "mem_17621");
    
    int64_t total_sizze_17674 = bytes_17588 * num_threads64_17672;
    struct memblock_device mem_17624;
    
    mem_17624.references = NULL;
    memblock_alloc_device(ctx, &mem_17624, total_sizze_17674, "mem_17624");
    
    struct memblock_local mem_17630;
    
    mem_17630.references = NULL;
    
    int64_t total_sizze_17675 = bytes_17588 * num_threads64_17672;
    struct memblock_device mem_17633;
    
    mem_17633.references = NULL;
    memblock_alloc_device(ctx, &mem_17633, total_sizze_17675, "mem_17633");
    
    int64_t total_sizze_17676 = bytes_17588 * num_threads64_17672;
    struct memblock_device mem_17636;
    
    mem_17636.references = NULL;
    memblock_alloc_device(ctx, &mem_17636, total_sizze_17676, "mem_17636");
    
    int64_t total_sizze_17677 = bytes_17588 * num_threads64_17672;
    struct memblock_device double_buffer_mem_17669;
    
    double_buffer_mem_17669.references = NULL;
    memblock_alloc_device(ctx, &double_buffer_mem_17669, total_sizze_17677,
                          "double_buffer_mem_17669");
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_16260);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 0,
                                  bytes_17627, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 1,
                                  sizeof(sizze_16260), &sizze_16260));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 2,
                                  sizeof(sizze_16261), &sizze_16261));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 3,
                                  sizeof(num_threads_16766),
                                  &num_threads_16766));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 4,
                                  sizeof(per_thread_elements_16769),
                                  &per_thread_elements_16769));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 5,
                                  sizeof(per_chunk_17513), &per_chunk_17513));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 6,
                                  sizeof(mem_17590.mem), &mem_17590.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 7,
                                  sizeof(binop_x_17600), &binop_x_17600));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 8,
                                  sizeof(mem_17603.mem), &mem_17603.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 9,
                                  sizeof(mem_17613.mem), &mem_17613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 10,
                                  sizeof(mem_17621.mem), &mem_17621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 11,
                                  sizeof(mem_17624.mem), &mem_17624.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 12,
                                  sizeof(mem_17633.mem), &mem_17633.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 13,
                                  sizeof(mem_17636.mem), &mem_17636.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 14,
                                  sizeof(mem_17640.mem), &mem_17640.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_16774, 15,
                                  sizeof(double_buffer_mem_17669.mem),
                                  &double_buffer_mem_17669.mem));
    if (1 * (num_groups_16765 * group_sizze_16758) != 0) {
        const size_t global_work_sizze_17871[1] = {num_groups_16765 *
                     group_sizze_16758};
        const size_t local_work_sizze_17875[1] = {group_sizze_16758};
        int64_t time_start_17872 = 0, time_end_17873 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_16774");
            fprintf(stderr, "%zu", global_work_sizze_17871[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17875[0]);
            fprintf(stderr, "].\n");
            time_start_17872 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_16774,
                                              1, NULL, global_work_sizze_17871,
                                              local_work_sizze_17875, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17873 = get_wall_time();
            
            long time_diff_17874 = time_end_17873 - time_start_17872;
            
            ctx->chunked_reduce_kernel_16774_total_runtime += time_diff_17874;
            ctx->chunked_reduce_kernel_16774_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_16774", time_diff_17874);
        }
    }
    memblock_unref_device(ctx, &mem_17603, "mem_17603");
    memblock_unref_device(ctx, &mem_17613, "mem_17613");
    memblock_unref_device(ctx, &mem_17621, "mem_17621");
    memblock_unref_device(ctx, &mem_17624, "mem_17624");
    memblock_unref_local(ctx, &mem_17630, "mem_17630");
    memblock_unref_device(ctx, &mem_17633, "mem_17633");
    memblock_unref_device(ctx, &mem_17636, "mem_17636");
    memblock_unref_device(ctx, &double_buffer_mem_17669,
                          "double_buffer_mem_17669");
    
    struct memblock_device mem_17659;
    
    mem_17659.references = NULL;
    memblock_alloc_device(ctx, &mem_17659, bytes_17588, "mem_17659");
    
    int32_t convop_x_17648 = sizze_16261 * max_num_groups_16760;
    int64_t binop_x_17649 = sext_i32_i64(convop_x_17648);
    int64_t bytes_17647 = 4 * binop_x_17649;
    int64_t num_threads64_17678 = sext_i32_i64(max_num_groups_16760);
    struct memblock_local mem_17650;
    
    mem_17650.references = NULL;
    
    int64_t total_sizze_17679 = bytes_17588 * num_threads64_17678;
    struct memblock_device mem_17653;
    
    mem_17653.references = NULL;
    memblock_alloc_device(ctx, &mem_17653, total_sizze_17679, "mem_17653");
    
    int64_t total_sizze_17680 = bytes_17588 * num_threads64_17678;
    struct memblock_device mem_17656;
    
    mem_17656.references = NULL;
    memblock_alloc_device(ctx, &mem_17656, total_sizze_17680, "mem_17656");
    
    int64_t total_sizze_17681 = bytes_17588 * num_threads64_17678;
    struct memblock_device mem_17666;
    
    mem_17666.references = NULL;
    memblock_alloc_device(ctx, &mem_17666, total_sizze_17681, "mem_17666");
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 0, bytes_17647,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 1,
                                  sizeof(sizze_16261), &sizze_16261));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 2,
                                  sizeof(num_groups_16765), &num_groups_16765));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 3,
                                  sizeof(mem_17590.mem), &mem_17590.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 4,
                                  sizeof(mem_17640.mem), &mem_17640.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 5,
                                  sizeof(mem_17653.mem), &mem_17653.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 6,
                                  sizeof(mem_17656.mem), &mem_17656.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 7,
                                  sizeof(mem_17659.mem), &mem_17659.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_16829, 8,
                                  sizeof(mem_17666.mem), &mem_17666.mem));
    if (1 * max_num_groups_16760 != 0) {
        const size_t global_work_sizze_17876[1] = {max_num_groups_16760};
        const size_t local_work_sizze_17880[1] = {max_num_groups_16760};
        int64_t time_start_17877 = 0, time_end_17878 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_16829");
            fprintf(stderr, "%zu", global_work_sizze_17876[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17880[0]);
            fprintf(stderr, "].\n");
            time_start_17877 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_16829, 1, NULL,
                                              global_work_sizze_17876,
                                              local_work_sizze_17880, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17878 = get_wall_time();
            
            long time_diff_17879 = time_end_17878 - time_start_17877;
            
            ctx->reduce_kernel_16829_total_runtime += time_diff_17879;
            ctx->reduce_kernel_16829_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_16829",
                    time_diff_17879);
        }
    }
    memblock_unref_device(ctx, &mem_17590, "mem_17590");
    memblock_unref_device(ctx, &mem_17640, "mem_17640");
    memblock_unref_local(ctx, &mem_17650, "mem_17650");
    memblock_unref_device(ctx, &mem_17653, "mem_17653");
    memblock_unref_device(ctx, &mem_17656, "mem_17656");
    memblock_unref_device(ctx, &mem_17666, "mem_17666");
    
    struct memblock_device mem_17662;
    
    mem_17662.references = NULL;
    memblock_alloc_device(ctx, &mem_17662, bytes_17588, "mem_17662");
    if (sizze_16261 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_17659.mem,
                                           mem_17662.mem, 0, 0, sizze_16261 *
                                           sizeof(float), 0, NULL, NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    memblock_unref_device(ctx, &mem_17659, "mem_17659");
    out_arrsizze_17700 = sizze_16261;
    out_memsizze_17699 = bytes_17588;
    memblock_set_device(ctx, &out_mem_17698, &mem_17662, "mem_17662");
    *out_out_memsizze_17861 = out_memsizze_17699;
    (*out_mem_p_17862).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17862, &out_mem_17698,
                        "out_mem_17698");
    *out_out_arrsizze_17863 = out_arrsizze_17700;
    memblock_unref_device(ctx, &mem_17662, "mem_17662");
    memblock_unref_device(ctx, &mem_17666, "mem_17666");
    memblock_unref_device(ctx, &mem_17656, "mem_17656");
    memblock_unref_device(ctx, &mem_17653, "mem_17653");
    memblock_unref_local(ctx, &mem_17650, "mem_17650");
    memblock_unref_device(ctx, &mem_17659, "mem_17659");
    memblock_unref_device(ctx, &double_buffer_mem_17669,
                          "double_buffer_mem_17669");
    memblock_unref_device(ctx, &mem_17636, "mem_17636");
    memblock_unref_device(ctx, &mem_17633, "mem_17633");
    memblock_unref_local(ctx, &mem_17630, "mem_17630");
    memblock_unref_device(ctx, &mem_17624, "mem_17624");
    memblock_unref_device(ctx, &mem_17621, "mem_17621");
    memblock_unref_device(ctx, &mem_17640, "mem_17640");
    memblock_unref_device(ctx, &mem_17613, "mem_17613");
    memblock_unref_device(ctx, &mem_17609, "mem_17609");
    memblock_unref_device(ctx, &mem_17606, "mem_17606");
    memblock_unref_device(ctx, &mem_17603, "mem_17603");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17590, "mem_17590");
    memblock_unref_device(ctx, &out_mem_17698, "out_mem_17698");
    return 0;
}
static int futrts_t1(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17881,
                     struct memblock_device *out_mem_p_17882,
                     int32_t *out_out_arrsizze_17883, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587, int32_t sizze_16285,
                     int32_t sizze_16286, int32_t sizze_16287)
{
    int64_t out_memsizze_17745;
    struct memblock_device out_mem_17744;
    
    out_mem_17744.references = NULL;
    
    int32_t out_arrsizze_17746;
    bool dim_zzero_16290 = 0 == sizze_16287;
    bool dim_zzero_16291 = 0 == sizze_16286;
    bool both_empty_16292 = dim_zzero_16290 && dim_zzero_16291;
    bool dim_match_16293 = sizze_16286 == sizze_16287;
    bool empty_or_match_16294 = both_empty_16292 || dim_match_16293;
    bool empty_or_match_cert_16295;
    
    if (!empty_or_match_16294) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:6:1-7:47",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17744, "out_mem_17744");
        return 1;
    }
    
    int32_t group_sizze_16851;
    
    group_sizze_16851 = ctx->sizes.group_sizze_16850;
    
    int32_t y_16852 = group_sizze_16851 - 1;
    int32_t x_16853 = sizze_16285 + y_16852;
    int32_t num_groups_16854 = squot32(x_16853, group_sizze_16851);
    int32_t num_threads_16855 = group_sizze_16851 * num_groups_16854;
    int32_t convop_x_17589 = sizze_16285 * sizze_16286;
    int64_t binop_x_17590 = sext_i32_i64(convop_x_17589);
    int64_t bytes_17588 = 4 * binop_x_17590;
    struct memblock_device mem_17591;
    
    mem_17591.references = NULL;
    memblock_alloc_device(ctx, &mem_17591, bytes_17588, "mem_17591");
    
    int call_ret_17884 = futrts_map_transpose_opencl_f32(ctx, mem_17591, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16286,
                                                         sizze_16285,
                                                         sizze_16285 *
                                                         sizze_16286,
                                                         sizze_16285 *
                                                         sizze_16286);
    
    assert(call_ret_17884 == 0);
    
    int64_t binop_x_17593 = sext_i32_i64(sizze_16285);
    int64_t bytes_17592 = 4 * binop_x_17593;
    struct memblock_device mem_17594;
    
    mem_17594.references = NULL;
    memblock_alloc_device(ctx, &mem_17594, bytes_17592, "mem_17594");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16856, 0, sizeof(sizze_16285),
                                  &sizze_16285));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16856, 1, sizeof(sizze_16286),
                                  &sizze_16286));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16856, 2,
                                  sizeof(B_mem_17587.mem), &B_mem_17587.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16856, 3,
                                  sizeof(mem_17591.mem), &mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16856, 4,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    if (1 * (num_groups_16854 * group_sizze_16851) != 0) {
        const size_t global_work_sizze_17885[1] = {num_groups_16854 *
                     group_sizze_16851};
        const size_t local_work_sizze_17889[1] = {group_sizze_16851};
        int64_t time_start_17886 = 0, time_end_17887 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_16856");
            fprintf(stderr, "%zu", global_work_sizze_17885[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17889[0]);
            fprintf(stderr, "].\n");
            time_start_17886 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_16856, 1, NULL,
                                              global_work_sizze_17885,
                                              local_work_sizze_17889, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17887 = get_wall_time();
            
            long time_diff_17888 = time_end_17887 - time_start_17886;
            
            ctx->map_kernel_16856_total_runtime += time_diff_17888;
            ctx->map_kernel_16856_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_16856",
                    time_diff_17888);
        }
    }
    memblock_unref_device(ctx, &mem_17591, "mem_17591");
    out_arrsizze_17746 = sizze_16285;
    out_memsizze_17745 = bytes_17592;
    memblock_set_device(ctx, &out_mem_17744, &mem_17594, "mem_17594");
    *out_out_memsizze_17881 = out_memsizze_17745;
    (*out_mem_p_17882).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17882, &out_mem_17744,
                        "out_mem_17744");
    *out_out_arrsizze_17883 = out_arrsizze_17746;
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17591, "mem_17591");
    memblock_unref_device(ctx, &out_mem_17744, "out_mem_17744");
    return 0;
}
static int futrts_t2(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17890,
                     struct memblock_device *out_mem_p_17891,
                     int32_t *out_out_arrsizze_17892, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589, int32_t sizze_16306,
                     int32_t sizze_16307, int32_t sizze_16308,
                     int32_t sizze_16309)
{
    int64_t out_memsizze_17752;
    struct memblock_device out_mem_17751;
    
    out_mem_17751.references = NULL;
    
    int32_t out_arrsizze_17753;
    bool dim_zzero_16313 = 0 == sizze_16308;
    bool dim_zzero_16314 = 0 == sizze_16307;
    bool both_empty_16315 = dim_zzero_16313 && dim_zzero_16314;
    bool dim_match_16316 = sizze_16307 == sizze_16308;
    bool empty_or_match_16317 = both_empty_16315 || dim_match_16316;
    bool empty_or_match_cert_16318;
    
    if (!empty_or_match_16317) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:9:1-10:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17751, "out_mem_17751");
        return 1;
    }
    
    bool dim_zzero_16319 = 0 == sizze_16309;
    bool dim_zzero_16320 = 0 == sizze_16306;
    bool both_empty_16321 = dim_zzero_16319 && dim_zzero_16320;
    bool dim_match_16322 = sizze_16306 == sizze_16309;
    bool empty_or_match_16323 = both_empty_16321 || dim_match_16322;
    bool empty_or_match_cert_16324;
    
    if (!empty_or_match_16323) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:9:1-10:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17751, "out_mem_17751");
        return 1;
    }
    
    int32_t group_sizze_16891;
    
    group_sizze_16891 = ctx->sizes.group_sizze_16890;
    
    int32_t y_16892 = group_sizze_16891 - 1;
    int32_t x_16893 = sizze_16306 + y_16892;
    int32_t num_groups_16894 = squot32(x_16893, group_sizze_16891);
    int32_t num_threads_16895 = group_sizze_16891 * num_groups_16894;
    int32_t convop_x_17591 = sizze_16306 * sizze_16307;
    int64_t binop_x_17592 = sext_i32_i64(convop_x_17591);
    int64_t bytes_17590 = 4 * binop_x_17592;
    struct memblock_device mem_17593;
    
    mem_17593.references = NULL;
    memblock_alloc_device(ctx, &mem_17593, bytes_17590, "mem_17593");
    
    int call_ret_17893 = futrts_map_transpose_opencl_f32(ctx, mem_17593, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16307,
                                                         sizze_16306,
                                                         sizze_16306 *
                                                         sizze_16307,
                                                         sizze_16306 *
                                                         sizze_16307);
    
    assert(call_ret_17893 == 0);
    
    int64_t binop_x_17595 = sext_i32_i64(sizze_16306);
    int64_t bytes_17594 = 4 * binop_x_17595;
    struct memblock_device mem_17596;
    
    mem_17596.references = NULL;
    memblock_alloc_device(ctx, &mem_17596, bytes_17594, "mem_17596");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 0, sizeof(sizze_16306),
                                  &sizze_16306));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 1, sizeof(sizze_16307),
                                  &sizze_16307));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 2,
                                  sizeof(B_mem_17587.mem), &B_mem_17587.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 3,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 4,
                                  sizeof(mem_17593.mem), &mem_17593.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16896, 5,
                                  sizeof(mem_17596.mem), &mem_17596.mem));
    if (1 * (num_groups_16894 * group_sizze_16891) != 0) {
        const size_t global_work_sizze_17894[1] = {num_groups_16894 *
                     group_sizze_16891};
        const size_t local_work_sizze_17898[1] = {group_sizze_16891};
        int64_t time_start_17895 = 0, time_end_17896 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_16896");
            fprintf(stderr, "%zu", global_work_sizze_17894[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17898[0]);
            fprintf(stderr, "].\n");
            time_start_17895 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_16896, 1, NULL,
                                              global_work_sizze_17894,
                                              local_work_sizze_17898, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17896 = get_wall_time();
            
            long time_diff_17897 = time_end_17896 - time_start_17895;
            
            ctx->map_kernel_16896_total_runtime += time_diff_17897;
            ctx->map_kernel_16896_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_16896",
                    time_diff_17897);
        }
    }
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    out_arrsizze_17753 = sizze_16306;
    out_memsizze_17752 = bytes_17594;
    memblock_set_device(ctx, &out_mem_17751, &mem_17596, "mem_17596");
    *out_out_memsizze_17890 = out_memsizze_17752;
    (*out_mem_p_17891).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17891, &out_mem_17751,
                        "out_mem_17751");
    *out_out_arrsizze_17892 = out_arrsizze_17753;
    memblock_unref_device(ctx, &mem_17596, "mem_17596");
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    memblock_unref_device(ctx, &out_mem_17751, "out_mem_17751");
    return 0;
}
static int futrts_t3(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17899,
                     struct memblock_device *out_mem_p_17900,
                     int32_t *out_out_arrsizze_17901, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589, int32_t sizze_16338,
                     int32_t sizze_16339, int32_t sizze_16340,
                     int32_t sizze_16341, int32_t sizze_16342)
{
    int64_t out_memsizze_17759;
    struct memblock_device out_mem_17758;
    
    out_mem_17758.references = NULL;
    
    int32_t out_arrsizze_17760;
    bool dim_zzero_16346 = 0 == sizze_16340;
    bool dim_zzero_16347 = 0 == sizze_16341;
    bool old_empty_16348 = dim_zzero_16346 || dim_zzero_16347;
    bool dim_zzero_16349 = 0 == sizze_16338;
    bool dim_zzero_16350 = 0 == sizze_16339;
    bool new_empty_16351 = dim_zzero_16349 || dim_zzero_16350;
    bool both_empty_16352 = old_empty_16348 && new_empty_16351;
    bool dim_match_16353 = sizze_16338 == sizze_16340;
    bool dim_match_16354 = sizze_16339 == sizze_16341;
    bool match_16355 = dim_match_16353 && dim_match_16354;
    bool empty_or_match_16356 = both_empty_16352 || match_16355;
    bool empty_or_match_cert_16357;
    
    if (!empty_or_match_16356) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:13:1-14:67",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17758, "out_mem_17758");
        return 1;
    }
    
    bool dim_zzero_16359 = 0 == sizze_16342;
    bool both_empty_16360 = dim_zzero_16350 && dim_zzero_16359;
    bool dim_match_16361 = sizze_16339 == sizze_16342;
    bool empty_or_match_16362 = both_empty_16360 || dim_match_16361;
    bool empty_or_match_cert_16363;
    
    if (!empty_or_match_16362) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:13:1-14:67",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17758, "out_mem_17758");
        return 1;
    }
    
    int32_t group_sizze_16935;
    
    group_sizze_16935 = ctx->sizes.group_sizze_16934;
    
    int32_t y_16936 = group_sizze_16935 - 1;
    int32_t x_16937 = sizze_16338 + y_16936;
    int32_t num_groups_16938 = squot32(x_16937, group_sizze_16935);
    int32_t num_threads_16939 = group_sizze_16935 * num_groups_16938;
    int32_t convop_x_17591 = sizze_16338 * sizze_16339;
    int64_t binop_x_17592 = sext_i32_i64(convop_x_17591);
    int64_t bytes_17590 = 4 * binop_x_17592;
    struct memblock_device mem_17593;
    
    mem_17593.references = NULL;
    memblock_alloc_device(ctx, &mem_17593, bytes_17590, "mem_17593");
    
    int call_ret_17902 = futrts_map_transpose_opencl_f32(ctx, mem_17593, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16339,
                                                         sizze_16338,
                                                         sizze_16338 *
                                                         sizze_16339,
                                                         sizze_16338 *
                                                         sizze_16339);
    
    assert(call_ret_17902 == 0);
    
    int32_t convop_x_17595 = sizze_16340 * sizze_16341;
    int64_t binop_x_17596 = sext_i32_i64(convop_x_17595);
    int64_t bytes_17594 = 4 * binop_x_17596;
    struct memblock_device mem_17597;
    
    mem_17597.references = NULL;
    memblock_alloc_device(ctx, &mem_17597, bytes_17594, "mem_17597");
    
    int call_ret_17903 = futrts_map_transpose_opencl_f32(ctx, mem_17597, 0,
                                                         B_mem_17587, 0, 1,
                                                         sizze_16341,
                                                         sizze_16340,
                                                         sizze_16340 *
                                                         sizze_16341,
                                                         sizze_16340 *
                                                         sizze_16341);
    
    assert(call_ret_17903 == 0);
    
    int64_t binop_x_17599 = sext_i32_i64(sizze_16338);
    int64_t bytes_17598 = 4 * binop_x_17599;
    struct memblock_device mem_17600;
    
    mem_17600.references = NULL;
    memblock_alloc_device(ctx, &mem_17600, bytes_17598, "mem_17600");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 0, sizeof(sizze_16338),
                                  &sizze_16338));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 1, sizeof(sizze_16339),
                                  &sizze_16339));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 2, sizeof(sizze_16340),
                                  &sizze_16340));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 3,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 4,
                                  sizeof(mem_17593.mem), &mem_17593.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 5,
                                  sizeof(mem_17597.mem), &mem_17597.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16940, 6,
                                  sizeof(mem_17600.mem), &mem_17600.mem));
    if (1 * (num_groups_16938 * group_sizze_16935) != 0) {
        const size_t global_work_sizze_17904[1] = {num_groups_16938 *
                     group_sizze_16935};
        const size_t local_work_sizze_17908[1] = {group_sizze_16935};
        int64_t time_start_17905 = 0, time_end_17906 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_16940");
            fprintf(stderr, "%zu", global_work_sizze_17904[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17908[0]);
            fprintf(stderr, "].\n");
            time_start_17905 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_16940, 1, NULL,
                                              global_work_sizze_17904,
                                              local_work_sizze_17908, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17906 = get_wall_time();
            
            long time_diff_17907 = time_end_17906 - time_start_17905;
            
            ctx->map_kernel_16940_total_runtime += time_diff_17907;
            ctx->map_kernel_16940_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_16940",
                    time_diff_17907);
        }
    }
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    memblock_unref_device(ctx, &mem_17597, "mem_17597");
    out_arrsizze_17760 = sizze_16338;
    out_memsizze_17759 = bytes_17598;
    memblock_set_device(ctx, &out_mem_17758, &mem_17600, "mem_17600");
    *out_out_memsizze_17899 = out_memsizze_17759;
    (*out_mem_p_17900).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17900, &out_mem_17758,
                        "out_mem_17758");
    *out_out_arrsizze_17901 = out_arrsizze_17760;
    memblock_unref_device(ctx, &mem_17600, "mem_17600");
    memblock_unref_device(ctx, &mem_17597, "mem_17597");
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    memblock_unref_device(ctx, &out_mem_17758, "out_mem_17758");
    return 0;
}
static int futrts_t4(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17909,
                     struct memblock_device *out_mem_p_17910,
                     int32_t *out_out_arrsizze_17911, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16377,
                     int32_t sizze_16378, int32_t sizze_16379,
                     int32_t sizze_16380, int32_t sizze_16381,
                     int32_t sizze_16382)
{
    int64_t out_memsizze_17766;
    struct memblock_device out_mem_17765;
    
    out_mem_17765.references = NULL;
    
    int32_t out_arrsizze_17767;
    bool dim_zzero_16387 = 0 == sizze_16379;
    bool dim_zzero_16388 = 0 == sizze_16380;
    bool old_empty_16389 = dim_zzero_16387 || dim_zzero_16388;
    bool dim_zzero_16390 = 0 == sizze_16377;
    bool dim_zzero_16391 = 0 == sizze_16378;
    bool new_empty_16392 = dim_zzero_16390 || dim_zzero_16391;
    bool both_empty_16393 = old_empty_16389 && new_empty_16392;
    bool dim_match_16394 = sizze_16377 == sizze_16379;
    bool dim_match_16395 = sizze_16378 == sizze_16380;
    bool match_16396 = dim_match_16394 && dim_match_16395;
    bool empty_or_match_16397 = both_empty_16393 || match_16396;
    bool empty_or_match_cert_16398;
    
    if (!empty_or_match_16397) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:16:1-21:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17765, "out_mem_17765");
        return 1;
    }
    
    bool dim_zzero_16400 = 0 == sizze_16381;
    bool both_empty_16401 = dim_zzero_16391 && dim_zzero_16400;
    bool dim_match_16402 = sizze_16378 == sizze_16381;
    bool empty_or_match_16403 = both_empty_16401 || dim_match_16402;
    bool empty_or_match_cert_16404;
    
    if (!empty_or_match_16403) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:16:1-21:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17765, "out_mem_17765");
        return 1;
    }
    
    bool dim_zzero_16406 = 0 == sizze_16382;
    bool both_empty_16407 = dim_zzero_16391 && dim_zzero_16406;
    bool dim_match_16408 = sizze_16378 == sizze_16382;
    bool empty_or_match_16409 = both_empty_16407 || dim_match_16408;
    bool empty_or_match_cert_16410;
    
    if (!empty_or_match_16409) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:16:1-21:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17765, "out_mem_17765");
        return 1;
    }
    
    int32_t group_sizze_16969;
    
    group_sizze_16969 = ctx->sizes.group_sizze_16968;
    
    int32_t y_16970 = group_sizze_16969 - 1;
    int32_t x_16971 = sizze_16378 + y_16970;
    int32_t num_groups_16972 = squot32(x_16971, group_sizze_16969);
    int32_t num_threads_16973 = group_sizze_16969 * num_groups_16972;
    int64_t binop_x_17593 = sext_i32_i64(sizze_16378);
    int64_t bytes_17592 = 4 * binop_x_17593;
    struct memblock_device mem_17594;
    
    mem_17594.references = NULL;
    memblock_alloc_device(ctx, &mem_17594, bytes_17592, "mem_17594");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16974, 0, sizeof(sizze_16378),
                                  &sizze_16378));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16974, 1,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16974, 2,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_16974, 3,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    if (1 * (num_groups_16972 * group_sizze_16969) != 0) {
        const size_t global_work_sizze_17912[1] = {num_groups_16972 *
                     group_sizze_16969};
        const size_t local_work_sizze_17916[1] = {group_sizze_16969};
        int64_t time_start_17913 = 0, time_end_17914 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_16974");
            fprintf(stderr, "%zu", global_work_sizze_17912[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17916[0]);
            fprintf(stderr, "].\n");
            time_start_17913 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_16974, 1, NULL,
                                              global_work_sizze_17912,
                                              local_work_sizze_17916, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17914 = get_wall_time();
            
            long time_diff_17915 = time_end_17914 - time_start_17913;
            
            ctx->map_kernel_16974_total_runtime += time_diff_17915;
            ctx->map_kernel_16974_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_16974",
                    time_diff_17915);
        }
    }
    
    int32_t group_sizze_16995;
    
    group_sizze_16995 = ctx->sizes.group_sizze_16994;
    
    int32_t y_16996 = group_sizze_16995 - 1;
    int32_t x_16997 = sizze_16377 + y_16996;
    int32_t num_groups_16998 = squot32(x_16997, group_sizze_16995);
    int32_t num_threads_16999 = group_sizze_16995 * num_groups_16998;
    int32_t convop_x_17596 = sizze_16377 * sizze_16378;
    int64_t binop_x_17597 = sext_i32_i64(convop_x_17596);
    int64_t bytes_17595 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17595, "mem_17598");
    
    int call_ret_17917 = futrts_map_transpose_opencl_f32(ctx, mem_17598, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16378,
                                                         sizze_16377,
                                                         sizze_16377 *
                                                         sizze_16378,
                                                         sizze_16377 *
                                                         sizze_16378);
    
    assert(call_ret_17917 == 0);
    
    int32_t convop_x_17600 = sizze_16379 * sizze_16380;
    int64_t binop_x_17601 = sext_i32_i64(convop_x_17600);
    int64_t bytes_17599 = 4 * binop_x_17601;
    struct memblock_device mem_17602;
    
    mem_17602.references = NULL;
    memblock_alloc_device(ctx, &mem_17602, bytes_17599, "mem_17602");
    
    int call_ret_17918 = futrts_map_transpose_opencl_f32(ctx, mem_17602, 0,
                                                         B_mem_17587, 0, 1,
                                                         sizze_16380,
                                                         sizze_16379,
                                                         sizze_16379 *
                                                         sizze_16380,
                                                         sizze_16379 *
                                                         sizze_16380);
    
    assert(call_ret_17918 == 0);
    
    int64_t binop_x_17604 = sext_i32_i64(sizze_16377);
    int64_t bytes_17603 = 4 * binop_x_17604;
    struct memblock_device mem_17605;
    
    mem_17605.references = NULL;
    memblock_alloc_device(ctx, &mem_17605, bytes_17603, "mem_17605");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 0, sizeof(sizze_16377),
                                  &sizze_16377));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 1, sizeof(sizze_16378),
                                  &sizze_16378));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 2, sizeof(sizze_16379),
                                  &sizze_16379));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 3,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 4,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 5,
                                  sizeof(mem_17602.mem), &mem_17602.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17000, 6,
                                  sizeof(mem_17605.mem), &mem_17605.mem));
    if (1 * (num_groups_16998 * group_sizze_16995) != 0) {
        const size_t global_work_sizze_17919[1] = {num_groups_16998 *
                     group_sizze_16995};
        const size_t local_work_sizze_17923[1] = {group_sizze_16995};
        int64_t time_start_17920 = 0, time_end_17921 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17000");
            fprintf(stderr, "%zu", global_work_sizze_17919[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17923[0]);
            fprintf(stderr, "].\n");
            time_start_17920 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17000, 1, NULL,
                                              global_work_sizze_17919,
                                              local_work_sizze_17923, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17921 = get_wall_time();
            
            long time_diff_17922 = time_end_17921 - time_start_17920;
            
            ctx->map_kernel_17000_total_runtime += time_diff_17922;
            ctx->map_kernel_17000_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17000",
                    time_diff_17922);
        }
    }
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    out_arrsizze_17767 = sizze_16377;
    out_memsizze_17766 = bytes_17603;
    memblock_set_device(ctx, &out_mem_17765, &mem_17605, "mem_17605");
    *out_out_memsizze_17909 = out_memsizze_17766;
    (*out_mem_p_17910).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17910, &out_mem_17765,
                        "out_mem_17765");
    *out_out_arrsizze_17911 = out_arrsizze_17767;
    memblock_unref_device(ctx, &mem_17605, "mem_17605");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &out_mem_17765, "out_mem_17765");
    return 0;
}
static int futrts_t5(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17924,
                     struct memblock_device *out_mem_p_17925,
                     int32_t *out_out_arrsizze_17926, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16428,
                     int32_t sizze_16429, int32_t sizze_16430,
                     int32_t sizze_16431, int32_t sizze_16432,
                     int32_t sizze_16433, float a_16434, float b_16436,
                     float c_16438, float d_16440)
{
    int64_t out_memsizze_17776;
    struct memblock_device out_mem_17775;
    
    out_mem_17775.references = NULL;
    
    int32_t out_arrsizze_17777;
    bool dim_zzero_16442 = 0 == sizze_16430;
    bool dim_zzero_16443 = 0 == sizze_16431;
    bool old_empty_16444 = dim_zzero_16442 || dim_zzero_16443;
    bool dim_zzero_16445 = 0 == sizze_16428;
    bool dim_zzero_16446 = 0 == sizze_16429;
    bool new_empty_16447 = dim_zzero_16445 || dim_zzero_16446;
    bool both_empty_16448 = old_empty_16444 && new_empty_16447;
    bool dim_match_16449 = sizze_16428 == sizze_16430;
    bool dim_match_16450 = sizze_16429 == sizze_16431;
    bool match_16451 = dim_match_16449 && dim_match_16450;
    bool empty_or_match_16452 = both_empty_16448 || match_16451;
    bool empty_or_match_cert_16453;
    
    if (!empty_or_match_16452) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:23:1-28:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17775, "out_mem_17775");
        return 1;
    }
    
    bool dim_zzero_16455 = 0 == sizze_16432;
    bool both_empty_16456 = dim_zzero_16446 && dim_zzero_16455;
    bool dim_match_16457 = sizze_16429 == sizze_16432;
    bool empty_or_match_16458 = both_empty_16456 || dim_match_16457;
    bool empty_or_match_cert_16459;
    
    if (!empty_or_match_16458) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:23:1-28:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17775, "out_mem_17775");
        return 1;
    }
    
    bool dim_zzero_16460 = 0 == sizze_16433;
    bool both_empty_16461 = dim_zzero_16446 && dim_zzero_16460;
    bool dim_match_16462 = sizze_16429 == sizze_16433;
    bool empty_or_match_16463 = both_empty_16461 || dim_match_16462;
    bool empty_or_match_cert_16464;
    
    if (!empty_or_match_16463) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:23:1-28:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17775, "out_mem_17775");
        return 1;
    }
    
    int32_t group_sizze_17029;
    
    group_sizze_17029 = ctx->sizes.group_sizze_17028;
    
    int32_t y_17030 = group_sizze_17029 - 1;
    int32_t x_17031 = sizze_16429 + y_17030;
    int32_t num_groups_17032 = squot32(x_17031, group_sizze_17029);
    int32_t num_threads_17033 = group_sizze_17029 * num_groups_17032;
    int64_t binop_x_17593 = sext_i32_i64(sizze_16429);
    int64_t bytes_17592 = 4 * binop_x_17593;
    struct memblock_device mem_17594;
    
    mem_17594.references = NULL;
    memblock_alloc_device(ctx, &mem_17594, bytes_17592, "mem_17594");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 0, sizeof(sizze_16429),
                                  &sizze_16429));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 1, sizeof(c_16438),
                                  &c_16438));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 2, sizeof(d_16440),
                                  &d_16440));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 3,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 4,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17034, 5,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    if (1 * (num_groups_17032 * group_sizze_17029) != 0) {
        const size_t global_work_sizze_17927[1] = {num_groups_17032 *
                     group_sizze_17029};
        const size_t local_work_sizze_17931[1] = {group_sizze_17029};
        int64_t time_start_17928 = 0, time_end_17929 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17034");
            fprintf(stderr, "%zu", global_work_sizze_17927[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17931[0]);
            fprintf(stderr, "].\n");
            time_start_17928 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17034, 1, NULL,
                                              global_work_sizze_17927,
                                              local_work_sizze_17931, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17929 = get_wall_time();
            
            long time_diff_17930 = time_end_17929 - time_start_17928;
            
            ctx->map_kernel_17034_total_runtime += time_diff_17930;
            ctx->map_kernel_17034_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17034",
                    time_diff_17930);
        }
    }
    
    int32_t group_sizze_17057;
    
    group_sizze_17057 = ctx->sizes.group_sizze_17056;
    
    int32_t y_17058 = group_sizze_17057 - 1;
    int32_t x_17059 = sizze_16428 + y_17058;
    int32_t num_groups_17060 = squot32(x_17059, group_sizze_17057);
    int32_t num_threads_17061 = group_sizze_17057 * num_groups_17060;
    int32_t convop_x_17596 = sizze_16428 * sizze_16429;
    int64_t binop_x_17597 = sext_i32_i64(convop_x_17596);
    int64_t bytes_17595 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17595, "mem_17598");
    
    int call_ret_17932 = futrts_map_transpose_opencl_f32(ctx, mem_17598, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16429,
                                                         sizze_16428,
                                                         sizze_16428 *
                                                         sizze_16429,
                                                         sizze_16428 *
                                                         sizze_16429);
    
    assert(call_ret_17932 == 0);
    
    int32_t convop_x_17600 = sizze_16430 * sizze_16431;
    int64_t binop_x_17601 = sext_i32_i64(convop_x_17600);
    int64_t bytes_17599 = 4 * binop_x_17601;
    struct memblock_device mem_17602;
    
    mem_17602.references = NULL;
    memblock_alloc_device(ctx, &mem_17602, bytes_17599, "mem_17602");
    
    int call_ret_17933 = futrts_map_transpose_opencl_f32(ctx, mem_17602, 0,
                                                         B_mem_17587, 0, 1,
                                                         sizze_16431,
                                                         sizze_16430,
                                                         sizze_16430 *
                                                         sizze_16431,
                                                         sizze_16430 *
                                                         sizze_16431);
    
    assert(call_ret_17933 == 0);
    
    int64_t binop_x_17604 = sext_i32_i64(sizze_16428);
    int64_t bytes_17603 = 4 * binop_x_17604;
    struct memblock_device mem_17605;
    
    mem_17605.references = NULL;
    memblock_alloc_device(ctx, &mem_17605, bytes_17603, "mem_17605");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 0, sizeof(sizze_16428),
                                  &sizze_16428));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 1, sizeof(sizze_16429),
                                  &sizze_16429));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 2, sizeof(sizze_16430),
                                  &sizze_16430));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 3, sizeof(a_16434),
                                  &a_16434));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 4, sizeof(b_16436),
                                  &b_16436));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 5,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 6,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 7,
                                  sizeof(mem_17602.mem), &mem_17602.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17062, 8,
                                  sizeof(mem_17605.mem), &mem_17605.mem));
    if (1 * (num_groups_17060 * group_sizze_17057) != 0) {
        const size_t global_work_sizze_17934[1] = {num_groups_17060 *
                     group_sizze_17057};
        const size_t local_work_sizze_17938[1] = {group_sizze_17057};
        int64_t time_start_17935 = 0, time_end_17936 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17062");
            fprintf(stderr, "%zu", global_work_sizze_17934[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17938[0]);
            fprintf(stderr, "].\n");
            time_start_17935 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17062, 1, NULL,
                                              global_work_sizze_17934,
                                              local_work_sizze_17938, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17936 = get_wall_time();
            
            long time_diff_17937 = time_end_17936 - time_start_17935;
            
            ctx->map_kernel_17062_total_runtime += time_diff_17937;
            ctx->map_kernel_17062_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17062",
                    time_diff_17937);
        }
    }
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    out_arrsizze_17777 = sizze_16428;
    out_memsizze_17776 = bytes_17603;
    memblock_set_device(ctx, &out_mem_17775, &mem_17605, "mem_17605");
    *out_out_memsizze_17924 = out_memsizze_17776;
    (*out_mem_p_17925).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17925, &out_mem_17775,
                        "out_mem_17775");
    *out_out_arrsizze_17926 = out_arrsizze_17777;
    memblock_unref_device(ctx, &mem_17605, "mem_17605");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &out_mem_17775, "out_mem_17775");
    return 0;
}
static int futrts_t6(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17939,
                     struct memblock_device *out_mem_p_17940,
                     int32_t *out_out_arrsizze_17941, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16487,
                     int32_t sizze_16488, int32_t sizze_16489,
                     int32_t sizze_16490)
{
    int64_t out_memsizze_17786;
    struct memblock_device out_mem_17785;
    
    out_mem_17785.references = NULL;
    
    int32_t out_arrsizze_17787;
    bool dim_zzero_16495 = 0 == sizze_16489;
    bool dim_zzero_16496 = 0 == sizze_16487;
    bool both_empty_16497 = dim_zzero_16495 && dim_zzero_16496;
    bool dim_match_16498 = sizze_16487 == sizze_16489;
    bool empty_or_match_16499 = both_empty_16497 || dim_match_16498;
    bool empty_or_match_cert_16500;
    
    if (!empty_or_match_16499) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:30:1-31:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17785, "out_mem_17785");
        return 1;
    }
    
    bool dim_zzero_16502 = 0 == sizze_16490;
    bool dim_zzero_16503 = 0 == sizze_16488;
    bool both_empty_16504 = dim_zzero_16502 && dim_zzero_16503;
    bool dim_match_16505 = sizze_16488 == sizze_16490;
    bool empty_or_match_16506 = both_empty_16504 || dim_match_16505;
    bool empty_or_match_cert_16507;
    
    if (!empty_or_match_16506) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:30:1-31:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17785, "out_mem_17785");
        return 1;
    }
    
    int32_t group_sizze_17104;
    
    group_sizze_17104 = ctx->sizes.group_sizze_17103;
    
    int32_t y_17105 = group_sizze_17104 - 1;
    int32_t x_17106 = sizze_16487 + y_17105;
    int32_t num_groups_17107 = squot32(x_17106, group_sizze_17104);
    int32_t num_threads_17108 = group_sizze_17104 * num_groups_17107;
    int64_t binop_x_17599 = sext_i32_i64(sizze_16487);
    int64_t bytes_17598 = 4 * binop_x_17599;
    struct memblock_device mem_17600;
    
    mem_17600.references = NULL;
    memblock_alloc_device(ctx, &mem_17600, bytes_17598, "mem_17600");
    
    int64_t binop_x_17593 = sext_i32_i64(group_sizze_17104);
    int64_t bytes_17592 = 4 * binop_x_17593;
    struct memblock_local mem_17594;
    
    mem_17594.references = NULL;
    
    struct memblock_local mem_17597;
    
    mem_17597.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 0, bytes_17592, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 1, bytes_17592, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 2, sizeof(sizze_16487),
                                  &sizze_16487));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 3, sizeof(sizze_16488),
                                  &sizze_16488));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 4,
                                  sizeof(A_mem_17585.mem), &A_mem_17585.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 5,
                                  sizeof(B_mem_17587.mem), &B_mem_17587.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 6,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 7,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17109, 8,
                                  sizeof(mem_17600.mem), &mem_17600.mem));
    if (1 * (num_groups_17107 * group_sizze_17104) != 0) {
        const size_t global_work_sizze_17942[1] = {num_groups_17107 *
                     group_sizze_17104};
        const size_t local_work_sizze_17946[1] = {group_sizze_17104};
        int64_t time_start_17943 = 0, time_end_17944 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17109");
            fprintf(stderr, "%zu", global_work_sizze_17942[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17946[0]);
            fprintf(stderr, "].\n");
            time_start_17943 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17109, 1, NULL,
                                              global_work_sizze_17942,
                                              local_work_sizze_17946, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17944 = get_wall_time();
            
            long time_diff_17945 = time_end_17944 - time_start_17943;
            
            ctx->map_kernel_17109_total_runtime += time_diff_17945;
            ctx->map_kernel_17109_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17109",
                    time_diff_17945);
        }
    }
    memblock_unref_local(ctx, &mem_17594, "mem_17594");
    memblock_unref_local(ctx, &mem_17597, "mem_17597");
    out_arrsizze_17787 = sizze_16487;
    out_memsizze_17786 = bytes_17598;
    memblock_set_device(ctx, &out_mem_17785, &mem_17600, "mem_17600");
    *out_out_memsizze_17939 = out_memsizze_17786;
    (*out_mem_p_17940).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17940, &out_mem_17785,
                        "out_mem_17785");
    *out_out_arrsizze_17941 = out_arrsizze_17787;
    memblock_unref_local(ctx, &mem_17597, "mem_17597");
    memblock_unref_local(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17600, "mem_17600");
    memblock_unref_device(ctx, &out_mem_17785, "out_mem_17785");
    return 0;
}
static int futrts_t7(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17947,
                     struct memblock_device *out_mem_p_17948,
                     int32_t *out_out_arrsizze_17949, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t C_mem_sizze_17586,
                     struct memblock_device C_mem_17587,
                     int64_t D_mem_sizze_17588,
                     struct memblock_device D_mem_17589, int32_t sizze_16521,
                     int32_t sizze_16522, int32_t sizze_16523,
                     int32_t sizze_16524, int32_t sizze_16525)
{
    int64_t out_memsizze_17797;
    struct memblock_device out_mem_17796;
    
    out_mem_17796.references = NULL;
    
    int32_t out_arrsizze_17798;
    bool dim_zzero_16529 = 0 == sizze_16523;
    bool dim_zzero_16530 = 0 == sizze_16524;
    bool old_empty_16531 = dim_zzero_16529 || dim_zzero_16530;
    bool dim_zzero_16532 = 0 == sizze_16522;
    bool new_empty_16533 = dim_zzero_16530 || dim_zzero_16532;
    bool both_empty_16534 = old_empty_16531 && new_empty_16533;
    bool dim_match_16535 = sizze_16522 == sizze_16523;
    bool empty_or_match_16536 = both_empty_16534 || dim_match_16535;
    bool empty_or_match_cert_16537;
    
    if (!empty_or_match_16536) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:33:1-36:9",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17796, "out_mem_17796");
        return 1;
    }
    
    bool dim_zzero_16539 = 0 == sizze_16525;
    bool both_empty_16540 = dim_zzero_16530 && dim_zzero_16539;
    bool dim_match_16541 = sizze_16524 == sizze_16525;
    bool empty_or_match_16542 = both_empty_16540 || dim_match_16541;
    bool empty_or_match_cert_16543;
    
    if (!empty_or_match_16542) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:33:1-36:9",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17796, "out_mem_17796");
        return 1;
    }
    
    int32_t group_sizze_17147;
    
    group_sizze_17147 = ctx->sizes.group_sizze_17146;
    
    int32_t y_17148 = group_sizze_17147 - 1;
    int32_t x_17149 = sizze_16522 + y_17148;
    int32_t num_groups_17150 = squot32(x_17149, group_sizze_17147);
    int32_t num_threads_17151 = group_sizze_17147 * num_groups_17150;
    int32_t convop_x_17591 = sizze_16523 * sizze_16524;
    int64_t binop_x_17592 = sext_i32_i64(convop_x_17591);
    int64_t bytes_17590 = 4 * binop_x_17592;
    struct memblock_device mem_17593;
    
    mem_17593.references = NULL;
    memblock_alloc_device(ctx, &mem_17593, bytes_17590, "mem_17593");
    
    int call_ret_17950 = futrts_map_transpose_opencl_f32(ctx, mem_17593, 0,
                                                         C_mem_17587, 0, 1,
                                                         sizze_16524,
                                                         sizze_16523,
                                                         sizze_16523 *
                                                         sizze_16524,
                                                         sizze_16523 *
                                                         sizze_16524);
    
    assert(call_ret_17950 == 0);
    
    int64_t binop_x_17595 = sext_i32_i64(sizze_16522);
    int64_t bytes_17594 = 4 * binop_x_17595;
    struct memblock_device mem_17596;
    
    mem_17596.references = NULL;
    memblock_alloc_device(ctx, &mem_17596, bytes_17594, "mem_17596");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 0, sizeof(sizze_16522),
                                  &sizze_16522));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 1, sizeof(sizze_16523),
                                  &sizze_16523));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 2, sizeof(sizze_16524),
                                  &sizze_16524));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 3,
                                  sizeof(D_mem_17589.mem), &D_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 4,
                                  sizeof(mem_17593.mem), &mem_17593.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17152, 5,
                                  sizeof(mem_17596.mem), &mem_17596.mem));
    if (1 * (num_groups_17150 * group_sizze_17147) != 0) {
        const size_t global_work_sizze_17951[1] = {num_groups_17150 *
                     group_sizze_17147};
        const size_t local_work_sizze_17955[1] = {group_sizze_17147};
        int64_t time_start_17952 = 0, time_end_17953 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17152");
            fprintf(stderr, "%zu", global_work_sizze_17951[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17955[0]);
            fprintf(stderr, "].\n");
            time_start_17952 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17152, 1, NULL,
                                              global_work_sizze_17951,
                                              local_work_sizze_17955, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17953 = get_wall_time();
            
            long time_diff_17954 = time_end_17953 - time_start_17952;
            
            ctx->map_kernel_17152_total_runtime += time_diff_17954;
            ctx->map_kernel_17152_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17152",
                    time_diff_17954);
        }
    }
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    
    int32_t group_sizze_17187;
    
    group_sizze_17187 = ctx->sizes.group_sizze_17186;
    
    int32_t y_17188 = group_sizze_17187 - 1;
    int32_t x_17189 = sizze_16521 + y_17188;
    int32_t num_groups_17190 = squot32(x_17189, group_sizze_17187);
    int32_t num_threads_17191 = group_sizze_17187 * num_groups_17190;
    int32_t convop_x_17598 = sizze_16521 * sizze_16522;
    int64_t binop_x_17599 = sext_i32_i64(convop_x_17598);
    int64_t bytes_17597 = 4 * binop_x_17599;
    struct memblock_device mem_17600;
    
    mem_17600.references = NULL;
    memblock_alloc_device(ctx, &mem_17600, bytes_17597, "mem_17600");
    
    int call_ret_17956 = futrts_map_transpose_opencl_f32(ctx, mem_17600, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16522,
                                                         sizze_16521,
                                                         sizze_16521 *
                                                         sizze_16522,
                                                         sizze_16521 *
                                                         sizze_16522);
    
    assert(call_ret_17956 == 0);
    
    int64_t binop_x_17602 = sext_i32_i64(sizze_16521);
    int64_t bytes_17601 = 4 * binop_x_17602;
    struct memblock_device mem_17603;
    
    mem_17603.references = NULL;
    memblock_alloc_device(ctx, &mem_17603, bytes_17601, "mem_17603");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17192, 0, sizeof(sizze_16521),
                                  &sizze_16521));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17192, 1, sizeof(sizze_16522),
                                  &sizze_16522));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17192, 2,
                                  sizeof(mem_17596.mem), &mem_17596.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17192, 3,
                                  sizeof(mem_17600.mem), &mem_17600.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17192, 4,
                                  sizeof(mem_17603.mem), &mem_17603.mem));
    if (1 * (num_groups_17190 * group_sizze_17187) != 0) {
        const size_t global_work_sizze_17957[1] = {num_groups_17190 *
                     group_sizze_17187};
        const size_t local_work_sizze_17961[1] = {group_sizze_17187};
        int64_t time_start_17958 = 0, time_end_17959 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17192");
            fprintf(stderr, "%zu", global_work_sizze_17957[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17961[0]);
            fprintf(stderr, "].\n");
            time_start_17958 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17192, 1, NULL,
                                              global_work_sizze_17957,
                                              local_work_sizze_17961, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17959 = get_wall_time();
            
            long time_diff_17960 = time_end_17959 - time_start_17958;
            
            ctx->map_kernel_17192_total_runtime += time_diff_17960;
            ctx->map_kernel_17192_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17192",
                    time_diff_17960);
        }
    }
    memblock_unref_device(ctx, &mem_17596, "mem_17596");
    memblock_unref_device(ctx, &mem_17600, "mem_17600");
    out_arrsizze_17798 = sizze_16521;
    out_memsizze_17797 = bytes_17601;
    memblock_set_device(ctx, &out_mem_17796, &mem_17603, "mem_17603");
    *out_out_memsizze_17947 = out_memsizze_17797;
    (*out_mem_p_17948).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17948, &out_mem_17796,
                        "out_mem_17796");
    *out_out_arrsizze_17949 = out_arrsizze_17798;
    memblock_unref_device(ctx, &mem_17603, "mem_17603");
    memblock_unref_device(ctx, &mem_17600, "mem_17600");
    memblock_unref_device(ctx, &mem_17596, "mem_17596");
    memblock_unref_device(ctx, &mem_17593, "mem_17593");
    memblock_unref_device(ctx, &out_mem_17796, "out_mem_17796");
    return 0;
}
static int futrts_t9(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17962,
                     struct memblock_device *out_mem_p_17963,
                     int32_t *out_out_arrsizze_17964, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16563,
                     int32_t sizze_16564, int32_t sizze_16565,
                     int32_t sizze_16566, int32_t sizze_16567,
                     int32_t sizze_16568)
{
    int64_t out_memsizze_17808;
    struct memblock_device out_mem_17807;
    
    out_mem_17807.references = NULL;
    
    int32_t out_arrsizze_17809;
    bool dim_zzero_16573 = 0 == sizze_16565;
    bool dim_zzero_16574 = 0 == sizze_16566;
    bool old_empty_16575 = dim_zzero_16573 || dim_zzero_16574;
    bool dim_zzero_16576 = 0 == sizze_16564;
    bool new_empty_16577 = dim_zzero_16574 || dim_zzero_16576;
    bool both_empty_16578 = old_empty_16575 && new_empty_16577;
    bool dim_match_16579 = sizze_16564 == sizze_16565;
    bool empty_or_match_16580 = both_empty_16578 || dim_match_16579;
    bool empty_or_match_cert_16581;
    
    if (!empty_or_match_16580) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:43:1-45:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17807, "out_mem_17807");
        return 1;
    }
    
    bool dim_zzero_16583 = 0 == sizze_16567;
    bool both_empty_16584 = dim_zzero_16574 && dim_zzero_16583;
    bool dim_match_16585 = sizze_16566 == sizze_16567;
    bool empty_or_match_16586 = both_empty_16584 || dim_match_16585;
    bool empty_or_match_cert_16587;
    
    if (!empty_or_match_16586) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:43:1-45:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17807, "out_mem_17807");
        return 1;
    }
    
    bool dim_zzero_16589 = 0 == sizze_16568;
    bool both_empty_16590 = dim_zzero_16574 && dim_zzero_16589;
    bool dim_match_16591 = sizze_16566 == sizze_16568;
    bool empty_or_match_16592 = both_empty_16590 || dim_match_16591;
    bool empty_or_match_cert_16593;
    
    if (!empty_or_match_16592) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:43:1-45:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17807, "out_mem_17807");
        return 1;
    }
    
    int32_t group_sizze_17216;
    
    group_sizze_17216 = ctx->sizes.group_sizze_17215;
    
    int32_t y_17217 = group_sizze_17216 - 1;
    int32_t x_17218 = sizze_16566 + y_17217;
    int32_t num_groups_17219 = squot32(x_17218, group_sizze_17216);
    int32_t num_threads_17220 = group_sizze_17216 * num_groups_17219;
    int64_t binop_x_17593 = sext_i32_i64(sizze_16566);
    int64_t bytes_17592 = 4 * binop_x_17593;
    struct memblock_device mem_17594;
    
    mem_17594.references = NULL;
    memblock_alloc_device(ctx, &mem_17594, bytes_17592, "mem_17594");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17221, 0, sizeof(sizze_16566),
                                  &sizze_16566));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17221, 1,
                                  sizeof(C_mem_17589.mem), &C_mem_17589.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17221, 2,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17221, 3,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    if (1 * (num_groups_17219 * group_sizze_17216) != 0) {
        const size_t global_work_sizze_17965[1] = {num_groups_17219 *
                     group_sizze_17216};
        const size_t local_work_sizze_17969[1] = {group_sizze_17216};
        int64_t time_start_17966 = 0, time_end_17967 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17221");
            fprintf(stderr, "%zu", global_work_sizze_17965[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17969[0]);
            fprintf(stderr, "].\n");
            time_start_17966 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17221, 1, NULL,
                                              global_work_sizze_17965,
                                              local_work_sizze_17969, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17967 = get_wall_time();
            
            long time_diff_17968 = time_end_17967 - time_start_17966;
            
            ctx->map_kernel_17221_total_runtime += time_diff_17968;
            ctx->map_kernel_17221_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17221",
                    time_diff_17968);
        }
    }
    
    int32_t group_sizze_17251;
    
    group_sizze_17251 = ctx->sizes.group_sizze_17250;
    
    int32_t y_17252 = group_sizze_17251 - 1;
    int32_t x_17253 = sizze_16563 + y_17252;
    int32_t num_groups_17254 = squot32(x_17253, group_sizze_17251);
    int32_t num_threads_17255 = group_sizze_17251 * num_groups_17254;
    int32_t convop_x_17596 = sizze_16563 * sizze_16564;
    int64_t binop_x_17597 = sext_i32_i64(convop_x_17596);
    int64_t bytes_17595 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17595, "mem_17598");
    
    int call_ret_17970 = futrts_map_transpose_opencl_f32(ctx, mem_17598, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16564,
                                                         sizze_16563,
                                                         sizze_16563 *
                                                         sizze_16564,
                                                         sizze_16563 *
                                                         sizze_16564);
    
    assert(call_ret_17970 == 0);
    
    int64_t binop_x_17600 = sext_i32_i64(sizze_16563);
    int64_t bytes_17599 = 4 * binop_x_17600;
    struct memblock_device mem_17601;
    
    mem_17601.references = NULL;
    memblock_alloc_device(ctx, &mem_17601, bytes_17599, "mem_17601");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 0, sizeof(sizze_16563),
                                  &sizze_16563));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 1, sizeof(sizze_16564),
                                  &sizze_16564));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 2, sizeof(sizze_16566),
                                  &sizze_16566));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 3,
                                  sizeof(B_mem_17587.mem), &B_mem_17587.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 4,
                                  sizeof(mem_17594.mem), &mem_17594.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 5,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17256, 6,
                                  sizeof(mem_17601.mem), &mem_17601.mem));
    if (1 * (num_groups_17254 * group_sizze_17251) != 0) {
        const size_t global_work_sizze_17971[1] = {num_groups_17254 *
                     group_sizze_17251};
        const size_t local_work_sizze_17975[1] = {group_sizze_17251};
        int64_t time_start_17972 = 0, time_end_17973 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17256");
            fprintf(stderr, "%zu", global_work_sizze_17971[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17975[0]);
            fprintf(stderr, "].\n");
            time_start_17972 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17256, 1, NULL,
                                              global_work_sizze_17971,
                                              local_work_sizze_17975, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17973 = get_wall_time();
            
            long time_diff_17974 = time_end_17973 - time_start_17972;
            
            ctx->map_kernel_17256_total_runtime += time_diff_17974;
            ctx->map_kernel_17256_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17256",
                    time_diff_17974);
        }
    }
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    out_arrsizze_17809 = sizze_16563;
    out_memsizze_17808 = bytes_17599;
    memblock_set_device(ctx, &out_mem_17807, &mem_17601, "mem_17601");
    *out_out_memsizze_17962 = out_memsizze_17808;
    (*out_mem_p_17963).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17963, &out_mem_17807,
                        "out_mem_17807");
    *out_out_arrsizze_17964 = out_arrsizze_17809;
    memblock_unref_device(ctx, &mem_17601, "mem_17601");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17594, "mem_17594");
    memblock_unref_device(ctx, &out_mem_17807, "out_mem_17807");
    return 0;
}
static int futrts_t10(struct futhark_context *ctx,
                      int64_t *out_out_memsizze_17976,
                      struct memblock_device *out_mem_p_17977,
                      int32_t *out_out_arrsizze_17978,
                      int64_t A_mem_sizze_17584,
                      struct memblock_device A_mem_17585,
                      int64_t B_mem_sizze_17586,
                      struct memblock_device B_mem_17587,
                      int64_t C_mem_sizze_17588,
                      struct memblock_device C_mem_17589,
                      int64_t D_mem_sizze_17590,
                      struct memblock_device D_mem_17591, int32_t sizze_16616,
                      int32_t sizze_16617, int32_t sizze_16618,
                      int32_t sizze_16619, int32_t sizze_16620,
                      int32_t sizze_16621, int32_t sizze_16622)
{
    int64_t out_memsizze_17819;
    struct memblock_device out_mem_17818;
    
    out_mem_17818.references = NULL;
    
    int32_t out_arrsizze_17820;
    bool dim_zzero_16627 = 0 == sizze_16618;
    bool dim_zzero_16628 = 0 == sizze_16619;
    bool old_empty_16629 = dim_zzero_16627 || dim_zzero_16628;
    bool dim_zzero_16630 = 0 == sizze_16617;
    bool new_empty_16631 = dim_zzero_16628 || dim_zzero_16630;
    bool both_empty_16632 = old_empty_16629 && new_empty_16631;
    bool dim_match_16633 = sizze_16617 == sizze_16618;
    bool empty_or_match_16634 = both_empty_16632 || dim_match_16633;
    bool empty_or_match_cert_16635;
    
    if (!empty_or_match_16634) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:47:1-50:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17818, "out_mem_17818");
        return 1;
    }
    
    bool dim_zzero_16637 = 0 == sizze_16620;
    bool dim_zzero_16638 = 0 == sizze_16621;
    bool old_empty_16639 = dim_zzero_16637 || dim_zzero_16638;
    bool both_empty_16640 = new_empty_16631 && old_empty_16639;
    bool dim_match_16641 = sizze_16619 == sizze_16620;
    bool dim_match_16642 = sizze_16617 == sizze_16621;
    bool match_16643 = dim_match_16641 && dim_match_16642;
    bool empty_or_match_16644 = both_empty_16640 || match_16643;
    bool empty_or_match_cert_16645;
    
    if (!empty_or_match_16644) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:47:1-50:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17818, "out_mem_17818");
        return 1;
    }
    
    bool dim_zzero_16647 = 0 == sizze_16622;
    bool both_empty_16648 = dim_zzero_16630 && dim_zzero_16647;
    bool dim_match_16649 = sizze_16617 == sizze_16622;
    bool empty_or_match_16650 = both_empty_16648 || dim_match_16649;
    bool empty_or_match_cert_16651;
    
    if (!empty_or_match_16650) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:47:1-50:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17818, "out_mem_17818");
        return 1;
    }
    
    int32_t group_sizze_17309;
    
    group_sizze_17309 = ctx->sizes.group_sizze_17308;
    
    int32_t y_17310 = group_sizze_17309 - 1;
    int32_t x_17311 = sizze_16619 + y_17310;
    int32_t num_groups_17312 = squot32(x_17311, group_sizze_17309);
    int32_t num_threads_17313 = group_sizze_17309 * num_groups_17312;
    int32_t convop_x_17593 = sizze_16620 * sizze_16621;
    int64_t binop_x_17594 = sext_i32_i64(convop_x_17593);
    int64_t bytes_17592 = 4 * binop_x_17594;
    struct memblock_device mem_17595;
    
    mem_17595.references = NULL;
    memblock_alloc_device(ctx, &mem_17595, bytes_17592, "mem_17595");
    
    int call_ret_17979 = futrts_map_transpose_opencl_f32(ctx, mem_17595, 0,
                                                         C_mem_17589, 0, 1,
                                                         sizze_16621,
                                                         sizze_16620,
                                                         sizze_16620 *
                                                         sizze_16621,
                                                         sizze_16620 *
                                                         sizze_16621);
    
    assert(call_ret_17979 == 0);
    
    int64_t binop_x_17597 = sext_i32_i64(sizze_16619);
    int64_t bytes_17596 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17596, "mem_17598");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 0, sizeof(sizze_16617),
                                  &sizze_16617));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 1, sizeof(sizze_16619),
                                  &sizze_16619));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 2, sizeof(sizze_16620),
                                  &sizze_16620));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 3,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 4,
                                  sizeof(mem_17595.mem), &mem_17595.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17314, 5,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    if (1 * (num_groups_17312 * group_sizze_17309) != 0) {
        const size_t global_work_sizze_17980[1] = {num_groups_17312 *
                     group_sizze_17309};
        const size_t local_work_sizze_17984[1] = {group_sizze_17309};
        int64_t time_start_17981 = 0, time_end_17982 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17314");
            fprintf(stderr, "%zu", global_work_sizze_17980[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17984[0]);
            fprintf(stderr, "].\n");
            time_start_17981 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17314, 1, NULL,
                                              global_work_sizze_17980,
                                              local_work_sizze_17984, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17982 = get_wall_time();
            
            long time_diff_17983 = time_end_17982 - time_start_17981;
            
            ctx->map_kernel_17314_total_runtime += time_diff_17983;
            ctx->map_kernel_17314_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17314",
                    time_diff_17983);
        }
    }
    memblock_unref_device(ctx, &mem_17595, "mem_17595");
    
    int32_t group_sizze_17360;
    
    group_sizze_17360 = ctx->sizes.group_sizze_17359;
    
    int32_t y_17361 = group_sizze_17360 - 1;
    int32_t x_17362 = sizze_16616 + y_17361;
    int32_t num_groups_17363 = squot32(x_17362, group_sizze_17360);
    int32_t num_threads_17364 = group_sizze_17360 * num_groups_17363;
    int32_t convop_x_17600 = sizze_16616 * sizze_16617;
    int64_t binop_x_17601 = sext_i32_i64(convop_x_17600);
    int64_t bytes_17599 = 4 * binop_x_17601;
    struct memblock_device mem_17602;
    
    mem_17602.references = NULL;
    memblock_alloc_device(ctx, &mem_17602, bytes_17599, "mem_17602");
    
    int call_ret_17985 = futrts_map_transpose_opencl_f32(ctx, mem_17602, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16617,
                                                         sizze_16616,
                                                         sizze_16616 *
                                                         sizze_16617,
                                                         sizze_16616 *
                                                         sizze_16617);
    
    assert(call_ret_17985 == 0);
    
    int64_t binop_x_17604 = sext_i32_i64(sizze_16616);
    int64_t bytes_17603 = 4 * binop_x_17604;
    struct memblock_device mem_17605;
    
    mem_17605.references = NULL;
    memblock_alloc_device(ctx, &mem_17605, bytes_17603, "mem_17605");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 0, sizeof(sizze_16616),
                                  &sizze_16616));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 1, sizeof(sizze_16617),
                                  &sizze_16617));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 2, sizeof(sizze_16619),
                                  &sizze_16619));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 3,
                                  sizeof(B_mem_17587.mem), &B_mem_17587.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 4,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 5,
                                  sizeof(mem_17602.mem), &mem_17602.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17365, 6,
                                  sizeof(mem_17605.mem), &mem_17605.mem));
    if (1 * (num_groups_17363 * group_sizze_17360) != 0) {
        const size_t global_work_sizze_17986[1] = {num_groups_17363 *
                     group_sizze_17360};
        const size_t local_work_sizze_17990[1] = {group_sizze_17360};
        int64_t time_start_17987 = 0, time_end_17988 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17365");
            fprintf(stderr, "%zu", global_work_sizze_17986[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17990[0]);
            fprintf(stderr, "].\n");
            time_start_17987 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17365, 1, NULL,
                                              global_work_sizze_17986,
                                              local_work_sizze_17990, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17988 = get_wall_time();
            
            long time_diff_17989 = time_end_17988 - time_start_17987;
            
            ctx->map_kernel_17365_total_runtime += time_diff_17989;
            ctx->map_kernel_17365_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17365",
                    time_diff_17989);
        }
    }
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    out_arrsizze_17820 = sizze_16616;
    out_memsizze_17819 = bytes_17603;
    memblock_set_device(ctx, &out_mem_17818, &mem_17605, "mem_17605");
    *out_out_memsizze_17976 = out_memsizze_17819;
    (*out_mem_p_17977).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17977, &out_mem_17818,
                        "out_mem_17818");
    *out_out_arrsizze_17978 = out_arrsizze_17820;
    memblock_unref_device(ctx, &mem_17605, "mem_17605");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17595, "mem_17595");
    memblock_unref_device(ctx, &out_mem_17818, "out_mem_17818");
    return 0;
}
static int futrts_t8(struct futhark_context *ctx,
                     int64_t *out_out_memsizze_17991,
                     struct memblock_device *out_mem_p_17992,
                     int32_t *out_out_arrsizze_17993, int64_t A_mem_sizze_17584,
                     struct memblock_device A_mem_17585,
                     int64_t B_mem_sizze_17586,
                     struct memblock_device B_mem_17587,
                     int64_t C_mem_sizze_17588,
                     struct memblock_device C_mem_17589,
                     int64_t D_mem_sizze_17590,
                     struct memblock_device D_mem_17591, int32_t sizze_16679,
                     int32_t sizze_16680, int32_t sizze_16681,
                     int32_t sizze_16682, int32_t sizze_16683,
                     int32_t sizze_16684, int32_t sizze_16685)
{
    int64_t out_memsizze_17831;
    struct memblock_device out_mem_17830;
    
    out_mem_17830.references = NULL;
    
    int32_t out_arrsizze_17832;
    bool dim_zzero_16690 = 0 == sizze_16681;
    bool dim_zzero_16691 = 0 == sizze_16682;
    bool old_empty_16692 = dim_zzero_16690 || dim_zzero_16691;
    bool dim_zzero_16693 = 0 == sizze_16679;
    bool dim_zzero_16694 = 0 == sizze_16680;
    bool new_empty_16695 = dim_zzero_16693 || dim_zzero_16694;
    bool both_empty_16696 = old_empty_16692 && new_empty_16695;
    bool dim_match_16697 = sizze_16679 == sizze_16681;
    bool dim_match_16698 = sizze_16680 == sizze_16682;
    bool match_16699 = dim_match_16697 && dim_match_16698;
    bool empty_or_match_16700 = both_empty_16696 || match_16699;
    bool empty_or_match_cert_16701;
    
    if (!empty_or_match_16700) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:38:1-41:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17830, "out_mem_17830");
        return 1;
    }
    
    bool dim_zzero_16703 = 0 == sizze_16683;
    bool dim_zzero_16704 = 0 == sizze_16684;
    bool old_empty_16705 = dim_zzero_16703 || dim_zzero_16704;
    bool new_empty_16706 = dim_zzero_16694 || dim_zzero_16704;
    bool both_empty_16707 = old_empty_16705 && new_empty_16706;
    bool dim_match_16708 = sizze_16680 == sizze_16683;
    bool empty_or_match_16709 = both_empty_16707 || dim_match_16708;
    bool empty_or_match_cert_16710;
    
    if (!empty_or_match_16709) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:38:1-41:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17830, "out_mem_17830");
        return 1;
    }
    
    bool dim_zzero_16712 = 0 == sizze_16685;
    bool both_empty_16713 = dim_zzero_16704 && dim_zzero_16712;
    bool dim_match_16714 = sizze_16684 == sizze_16685;
    bool empty_or_match_16715 = both_empty_16713 || dim_match_16714;
    bool empty_or_match_cert_16716;
    
    if (!empty_or_match_16715) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "2d.fut:38:1-41:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_17830, "out_mem_17830");
        return 1;
    }
    
    int32_t group_sizze_17418;
    
    group_sizze_17418 = ctx->sizes.group_sizze_17417;
    
    int32_t y_17419 = group_sizze_17418 - 1;
    int32_t x_17420 = sizze_16680 + y_17419;
    int32_t num_groups_17421 = squot32(x_17420, group_sizze_17418);
    int32_t num_threads_17422 = group_sizze_17418 * num_groups_17421;
    int32_t convop_x_17593 = sizze_16683 * sizze_16684;
    int64_t binop_x_17594 = sext_i32_i64(convop_x_17593);
    int64_t bytes_17592 = 4 * binop_x_17594;
    struct memblock_device mem_17595;
    
    mem_17595.references = NULL;
    memblock_alloc_device(ctx, &mem_17595, bytes_17592, "mem_17595");
    
    int call_ret_17994 = futrts_map_transpose_opencl_f32(ctx, mem_17595, 0,
                                                         C_mem_17589, 0, 1,
                                                         sizze_16684,
                                                         sizze_16683,
                                                         sizze_16683 *
                                                         sizze_16684,
                                                         sizze_16683 *
                                                         sizze_16684);
    
    assert(call_ret_17994 == 0);
    
    int64_t binop_x_17597 = sext_i32_i64(sizze_16680);
    int64_t bytes_17596 = 4 * binop_x_17597;
    struct memblock_device mem_17598;
    
    mem_17598.references = NULL;
    memblock_alloc_device(ctx, &mem_17598, bytes_17596, "mem_17598");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 0, sizeof(sizze_16680),
                                  &sizze_16680));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 1, sizeof(sizze_16683),
                                  &sizze_16683));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 2, sizeof(sizze_16684),
                                  &sizze_16684));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 3,
                                  sizeof(D_mem_17591.mem), &D_mem_17591.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 4,
                                  sizeof(mem_17595.mem), &mem_17595.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17423, 5,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    if (1 * (num_groups_17421 * group_sizze_17418) != 0) {
        const size_t global_work_sizze_17995[1] = {num_groups_17421 *
                     group_sizze_17418};
        const size_t local_work_sizze_17999[1] = {group_sizze_17418};
        int64_t time_start_17996 = 0, time_end_17997 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17423");
            fprintf(stderr, "%zu", global_work_sizze_17995[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_17999[0]);
            fprintf(stderr, "].\n");
            time_start_17996 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17423, 1, NULL,
                                              global_work_sizze_17995,
                                              local_work_sizze_17999, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_17997 = get_wall_time();
            
            long time_diff_17998 = time_end_17997 - time_start_17996;
            
            ctx->map_kernel_17423_total_runtime += time_diff_17998;
            ctx->map_kernel_17423_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17423",
                    time_diff_17998);
        }
    }
    memblock_unref_device(ctx, &mem_17595, "mem_17595");
    
    int32_t group_sizze_17460;
    
    group_sizze_17460 = ctx->sizes.group_sizze_17459;
    
    int32_t y_17461 = group_sizze_17460 - 1;
    int32_t x_17462 = sizze_16679 + y_17461;
    int32_t num_groups_17463 = squot32(x_17462, group_sizze_17460);
    int32_t num_threads_17464 = group_sizze_17460 * num_groups_17463;
    int32_t convop_x_17600 = sizze_16679 * sizze_16680;
    int64_t binop_x_17601 = sext_i32_i64(convop_x_17600);
    int64_t bytes_17599 = 4 * binop_x_17601;
    struct memblock_device mem_17602;
    
    mem_17602.references = NULL;
    memblock_alloc_device(ctx, &mem_17602, bytes_17599, "mem_17602");
    
    int call_ret_18000 = futrts_map_transpose_opencl_f32(ctx, mem_17602, 0,
                                                         A_mem_17585, 0, 1,
                                                         sizze_16680,
                                                         sizze_16679,
                                                         sizze_16679 *
                                                         sizze_16680,
                                                         sizze_16679 *
                                                         sizze_16680);
    
    assert(call_ret_18000 == 0);
    
    int32_t convop_x_17604 = sizze_16681 * sizze_16682;
    int64_t binop_x_17605 = sext_i32_i64(convop_x_17604);
    int64_t bytes_17603 = 4 * binop_x_17605;
    struct memblock_device mem_17606;
    
    mem_17606.references = NULL;
    memblock_alloc_device(ctx, &mem_17606, bytes_17603, "mem_17606");
    
    int call_ret_18001 = futrts_map_transpose_opencl_f32(ctx, mem_17606, 0,
                                                         B_mem_17587, 0, 1,
                                                         sizze_16682,
                                                         sizze_16681,
                                                         sizze_16681 *
                                                         sizze_16682,
                                                         sizze_16681 *
                                                         sizze_16682);
    
    assert(call_ret_18001 == 0);
    
    int64_t binop_x_17608 = sext_i32_i64(sizze_16679);
    int64_t bytes_17607 = 4 * binop_x_17608;
    struct memblock_device mem_17609;
    
    mem_17609.references = NULL;
    memblock_alloc_device(ctx, &mem_17609, bytes_17607, "mem_17609");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 0, sizeof(sizze_16679),
                                  &sizze_16679));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 1, sizeof(sizze_16680),
                                  &sizze_16680));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 2, sizeof(sizze_16681),
                                  &sizze_16681));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 3,
                                  sizeof(mem_17598.mem), &mem_17598.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 4,
                                  sizeof(mem_17602.mem), &mem_17602.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 5,
                                  sizeof(mem_17606.mem), &mem_17606.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_17465, 6,
                                  sizeof(mem_17609.mem), &mem_17609.mem));
    if (1 * (num_groups_17463 * group_sizze_17460) != 0) {
        const size_t global_work_sizze_18002[1] = {num_groups_17463 *
                     group_sizze_17460};
        const size_t local_work_sizze_18006[1] = {group_sizze_17460};
        int64_t time_start_18003 = 0, time_end_18004 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_17465");
            fprintf(stderr, "%zu", global_work_sizze_18002[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_18006[0]);
            fprintf(stderr, "].\n");
            time_start_18003 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_17465, 1, NULL,
                                              global_work_sizze_18002,
                                              local_work_sizze_18006, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_18004 = get_wall_time();
            
            long time_diff_18005 = time_end_18004 - time_start_18003;
            
            ctx->map_kernel_17465_total_runtime += time_diff_18005;
            ctx->map_kernel_17465_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_17465",
                    time_diff_18005);
        }
    }
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    memblock_unref_device(ctx, &mem_17606, "mem_17606");
    out_arrsizze_17832 = sizze_16679;
    out_memsizze_17831 = bytes_17607;
    memblock_set_device(ctx, &out_mem_17830, &mem_17609, "mem_17609");
    *out_out_memsizze_17991 = out_memsizze_17831;
    (*out_mem_p_17992).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_17992, &out_mem_17830,
                        "out_mem_17830");
    *out_out_arrsizze_17993 = out_arrsizze_17832;
    memblock_unref_device(ctx, &mem_17609, "mem_17609");
    memblock_unref_device(ctx, &mem_17606, "mem_17606");
    memblock_unref_device(ctx, &mem_17602, "mem_17602");
    memblock_unref_device(ctx, &mem_17598, "mem_17598");
    memblock_unref_device(ctx, &mem_17595, "mem_17595");
    memblock_unref_device(ctx, &out_mem_17830, "out_mem_17830");
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
int futhark_entry_t1_(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int32_t sizze_16260;
    int32_t sizze_16261;
    int32_t sizze_16262;
    int64_t out_memsizze_17699;
    struct memblock_device out_mem_17698;
    
    out_mem_17698.references = NULL;
    
    int32_t out_arrsizze_17700;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16260 = in0->shape[0];
    sizze_16261 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16262 = in1->shape[0];
    
    int ret = futrts_t1_(ctx, &out_memsizze_17699, &out_mem_17698,
                         &out_arrsizze_17700, A_mem_sizze_17584, A_mem_17585,
                         B_mem_sizze_17586, B_mem_17587, sizze_16260,
                         sizze_16261, sizze_16262);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17698;
        (*out0)->shape[0] = out_arrsizze_17700;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t1(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int32_t sizze_16285;
    int32_t sizze_16286;
    int32_t sizze_16287;
    int64_t out_memsizze_17745;
    struct memblock_device out_mem_17744;
    
    out_mem_17744.references = NULL;
    
    int32_t out_arrsizze_17746;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16285 = in0->shape[0];
    sizze_16286 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16287 = in1->shape[0];
    
    int ret = futrts_t1(ctx, &out_memsizze_17745, &out_mem_17744,
                        &out_arrsizze_17746, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, sizze_16285,
                        sizze_16286, sizze_16287);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17744;
        (*out0)->shape[0] = out_arrsizze_17746;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t2(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int32_t sizze_16306;
    int32_t sizze_16307;
    int32_t sizze_16308;
    int32_t sizze_16309;
    int64_t out_memsizze_17752;
    struct memblock_device out_mem_17751;
    
    out_mem_17751.references = NULL;
    
    int32_t out_arrsizze_17753;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16306 = in0->shape[0];
    sizze_16307 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16308 = in1->shape[0];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16309 = in2->shape[0];
    
    int ret = futrts_t2(ctx, &out_memsizze_17752, &out_mem_17751,
                        &out_arrsizze_17753, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, sizze_16306, sizze_16307, sizze_16308,
                        sizze_16309);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17751;
        (*out0)->shape[0] = out_arrsizze_17753;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t3(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int32_t sizze_16338;
    int32_t sizze_16339;
    int32_t sizze_16340;
    int32_t sizze_16341;
    int32_t sizze_16342;
    int64_t out_memsizze_17759;
    struct memblock_device out_mem_17758;
    
    out_mem_17758.references = NULL;
    
    int32_t out_arrsizze_17760;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16338 = in0->shape[0];
    sizze_16339 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16340 = in1->shape[0];
    sizze_16341 = in1->shape[1];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16342 = in2->shape[0];
    
    int ret = futrts_t3(ctx, &out_memsizze_17759, &out_mem_17758,
                        &out_arrsizze_17760, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, sizze_16338, sizze_16339, sizze_16340,
                        sizze_16341, sizze_16342);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17758;
        (*out0)->shape[0] = out_arrsizze_17760;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t4(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16377;
    int32_t sizze_16378;
    int32_t sizze_16379;
    int32_t sizze_16380;
    int32_t sizze_16381;
    int32_t sizze_16382;
    int64_t out_memsizze_17766;
    struct memblock_device out_mem_17765;
    
    out_mem_17765.references = NULL;
    
    int32_t out_arrsizze_17767;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16377 = in0->shape[0];
    sizze_16378 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16379 = in1->shape[0];
    sizze_16380 = in1->shape[1];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16381 = in2->shape[0];
    D_mem_17591 = in3->mem;
    D_mem_sizze_17590 = in3->mem.size;
    sizze_16382 = in3->shape[0];
    
    int ret = futrts_t4(ctx, &out_memsizze_17766, &out_mem_17765,
                        &out_arrsizze_17767, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                        sizze_16377, sizze_16378, sizze_16379, sizze_16380,
                        sizze_16381, sizze_16382);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17765;
        (*out0)->shape[0] = out_arrsizze_17767;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t5(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const float in0, const struct futhark_f32_2d *in1, const
                     float in2, const struct futhark_f32_2d *in3, const
                     float in4, const struct futhark_f32_1d *in5, const
                     float in6, const struct futhark_f32_1d *in7)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16428;
    int32_t sizze_16429;
    int32_t sizze_16430;
    int32_t sizze_16431;
    int32_t sizze_16432;
    int32_t sizze_16433;
    float a_16434;
    float b_16436;
    float c_16438;
    float d_16440;
    int64_t out_memsizze_17776;
    struct memblock_device out_mem_17775;
    
    out_mem_17775.references = NULL;
    
    int32_t out_arrsizze_17777;
    
    lock_lock(&ctx->lock);
    a_16434 = in0;
    A_mem_17585 = in1->mem;
    A_mem_sizze_17584 = in1->mem.size;
    sizze_16428 = in1->shape[0];
    sizze_16429 = in1->shape[1];
    b_16436 = in2;
    B_mem_17587 = in3->mem;
    B_mem_sizze_17586 = in3->mem.size;
    sizze_16430 = in3->shape[0];
    sizze_16431 = in3->shape[1];
    c_16438 = in4;
    C_mem_17589 = in5->mem;
    C_mem_sizze_17588 = in5->mem.size;
    sizze_16432 = in5->shape[0];
    d_16440 = in6;
    D_mem_17591 = in7->mem;
    D_mem_sizze_17590 = in7->mem.size;
    sizze_16433 = in7->shape[0];
    
    int ret = futrts_t5(ctx, &out_memsizze_17776, &out_mem_17775,
                        &out_arrsizze_17777, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                        sizze_16428, sizze_16429, sizze_16430, sizze_16431,
                        sizze_16432, sizze_16433, a_16434, b_16436, c_16438,
                        d_16440);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17775;
        (*out0)->shape[0] = out_arrsizze_17777;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t6(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_1d *in0, const
                     struct futhark_f32_1d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16487;
    int32_t sizze_16488;
    int32_t sizze_16489;
    int32_t sizze_16490;
    int64_t out_memsizze_17786;
    struct memblock_device out_mem_17785;
    
    out_mem_17785.references = NULL;
    
    int32_t out_arrsizze_17787;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16487 = in0->shape[0];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16488 = in1->shape[0];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16489 = in2->shape[0];
    D_mem_17591 = in3->mem;
    D_mem_sizze_17590 = in3->mem.size;
    sizze_16490 = in3->shape[0];
    
    int ret = futrts_t6(ctx, &out_memsizze_17786, &out_mem_17785,
                        &out_arrsizze_17787, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                        sizze_16487, sizze_16488, sizze_16489, sizze_16490);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17785;
        (*out0)->shape[0] = out_arrsizze_17787;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t7(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t C_mem_sizze_17586;
    struct memblock_device C_mem_17587;
    
    C_mem_17587.references = NULL;
    
    int64_t D_mem_sizze_17588;
    struct memblock_device D_mem_17589;
    
    D_mem_17589.references = NULL;
    
    int32_t sizze_16521;
    int32_t sizze_16522;
    int32_t sizze_16523;
    int32_t sizze_16524;
    int32_t sizze_16525;
    int64_t out_memsizze_17797;
    struct memblock_device out_mem_17796;
    
    out_mem_17796.references = NULL;
    
    int32_t out_arrsizze_17798;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16521 = in0->shape[0];
    sizze_16522 = in0->shape[1];
    C_mem_17587 = in1->mem;
    C_mem_sizze_17586 = in1->mem.size;
    sizze_16523 = in1->shape[0];
    sizze_16524 = in1->shape[1];
    D_mem_17589 = in2->mem;
    D_mem_sizze_17588 = in2->mem.size;
    sizze_16525 = in2->shape[0];
    
    int ret = futrts_t7(ctx, &out_memsizze_17797, &out_mem_17796,
                        &out_arrsizze_17798, A_mem_sizze_17584, A_mem_17585,
                        C_mem_sizze_17586, C_mem_17587, D_mem_sizze_17588,
                        D_mem_17589, sizze_16521, sizze_16522, sizze_16523,
                        sizze_16524, sizze_16525);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17796;
        (*out0)->shape[0] = out_arrsizze_17798;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t9(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_1d *in2, const
                     struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16563;
    int32_t sizze_16564;
    int32_t sizze_16565;
    int32_t sizze_16566;
    int32_t sizze_16567;
    int32_t sizze_16568;
    int64_t out_memsizze_17808;
    struct memblock_device out_mem_17807;
    
    out_mem_17807.references = NULL;
    
    int32_t out_arrsizze_17809;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16563 = in0->shape[0];
    sizze_16564 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16565 = in1->shape[0];
    sizze_16566 = in1->shape[1];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16567 = in2->shape[0];
    D_mem_17591 = in3->mem;
    D_mem_sizze_17590 = in3->mem.size;
    sizze_16568 = in3->shape[0];
    
    int ret = futrts_t9(ctx, &out_memsizze_17808, &out_mem_17807,
                        &out_arrsizze_17809, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                        sizze_16563, sizze_16564, sizze_16565, sizze_16566,
                        sizze_16567, sizze_16568);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17807;
        (*out0)->shape[0] = out_arrsizze_17809;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t10(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                      const struct futhark_f32_2d *in0, const
                      struct futhark_f32_2d *in1, const
                      struct futhark_f32_2d *in2, const
                      struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16616;
    int32_t sizze_16617;
    int32_t sizze_16618;
    int32_t sizze_16619;
    int32_t sizze_16620;
    int32_t sizze_16621;
    int32_t sizze_16622;
    int64_t out_memsizze_17819;
    struct memblock_device out_mem_17818;
    
    out_mem_17818.references = NULL;
    
    int32_t out_arrsizze_17820;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16616 = in0->shape[0];
    sizze_16617 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16618 = in1->shape[0];
    sizze_16619 = in1->shape[1];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16620 = in2->shape[0];
    sizze_16621 = in2->shape[1];
    D_mem_17591 = in3->mem;
    D_mem_sizze_17590 = in3->mem.size;
    sizze_16622 = in3->shape[0];
    
    int ret = futrts_t10(ctx, &out_memsizze_17819, &out_mem_17818,
                         &out_arrsizze_17820, A_mem_sizze_17584, A_mem_17585,
                         B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                         C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                         sizze_16616, sizze_16617, sizze_16618, sizze_16619,
                         sizze_16620, sizze_16621, sizze_16622);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17818;
        (*out0)->shape[0] = out_arrsizze_17820;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t8(struct futhark_context *ctx, struct futhark_f32_1d **out0,
                     const struct futhark_f32_2d *in0, const
                     struct futhark_f32_2d *in1, const
                     struct futhark_f32_2d *in2, const
                     struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_17584;
    struct memblock_device A_mem_17585;
    
    A_mem_17585.references = NULL;
    
    int64_t B_mem_sizze_17586;
    struct memblock_device B_mem_17587;
    
    B_mem_17587.references = NULL;
    
    int64_t C_mem_sizze_17588;
    struct memblock_device C_mem_17589;
    
    C_mem_17589.references = NULL;
    
    int64_t D_mem_sizze_17590;
    struct memblock_device D_mem_17591;
    
    D_mem_17591.references = NULL;
    
    int32_t sizze_16679;
    int32_t sizze_16680;
    int32_t sizze_16681;
    int32_t sizze_16682;
    int32_t sizze_16683;
    int32_t sizze_16684;
    int32_t sizze_16685;
    int64_t out_memsizze_17831;
    struct memblock_device out_mem_17830;
    
    out_mem_17830.references = NULL;
    
    int32_t out_arrsizze_17832;
    
    lock_lock(&ctx->lock);
    A_mem_17585 = in0->mem;
    A_mem_sizze_17584 = in0->mem.size;
    sizze_16679 = in0->shape[0];
    sizze_16680 = in0->shape[1];
    B_mem_17587 = in1->mem;
    B_mem_sizze_17586 = in1->mem.size;
    sizze_16681 = in1->shape[0];
    sizze_16682 = in1->shape[1];
    C_mem_17589 = in2->mem;
    C_mem_sizze_17588 = in2->mem.size;
    sizze_16683 = in2->shape[0];
    sizze_16684 = in2->shape[1];
    D_mem_17591 = in3->mem;
    D_mem_sizze_17590 = in3->mem.size;
    sizze_16685 = in3->shape[0];
    
    int ret = futrts_t8(ctx, &out_memsizze_17831, &out_mem_17830,
                        &out_arrsizze_17832, A_mem_sizze_17584, A_mem_17585,
                        B_mem_sizze_17586, B_mem_17587, C_mem_sizze_17588,
                        C_mem_17589, D_mem_sizze_17590, D_mem_17591,
                        sizze_16679, sizze_16680, sizze_16681, sizze_16682,
                        sizze_16683, sizze_16684, sizze_16685);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_17830;
        (*out0)->shape[0] = out_arrsizze_17832;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
