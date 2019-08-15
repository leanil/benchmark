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
struct futhark_f32_1d ;

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1);
int futhark_entry_dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_dot2(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_dot3(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_dot4(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3);
int futhark_entry_dot5(struct futhark_context *ctx, float *out0, const
                       float in0, const struct futhark_f32_1d *in1, const
                       float in2, const struct futhark_f32_1d *in3, const
                       float in4, const struct futhark_f32_1d *in5, const
                       float in6, const struct futhark_f32_1d *in7);
int futhark_entry_dot6(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
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
static void futrts_cli_entry_dot(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9739;
    int64_t read_shape_9740[1];
    float *read_arr_9741 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9741, read_shape_9740, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9742;
    int64_t read_shape_9743[1];
    float *read_arr_9744 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9744, read_shape_9743, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9745;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9739 = futhark_new_f32_1d(ctx, read_arr_9741,
                                                     read_shape_9740[0])) != 0);
        assert((read_value_9742 = futhark_new_f32_1d(ctx, read_arr_9744,
                                                     read_shape_9743[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot(ctx, &result_9745, read_value_9739,
                              read_value_9742);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9739) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9742) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9739 = futhark_new_f32_1d(ctx, read_arr_9741,
                                                     read_shape_9740[0])) != 0);
        assert((read_value_9742 = futhark_new_f32_1d(ctx, read_arr_9744,
                                                     read_shape_9743[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot(ctx, &result_9745, read_value_9739,
                              read_value_9742);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9739) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9742) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9741);
    free(read_arr_9744);
    write_scalar(stdout, binary_output, &f32_info, &result_9745);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot1(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9746;
    int64_t read_shape_9747[1];
    float *read_arr_9748 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9748, read_shape_9747, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9749;
    int64_t read_shape_9750[1];
    float *read_arr_9751 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9751, read_shape_9750, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9752;
    int64_t read_shape_9753[1];
    float *read_arr_9754 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9754, read_shape_9753, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9755;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9746 = futhark_new_f32_1d(ctx, read_arr_9748,
                                                     read_shape_9747[0])) != 0);
        assert((read_value_9749 = futhark_new_f32_1d(ctx, read_arr_9751,
                                                     read_shape_9750[0])) != 0);
        assert((read_value_9752 = futhark_new_f32_1d(ctx, read_arr_9754,
                                                     read_shape_9753[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot1(ctx, &result_9755, read_value_9746,
                               read_value_9749, read_value_9752);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9746) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9749) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9752) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9746 = futhark_new_f32_1d(ctx, read_arr_9748,
                                                     read_shape_9747[0])) != 0);
        assert((read_value_9749 = futhark_new_f32_1d(ctx, read_arr_9751,
                                                     read_shape_9750[0])) != 0);
        assert((read_value_9752 = futhark_new_f32_1d(ctx, read_arr_9754,
                                                     read_shape_9753[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot1(ctx, &result_9755, read_value_9746,
                               read_value_9749, read_value_9752);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9746) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9749) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9752) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9748);
    free(read_arr_9751);
    free(read_arr_9754);
    write_scalar(stdout, binary_output, &f32_info, &result_9755);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot2(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9756;
    int64_t read_shape_9757[1];
    float *read_arr_9758 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9758, read_shape_9757, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9759;
    int64_t read_shape_9760[1];
    float *read_arr_9761 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9761, read_shape_9760, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9762;
    int64_t read_shape_9763[1];
    float *read_arr_9764 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9764, read_shape_9763, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9765;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9756 = futhark_new_f32_1d(ctx, read_arr_9758,
                                                     read_shape_9757[0])) != 0);
        assert((read_value_9759 = futhark_new_f32_1d(ctx, read_arr_9761,
                                                     read_shape_9760[0])) != 0);
        assert((read_value_9762 = futhark_new_f32_1d(ctx, read_arr_9764,
                                                     read_shape_9763[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot2(ctx, &result_9765, read_value_9756,
                               read_value_9759, read_value_9762);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9756) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9759) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9762) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9756 = futhark_new_f32_1d(ctx, read_arr_9758,
                                                     read_shape_9757[0])) != 0);
        assert((read_value_9759 = futhark_new_f32_1d(ctx, read_arr_9761,
                                                     read_shape_9760[0])) != 0);
        assert((read_value_9762 = futhark_new_f32_1d(ctx, read_arr_9764,
                                                     read_shape_9763[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot2(ctx, &result_9765, read_value_9756,
                               read_value_9759, read_value_9762);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9756) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9759) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9762) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9758);
    free(read_arr_9761);
    free(read_arr_9764);
    write_scalar(stdout, binary_output, &f32_info, &result_9765);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot3(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9766;
    int64_t read_shape_9767[1];
    float *read_arr_9768 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9768, read_shape_9767, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9769;
    int64_t read_shape_9770[1];
    float *read_arr_9771 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9771, read_shape_9770, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9772;
    int64_t read_shape_9773[1];
    float *read_arr_9774 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9774, read_shape_9773, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9775;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9766 = futhark_new_f32_1d(ctx, read_arr_9768,
                                                     read_shape_9767[0])) != 0);
        assert((read_value_9769 = futhark_new_f32_1d(ctx, read_arr_9771,
                                                     read_shape_9770[0])) != 0);
        assert((read_value_9772 = futhark_new_f32_1d(ctx, read_arr_9774,
                                                     read_shape_9773[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot3(ctx, &result_9775, read_value_9766,
                               read_value_9769, read_value_9772);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9766) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9769) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9772) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9766 = futhark_new_f32_1d(ctx, read_arr_9768,
                                                     read_shape_9767[0])) != 0);
        assert((read_value_9769 = futhark_new_f32_1d(ctx, read_arr_9771,
                                                     read_shape_9770[0])) != 0);
        assert((read_value_9772 = futhark_new_f32_1d(ctx, read_arr_9774,
                                                     read_shape_9773[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot3(ctx, &result_9775, read_value_9766,
                               read_value_9769, read_value_9772);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9766) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9769) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9772) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9768);
    free(read_arr_9771);
    free(read_arr_9774);
    write_scalar(stdout, binary_output, &f32_info, &result_9775);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot4(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9776;
    int64_t read_shape_9777[1];
    float *read_arr_9778 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9778, read_shape_9777, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9779;
    int64_t read_shape_9780[1];
    float *read_arr_9781 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9781, read_shape_9780, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9782;
    int64_t read_shape_9783[1];
    float *read_arr_9784 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9784, read_shape_9783, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9785;
    int64_t read_shape_9786[1];
    float *read_arr_9787 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9787, read_shape_9786, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9788;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9776 = futhark_new_f32_1d(ctx, read_arr_9778,
                                                     read_shape_9777[0])) != 0);
        assert((read_value_9779 = futhark_new_f32_1d(ctx, read_arr_9781,
                                                     read_shape_9780[0])) != 0);
        assert((read_value_9782 = futhark_new_f32_1d(ctx, read_arr_9784,
                                                     read_shape_9783[0])) != 0);
        assert((read_value_9785 = futhark_new_f32_1d(ctx, read_arr_9787,
                                                     read_shape_9786[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot4(ctx, &result_9788, read_value_9776,
                               read_value_9779, read_value_9782,
                               read_value_9785);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9776) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9779) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9782) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9785) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9776 = futhark_new_f32_1d(ctx, read_arr_9778,
                                                     read_shape_9777[0])) != 0);
        assert((read_value_9779 = futhark_new_f32_1d(ctx, read_arr_9781,
                                                     read_shape_9780[0])) != 0);
        assert((read_value_9782 = futhark_new_f32_1d(ctx, read_arr_9784,
                                                     read_shape_9783[0])) != 0);
        assert((read_value_9785 = futhark_new_f32_1d(ctx, read_arr_9787,
                                                     read_shape_9786[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot4(ctx, &result_9788, read_value_9776,
                               read_value_9779, read_value_9782,
                               read_value_9785);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9776) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9779) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9782) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9785) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9778);
    free(read_arr_9781);
    free(read_arr_9784);
    free(read_arr_9787);
    write_scalar(stdout, binary_output, &f32_info, &result_9788);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot5(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    float read_value_9789;
    
    if (read_scalar(&f32_info, &read_value_9789) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 0,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9790;
    int64_t read_shape_9791[1];
    float *read_arr_9792 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9792, read_shape_9791, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_9793;
    
    if (read_scalar(&f32_info, &read_value_9793) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 2,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9794;
    int64_t read_shape_9795[1];
    float *read_arr_9796 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9796, read_shape_9795, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_9797;
    
    if (read_scalar(&f32_info, &read_value_9797) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 4,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9798;
    int64_t read_shape_9799[1];
    float *read_arr_9800 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9800, read_shape_9799, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 5, "[]",
              f32_info.type_name, strerror(errno));
    
    float read_value_9801;
    
    if (read_scalar(&f32_info, &read_value_9801) != 0)
        panic(1, "Error when reading input #%d of type %s (errno: %s).\n", 6,
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9802;
    int64_t read_shape_9803[1];
    float *read_arr_9804 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9804, read_shape_9803, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 7, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9805;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        ;
        assert((read_value_9790 = futhark_new_f32_1d(ctx, read_arr_9792,
                                                     read_shape_9791[0])) != 0);
        ;
        assert((read_value_9794 = futhark_new_f32_1d(ctx, read_arr_9796,
                                                     read_shape_9795[0])) != 0);
        ;
        assert((read_value_9798 = futhark_new_f32_1d(ctx, read_arr_9800,
                                                     read_shape_9799[0])) != 0);
        ;
        assert((read_value_9802 = futhark_new_f32_1d(ctx, read_arr_9804,
                                                     read_shape_9803[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot5(ctx, &result_9805, read_value_9789,
                               read_value_9790, read_value_9793,
                               read_value_9794, read_value_9797,
                               read_value_9798, read_value_9801,
                               read_value_9802);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9790) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9794) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9798) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9802) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        ;
        assert((read_value_9790 = futhark_new_f32_1d(ctx, read_arr_9792,
                                                     read_shape_9791[0])) != 0);
        ;
        assert((read_value_9794 = futhark_new_f32_1d(ctx, read_arr_9796,
                                                     read_shape_9795[0])) != 0);
        ;
        assert((read_value_9798 = futhark_new_f32_1d(ctx, read_arr_9800,
                                                     read_shape_9799[0])) != 0);
        ;
        assert((read_value_9802 = futhark_new_f32_1d(ctx, read_arr_9804,
                                                     read_shape_9803[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot5(ctx, &result_9805, read_value_9789,
                               read_value_9790, read_value_9793,
                               read_value_9794, read_value_9797,
                               read_value_9798, read_value_9801,
                               read_value_9802);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9790) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9794) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9798) == 0);
        ;
        assert(futhark_free_f32_1d(ctx, read_value_9802) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    ;
    free(read_arr_9792);
    ;
    free(read_arr_9796);
    ;
    free(read_arr_9800);
    ;
    free(read_arr_9804);
    write_scalar(stdout, binary_output, &f32_info, &result_9805);
    printf("\n");
    ;
}
static void futrts_cli_entry_dot6(struct futhark_context *ctx)
{
    int64_t t_start, t_end;
    int time_runs;
    struct futhark_f32_1d *read_value_9806;
    int64_t read_shape_9807[1];
    float *read_arr_9808 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9808, read_shape_9807, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 0, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9809;
    int64_t read_shape_9810[1];
    float *read_arr_9811 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9811, read_shape_9810, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 1, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9812;
    int64_t read_shape_9813[1];
    float *read_arr_9814 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9814, read_shape_9813, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 2, "[]",
              f32_info.type_name, strerror(errno));
    
    struct futhark_f32_1d *read_value_9815;
    int64_t read_shape_9816[1];
    float *read_arr_9817 = NULL;
    
    errno = 0;
    if (read_array(&f32_info, (void **) &read_arr_9817, read_shape_9816, 1) !=
        0)
        panic(1, "Cannot read input #%d of type %s%s (errno: %s).\n", 3, "[]",
              f32_info.type_name, strerror(errno));
    
    float result_9818;
    
    if (perform_warmup) {
        time_runs = 0;
        
        int r;
        
        assert((read_value_9806 = futhark_new_f32_1d(ctx, read_arr_9808,
                                                     read_shape_9807[0])) != 0);
        assert((read_value_9809 = futhark_new_f32_1d(ctx, read_arr_9811,
                                                     read_shape_9810[0])) != 0);
        assert((read_value_9812 = futhark_new_f32_1d(ctx, read_arr_9814,
                                                     read_shape_9813[0])) != 0);
        assert((read_value_9815 = futhark_new_f32_1d(ctx, read_arr_9817,
                                                     read_shape_9816[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot6(ctx, &result_9818, read_value_9806,
                               read_value_9809, read_value_9812,
                               read_value_9815);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9806) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9809) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9812) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9815) == 0);
        ;
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        int r;
        
        assert((read_value_9806 = futhark_new_f32_1d(ctx, read_arr_9808,
                                                     read_shape_9807[0])) != 0);
        assert((read_value_9809 = futhark_new_f32_1d(ctx, read_arr_9811,
                                                     read_shape_9810[0])) != 0);
        assert((read_value_9812 = futhark_new_f32_1d(ctx, read_arr_9814,
                                                     read_shape_9813[0])) != 0);
        assert((read_value_9815 = futhark_new_f32_1d(ctx, read_arr_9817,
                                                     read_shape_9816[0])) != 0);
        assert(futhark_context_sync(ctx) == 0);
        t_start = get_wall_time();
        r = futhark_entry_dot6(ctx, &result_9818, read_value_9806,
                               read_value_9809, read_value_9812,
                               read_value_9815);
        if (r != 0)
            panic(1, "%s", futhark_context_get_error(ctx));
        assert(futhark_context_sync(ctx) == 0);
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%lld\n", (long long) elapsed_usec);
        assert(futhark_free_f32_1d(ctx, read_value_9806) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9809) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9812) == 0);
        assert(futhark_free_f32_1d(ctx, read_value_9815) == 0);
        if (run < num_runs - 1) {
            ;
        }
    }
    free(read_arr_9808);
    free(read_arr_9811);
    free(read_arr_9814);
    free(read_arr_9817);
    write_scalar(stdout, binary_output, &f32_info, &result_9818);
    printf("\n");
    ;
}
typedef void entry_point_fun(struct futhark_context *);
struct entry_point_entry {
    const char *name;
    entry_point_fun *fun;
} ;
int main(int argc, char **argv)
{
    fut_progname = argv[0];
    
    struct entry_point_entry entry_points[] = {{.name ="dot", .fun =
                                                futrts_cli_entry_dot}, {.name =
                                                                        "dot1",
                                                                        .fun =
                                                                        futrts_cli_entry_dot1},
                                               {.name ="dot2", .fun =
                                                futrts_cli_entry_dot2}, {.name =
                                                                         "dot3",
                                                                         .fun =
                                                                         futrts_cli_entry_dot3},
                                               {.name ="dot4", .fun =
                                                futrts_cli_entry_dot4}, {.name =
                                                                         "dot5",
                                                                         .fun =
                                                                         futrts_cli_entry_dot5},
                                               {.name ="dot6", .fun =
                                                futrts_cli_entry_dot6}};
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
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define group_sizze_8973 (group_size_8972)\n#define max_num_groups_8975 (max_num_groups_8974)\n#define group_sizze_9042 (group_size_9041)\n#define max_num_groups_9044 (max_num_g",
            "roups_9043)\n#define group_sizze_9115 (group_size_9114)\n#define max_num_groups_9117 (max_num_groups_9116)\n#define group_sizze_9188 (group_size_9187)\n#define max_num_groups_9190 (max_num_groups_9189)\n#define group_sizze_9264 (group_size_9263)\n#define max_num_groups_9266 (max_num_groups_9265)\n#define group_sizze_9343 (group_size_9342)\n#define max_num_groups_9345 (max_num_groups_9344)\n#define group_sizze_9426 (group_size_9425)\n#define max_num_groups_9428 (max_num_groups_9427)\n__kernel void chunked_reduce_kernel_8989(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8738,\n                                         int32_t num_threads_8981,\n                                         int32_t per_thread_elements_8984,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *mem_9518)\n{\n    __local volatile char *restrict mem_9515 = mem_aligned_0;\n    int32_t wave_sizze_9544;\n    int32_t group_sizze_9545;\n    bool thread_active_9546;\n    int32_t global_tid_8989;\n    int32_t local_tid_8990;\n    int32_t group_id_8991;\n    \n    global_tid_8989 = get_global_id(0);\n    local_tid_8990 = get_local_id(0);\n    group_sizze_9545 = get_local_size(0);\n    wave_sizze_9544 = LOCKSTEP_WIDTH;\n    group_id_8991 = get_group_id(0);\n    thread_active_9546 = 1;\n    \n    int32_t chunk_sizze_8996 = smin32(per_thread_elements_8984,\n                                      squot32(sizze_8738 - global_tid_8989 +\n                                              num_threads_8981 - 1,\n                                              num_threads_8981));\n    float res_9000;\n    \n    if (thread_active_9546) {\n        float acc_9003 = 0.0F;\n        \n        for (int32_t i_9002 = 0; i_9002 < chunk_sizze_8996; i_9002++) {\n            int32_t j_t_s_9499 = num_threads_8981 * i_900",
            "2;\n            int32_t j_p_i_t_s_9500 = global_tid_8989 + j_t_s_9499;\n            float x_9006 = *(__global float *) &A_mem_9510[j_p_i_t_s_9500 * 4];\n            float x_9007 = *(__global float *) &B_mem_9512[j_p_i_t_s_9500 * 4];\n            float res_9009 = x_9006 * x_9007;\n            float res_9011 = acc_9003 + res_9009;\n            float acc_tmp_9547 = res_9011;\n            \n            acc_9003 = acc_tmp_9547;\n        }\n        res_9000 = acc_9003;\n    }\n    \n    float final_result_9014;\n    \n    for (int32_t comb_iter_9548 = 0; comb_iter_9548 < squot32(group_sizze_8973 +\n                                                              group_sizze_8973 -\n                                                              1,\n                                                              group_sizze_8973);\n         comb_iter_9548++) {\n        int32_t combine_id_8994;\n        int32_t flat_comb_id_9549 = comb_iter_9548 * group_sizze_8973 +\n                local_tid_8990;\n        \n        combine_id_8994 = flat_comb_id_9549;\n        if (slt32(combine_id_8994, group_sizze_8973) && 1) {\n            *(__local float *) &mem_9515[combine_id_8994 * 4] = res_9000;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9550;\n    int32_t my_index_9015;\n    int32_t other_offset_9016;\n    float x_9017;\n    float x_9018;\n    \n    my_index_9015 = local_tid_8990;\n    other_offset_9016 = 0;\n    if (slt32(local_tid_8990, group_sizze_8973)) {\n        x_9017 = *(__local float *) &mem_9515[(local_tid_8990 +\n                                               other_offset_9016) * 4];\n    }\n    other_offset_9016 = 1;\n    while (slt32(other_offset_9016, wave_sizze_9544)) {\n        if (slt32(local_tid_8990 + other_offset_9016, group_sizze_8973) &&\n            ((local_tid_8990 - squot32(local_tid_8990, wave_sizze_9544) *\n              wave_sizze_9544) & (2 * other_offset_9016 - 1)) == 0) {\n            // read array element\n            {\n                x_9018 = *(volatile __local f",
            "loat *) &mem_9515[(local_tid_8990 +\n                                                                other_offset_9016) *\n                                                               4];\n            }\n            \n            float res_9019;\n            \n            if (thread_active_9546) {\n                res_9019 = x_9017 + x_9018;\n            }\n            x_9017 = res_9019;\n            *(volatile __local float *) &mem_9515[local_tid_8990 * 4] = x_9017;\n        }\n        other_offset_9016 *= 2;\n    }\n    skip_waves_9550 = 1;\n    while (slt32(skip_waves_9550, squot32(group_sizze_8973 + wave_sizze_9544 -\n                                          1, wave_sizze_9544))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9016 = skip_waves_9550 * wave_sizze_9544;\n        if (slt32(local_tid_8990 + other_offset_9016, group_sizze_8973) &&\n            ((local_tid_8990 - squot32(local_tid_8990, wave_sizze_9544) *\n              wave_sizze_9544) == 0 && (squot32(local_tid_8990,\n                                                wave_sizze_9544) & (2 *\n                                                                    skip_waves_9550 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9018 = *(__local float *) &mem_9515[(local_tid_8990 +\n                                                       other_offset_9016) * 4];\n            }\n            \n            float res_9019;\n            \n            if (thread_active_9546) {\n                res_9019 = x_9017 + x_9018;\n            }\n            x_9017 = res_9019;\n            *(__local float *) &mem_9515[local_tid_8990 * 4] = x_9017;\n        }\n        skip_waves_9550 *= 2;\n    }\n    final_result_9014 = x_9017;\n    if (local_tid_8990 == 0) {\n        *(__global float *) &mem_9518[group_id_8991 * 4] = final_result_9014;\n    }\n}\n__kernel void chunked_reduce_kernel_9058(__local volatile\n                                         int64_",
            "t *mem_aligned_0,\n                                         int32_t sizze_8756,\n                                         int32_t num_threads_9050,\n                                         int32_t per_thread_elements_9053,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *C_mem_9514,\n                                         __global unsigned char *mem_9520)\n{\n    __local volatile char *restrict mem_9517 = mem_aligned_0;\n    int32_t wave_sizze_9560;\n    int32_t group_sizze_9561;\n    bool thread_active_9562;\n    int32_t global_tid_9058;\n    int32_t local_tid_9059;\n    int32_t group_id_9060;\n    \n    global_tid_9058 = get_global_id(0);\n    local_tid_9059 = get_local_id(0);\n    group_sizze_9561 = get_local_size(0);\n    wave_sizze_9560 = LOCKSTEP_WIDTH;\n    group_id_9060 = get_group_id(0);\n    thread_active_9562 = 1;\n    \n    int32_t chunk_sizze_9065 = smin32(per_thread_elements_9053,\n                                      squot32(sizze_8756 - global_tid_9058 +\n                                              num_threads_9050 - 1,\n                                              num_threads_9050));\n    float res_9070;\n    \n    if (thread_active_9562) {\n        float acc_9073 = 0.0F;\n        \n        for (int32_t i_9072 = 0; i_9072 < chunk_sizze_9065; i_9072++) {\n            int32_t j_t_s_9503 = num_threads_9050 * i_9072;\n            int32_t j_p_i_t_s_9504 = global_tid_9058 + j_t_s_9503;\n            float x_9077 = *(__global float *) &A_mem_9510[j_p_i_t_s_9504 * 4];\n            float x_9078 = *(__global float *) &B_mem_9512[j_p_i_t_s_9504 * 4];\n            float x_9079 = *(__global float *) &C_mem_9514[j_p_i_t_s_9504 * 4];\n            float res_9081 = x_9077 * x_9078;\n            float res_9082 = x_9079 * res_9081;\n            float res_9084 = acc_9073 + res_9082;\n            float acc_tmp_9563 = res_9084;\n        ",
            "    \n            acc_9073 = acc_tmp_9563;\n        }\n        res_9070 = acc_9073;\n    }\n    \n    float final_result_9087;\n    \n    for (int32_t comb_iter_9564 = 0; comb_iter_9564 < squot32(group_sizze_9042 +\n                                                              group_sizze_9042 -\n                                                              1,\n                                                              group_sizze_9042);\n         comb_iter_9564++) {\n        int32_t combine_id_9063;\n        int32_t flat_comb_id_9565 = comb_iter_9564 * group_sizze_9042 +\n                local_tid_9059;\n        \n        combine_id_9063 = flat_comb_id_9565;\n        if (slt32(combine_id_9063, group_sizze_9042) && 1) {\n            *(__local float *) &mem_9517[combine_id_9063 * 4] = res_9070;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9566;\n    int32_t my_index_9088;\n    int32_t other_offset_9089;\n    float x_9090;\n    float x_9091;\n    \n    my_index_9088 = local_tid_9059;\n    other_offset_9089 = 0;\n    if (slt32(local_tid_9059, group_sizze_9042)) {\n        x_9090 = *(__local float *) &mem_9517[(local_tid_9059 +\n                                               other_offset_9089) * 4];\n    }\n    other_offset_9089 = 1;\n    while (slt32(other_offset_9089, wave_sizze_9560)) {\n        if (slt32(local_tid_9059 + other_offset_9089, group_sizze_9042) &&\n            ((local_tid_9059 - squot32(local_tid_9059, wave_sizze_9560) *\n              wave_sizze_9560) & (2 * other_offset_9089 - 1)) == 0) {\n            // read array element\n            {\n                x_9091 = *(volatile __local float *) &mem_9517[(local_tid_9059 +\n                                                                other_offset_9089) *\n                                                               4];\n            }\n            \n            float res_9092;\n            \n            if (thread_active_9562) {\n                res_9092 = x_9090 + x_9091;\n            }\n            x_9090 = res_",
            "9092;\n            *(volatile __local float *) &mem_9517[local_tid_9059 * 4] = x_9090;\n        }\n        other_offset_9089 *= 2;\n    }\n    skip_waves_9566 = 1;\n    while (slt32(skip_waves_9566, squot32(group_sizze_9042 + wave_sizze_9560 -\n                                          1, wave_sizze_9560))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9089 = skip_waves_9566 * wave_sizze_9560;\n        if (slt32(local_tid_9059 + other_offset_9089, group_sizze_9042) &&\n            ((local_tid_9059 - squot32(local_tid_9059, wave_sizze_9560) *\n              wave_sizze_9560) == 0 && (squot32(local_tid_9059,\n                                                wave_sizze_9560) & (2 *\n                                                                    skip_waves_9566 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9091 = *(__local float *) &mem_9517[(local_tid_9059 +\n                                                       other_offset_9089) * 4];\n            }\n            \n            float res_9092;\n            \n            if (thread_active_9562) {\n                res_9092 = x_9090 + x_9091;\n            }\n            x_9090 = res_9092;\n            *(__local float *) &mem_9517[local_tid_9059 * 4] = x_9090;\n        }\n        skip_waves_9566 *= 2;\n    }\n    final_result_9087 = x_9090;\n    if (local_tid_9059 == 0) {\n        *(__global float *) &mem_9520[group_id_9060 * 4] = final_result_9087;\n    }\n}\n__kernel void chunked_reduce_kernel_9131(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8784,\n                                         int32_t num_threads_9123,\n                                         int32_t per_thread_elements_9126,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n  ",
            "                                       __global unsigned char *C_mem_9514,\n                                         __global unsigned char *mem_9520)\n{\n    __local volatile char *restrict mem_9517 = mem_aligned_0;\n    int32_t wave_sizze_9576;\n    int32_t group_sizze_9577;\n    bool thread_active_9578;\n    int32_t global_tid_9131;\n    int32_t local_tid_9132;\n    int32_t group_id_9133;\n    \n    global_tid_9131 = get_global_id(0);\n    local_tid_9132 = get_local_id(0);\n    group_sizze_9577 = get_local_size(0);\n    wave_sizze_9576 = LOCKSTEP_WIDTH;\n    group_id_9133 = get_group_id(0);\n    thread_active_9578 = 1;\n    \n    int32_t chunk_sizze_9138 = smin32(per_thread_elements_9126,\n                                      squot32(sizze_8784 - global_tid_9131 +\n                                              num_threads_9123 - 1,\n                                              num_threads_9123));\n    float res_9143;\n    \n    if (thread_active_9578) {\n        float acc_9146 = 0.0F;\n        \n        for (int32_t i_9145 = 0; i_9145 < chunk_sizze_9138; i_9145++) {\n            int32_t j_t_s_9503 = num_threads_9123 * i_9145;\n            int32_t j_p_i_t_s_9504 = global_tid_9131 + j_t_s_9503;\n            float x_9150 = *(__global float *) &A_mem_9510[j_p_i_t_s_9504 * 4];\n            float x_9151 = *(__global float *) &B_mem_9512[j_p_i_t_s_9504 * 4];\n            float x_9152 = *(__global float *) &C_mem_9514[j_p_i_t_s_9504 * 4];\n            float res_9154 = x_9150 + x_9151;\n            float res_9155 = x_9152 * res_9154;\n            float res_9157 = acc_9146 + res_9155;\n            float acc_tmp_9579 = res_9157;\n            \n            acc_9146 = acc_tmp_9579;\n        }\n        res_9143 = acc_9146;\n    }\n    \n    float final_result_9160;\n    \n    for (int32_t comb_iter_9580 = 0; comb_iter_9580 < squot32(group_sizze_9115 +\n                                                              group_sizze_9115 -\n                                                              1,\n                        ",
            "                                      group_sizze_9115);\n         comb_iter_9580++) {\n        int32_t combine_id_9136;\n        int32_t flat_comb_id_9581 = comb_iter_9580 * group_sizze_9115 +\n                local_tid_9132;\n        \n        combine_id_9136 = flat_comb_id_9581;\n        if (slt32(combine_id_9136, group_sizze_9115) && 1) {\n            *(__local float *) &mem_9517[combine_id_9136 * 4] = res_9143;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9582;\n    int32_t my_index_9161;\n    int32_t other_offset_9162;\n    float x_9163;\n    float x_9164;\n    \n    my_index_9161 = local_tid_9132;\n    other_offset_9162 = 0;\n    if (slt32(local_tid_9132, group_sizze_9115)) {\n        x_9163 = *(__local float *) &mem_9517[(local_tid_9132 +\n                                               other_offset_9162) * 4];\n    }\n    other_offset_9162 = 1;\n    while (slt32(other_offset_9162, wave_sizze_9576)) {\n        if (slt32(local_tid_9132 + other_offset_9162, group_sizze_9115) &&\n            ((local_tid_9132 - squot32(local_tid_9132, wave_sizze_9576) *\n              wave_sizze_9576) & (2 * other_offset_9162 - 1)) == 0) {\n            // read array element\n            {\n                x_9164 = *(volatile __local float *) &mem_9517[(local_tid_9132 +\n                                                                other_offset_9162) *\n                                                               4];\n            }\n            \n            float res_9165;\n            \n            if (thread_active_9578) {\n                res_9165 = x_9163 + x_9164;\n            }\n            x_9163 = res_9165;\n            *(volatile __local float *) &mem_9517[local_tid_9132 * 4] = x_9163;\n        }\n        other_offset_9162 *= 2;\n    }\n    skip_waves_9582 = 1;\n    while (slt32(skip_waves_9582, squot32(group_sizze_9115 + wave_sizze_9576 -\n                                          1, wave_sizze_9576))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9162 = skip_wa",
            "ves_9582 * wave_sizze_9576;\n        if (slt32(local_tid_9132 + other_offset_9162, group_sizze_9115) &&\n            ((local_tid_9132 - squot32(local_tid_9132, wave_sizze_9576) *\n              wave_sizze_9576) == 0 && (squot32(local_tid_9132,\n                                                wave_sizze_9576) & (2 *\n                                                                    skip_waves_9582 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9164 = *(__local float *) &mem_9517[(local_tid_9132 +\n                                                       other_offset_9162) * 4];\n            }\n            \n            float res_9165;\n            \n            if (thread_active_9578) {\n                res_9165 = x_9163 + x_9164;\n            }\n            x_9163 = res_9165;\n            *(__local float *) &mem_9517[local_tid_9132 * 4] = x_9163;\n        }\n        skip_waves_9582 *= 2;\n    }\n    final_result_9160 = x_9163;\n    if (local_tid_9132 == 0) {\n        *(__global float *) &mem_9520[group_id_9133 * 4] = final_result_9160;\n    }\n}\n__kernel void chunked_reduce_kernel_9204(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8812,\n                                         int32_t num_threads_9196,\n                                         int32_t per_thread_elements_9199,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *C_mem_9514,\n                                         __global unsigned char *mem_9520)\n{\n    __local volatile char *restrict mem_9517 = mem_aligned_0;\n    int32_t wave_sizze_9592;\n    int32_t group_sizze_9593;\n    bool thread_active_9594;\n    int32_t global_tid_9204;\n    int32_t local_tid_9205;\n    int32_t group_",
            "id_9206;\n    \n    global_tid_9204 = get_global_id(0);\n    local_tid_9205 = get_local_id(0);\n    group_sizze_9593 = get_local_size(0);\n    wave_sizze_9592 = LOCKSTEP_WIDTH;\n    group_id_9206 = get_group_id(0);\n    thread_active_9594 = 1;\n    \n    int32_t chunk_sizze_9211 = smin32(per_thread_elements_9199,\n                                      squot32(sizze_8812 - global_tid_9204 +\n                                              num_threads_9196 - 1,\n                                              num_threads_9196));\n    float res_9216;\n    \n    if (thread_active_9594) {\n        float acc_9219 = 0.0F;\n        \n        for (int32_t i_9218 = 0; i_9218 < chunk_sizze_9211; i_9218++) {\n            int32_t j_t_s_9503 = num_threads_9196 * i_9218;\n            int32_t j_p_i_t_s_9504 = global_tid_9204 + j_t_s_9503;\n            float x_9223 = *(__global float *) &A_mem_9510[j_p_i_t_s_9504 * 4];\n            float x_9224 = *(__global float *) &B_mem_9512[j_p_i_t_s_9504 * 4];\n            float x_9225 = *(__global float *) &C_mem_9514[j_p_i_t_s_9504 * 4];\n            float res_9227 = x_9223 + x_9224;\n            float res_9228 = x_9223 - x_9225;\n            float res_9229 = res_9227 * res_9228;\n            float res_9231 = acc_9219 + res_9229;\n            float acc_tmp_9595 = res_9231;\n            \n            acc_9219 = acc_tmp_9595;\n        }\n        res_9216 = acc_9219;\n    }\n    \n    float final_result_9234;\n    \n    for (int32_t comb_iter_9596 = 0; comb_iter_9596 < squot32(group_sizze_9188 +\n                                                              group_sizze_9188 -\n                                                              1,\n                                                              group_sizze_9188);\n         comb_iter_9596++) {\n        int32_t combine_id_9209;\n        int32_t flat_comb_id_9597 = comb_iter_9596 * group_sizze_9188 +\n                local_tid_9205;\n        \n        combine_id_9209 = flat_comb_id_9597;\n        if (slt32(combine_id_9209, group_sizze_9188)",
            " && 1) {\n            *(__local float *) &mem_9517[combine_id_9209 * 4] = res_9216;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9598;\n    int32_t my_index_9235;\n    int32_t other_offset_9236;\n    float x_9237;\n    float x_9238;\n    \n    my_index_9235 = local_tid_9205;\n    other_offset_9236 = 0;\n    if (slt32(local_tid_9205, group_sizze_9188)) {\n        x_9237 = *(__local float *) &mem_9517[(local_tid_9205 +\n                                               other_offset_9236) * 4];\n    }\n    other_offset_9236 = 1;\n    while (slt32(other_offset_9236, wave_sizze_9592)) {\n        if (slt32(local_tid_9205 + other_offset_9236, group_sizze_9188) &&\n            ((local_tid_9205 - squot32(local_tid_9205, wave_sizze_9592) *\n              wave_sizze_9592) & (2 * other_offset_9236 - 1)) == 0) {\n            // read array element\n            {\n                x_9238 = *(volatile __local float *) &mem_9517[(local_tid_9205 +\n                                                                other_offset_9236) *\n                                                               4];\n            }\n            \n            float res_9239;\n            \n            if (thread_active_9594) {\n                res_9239 = x_9237 + x_9238;\n            }\n            x_9237 = res_9239;\n            *(volatile __local float *) &mem_9517[local_tid_9205 * 4] = x_9237;\n        }\n        other_offset_9236 *= 2;\n    }\n    skip_waves_9598 = 1;\n    while (slt32(skip_waves_9598, squot32(group_sizze_9188 + wave_sizze_9592 -\n                                          1, wave_sizze_9592))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9236 = skip_waves_9598 * wave_sizze_9592;\n        if (slt32(local_tid_9205 + other_offset_9236, group_sizze_9188) &&\n            ((local_tid_9205 - squot32(local_tid_9205, wave_sizze_9592) *\n              wave_sizze_9592) == 0 && (squot32(local_tid_9205,\n                                                wave_sizze_9592) & (2 *\n                ",
            "                                                    skip_waves_9598 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9238 = *(__local float *) &mem_9517[(local_tid_9205 +\n                                                       other_offset_9236) * 4];\n            }\n            \n            float res_9239;\n            \n            if (thread_active_9594) {\n                res_9239 = x_9237 + x_9238;\n            }\n            x_9237 = res_9239;\n            *(__local float *) &mem_9517[local_tid_9205 * 4] = x_9237;\n        }\n        skip_waves_9598 *= 2;\n    }\n    final_result_9234 = x_9237;\n    if (local_tid_9205 == 0) {\n        *(__global float *) &mem_9520[group_id_9206 * 4] = final_result_9234;\n    }\n}\n__kernel void chunked_reduce_kernel_9280(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8841,\n                                         int32_t num_threads_9272,\n                                         int32_t per_thread_elements_9275,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *C_mem_9514,\n                                         __global unsigned char *D_mem_9516,\n                                         __global unsigned char *mem_9522)\n{\n    __local volatile char *restrict mem_9519 = mem_aligned_0;\n    int32_t wave_sizze_9608;\n    int32_t group_sizze_9609;\n    bool thread_active_9610;\n    int32_t global_tid_9280;\n    int32_t local_tid_9281;\n    int32_t group_id_9282;\n    \n    global_tid_9280 = get_global_id(0);\n    local_tid_9281 = get_local_id(0);\n    group_sizze_9609 = get_local_size(0);\n    wave_sizze_9608 = LOCKSTEP_WIDTH;\n    group_id_9282 = get_group_id(0);\n    thread_active_9610 = 1;\n    \n    int32_",
            "t chunk_sizze_9287 = smin32(per_thread_elements_9275,\n                                      squot32(sizze_8841 - global_tid_9280 +\n                                              num_threads_9272 - 1,\n                                              num_threads_9272));\n    float res_9293;\n    \n    if (thread_active_9610) {\n        float acc_9296 = 0.0F;\n        \n        for (int32_t i_9295 = 0; i_9295 < chunk_sizze_9287; i_9295++) {\n            int32_t j_t_s_9507 = num_threads_9272 * i_9295;\n            int32_t j_p_i_t_s_9508 = global_tid_9280 + j_t_s_9507;\n            float x_9301 = *(__global float *) &A_mem_9510[j_p_i_t_s_9508 * 4];\n            float x_9302 = *(__global float *) &B_mem_9512[j_p_i_t_s_9508 * 4];\n            float x_9303 = *(__global float *) &C_mem_9514[j_p_i_t_s_9508 * 4];\n            float x_9304 = *(__global float *) &D_mem_9516[j_p_i_t_s_9508 * 4];\n            float res_9306 = x_9301 + x_9302;\n            float res_9307 = x_9303 - x_9304;\n            float res_9308 = res_9306 * res_9307;\n            float res_9310 = acc_9296 + res_9308;\n            float acc_tmp_9611 = res_9310;\n            \n            acc_9296 = acc_tmp_9611;\n        }\n        res_9293 = acc_9296;\n    }\n    \n    float final_result_9313;\n    \n    for (int32_t comb_iter_9612 = 0; comb_iter_9612 < squot32(group_sizze_9264 +\n                                                              group_sizze_9264 -\n                                                              1,\n                                                              group_sizze_9264);\n         comb_iter_9612++) {\n        int32_t combine_id_9285;\n        int32_t flat_comb_id_9613 = comb_iter_9612 * group_sizze_9264 +\n                local_tid_9281;\n        \n        combine_id_9285 = flat_comb_id_9613;\n        if (slt32(combine_id_9285, group_sizze_9264) && 1) {\n            *(__local float *) &mem_9519[combine_id_9285 * 4] = res_9293;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9614;\n    i",
            "nt32_t my_index_9314;\n    int32_t other_offset_9315;\n    float x_9316;\n    float x_9317;\n    \n    my_index_9314 = local_tid_9281;\n    other_offset_9315 = 0;\n    if (slt32(local_tid_9281, group_sizze_9264)) {\n        x_9316 = *(__local float *) &mem_9519[(local_tid_9281 +\n                                               other_offset_9315) * 4];\n    }\n    other_offset_9315 = 1;\n    while (slt32(other_offset_9315, wave_sizze_9608)) {\n        if (slt32(local_tid_9281 + other_offset_9315, group_sizze_9264) &&\n            ((local_tid_9281 - squot32(local_tid_9281, wave_sizze_9608) *\n              wave_sizze_9608) & (2 * other_offset_9315 - 1)) == 0) {\n            // read array element\n            {\n                x_9317 = *(volatile __local float *) &mem_9519[(local_tid_9281 +\n                                                                other_offset_9315) *\n                                                               4];\n            }\n            \n            float res_9318;\n            \n            if (thread_active_9610) {\n                res_9318 = x_9316 + x_9317;\n            }\n            x_9316 = res_9318;\n            *(volatile __local float *) &mem_9519[local_tid_9281 * 4] = x_9316;\n        }\n        other_offset_9315 *= 2;\n    }\n    skip_waves_9614 = 1;\n    while (slt32(skip_waves_9614, squot32(group_sizze_9264 + wave_sizze_9608 -\n                                          1, wave_sizze_9608))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9315 = skip_waves_9614 * wave_sizze_9608;\n        if (slt32(local_tid_9281 + other_offset_9315, group_sizze_9264) &&\n            ((local_tid_9281 - squot32(local_tid_9281, wave_sizze_9608) *\n              wave_sizze_9608) == 0 && (squot32(local_tid_9281,\n                                                wave_sizze_9608) & (2 *\n                                                                    skip_waves_9614 -\n                                                                    1)) == 0)) {\n            // read a",
            "rray element\n            {\n                x_9317 = *(__local float *) &mem_9519[(local_tid_9281 +\n                                                       other_offset_9315) * 4];\n            }\n            \n            float res_9318;\n            \n            if (thread_active_9610) {\n                res_9318 = x_9316 + x_9317;\n            }\n            x_9316 = res_9318;\n            *(__local float *) &mem_9519[local_tid_9281 * 4] = x_9316;\n        }\n        skip_waves_9614 *= 2;\n    }\n    final_result_9313 = x_9316;\n    if (local_tid_9281 == 0) {\n        *(__global float *) &mem_9522[group_id_9282 * 4] = final_result_9313;\n    }\n}\n__kernel void chunked_reduce_kernel_9359(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8879, float a_8883,\n                                         float b_8885, float c_8887,\n                                         float d_8889, int32_t num_threads_9351,\n                                         int32_t per_thread_elements_9354,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *C_mem_9514,\n                                         __global unsigned char *D_mem_9516,\n                                         __global unsigned char *mem_9522)\n{\n    __local volatile char *restrict mem_9519 = mem_aligned_0;\n    int32_t wave_sizze_9624;\n    int32_t group_sizze_9625;\n    bool thread_active_9626;\n    int32_t global_tid_9359;\n    int32_t local_tid_9360;\n    int32_t group_id_9361;\n    \n    global_tid_9359 = get_global_id(0);\n    local_tid_9360 = get_local_id(0);\n    group_sizze_9625 = get_local_size(0);\n    wave_sizze_9624 = LOCKSTEP_WIDTH;\n    group_id_9361 = get_group_id(0);\n    thread_active_9626 = 1;\n    \n    int32_t chunk_sizze_9366 = smin32(per_thread_elements_9354,\n                     ",
            "                 squot32(sizze_8879 - global_tid_9359 +\n                                              num_threads_9351 - 1,\n                                              num_threads_9351));\n    float res_9372;\n    \n    if (thread_active_9626) {\n        float acc_9375 = 0.0F;\n        \n        for (int32_t i_9374 = 0; i_9374 < chunk_sizze_9366; i_9374++) {\n            int32_t j_t_s_9507 = num_threads_9351 * i_9374;\n            int32_t j_p_i_t_s_9508 = global_tid_9359 + j_t_s_9507;\n            float x_9380 = *(__global float *) &A_mem_9510[j_p_i_t_s_9508 * 4];\n            float x_9381 = *(__global float *) &B_mem_9512[j_p_i_t_s_9508 * 4];\n            float x_9382 = *(__global float *) &C_mem_9514[j_p_i_t_s_9508 * 4];\n            float x_9383 = *(__global float *) &D_mem_9516[j_p_i_t_s_9508 * 4];\n            float res_9385 = a_8883 * x_9380;\n            float res_9386 = b_8885 * x_9381;\n            float res_9387 = res_9385 + res_9386;\n            float res_9388 = c_8887 * x_9382;\n            float res_9389 = d_8889 * x_9383;\n            float res_9390 = res_9388 + res_9389;\n            float res_9391 = res_9387 * res_9390;\n            float res_9393 = acc_9375 + res_9391;\n            float acc_tmp_9627 = res_9393;\n            \n            acc_9375 = acc_tmp_9627;\n        }\n        res_9372 = acc_9375;\n    }\n    \n    float final_result_9396;\n    \n    for (int32_t comb_iter_9628 = 0; comb_iter_9628 < squot32(group_sizze_9343 +\n                                                              group_sizze_9343 -\n                                                              1,\n                                                              group_sizze_9343);\n         comb_iter_9628++) {\n        int32_t combine_id_9364;\n        int32_t flat_comb_id_9629 = comb_iter_9628 * group_sizze_9343 +\n                local_tid_9360;\n        \n        combine_id_9364 = flat_comb_id_9629;\n        if (slt32(combine_id_9364, group_sizze_9343) && 1) {\n            *(__local float *) &mem_9519[combi",
            "ne_id_9364 * 4] = res_9372;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9630;\n    int32_t my_index_9397;\n    int32_t other_offset_9398;\n    float x_9399;\n    float x_9400;\n    \n    my_index_9397 = local_tid_9360;\n    other_offset_9398 = 0;\n    if (slt32(local_tid_9360, group_sizze_9343)) {\n        x_9399 = *(__local float *) &mem_9519[(local_tid_9360 +\n                                               other_offset_9398) * 4];\n    }\n    other_offset_9398 = 1;\n    while (slt32(other_offset_9398, wave_sizze_9624)) {\n        if (slt32(local_tid_9360 + other_offset_9398, group_sizze_9343) &&\n            ((local_tid_9360 - squot32(local_tid_9360, wave_sizze_9624) *\n              wave_sizze_9624) & (2 * other_offset_9398 - 1)) == 0) {\n            // read array element\n            {\n                x_9400 = *(volatile __local float *) &mem_9519[(local_tid_9360 +\n                                                                other_offset_9398) *\n                                                               4];\n            }\n            \n            float res_9401;\n            \n            if (thread_active_9626) {\n                res_9401 = x_9399 + x_9400;\n            }\n            x_9399 = res_9401;\n            *(volatile __local float *) &mem_9519[local_tid_9360 * 4] = x_9399;\n        }\n        other_offset_9398 *= 2;\n    }\n    skip_waves_9630 = 1;\n    while (slt32(skip_waves_9630, squot32(group_sizze_9343 + wave_sizze_9624 -\n                                          1, wave_sizze_9624))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9398 = skip_waves_9630 * wave_sizze_9624;\n        if (slt32(local_tid_9360 + other_offset_9398, group_sizze_9343) &&\n            ((local_tid_9360 - squot32(local_tid_9360, wave_sizze_9624) *\n              wave_sizze_9624) == 0 && (squot32(local_tid_9360,\n                                                wave_sizze_9624) & (2 *\n                                                                    ski",
            "p_waves_9630 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9400 = *(__local float *) &mem_9519[(local_tid_9360 +\n                                                       other_offset_9398) * 4];\n            }\n            \n            float res_9401;\n            \n            if (thread_active_9626) {\n                res_9401 = x_9399 + x_9400;\n            }\n            x_9399 = res_9401;\n            *(__local float *) &mem_9519[local_tid_9360 * 4] = x_9399;\n        }\n        skip_waves_9630 *= 2;\n    }\n    final_result_9396 = x_9399;\n    if (local_tid_9360 == 0) {\n        *(__global float *) &mem_9522[group_id_9361 * 4] = final_result_9396;\n    }\n}\n__kernel void chunked_reduce_kernel_9442(__local volatile\n                                         int64_t *mem_aligned_0,\n                                         int32_t sizze_8925,\n                                         int32_t num_threads_9434,\n                                         int32_t per_thread_elements_9437,\n                                         __global unsigned char *A_mem_9510,\n                                         __global unsigned char *B_mem_9512,\n                                         __global unsigned char *C_mem_9514,\n                                         __global unsigned char *D_mem_9516,\n                                         __global unsigned char *mem_9522)\n{\n    __local volatile char *restrict mem_9519 = mem_aligned_0;\n    int32_t wave_sizze_9640;\n    int32_t group_sizze_9641;\n    bool thread_active_9642;\n    int32_t global_tid_9442;\n    int32_t local_tid_9443;\n    int32_t group_id_9444;\n    \n    global_tid_9442 = get_global_id(0);\n    local_tid_9443 = get_local_id(0);\n    group_sizze_9641 = get_local_size(0);\n    wave_sizze_9640 = LOCKSTEP_WIDTH;\n    group_id_9444 = get_group_id(0);\n    thread_active_9642 = 1;\n    \n    int32_t chunk_sizze_9449 = smin32(per_thread_elements_9437,\n ",
            "                                     squot32(sizze_8925 - global_tid_9442 +\n                                              num_threads_9434 - 1,\n                                              num_threads_9434));\n    float res_9455;\n    \n    if (thread_active_9642) {\n        float acc_9458 = 0.0F;\n        \n        for (int32_t i_9457 = 0; i_9457 < chunk_sizze_9449; i_9457++) {\n            int32_t j_t_s_9507 = num_threads_9434 * i_9457;\n            int32_t j_p_i_t_s_9508 = global_tid_9442 + j_t_s_9507;\n            float x_9463 = *(__global float *) &A_mem_9510[j_p_i_t_s_9508 * 4];\n            float x_9464 = *(__global float *) &B_mem_9512[j_p_i_t_s_9508 * 4];\n            float x_9465 = *(__global float *) &C_mem_9514[j_p_i_t_s_9508 * 4];\n            float x_9466 = *(__global float *) &D_mem_9516[j_p_i_t_s_9508 * 4];\n            float res_9468 = x_9463 + x_9464;\n            float res_9469 = x_9465 - x_9466;\n            float res_9470 = res_9468 * res_9469;\n            float res_9472 = acc_9458 + res_9470;\n            float acc_tmp_9643 = res_9472;\n            \n            acc_9458 = acc_tmp_9643;\n        }\n        res_9455 = acc_9458;\n    }\n    \n    float final_result_9475;\n    \n    for (int32_t comb_iter_9644 = 0; comb_iter_9644 < squot32(group_sizze_9426 +\n                                                              group_sizze_9426 -\n                                                              1,\n                                                              group_sizze_9426);\n         comb_iter_9644++) {\n        int32_t combine_id_9447;\n        int32_t flat_comb_id_9645 = comb_iter_9644 * group_sizze_9426 +\n                local_tid_9443;\n        \n        combine_id_9447 = flat_comb_id_9645;\n        if (slt32(combine_id_9447, group_sizze_9426) && 1) {\n            *(__local float *) &mem_9519[combine_id_9447 * 4] = res_9455;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9646;\n    int32_t my_index_9476;\n    int32_t other_offset_9477;\n  ",
            "  float x_9478;\n    float x_9479;\n    \n    my_index_9476 = local_tid_9443;\n    other_offset_9477 = 0;\n    if (slt32(local_tid_9443, group_sizze_9426)) {\n        x_9478 = *(__local float *) &mem_9519[(local_tid_9443 +\n                                               other_offset_9477) * 4];\n    }\n    other_offset_9477 = 1;\n    while (slt32(other_offset_9477, wave_sizze_9640)) {\n        if (slt32(local_tid_9443 + other_offset_9477, group_sizze_9426) &&\n            ((local_tid_9443 - squot32(local_tid_9443, wave_sizze_9640) *\n              wave_sizze_9640) & (2 * other_offset_9477 - 1)) == 0) {\n            // read array element\n            {\n                x_9479 = *(volatile __local float *) &mem_9519[(local_tid_9443 +\n                                                                other_offset_9477) *\n                                                               4];\n            }\n            \n            float res_9480;\n            \n            if (thread_active_9642) {\n                res_9480 = x_9478 + x_9479;\n            }\n            x_9478 = res_9480;\n            *(volatile __local float *) &mem_9519[local_tid_9443 * 4] = x_9478;\n        }\n        other_offset_9477 *= 2;\n    }\n    skip_waves_9646 = 1;\n    while (slt32(skip_waves_9646, squot32(group_sizze_9426 + wave_sizze_9640 -\n                                          1, wave_sizze_9640))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9477 = skip_waves_9646 * wave_sizze_9640;\n        if (slt32(local_tid_9443 + other_offset_9477, group_sizze_9426) &&\n            ((local_tid_9443 - squot32(local_tid_9443, wave_sizze_9640) *\n              wave_sizze_9640) == 0 && (squot32(local_tid_9443,\n                                                wave_sizze_9640) & (2 *\n                                                                    skip_waves_9646 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_9479 = *(_",
            "_local float *) &mem_9519[(local_tid_9443 +\n                                                       other_offset_9477) * 4];\n            }\n            \n            float res_9480;\n            \n            if (thread_active_9642) {\n                res_9480 = x_9478 + x_9479;\n            }\n            x_9478 = res_9480;\n            *(__local float *) &mem_9519[local_tid_9443 * 4] = x_9478;\n        }\n        skip_waves_9646 *= 2;\n    }\n    final_result_9475 = x_9478;\n    if (local_tid_9443 == 0) {\n        *(__global float *) &mem_9522[group_id_9444 * 4] = final_result_9475;\n    }\n}\n__kernel void reduce_kernel_9021(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_8980, __global\n                                 unsigned char *mem_9518, __global\n                                 unsigned char *mem_9524)\n{\n    __local volatile char *restrict mem_9521 = mem_aligned_0;\n    int32_t wave_sizze_9552;\n    int32_t group_sizze_9553;\n    bool thread_active_9554;\n    int32_t global_tid_9021;\n    int32_t local_tid_9022;\n    int32_t group_id_9023;\n    \n    global_tid_9021 = get_global_id(0);\n    local_tid_9022 = get_local_id(0);\n    group_sizze_9553 = get_local_size(0);\n    wave_sizze_9552 = LOCKSTEP_WIDTH;\n    group_id_9023 = get_group_id(0);\n    thread_active_9554 = 1;\n    \n    bool in_bounds_9024;\n    float x_9491;\n    \n    if (thread_active_9554) {\n        in_bounds_9024 = slt32(local_tid_9022, num_groups_8980);\n        if (in_bounds_9024) {\n            float x_9025 = *(__global float *) &mem_9518[global_tid_9021 * 4];\n            \n            x_9491 = x_9025;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9029;\n    \n    for (int32_t comb_iter_9555 = 0; comb_iter_9555 <\n         squot32(max_num_groups_8975 + max_num_groups_8975 - 1,\n                 max_num_groups_8975); comb_iter_9555++) {\n        int32_t combine_id_9028;\n        int32_t flat_comb_id_9556 = comb_iter_9555 * max_num_groups_8975 +\n  ",
            "              local_tid_9022;\n        \n        combine_id_9028 = flat_comb_id_9556;\n        if (slt32(combine_id_9028, max_num_groups_8975) && 1) {\n            *(__local float *) &mem_9521[combine_id_9028 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9557;\n    float x_8750;\n    float x_8751;\n    int32_t my_index_8987;\n    int32_t other_offset_8988;\n    \n    my_index_8987 = local_tid_9022;\n    other_offset_8988 = 0;\n    if (slt32(local_tid_9022, max_num_groups_8975)) {\n        x_8750 = *(__local float *) &mem_9521[(local_tid_9022 +\n                                               other_offset_8988) * 4];\n    }\n    other_offset_8988 = 1;\n    while (slt32(other_offset_8988, wave_sizze_9552)) {\n        if (slt32(local_tid_9022 + other_offset_8988, max_num_groups_8975) &&\n            ((local_tid_9022 - squot32(local_tid_9022, wave_sizze_9552) *\n              wave_sizze_9552) & (2 * other_offset_8988 - 1)) == 0) {\n            // read array element\n            {\n                x_8751 = *(volatile __local float *) &mem_9521[(local_tid_9022 +\n                                                                other_offset_8988) *\n                                                               4];\n            }\n            \n            float res_8752;\n            \n            if (thread_active_9554) {\n                res_8752 = x_8750 + x_8751;\n            }\n            x_8750 = res_8752;\n            *(volatile __local float *) &mem_9521[local_tid_9022 * 4] = x_8750;\n        }\n        other_offset_8988 *= 2;\n    }\n    skip_waves_9557 = 1;\n    while (slt32(skip_waves_9557, squot32(max_num_groups_8975 +\n                                          wave_sizze_9552 - 1,\n                                          wave_sizze_9552))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_8988 = skip_waves_9557 * wave_sizze_9552;\n        if (slt32(local_tid_9022 + other_offset_8988, max_num_groups_8975) &&\n            ((local_tid_9022 - squo",
            "t32(local_tid_9022, wave_sizze_9552) *\n              wave_sizze_9552) == 0 && (squot32(local_tid_9022,\n                                                wave_sizze_9552) & (2 *\n                                                                    skip_waves_9557 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8751 = *(__local float *) &mem_9521[(local_tid_9022 +\n                                                       other_offset_8988) * 4];\n            }\n            \n            float res_8752;\n            \n            if (thread_active_9554) {\n                res_8752 = x_8750 + x_8751;\n            }\n            x_8750 = res_8752;\n            *(__local float *) &mem_9521[local_tid_9022 * 4] = x_8750;\n        }\n        skip_waves_9557 *= 2;\n    }\n    final_result_9029 = x_8750;\n    if (local_tid_9022 == 0) {\n        *(__global float *) &mem_9524[group_id_9023 * 4] = final_result_9029;\n    }\n}\n__kernel void reduce_kernel_9094(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9049, __global\n                                 unsigned char *mem_9520, __global\n                                 unsigned char *mem_9526)\n{\n    __local volatile char *restrict mem_9523 = mem_aligned_0;\n    int32_t wave_sizze_9568;\n    int32_t group_sizze_9569;\n    bool thread_active_9570;\n    int32_t global_tid_9094;\n    int32_t local_tid_9095;\n    int32_t group_id_9096;\n    \n    global_tid_9094 = get_global_id(0);\n    local_tid_9095 = get_local_id(0);\n    group_sizze_9569 = get_local_size(0);\n    wave_sizze_9568 = LOCKSTEP_WIDTH;\n    group_id_9096 = get_group_id(0);\n    thread_active_9570 = 1;\n    \n    bool in_bounds_9097;\n    float x_9491;\n    \n    if (thread_active_9570) {\n        in_bounds_9097 = slt32(local_tid_9095, num_groups_9049);\n        if (in_bounds_9097) {\n            float x_9098 = *(__global float *) &mem_9520[global_tid_9094 * 4];\n          ",
            "  \n            x_9491 = x_9098;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9102;\n    \n    for (int32_t comb_iter_9571 = 0; comb_iter_9571 <\n         squot32(max_num_groups_9044 + max_num_groups_9044 - 1,\n                 max_num_groups_9044); comb_iter_9571++) {\n        int32_t combine_id_9101;\n        int32_t flat_comb_id_9572 = comb_iter_9571 * max_num_groups_9044 +\n                local_tid_9095;\n        \n        combine_id_9101 = flat_comb_id_9572;\n        if (slt32(combine_id_9101, max_num_groups_9044) && 1) {\n            *(__local float *) &mem_9523[combine_id_9101 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9573;\n    float x_8776;\n    float x_8777;\n    int32_t my_index_9056;\n    int32_t other_offset_9057;\n    \n    my_index_9056 = local_tid_9095;\n    other_offset_9057 = 0;\n    if (slt32(local_tid_9095, max_num_groups_9044)) {\n        x_8776 = *(__local float *) &mem_9523[(local_tid_9095 +\n                                               other_offset_9057) * 4];\n    }\n    other_offset_9057 = 1;\n    while (slt32(other_offset_9057, wave_sizze_9568)) {\n        if (slt32(local_tid_9095 + other_offset_9057, max_num_groups_9044) &&\n            ((local_tid_9095 - squot32(local_tid_9095, wave_sizze_9568) *\n              wave_sizze_9568) & (2 * other_offset_9057 - 1)) == 0) {\n            // read array element\n            {\n                x_8777 = *(volatile __local float *) &mem_9523[(local_tid_9095 +\n                                                                other_offset_9057) *\n                                                               4];\n            }\n            \n            float res_8778;\n            \n            if (thread_active_9570) {\n                res_8778 = x_8776 + x_8777;\n            }\n            x_8776 = res_8778;\n            *(volatile __local float *) &mem_9523[local_tid_9095 * 4] = x_8776;\n        }\n        other_offset_9057 *= 2;\n    }\n    skip_wa",
            "ves_9573 = 1;\n    while (slt32(skip_waves_9573, squot32(max_num_groups_9044 +\n                                          wave_sizze_9568 - 1,\n                                          wave_sizze_9568))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9057 = skip_waves_9573 * wave_sizze_9568;\n        if (slt32(local_tid_9095 + other_offset_9057, max_num_groups_9044) &&\n            ((local_tid_9095 - squot32(local_tid_9095, wave_sizze_9568) *\n              wave_sizze_9568) == 0 && (squot32(local_tid_9095,\n                                                wave_sizze_9568) & (2 *\n                                                                    skip_waves_9573 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8777 = *(__local float *) &mem_9523[(local_tid_9095 +\n                                                       other_offset_9057) * 4];\n            }\n            \n            float res_8778;\n            \n            if (thread_active_9570) {\n                res_8778 = x_8776 + x_8777;\n            }\n            x_8776 = res_8778;\n            *(__local float *) &mem_9523[local_tid_9095 * 4] = x_8776;\n        }\n        skip_waves_9573 *= 2;\n    }\n    final_result_9102 = x_8776;\n    if (local_tid_9095 == 0) {\n        *(__global float *) &mem_9526[group_id_9096 * 4] = final_result_9102;\n    }\n}\n__kernel void reduce_kernel_9167(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9122, __global\n                                 unsigned char *mem_9520, __global\n                                 unsigned char *mem_9526)\n{\n    __local volatile char *restrict mem_9523 = mem_aligned_0;\n    int32_t wave_sizze_9584;\n    int32_t group_sizze_9585;\n    bool thread_active_9586;\n    int32_t global_tid_9167;\n    int32_t local_tid_9168;\n    int32_t group_id_9169;\n    \n    global_tid_9167 = get_global_id(0);\n    local_tid_9168 = get_local_i",
            "d(0);\n    group_sizze_9585 = get_local_size(0);\n    wave_sizze_9584 = LOCKSTEP_WIDTH;\n    group_id_9169 = get_group_id(0);\n    thread_active_9586 = 1;\n    \n    bool in_bounds_9170;\n    float x_9491;\n    \n    if (thread_active_9586) {\n        in_bounds_9170 = slt32(local_tid_9168, num_groups_9122);\n        if (in_bounds_9170) {\n            float x_9171 = *(__global float *) &mem_9520[global_tid_9167 * 4];\n            \n            x_9491 = x_9171;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9175;\n    \n    for (int32_t comb_iter_9587 = 0; comb_iter_9587 <\n         squot32(max_num_groups_9117 + max_num_groups_9117 - 1,\n                 max_num_groups_9117); comb_iter_9587++) {\n        int32_t combine_id_9174;\n        int32_t flat_comb_id_9588 = comb_iter_9587 * max_num_groups_9117 +\n                local_tid_9168;\n        \n        combine_id_9174 = flat_comb_id_9588;\n        if (slt32(combine_id_9174, max_num_groups_9117) && 1) {\n            *(__local float *) &mem_9523[combine_id_9174 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9589;\n    float x_8804;\n    float x_8805;\n    int32_t my_index_9129;\n    int32_t other_offset_9130;\n    \n    my_index_9129 = local_tid_9168;\n    other_offset_9130 = 0;\n    if (slt32(local_tid_9168, max_num_groups_9117)) {\n        x_8804 = *(__local float *) &mem_9523[(local_tid_9168 +\n                                               other_offset_9130) * 4];\n    }\n    other_offset_9130 = 1;\n    while (slt32(other_offset_9130, wave_sizze_9584)) {\n        if (slt32(local_tid_9168 + other_offset_9130, max_num_groups_9117) &&\n            ((local_tid_9168 - squot32(local_tid_9168, wave_sizze_9584) *\n              wave_sizze_9584) & (2 * other_offset_9130 - 1)) == 0) {\n            // read array element\n            {\n                x_8805 = *(volatile __local float *) &mem_9523[(local_tid_9168 +\n                                                                othe",
            "r_offset_9130) *\n                                                               4];\n            }\n            \n            float res_8806;\n            \n            if (thread_active_9586) {\n                res_8806 = x_8804 + x_8805;\n            }\n            x_8804 = res_8806;\n            *(volatile __local float *) &mem_9523[local_tid_9168 * 4] = x_8804;\n        }\n        other_offset_9130 *= 2;\n    }\n    skip_waves_9589 = 1;\n    while (slt32(skip_waves_9589, squot32(max_num_groups_9117 +\n                                          wave_sizze_9584 - 1,\n                                          wave_sizze_9584))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9130 = skip_waves_9589 * wave_sizze_9584;\n        if (slt32(local_tid_9168 + other_offset_9130, max_num_groups_9117) &&\n            ((local_tid_9168 - squot32(local_tid_9168, wave_sizze_9584) *\n              wave_sizze_9584) == 0 && (squot32(local_tid_9168,\n                                                wave_sizze_9584) & (2 *\n                                                                    skip_waves_9589 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8805 = *(__local float *) &mem_9523[(local_tid_9168 +\n                                                       other_offset_9130) * 4];\n            }\n            \n            float res_8806;\n            \n            if (thread_active_9586) {\n                res_8806 = x_8804 + x_8805;\n            }\n            x_8804 = res_8806;\n            *(__local float *) &mem_9523[local_tid_9168 * 4] = x_8804;\n        }\n        skip_waves_9589 *= 2;\n    }\n    final_result_9175 = x_8804;\n    if (local_tid_9168 == 0) {\n        *(__global float *) &mem_9526[group_id_9169 * 4] = final_result_9175;\n    }\n}\n__kernel void reduce_kernel_9241(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9195, __global\n                    ",
            "             unsigned char *mem_9520, __global\n                                 unsigned char *mem_9526)\n{\n    __local volatile char *restrict mem_9523 = mem_aligned_0;\n    int32_t wave_sizze_9600;\n    int32_t group_sizze_9601;\n    bool thread_active_9602;\n    int32_t global_tid_9241;\n    int32_t local_tid_9242;\n    int32_t group_id_9243;\n    \n    global_tid_9241 = get_global_id(0);\n    local_tid_9242 = get_local_id(0);\n    group_sizze_9601 = get_local_size(0);\n    wave_sizze_9600 = LOCKSTEP_WIDTH;\n    group_id_9243 = get_group_id(0);\n    thread_active_9602 = 1;\n    \n    bool in_bounds_9244;\n    float x_9491;\n    \n    if (thread_active_9602) {\n        in_bounds_9244 = slt32(local_tid_9242, num_groups_9195);\n        if (in_bounds_9244) {\n            float x_9245 = *(__global float *) &mem_9520[global_tid_9241 * 4];\n            \n            x_9491 = x_9245;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9249;\n    \n    for (int32_t comb_iter_9603 = 0; comb_iter_9603 <\n         squot32(max_num_groups_9190 + max_num_groups_9190 - 1,\n                 max_num_groups_9190); comb_iter_9603++) {\n        int32_t combine_id_9248;\n        int32_t flat_comb_id_9604 = comb_iter_9603 * max_num_groups_9190 +\n                local_tid_9242;\n        \n        combine_id_9248 = flat_comb_id_9604;\n        if (slt32(combine_id_9248, max_num_groups_9190) && 1) {\n            *(__local float *) &mem_9523[combine_id_9248 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9605;\n    float x_8832;\n    float x_8833;\n    int32_t my_index_9202;\n    int32_t other_offset_9203;\n    \n    my_index_9202 = local_tid_9242;\n    other_offset_9203 = 0;\n    if (slt32(local_tid_9242, max_num_groups_9190)) {\n        x_8832 = *(__local float *) &mem_9523[(local_tid_9242 +\n                                               other_offset_9203) * 4];\n    }\n    other_offset_9203 = 1;\n    while (slt32(other_offset_9203, wave_sizze_9600)) {\n ",
            "       if (slt32(local_tid_9242 + other_offset_9203, max_num_groups_9190) &&\n            ((local_tid_9242 - squot32(local_tid_9242, wave_sizze_9600) *\n              wave_sizze_9600) & (2 * other_offset_9203 - 1)) == 0) {\n            // read array element\n            {\n                x_8833 = *(volatile __local float *) &mem_9523[(local_tid_9242 +\n                                                                other_offset_9203) *\n                                                               4];\n            }\n            \n            float res_8834;\n            \n            if (thread_active_9602) {\n                res_8834 = x_8832 + x_8833;\n            }\n            x_8832 = res_8834;\n            *(volatile __local float *) &mem_9523[local_tid_9242 * 4] = x_8832;\n        }\n        other_offset_9203 *= 2;\n    }\n    skip_waves_9605 = 1;\n    while (slt32(skip_waves_9605, squot32(max_num_groups_9190 +\n                                          wave_sizze_9600 - 1,\n                                          wave_sizze_9600))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9203 = skip_waves_9605 * wave_sizze_9600;\n        if (slt32(local_tid_9242 + other_offset_9203, max_num_groups_9190) &&\n            ((local_tid_9242 - squot32(local_tid_9242, wave_sizze_9600) *\n              wave_sizze_9600) == 0 && (squot32(local_tid_9242,\n                                                wave_sizze_9600) & (2 *\n                                                                    skip_waves_9605 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8833 = *(__local float *) &mem_9523[(local_tid_9242 +\n                                                       other_offset_9203) * 4];\n            }\n            \n            float res_8834;\n            \n            if (thread_active_9602) {\n                res_8834 = x_8832 + x_8833;\n            }\n            x_8832 = res_8834;\n         ",
            "   *(__local float *) &mem_9523[local_tid_9242 * 4] = x_8832;\n        }\n        skip_waves_9605 *= 2;\n    }\n    final_result_9249 = x_8832;\n    if (local_tid_9242 == 0) {\n        *(__global float *) &mem_9526[group_id_9243 * 4] = final_result_9249;\n    }\n}\n__kernel void reduce_kernel_9320(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9271, __global\n                                 unsigned char *mem_9522, __global\n                                 unsigned char *mem_9528)\n{\n    __local volatile char *restrict mem_9525 = mem_aligned_0;\n    int32_t wave_sizze_9616;\n    int32_t group_sizze_9617;\n    bool thread_active_9618;\n    int32_t global_tid_9320;\n    int32_t local_tid_9321;\n    int32_t group_id_9322;\n    \n    global_tid_9320 = get_global_id(0);\n    local_tid_9321 = get_local_id(0);\n    group_sizze_9617 = get_local_size(0);\n    wave_sizze_9616 = LOCKSTEP_WIDTH;\n    group_id_9322 = get_group_id(0);\n    thread_active_9618 = 1;\n    \n    bool in_bounds_9323;\n    float x_9491;\n    \n    if (thread_active_9618) {\n        in_bounds_9323 = slt32(local_tid_9321, num_groups_9271);\n        if (in_bounds_9323) {\n            float x_9324 = *(__global float *) &mem_9522[global_tid_9320 * 4];\n            \n            x_9491 = x_9324;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9328;\n    \n    for (int32_t comb_iter_9619 = 0; comb_iter_9619 <\n         squot32(max_num_groups_9266 + max_num_groups_9266 - 1,\n                 max_num_groups_9266); comb_iter_9619++) {\n        int32_t combine_id_9327;\n        int32_t flat_comb_id_9620 = comb_iter_9619 * max_num_groups_9266 +\n                local_tid_9321;\n        \n        combine_id_9327 = flat_comb_id_9620;\n        if (slt32(combine_id_9327, max_num_groups_9266) && 1) {\n            *(__local float *) &mem_9525[combine_id_9327 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9621;\n    float x_8869;\n    fl",
            "oat x_8870;\n    int32_t my_index_9278;\n    int32_t other_offset_9279;\n    \n    my_index_9278 = local_tid_9321;\n    other_offset_9279 = 0;\n    if (slt32(local_tid_9321, max_num_groups_9266)) {\n        x_8869 = *(__local float *) &mem_9525[(local_tid_9321 +\n                                               other_offset_9279) * 4];\n    }\n    other_offset_9279 = 1;\n    while (slt32(other_offset_9279, wave_sizze_9616)) {\n        if (slt32(local_tid_9321 + other_offset_9279, max_num_groups_9266) &&\n            ((local_tid_9321 - squot32(local_tid_9321, wave_sizze_9616) *\n              wave_sizze_9616) & (2 * other_offset_9279 - 1)) == 0) {\n            // read array element\n            {\n                x_8870 = *(volatile __local float *) &mem_9525[(local_tid_9321 +\n                                                                other_offset_9279) *\n                                                               4];\n            }\n            \n            float res_8871;\n            \n            if (thread_active_9618) {\n                res_8871 = x_8869 + x_8870;\n            }\n            x_8869 = res_8871;\n            *(volatile __local float *) &mem_9525[local_tid_9321 * 4] = x_8869;\n        }\n        other_offset_9279 *= 2;\n    }\n    skip_waves_9621 = 1;\n    while (slt32(skip_waves_9621, squot32(max_num_groups_9266 +\n                                          wave_sizze_9616 - 1,\n                                          wave_sizze_9616))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9279 = skip_waves_9621 * wave_sizze_9616;\n        if (slt32(local_tid_9321 + other_offset_9279, max_num_groups_9266) &&\n            ((local_tid_9321 - squot32(local_tid_9321, wave_sizze_9616) *\n              wave_sizze_9616) == 0 && (squot32(local_tid_9321,\n                                                wave_sizze_9616) & (2 *\n                                                                    skip_waves_9621 -\n                                                                   ",
            " 1)) == 0)) {\n            // read array element\n            {\n                x_8870 = *(__local float *) &mem_9525[(local_tid_9321 +\n                                                       other_offset_9279) * 4];\n            }\n            \n            float res_8871;\n            \n            if (thread_active_9618) {\n                res_8871 = x_8869 + x_8870;\n            }\n            x_8869 = res_8871;\n            *(__local float *) &mem_9525[local_tid_9321 * 4] = x_8869;\n        }\n        skip_waves_9621 *= 2;\n    }\n    final_result_9328 = x_8869;\n    if (local_tid_9321 == 0) {\n        *(__global float *) &mem_9528[group_id_9322 * 4] = final_result_9328;\n    }\n}\n__kernel void reduce_kernel_9403(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9350, __global\n                                 unsigned char *mem_9522, __global\n                                 unsigned char *mem_9528)\n{\n    __local volatile char *restrict mem_9525 = mem_aligned_0;\n    int32_t wave_sizze_9632;\n    int32_t group_sizze_9633;\n    bool thread_active_9634;\n    int32_t global_tid_9403;\n    int32_t local_tid_9404;\n    int32_t group_id_9405;\n    \n    global_tid_9403 = get_global_id(0);\n    local_tid_9404 = get_local_id(0);\n    group_sizze_9633 = get_local_size(0);\n    wave_sizze_9632 = LOCKSTEP_WIDTH;\n    group_id_9405 = get_group_id(0);\n    thread_active_9634 = 1;\n    \n    bool in_bounds_9406;\n    float x_9491;\n    \n    if (thread_active_9634) {\n        in_bounds_9406 = slt32(local_tid_9404, num_groups_9350);\n        if (in_bounds_9406) {\n            float x_9407 = *(__global float *) &mem_9522[global_tid_9403 * 4];\n            \n            x_9491 = x_9407;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9411;\n    \n    for (int32_t comb_iter_9635 = 0; comb_iter_9635 <\n         squot32(max_num_groups_9345 + max_num_groups_9345 - 1,\n                 max_num_groups_9345); comb_iter_9635++) {\n        int32_t comb",
            "ine_id_9410;\n        int32_t flat_comb_id_9636 = comb_iter_9635 * max_num_groups_9345 +\n                local_tid_9404;\n        \n        combine_id_9410 = flat_comb_id_9636;\n        if (slt32(combine_id_9410, max_num_groups_9345) && 1) {\n            *(__local float *) &mem_9525[combine_id_9410 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9637;\n    float x_8911;\n    float x_8912;\n    int32_t my_index_9357;\n    int32_t other_offset_9358;\n    \n    my_index_9357 = local_tid_9404;\n    other_offset_9358 = 0;\n    if (slt32(local_tid_9404, max_num_groups_9345)) {\n        x_8911 = *(__local float *) &mem_9525[(local_tid_9404 +\n                                               other_offset_9358) * 4];\n    }\n    other_offset_9358 = 1;\n    while (slt32(other_offset_9358, wave_sizze_9632)) {\n        if (slt32(local_tid_9404 + other_offset_9358, max_num_groups_9345) &&\n            ((local_tid_9404 - squot32(local_tid_9404, wave_sizze_9632) *\n              wave_sizze_9632) & (2 * other_offset_9358 - 1)) == 0) {\n            // read array element\n            {\n                x_8912 = *(volatile __local float *) &mem_9525[(local_tid_9404 +\n                                                                other_offset_9358) *\n                                                               4];\n            }\n            \n            float res_8913;\n            \n            if (thread_active_9634) {\n                res_8913 = x_8911 + x_8912;\n            }\n            x_8911 = res_8913;\n            *(volatile __local float *) &mem_9525[local_tid_9404 * 4] = x_8911;\n        }\n        other_offset_9358 *= 2;\n    }\n    skip_waves_9637 = 1;\n    while (slt32(skip_waves_9637, squot32(max_num_groups_9345 +\n                                          wave_sizze_9632 - 1,\n                                          wave_sizze_9632))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9358 = skip_waves_9637 * wave_sizze_9632;\n        if (slt32(local",
            "_tid_9404 + other_offset_9358, max_num_groups_9345) &&\n            ((local_tid_9404 - squot32(local_tid_9404, wave_sizze_9632) *\n              wave_sizze_9632) == 0 && (squot32(local_tid_9404,\n                                                wave_sizze_9632) & (2 *\n                                                                    skip_waves_9637 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8912 = *(__local float *) &mem_9525[(local_tid_9404 +\n                                                       other_offset_9358) * 4];\n            }\n            \n            float res_8913;\n            \n            if (thread_active_9634) {\n                res_8913 = x_8911 + x_8912;\n            }\n            x_8911 = res_8913;\n            *(__local float *) &mem_9525[local_tid_9404 * 4] = x_8911;\n        }\n        skip_waves_9637 *= 2;\n    }\n    final_result_9411 = x_8911;\n    if (local_tid_9404 == 0) {\n        *(__global float *) &mem_9528[group_id_9405 * 4] = final_result_9411;\n    }\n}\n__kernel void reduce_kernel_9482(__local volatile int64_t *mem_aligned_0,\n                                 int32_t num_groups_9433, __global\n                                 unsigned char *mem_9522, __global\n                                 unsigned char *mem_9528)\n{\n    __local volatile char *restrict mem_9525 = mem_aligned_0;\n    int32_t wave_sizze_9648;\n    int32_t group_sizze_9649;\n    bool thread_active_9650;\n    int32_t global_tid_9482;\n    int32_t local_tid_9483;\n    int32_t group_id_9484;\n    \n    global_tid_9482 = get_global_id(0);\n    local_tid_9483 = get_local_id(0);\n    group_sizze_9649 = get_local_size(0);\n    wave_sizze_9648 = LOCKSTEP_WIDTH;\n    group_id_9484 = get_group_id(0);\n    thread_active_9650 = 1;\n    \n    bool in_bounds_9485;\n    float x_9491;\n    \n    if (thread_active_9650) {\n        in_bounds_9485 = slt32(local_tid_9483, num_groups_9433);\n        if (in_bounds_9485) {",
            "\n            float x_9486 = *(__global float *) &mem_9522[global_tid_9482 * 4];\n            \n            x_9491 = x_9486;\n        } else {\n            x_9491 = 0.0F;\n        }\n    }\n    \n    float final_result_9490;\n    \n    for (int32_t comb_iter_9651 = 0; comb_iter_9651 <\n         squot32(max_num_groups_9428 + max_num_groups_9428 - 1,\n                 max_num_groups_9428); comb_iter_9651++) {\n        int32_t combine_id_9489;\n        int32_t flat_comb_id_9652 = comb_iter_9651 * max_num_groups_9428 +\n                local_tid_9483;\n        \n        combine_id_9489 = flat_comb_id_9652;\n        if (slt32(combine_id_9489, max_num_groups_9428) && 1) {\n            *(__local float *) &mem_9525[combine_id_9489 * 4] = x_9491;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_9653;\n    float x_8953;\n    float x_8954;\n    int32_t my_index_9440;\n    int32_t other_offset_9441;\n    \n    my_index_9440 = local_tid_9483;\n    other_offset_9441 = 0;\n    if (slt32(local_tid_9483, max_num_groups_9428)) {\n        x_8953 = *(__local float *) &mem_9525[(local_tid_9483 +\n                                               other_offset_9441) * 4];\n    }\n    other_offset_9441 = 1;\n    while (slt32(other_offset_9441, wave_sizze_9648)) {\n        if (slt32(local_tid_9483 + other_offset_9441, max_num_groups_9428) &&\n            ((local_tid_9483 - squot32(local_tid_9483, wave_sizze_9648) *\n              wave_sizze_9648) & (2 * other_offset_9441 - 1)) == 0) {\n            // read array element\n            {\n                x_8954 = *(volatile __local float *) &mem_9525[(local_tid_9483 +\n                                                                other_offset_9441) *\n                                                               4];\n            }\n            \n            float res_8955;\n            \n            if (thread_active_9650) {\n                res_8955 = x_8953 + x_8954;\n            }\n            x_8953 = res_8955;\n            *(volatile __local float *) &mem_9525",
            "[local_tid_9483 * 4] = x_8953;\n        }\n        other_offset_9441 *= 2;\n    }\n    skip_waves_9653 = 1;\n    while (slt32(skip_waves_9653, squot32(max_num_groups_9428 +\n                                          wave_sizze_9648 - 1,\n                                          wave_sizze_9648))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_9441 = skip_waves_9653 * wave_sizze_9648;\n        if (slt32(local_tid_9483 + other_offset_9441, max_num_groups_9428) &&\n            ((local_tid_9483 - squot32(local_tid_9483, wave_sizze_9648) *\n              wave_sizze_9648) == 0 && (squot32(local_tid_9483,\n                                                wave_sizze_9648) & (2 *\n                                                                    skip_waves_9653 -\n                                                                    1)) == 0)) {\n            // read array element\n            {\n                x_8954 = *(__local float *) &mem_9525[(local_tid_9483 +\n                                                       other_offset_9441) * 4];\n            }\n            \n            float res_8955;\n            \n            if (thread_active_9650) {\n                res_8955 = x_8953 + x_8954;\n            }\n            x_8953 = res_8955;\n            *(__local float *) &mem_9525[local_tid_9483 * 4] = x_8953;\n        }\n        skip_waves_9653 *= 2;\n    }\n    final_result_9490 = x_8953;\n    if (local_tid_9483 == 0) {\n        *(__global float *) &mem_9528[group_id_9484 * 4] = final_result_9490;\n    }\n}\n",
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
static const char *size_names[] = {"group_size_8972", "max_num_groups_8974",
                                   "group_size_9041", "max_num_groups_9043",
                                   "group_size_9114", "max_num_groups_9116",
                                   "group_size_9187", "max_num_groups_9189",
                                   "group_size_9263", "max_num_groups_9265",
                                   "group_size_9342", "max_num_groups_9344",
                                   "group_size_9425", "max_num_groups_9427"};
static const char *size_classes[] = {"group_size", "num_groups", "group_size",
                                     "num_groups", "group_size", "num_groups",
                                     "group_size", "num_groups", "group_size",
                                     "num_groups", "group_size", "num_groups",
                                     "group_size", "num_groups"};
static const char *size_entry_points[] = {"dot", "dot", "dot1", "dot1", "dot2",
                                          "dot2", "dot3", "dot3", "dot4",
                                          "dot4", "dot5", "dot5", "dot6",
                                          "dot6"};
int futhark_get_num_sizes(void)
{
    return 14;
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
    size_t group_sizze_8972;
    size_t max_num_groups_8974;
    size_t group_sizze_9041;
    size_t max_num_groups_9043;
    size_t group_sizze_9114;
    size_t max_num_groups_9116;
    size_t group_sizze_9187;
    size_t max_num_groups_9189;
    size_t group_sizze_9263;
    size_t max_num_groups_9265;
    size_t group_sizze_9342;
    size_t max_num_groups_9344;
    size_t group_sizze_9425;
    size_t max_num_groups_9427;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[14];
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
    opencl_config_init(&cfg->opencl, 14, size_names, cfg->sizes, size_classes,
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
    for (int i = 0; i < 14; i++) {
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
    cl_kernel chunked_reduce_kernel_8989;
    int chunked_reduce_kernel_8989_total_runtime;
    int chunked_reduce_kernel_8989_runs;
    cl_kernel chunked_reduce_kernel_9058;
    int chunked_reduce_kernel_9058_total_runtime;
    int chunked_reduce_kernel_9058_runs;
    cl_kernel chunked_reduce_kernel_9131;
    int chunked_reduce_kernel_9131_total_runtime;
    int chunked_reduce_kernel_9131_runs;
    cl_kernel chunked_reduce_kernel_9204;
    int chunked_reduce_kernel_9204_total_runtime;
    int chunked_reduce_kernel_9204_runs;
    cl_kernel chunked_reduce_kernel_9280;
    int chunked_reduce_kernel_9280_total_runtime;
    int chunked_reduce_kernel_9280_runs;
    cl_kernel chunked_reduce_kernel_9359;
    int chunked_reduce_kernel_9359_total_runtime;
    int chunked_reduce_kernel_9359_runs;
    cl_kernel chunked_reduce_kernel_9442;
    int chunked_reduce_kernel_9442_total_runtime;
    int chunked_reduce_kernel_9442_runs;
    cl_kernel reduce_kernel_9021;
    int reduce_kernel_9021_total_runtime;
    int reduce_kernel_9021_runs;
    cl_kernel reduce_kernel_9094;
    int reduce_kernel_9094_total_runtime;
    int reduce_kernel_9094_runs;
    cl_kernel reduce_kernel_9167;
    int reduce_kernel_9167_total_runtime;
    int reduce_kernel_9167_runs;
    cl_kernel reduce_kernel_9241;
    int reduce_kernel_9241_total_runtime;
    int reduce_kernel_9241_runs;
    cl_kernel reduce_kernel_9320;
    int reduce_kernel_9320_total_runtime;
    int reduce_kernel_9320_runs;
    cl_kernel reduce_kernel_9403;
    int reduce_kernel_9403_total_runtime;
    int reduce_kernel_9403_runs;
    cl_kernel reduce_kernel_9482;
    int reduce_kernel_9482_total_runtime;
    int reduce_kernel_9482_runs;
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
    ctx->chunked_reduce_kernel_8989_total_runtime = 0;
    ctx->chunked_reduce_kernel_8989_runs = 0;
    ctx->chunked_reduce_kernel_9058_total_runtime = 0;
    ctx->chunked_reduce_kernel_9058_runs = 0;
    ctx->chunked_reduce_kernel_9131_total_runtime = 0;
    ctx->chunked_reduce_kernel_9131_runs = 0;
    ctx->chunked_reduce_kernel_9204_total_runtime = 0;
    ctx->chunked_reduce_kernel_9204_runs = 0;
    ctx->chunked_reduce_kernel_9280_total_runtime = 0;
    ctx->chunked_reduce_kernel_9280_runs = 0;
    ctx->chunked_reduce_kernel_9359_total_runtime = 0;
    ctx->chunked_reduce_kernel_9359_runs = 0;
    ctx->chunked_reduce_kernel_9442_total_runtime = 0;
    ctx->chunked_reduce_kernel_9442_runs = 0;
    ctx->reduce_kernel_9021_total_runtime = 0;
    ctx->reduce_kernel_9021_runs = 0;
    ctx->reduce_kernel_9094_total_runtime = 0;
    ctx->reduce_kernel_9094_runs = 0;
    ctx->reduce_kernel_9167_total_runtime = 0;
    ctx->reduce_kernel_9167_runs = 0;
    ctx->reduce_kernel_9241_total_runtime = 0;
    ctx->reduce_kernel_9241_runs = 0;
    ctx->reduce_kernel_9320_total_runtime = 0;
    ctx->reduce_kernel_9320_runs = 0;
    ctx->reduce_kernel_9403_total_runtime = 0;
    ctx->reduce_kernel_9403_runs = 0;
    ctx->reduce_kernel_9482_total_runtime = 0;
    ctx->reduce_kernel_9482_runs = 0;
    
    int required_types = 0;
    cl_int error;
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    {
        ctx->chunked_reduce_kernel_8989 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_8989",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_8989");
    }
    {
        ctx->chunked_reduce_kernel_9058 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9058",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9058");
    }
    {
        ctx->chunked_reduce_kernel_9131 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9131",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9131");
    }
    {
        ctx->chunked_reduce_kernel_9204 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9204",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9204");
    }
    {
        ctx->chunked_reduce_kernel_9280 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9280",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9280");
    }
    {
        ctx->chunked_reduce_kernel_9359 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9359",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9359");
    }
    {
        ctx->chunked_reduce_kernel_9442 = clCreateKernel(prog,
                                                         "chunked_reduce_kernel_9442",
                                                         &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_9442");
    }
    {
        ctx->reduce_kernel_9021 = clCreateKernel(prog, "reduce_kernel_9021",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9021");
    }
    {
        ctx->reduce_kernel_9094 = clCreateKernel(prog, "reduce_kernel_9094",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9094");
    }
    {
        ctx->reduce_kernel_9167 = clCreateKernel(prog, "reduce_kernel_9167",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9167");
    }
    {
        ctx->reduce_kernel_9241 = clCreateKernel(prog, "reduce_kernel_9241",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9241");
    }
    {
        ctx->reduce_kernel_9320 = clCreateKernel(prog, "reduce_kernel_9320",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9320");
    }
    {
        ctx->reduce_kernel_9403 = clCreateKernel(prog, "reduce_kernel_9403",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9403");
    }
    {
        ctx->reduce_kernel_9482 = clCreateKernel(prog, "reduce_kernel_9482",
                                                 &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_9482");
    }
    ctx->sizes.group_sizze_8972 = cfg->sizes[0];
    ctx->sizes.max_num_groups_8974 = cfg->sizes[1];
    ctx->sizes.group_sizze_9041 = cfg->sizes[2];
    ctx->sizes.max_num_groups_9043 = cfg->sizes[3];
    ctx->sizes.group_sizze_9114 = cfg->sizes[4];
    ctx->sizes.max_num_groups_9116 = cfg->sizes[5];
    ctx->sizes.group_sizze_9187 = cfg->sizes[6];
    ctx->sizes.max_num_groups_9189 = cfg->sizes[7];
    ctx->sizes.group_sizze_9263 = cfg->sizes[8];
    ctx->sizes.max_num_groups_9265 = cfg->sizes[9];
    ctx->sizes.group_sizze_9342 = cfg->sizes[10];
    ctx->sizes.max_num_groups_9344 = cfg->sizes[11];
    ctx->sizes.group_sizze_9425 = cfg->sizes[12];
    ctx->sizes.max_num_groups_9427 = cfg->sizes[13];
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
                "Kernel chunked_reduce_kernel_8989 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_8989_runs,
                (long) ctx->chunked_reduce_kernel_8989_total_runtime /
                (ctx->chunked_reduce_kernel_8989_runs !=
                 0 ? ctx->chunked_reduce_kernel_8989_runs : 1),
                (long) ctx->chunked_reduce_kernel_8989_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_8989_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_8989_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9058 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9058_runs,
                (long) ctx->chunked_reduce_kernel_9058_total_runtime /
                (ctx->chunked_reduce_kernel_9058_runs !=
                 0 ? ctx->chunked_reduce_kernel_9058_runs : 1),
                (long) ctx->chunked_reduce_kernel_9058_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9058_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9058_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9131 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9131_runs,
                (long) ctx->chunked_reduce_kernel_9131_total_runtime /
                (ctx->chunked_reduce_kernel_9131_runs !=
                 0 ? ctx->chunked_reduce_kernel_9131_runs : 1),
                (long) ctx->chunked_reduce_kernel_9131_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9131_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9131_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9204 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9204_runs,
                (long) ctx->chunked_reduce_kernel_9204_total_runtime /
                (ctx->chunked_reduce_kernel_9204_runs !=
                 0 ? ctx->chunked_reduce_kernel_9204_runs : 1),
                (long) ctx->chunked_reduce_kernel_9204_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9204_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9204_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9280 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9280_runs,
                (long) ctx->chunked_reduce_kernel_9280_total_runtime /
                (ctx->chunked_reduce_kernel_9280_runs !=
                 0 ? ctx->chunked_reduce_kernel_9280_runs : 1),
                (long) ctx->chunked_reduce_kernel_9280_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9280_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9280_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9359 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9359_runs,
                (long) ctx->chunked_reduce_kernel_9359_total_runtime /
                (ctx->chunked_reduce_kernel_9359_runs !=
                 0 ? ctx->chunked_reduce_kernel_9359_runs : 1),
                (long) ctx->chunked_reduce_kernel_9359_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9359_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9359_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_9442 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_9442_runs,
                (long) ctx->chunked_reduce_kernel_9442_total_runtime /
                (ctx->chunked_reduce_kernel_9442_runs !=
                 0 ? ctx->chunked_reduce_kernel_9442_runs : 1),
                (long) ctx->chunked_reduce_kernel_9442_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_9442_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_9442_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9021         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9021_runs,
                (long) ctx->reduce_kernel_9021_total_runtime /
                (ctx->reduce_kernel_9021_runs !=
                 0 ? ctx->reduce_kernel_9021_runs : 1),
                (long) ctx->reduce_kernel_9021_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9021_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9021_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9094         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9094_runs,
                (long) ctx->reduce_kernel_9094_total_runtime /
                (ctx->reduce_kernel_9094_runs !=
                 0 ? ctx->reduce_kernel_9094_runs : 1),
                (long) ctx->reduce_kernel_9094_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9094_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9094_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9167         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9167_runs,
                (long) ctx->reduce_kernel_9167_total_runtime /
                (ctx->reduce_kernel_9167_runs !=
                 0 ? ctx->reduce_kernel_9167_runs : 1),
                (long) ctx->reduce_kernel_9167_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9167_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9167_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9241         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9241_runs,
                (long) ctx->reduce_kernel_9241_total_runtime /
                (ctx->reduce_kernel_9241_runs !=
                 0 ? ctx->reduce_kernel_9241_runs : 1),
                (long) ctx->reduce_kernel_9241_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9241_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9241_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9320         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9320_runs,
                (long) ctx->reduce_kernel_9320_total_runtime /
                (ctx->reduce_kernel_9320_runs !=
                 0 ? ctx->reduce_kernel_9320_runs : 1),
                (long) ctx->reduce_kernel_9320_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9320_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9320_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9403         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9403_runs,
                (long) ctx->reduce_kernel_9403_total_runtime /
                (ctx->reduce_kernel_9403_runs !=
                 0 ? ctx->reduce_kernel_9403_runs : 1),
                (long) ctx->reduce_kernel_9403_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9403_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9403_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_9482         executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_9482_runs,
                (long) ctx->reduce_kernel_9482_total_runtime /
                (ctx->reduce_kernel_9482_runs !=
                 0 ? ctx->reduce_kernel_9482_runs : 1),
                (long) ctx->reduce_kernel_9482_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_9482_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_9482_runs;
        if (ctx->debugging)
            fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                    ctx->total_runs, ctx->total_runtime);
    }
}
static int futrts_dot(struct futhark_context *ctx, float *out_scalar_out_9655,
                      int64_t A_mem_sizze_9509,
                      struct memblock_device A_mem_9510,
                      int64_t B_mem_sizze_9511,
                      struct memblock_device B_mem_9512, int32_t sizze_8738,
                      int32_t sizze_8739);
static int futrts_dot1(struct futhark_context *ctx, float *out_scalar_out_9667,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8756,
                       int32_t sizze_8757, int32_t sizze_8758);
static int futrts_dot2(struct futhark_context *ctx, float *out_scalar_out_9679,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8784,
                       int32_t sizze_8785, int32_t sizze_8786);
static int futrts_dot3(struct futhark_context *ctx, float *out_scalar_out_9691,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8812,
                       int32_t sizze_8813, int32_t sizze_8814);
static int futrts_dot4(struct futhark_context *ctx, float *out_scalar_out_9703,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8841,
                       int32_t sizze_8842, int32_t sizze_8843,
                       int32_t sizze_8844);
static int futrts_dot5(struct futhark_context *ctx, float *out_scalar_out_9715,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8879,
                       int32_t sizze_8880, int32_t sizze_8881,
                       int32_t sizze_8882, float a_8883, float b_8885,
                       float c_8887, float d_8889);
static int futrts_dot6(struct futhark_context *ctx, float *out_scalar_out_9727,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8925,
                       int32_t sizze_8926, int32_t sizze_8927,
                       int32_t sizze_8928);
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
static int futrts_dot(struct futhark_context *ctx, float *out_scalar_out_9655,
                      int64_t A_mem_sizze_9509,
                      struct memblock_device A_mem_9510,
                      int64_t B_mem_sizze_9511,
                      struct memblock_device B_mem_9512, int32_t sizze_8738,
                      int32_t sizze_8739)
{
    float scalar_out_9543;
    bool dim_zzero_8742 = 0 == sizze_8739;
    bool dim_zzero_8743 = 0 == sizze_8738;
    bool both_empty_8744 = dim_zzero_8742 && dim_zzero_8743;
    bool dim_match_8745 = sizze_8738 == sizze_8739;
    bool empty_or_match_8746 = both_empty_8744 || dim_match_8745;
    bool empty_or_match_cert_8747;
    
    if (!empty_or_match_8746) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:1:1-2:31",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_8973;
    
    group_sizze_8973 = ctx->sizes.group_sizze_8972;
    
    int32_t max_num_groups_8975;
    
    max_num_groups_8975 = ctx->sizes.max_num_groups_8974;
    
    int32_t y_8976 = group_sizze_8973 - 1;
    int32_t x_8977 = sizze_8738 + y_8976;
    int32_t w_div_group_sizze_8978 = squot32(x_8977, group_sizze_8973);
    int32_t num_groups_maybe_zzero_8979 = smin32(max_num_groups_8975,
                                                 w_div_group_sizze_8978);
    int32_t num_groups_8980 = smax32(1, num_groups_maybe_zzero_8979);
    int32_t num_threads_8981 = group_sizze_8973 * num_groups_8980;
    int32_t y_8982 = num_threads_8981 - 1;
    int32_t x_8983 = sizze_8738 + y_8982;
    int32_t per_thread_elements_8984 = squot32(x_8983, num_threads_8981);
    int64_t binop_x_9517 = sext_i32_i64(num_groups_8980);
    int64_t bytes_9516 = 4 * binop_x_9517;
    struct memblock_device mem_9518;
    
    mem_9518.references = NULL;
    memblock_alloc_device(ctx, &mem_9518, bytes_9516, "mem_9518");
    
    int64_t binop_x_9514 = sext_i32_i64(group_sizze_8973);
    int64_t bytes_9513 = 4 * binop_x_9514;
    struct memblock_local mem_9515;
    
    mem_9515.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8738);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 0,
                                  bytes_9513, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 1,
                                  sizeof(sizze_8738), &sizze_8738));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 2,
                                  sizeof(num_threads_8981), &num_threads_8981));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 3,
                                  sizeof(per_thread_elements_8984),
                                  &per_thread_elements_8984));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_8989, 6,
                                  sizeof(mem_9518.mem), &mem_9518.mem));
    if (1 * (num_groups_8980 * group_sizze_8973) != 0) {
        const size_t global_work_sizze_9656[1] = {num_groups_8980 *
                     group_sizze_8973};
        const size_t local_work_sizze_9660[1] = {group_sizze_8973};
        int64_t time_start_9657 = 0, time_end_9658 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_8989");
            fprintf(stderr, "%zu", global_work_sizze_9656[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9660[0]);
            fprintf(stderr, "].\n");
            time_start_9657 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_8989,
                                              1, NULL, global_work_sizze_9656,
                                              local_work_sizze_9660, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9658 = get_wall_time();
            
            long time_diff_9659 = time_end_9658 - time_start_9657;
            
            ctx->chunked_reduce_kernel_8989_total_runtime += time_diff_9659;
            ctx->chunked_reduce_kernel_8989_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_8989", time_diff_9659);
        }
    }
    memblock_unref_local(ctx, &mem_9515, "mem_9515");
    
    struct memblock_device mem_9524;
    
    mem_9524.references = NULL;
    memblock_alloc_device(ctx, &mem_9524, 4, "mem_9524");
    
    int64_t binop_x_9520 = sext_i32_i64(max_num_groups_8975);
    int64_t bytes_9519 = 4 * binop_x_9520;
    struct memblock_local mem_9521;
    
    mem_9521.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9021, 0, bytes_9519,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9021, 1,
                                  sizeof(num_groups_8980), &num_groups_8980));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9021, 2,
                                  sizeof(mem_9518.mem), &mem_9518.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9021, 3,
                                  sizeof(mem_9524.mem), &mem_9524.mem));
    if (1 * max_num_groups_8975 != 0) {
        const size_t global_work_sizze_9661[1] = {max_num_groups_8975};
        const size_t local_work_sizze_9665[1] = {max_num_groups_8975};
        int64_t time_start_9662 = 0, time_end_9663 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9021");
            fprintf(stderr, "%zu", global_work_sizze_9661[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9665[0]);
            fprintf(stderr, "].\n");
            time_start_9662 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9021, 1, NULL,
                                              global_work_sizze_9661,
                                              local_work_sizze_9665, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9663 = get_wall_time();
            
            long time_diff_9664 = time_end_9663 - time_start_9662;
            
            ctx->reduce_kernel_9021_total_runtime += time_diff_9664;
            ctx->reduce_kernel_9021_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9021",
                    time_diff_9664);
        }
    }
    memblock_unref_device(ctx, &mem_9518, "mem_9518");
    memblock_unref_local(ctx, &mem_9521, "mem_9521");
    
    float read_res_9666;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9524.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9666, 0,
                                       NULL, NULL));
    
    float res_8749 = read_res_9666;
    
    memblock_unref_device(ctx, &mem_9524, "mem_9524");
    scalar_out_9543 = res_8749;
    *out_scalar_out_9655 = scalar_out_9543;
    memblock_unref_local(ctx, &mem_9521, "mem_9521");
    memblock_unref_device(ctx, &mem_9524, "mem_9524");
    memblock_unref_local(ctx, &mem_9515, "mem_9515");
    memblock_unref_device(ctx, &mem_9518, "mem_9518");
    return 0;
}
static int futrts_dot1(struct futhark_context *ctx, float *out_scalar_out_9667,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8756,
                       int32_t sizze_8757, int32_t sizze_8758)
{
    float scalar_out_9559;
    bool dim_zzero_8762 = 0 == sizze_8757;
    bool dim_zzero_8763 = 0 == sizze_8756;
    bool both_empty_8764 = dim_zzero_8762 && dim_zzero_8763;
    bool dim_match_8765 = sizze_8756 == sizze_8757;
    bool empty_or_match_8766 = both_empty_8764 || dim_match_8765;
    bool empty_or_match_cert_8767;
    
    if (!empty_or_match_8766) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:4:1-5:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8768 = 0 == sizze_8758;
    bool both_empty_8769 = dim_zzero_8763 && dim_zzero_8768;
    bool dim_match_8770 = sizze_8756 == sizze_8758;
    bool empty_or_match_8771 = both_empty_8769 || dim_match_8770;
    bool empty_or_match_cert_8772;
    
    if (!empty_or_match_8771) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:4:1-5:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9042;
    
    group_sizze_9042 = ctx->sizes.group_sizze_9041;
    
    int32_t max_num_groups_9044;
    
    max_num_groups_9044 = ctx->sizes.max_num_groups_9043;
    
    int32_t y_9045 = group_sizze_9042 - 1;
    int32_t x_9046 = sizze_8756 + y_9045;
    int32_t w_div_group_sizze_9047 = squot32(x_9046, group_sizze_9042);
    int32_t num_groups_maybe_zzero_9048 = smin32(max_num_groups_9044,
                                                 w_div_group_sizze_9047);
    int32_t num_groups_9049 = smax32(1, num_groups_maybe_zzero_9048);
    int32_t num_threads_9050 = group_sizze_9042 * num_groups_9049;
    int32_t y_9051 = num_threads_9050 - 1;
    int32_t x_9052 = sizze_8756 + y_9051;
    int32_t per_thread_elements_9053 = squot32(x_9052, num_threads_9050);
    int64_t binop_x_9519 = sext_i32_i64(num_groups_9049);
    int64_t bytes_9518 = 4 * binop_x_9519;
    struct memblock_device mem_9520;
    
    mem_9520.references = NULL;
    memblock_alloc_device(ctx, &mem_9520, bytes_9518, "mem_9520");
    
    int64_t binop_x_9516 = sext_i32_i64(group_sizze_9042);
    int64_t bytes_9515 = 4 * binop_x_9516;
    struct memblock_local mem_9517;
    
    mem_9517.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8756);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 0,
                                  bytes_9515, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 1,
                                  sizeof(sizze_8756), &sizze_8756));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 2,
                                  sizeof(num_threads_9050), &num_threads_9050));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 3,
                                  sizeof(per_thread_elements_9053),
                                  &per_thread_elements_9053));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 6,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9058, 7,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    if (1 * (num_groups_9049 * group_sizze_9042) != 0) {
        const size_t global_work_sizze_9668[1] = {num_groups_9049 *
                     group_sizze_9042};
        const size_t local_work_sizze_9672[1] = {group_sizze_9042};
        int64_t time_start_9669 = 0, time_end_9670 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9058");
            fprintf(stderr, "%zu", global_work_sizze_9668[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9672[0]);
            fprintf(stderr, "].\n");
            time_start_9669 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9058,
                                              1, NULL, global_work_sizze_9668,
                                              local_work_sizze_9672, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9670 = get_wall_time();
            
            long time_diff_9671 = time_end_9670 - time_start_9669;
            
            ctx->chunked_reduce_kernel_9058_total_runtime += time_diff_9671;
            ctx->chunked_reduce_kernel_9058_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9058", time_diff_9671);
        }
    }
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    
    struct memblock_device mem_9526;
    
    mem_9526.references = NULL;
    memblock_alloc_device(ctx, &mem_9526, 4, "mem_9526");
    
    int64_t binop_x_9522 = sext_i32_i64(max_num_groups_9044);
    int64_t bytes_9521 = 4 * binop_x_9522;
    struct memblock_local mem_9523;
    
    mem_9523.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9094, 0, bytes_9521,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9094, 1,
                                  sizeof(num_groups_9049), &num_groups_9049));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9094, 2,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9094, 3,
                                  sizeof(mem_9526.mem), &mem_9526.mem));
    if (1 * max_num_groups_9044 != 0) {
        const size_t global_work_sizze_9673[1] = {max_num_groups_9044};
        const size_t local_work_sizze_9677[1] = {max_num_groups_9044};
        int64_t time_start_9674 = 0, time_end_9675 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9094");
            fprintf(stderr, "%zu", global_work_sizze_9673[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9677[0]);
            fprintf(stderr, "].\n");
            time_start_9674 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9094, 1, NULL,
                                              global_work_sizze_9673,
                                              local_work_sizze_9677, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9675 = get_wall_time();
            
            long time_diff_9676 = time_end_9675 - time_start_9674;
            
            ctx->reduce_kernel_9094_total_runtime += time_diff_9676;
            ctx->reduce_kernel_9094_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9094",
                    time_diff_9676);
        }
    }
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    
    float read_res_9678;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9526.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9678, 0,
                                       NULL, NULL));
    
    float res_8775 = read_res_9678;
    
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    scalar_out_9559 = res_8775;
    *out_scalar_out_9667 = scalar_out_9559;
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    return 0;
}
static int futrts_dot2(struct futhark_context *ctx, float *out_scalar_out_9679,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8784,
                       int32_t sizze_8785, int32_t sizze_8786)
{
    float scalar_out_9575;
    bool dim_zzero_8790 = 0 == sizze_8785;
    bool dim_zzero_8791 = 0 == sizze_8784;
    bool both_empty_8792 = dim_zzero_8790 && dim_zzero_8791;
    bool dim_match_8793 = sizze_8784 == sizze_8785;
    bool empty_or_match_8794 = both_empty_8792 || dim_match_8793;
    bool empty_or_match_cert_8795;
    
    if (!empty_or_match_8794) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:7:1-8:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8796 = 0 == sizze_8786;
    bool both_empty_8797 = dim_zzero_8791 && dim_zzero_8796;
    bool dim_match_8798 = sizze_8784 == sizze_8786;
    bool empty_or_match_8799 = both_empty_8797 || dim_match_8798;
    bool empty_or_match_cert_8800;
    
    if (!empty_or_match_8799) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:7:1-8:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9115;
    
    group_sizze_9115 = ctx->sizes.group_sizze_9114;
    
    int32_t max_num_groups_9117;
    
    max_num_groups_9117 = ctx->sizes.max_num_groups_9116;
    
    int32_t y_9118 = group_sizze_9115 - 1;
    int32_t x_9119 = sizze_8784 + y_9118;
    int32_t w_div_group_sizze_9120 = squot32(x_9119, group_sizze_9115);
    int32_t num_groups_maybe_zzero_9121 = smin32(max_num_groups_9117,
                                                 w_div_group_sizze_9120);
    int32_t num_groups_9122 = smax32(1, num_groups_maybe_zzero_9121);
    int32_t num_threads_9123 = group_sizze_9115 * num_groups_9122;
    int32_t y_9124 = num_threads_9123 - 1;
    int32_t x_9125 = sizze_8784 + y_9124;
    int32_t per_thread_elements_9126 = squot32(x_9125, num_threads_9123);
    int64_t binop_x_9519 = sext_i32_i64(num_groups_9122);
    int64_t bytes_9518 = 4 * binop_x_9519;
    struct memblock_device mem_9520;
    
    mem_9520.references = NULL;
    memblock_alloc_device(ctx, &mem_9520, bytes_9518, "mem_9520");
    
    int64_t binop_x_9516 = sext_i32_i64(group_sizze_9115);
    int64_t bytes_9515 = 4 * binop_x_9516;
    struct memblock_local mem_9517;
    
    mem_9517.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8784);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 0,
                                  bytes_9515, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 1,
                                  sizeof(sizze_8784), &sizze_8784));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 2,
                                  sizeof(num_threads_9123), &num_threads_9123));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 3,
                                  sizeof(per_thread_elements_9126),
                                  &per_thread_elements_9126));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 6,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9131, 7,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    if (1 * (num_groups_9122 * group_sizze_9115) != 0) {
        const size_t global_work_sizze_9680[1] = {num_groups_9122 *
                     group_sizze_9115};
        const size_t local_work_sizze_9684[1] = {group_sizze_9115};
        int64_t time_start_9681 = 0, time_end_9682 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9131");
            fprintf(stderr, "%zu", global_work_sizze_9680[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9684[0]);
            fprintf(stderr, "].\n");
            time_start_9681 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9131,
                                              1, NULL, global_work_sizze_9680,
                                              local_work_sizze_9684, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9682 = get_wall_time();
            
            long time_diff_9683 = time_end_9682 - time_start_9681;
            
            ctx->chunked_reduce_kernel_9131_total_runtime += time_diff_9683;
            ctx->chunked_reduce_kernel_9131_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9131", time_diff_9683);
        }
    }
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    
    struct memblock_device mem_9526;
    
    mem_9526.references = NULL;
    memblock_alloc_device(ctx, &mem_9526, 4, "mem_9526");
    
    int64_t binop_x_9522 = sext_i32_i64(max_num_groups_9117);
    int64_t bytes_9521 = 4 * binop_x_9522;
    struct memblock_local mem_9523;
    
    mem_9523.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9167, 0, bytes_9521,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9167, 1,
                                  sizeof(num_groups_9122), &num_groups_9122));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9167, 2,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9167, 3,
                                  sizeof(mem_9526.mem), &mem_9526.mem));
    if (1 * max_num_groups_9117 != 0) {
        const size_t global_work_sizze_9685[1] = {max_num_groups_9117};
        const size_t local_work_sizze_9689[1] = {max_num_groups_9117};
        int64_t time_start_9686 = 0, time_end_9687 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9167");
            fprintf(stderr, "%zu", global_work_sizze_9685[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9689[0]);
            fprintf(stderr, "].\n");
            time_start_9686 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9167, 1, NULL,
                                              global_work_sizze_9685,
                                              local_work_sizze_9689, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9687 = get_wall_time();
            
            long time_diff_9688 = time_end_9687 - time_start_9686;
            
            ctx->reduce_kernel_9167_total_runtime += time_diff_9688;
            ctx->reduce_kernel_9167_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9167",
                    time_diff_9688);
        }
    }
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    
    float read_res_9690;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9526.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9690, 0,
                                       NULL, NULL));
    
    float res_8803 = read_res_9690;
    
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    scalar_out_9575 = res_8803;
    *out_scalar_out_9679 = scalar_out_9575;
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    return 0;
}
static int futrts_dot3(struct futhark_context *ctx, float *out_scalar_out_9691,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514, int32_t sizze_8812,
                       int32_t sizze_8813, int32_t sizze_8814)
{
    float scalar_out_9591;
    bool dim_zzero_8818 = 0 == sizze_8813;
    bool dim_zzero_8819 = 0 == sizze_8812;
    bool both_empty_8820 = dim_zzero_8818 && dim_zzero_8819;
    bool dim_match_8821 = sizze_8812 == sizze_8813;
    bool empty_or_match_8822 = both_empty_8820 || dim_match_8821;
    bool empty_or_match_cert_8823;
    
    if (!empty_or_match_8822) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:10:1-11:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8824 = 0 == sizze_8814;
    bool both_empty_8825 = dim_zzero_8819 && dim_zzero_8824;
    bool dim_match_8826 = sizze_8812 == sizze_8814;
    bool empty_or_match_8827 = both_empty_8825 || dim_match_8826;
    bool empty_or_match_cert_8828;
    
    if (!empty_or_match_8827) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:10:1-11:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9188;
    
    group_sizze_9188 = ctx->sizes.group_sizze_9187;
    
    int32_t max_num_groups_9190;
    
    max_num_groups_9190 = ctx->sizes.max_num_groups_9189;
    
    int32_t y_9191 = group_sizze_9188 - 1;
    int32_t x_9192 = sizze_8812 + y_9191;
    int32_t w_div_group_sizze_9193 = squot32(x_9192, group_sizze_9188);
    int32_t num_groups_maybe_zzero_9194 = smin32(max_num_groups_9190,
                                                 w_div_group_sizze_9193);
    int32_t num_groups_9195 = smax32(1, num_groups_maybe_zzero_9194);
    int32_t num_threads_9196 = group_sizze_9188 * num_groups_9195;
    int32_t y_9197 = num_threads_9196 - 1;
    int32_t x_9198 = sizze_8812 + y_9197;
    int32_t per_thread_elements_9199 = squot32(x_9198, num_threads_9196);
    int64_t binop_x_9519 = sext_i32_i64(num_groups_9195);
    int64_t bytes_9518 = 4 * binop_x_9519;
    struct memblock_device mem_9520;
    
    mem_9520.references = NULL;
    memblock_alloc_device(ctx, &mem_9520, bytes_9518, "mem_9520");
    
    int64_t binop_x_9516 = sext_i32_i64(group_sizze_9188);
    int64_t bytes_9515 = 4 * binop_x_9516;
    struct memblock_local mem_9517;
    
    mem_9517.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8812);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 0,
                                  bytes_9515, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 1,
                                  sizeof(sizze_8812), &sizze_8812));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 2,
                                  sizeof(num_threads_9196), &num_threads_9196));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 3,
                                  sizeof(per_thread_elements_9199),
                                  &per_thread_elements_9199));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 6,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9204, 7,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    if (1 * (num_groups_9195 * group_sizze_9188) != 0) {
        const size_t global_work_sizze_9692[1] = {num_groups_9195 *
                     group_sizze_9188};
        const size_t local_work_sizze_9696[1] = {group_sizze_9188};
        int64_t time_start_9693 = 0, time_end_9694 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9204");
            fprintf(stderr, "%zu", global_work_sizze_9692[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9696[0]);
            fprintf(stderr, "].\n");
            time_start_9693 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9204,
                                              1, NULL, global_work_sizze_9692,
                                              local_work_sizze_9696, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9694 = get_wall_time();
            
            long time_diff_9695 = time_end_9694 - time_start_9693;
            
            ctx->chunked_reduce_kernel_9204_total_runtime += time_diff_9695;
            ctx->chunked_reduce_kernel_9204_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9204", time_diff_9695);
        }
    }
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    
    struct memblock_device mem_9526;
    
    mem_9526.references = NULL;
    memblock_alloc_device(ctx, &mem_9526, 4, "mem_9526");
    
    int64_t binop_x_9522 = sext_i32_i64(max_num_groups_9190);
    int64_t bytes_9521 = 4 * binop_x_9522;
    struct memblock_local mem_9523;
    
    mem_9523.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9241, 0, bytes_9521,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9241, 1,
                                  sizeof(num_groups_9195), &num_groups_9195));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9241, 2,
                                  sizeof(mem_9520.mem), &mem_9520.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9241, 3,
                                  sizeof(mem_9526.mem), &mem_9526.mem));
    if (1 * max_num_groups_9190 != 0) {
        const size_t global_work_sizze_9697[1] = {max_num_groups_9190};
        const size_t local_work_sizze_9701[1] = {max_num_groups_9190};
        int64_t time_start_9698 = 0, time_end_9699 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9241");
            fprintf(stderr, "%zu", global_work_sizze_9697[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9701[0]);
            fprintf(stderr, "].\n");
            time_start_9698 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9241, 1, NULL,
                                              global_work_sizze_9697,
                                              local_work_sizze_9701, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9699 = get_wall_time();
            
            long time_diff_9700 = time_end_9699 - time_start_9698;
            
            ctx->reduce_kernel_9241_total_runtime += time_diff_9700;
            ctx->reduce_kernel_9241_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9241",
                    time_diff_9700);
        }
    }
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    
    float read_res_9702;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9526.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9702, 0,
                                       NULL, NULL));
    
    float res_8831 = read_res_9702;
    
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    scalar_out_9591 = res_8831;
    *out_scalar_out_9691 = scalar_out_9591;
    memblock_unref_local(ctx, &mem_9523, "mem_9523");
    memblock_unref_device(ctx, &mem_9526, "mem_9526");
    memblock_unref_local(ctx, &mem_9517, "mem_9517");
    memblock_unref_device(ctx, &mem_9520, "mem_9520");
    return 0;
}
static int futrts_dot4(struct futhark_context *ctx, float *out_scalar_out_9703,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8841,
                       int32_t sizze_8842, int32_t sizze_8843,
                       int32_t sizze_8844)
{
    float scalar_out_9607;
    bool dim_zzero_8849 = 0 == sizze_8842;
    bool dim_zzero_8850 = 0 == sizze_8841;
    bool both_empty_8851 = dim_zzero_8849 && dim_zzero_8850;
    bool dim_match_8852 = sizze_8841 == sizze_8842;
    bool empty_or_match_8853 = both_empty_8851 || dim_match_8852;
    bool empty_or_match_cert_8854;
    
    if (!empty_or_match_8853) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:13:1-14:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8855 = 0 == sizze_8843;
    bool both_empty_8856 = dim_zzero_8850 && dim_zzero_8855;
    bool dim_match_8857 = sizze_8841 == sizze_8843;
    bool empty_or_match_8858 = both_empty_8856 || dim_match_8857;
    bool empty_or_match_cert_8859;
    
    if (!empty_or_match_8858) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:13:1-14:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8860 = 0 == sizze_8844;
    bool both_empty_8861 = dim_zzero_8850 && dim_zzero_8860;
    bool dim_match_8862 = sizze_8841 == sizze_8844;
    bool empty_or_match_8863 = both_empty_8861 || dim_match_8862;
    bool empty_or_match_cert_8864;
    
    if (!empty_or_match_8863) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:13:1-14:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9264;
    
    group_sizze_9264 = ctx->sizes.group_sizze_9263;
    
    int32_t max_num_groups_9266;
    
    max_num_groups_9266 = ctx->sizes.max_num_groups_9265;
    
    int32_t y_9267 = group_sizze_9264 - 1;
    int32_t x_9268 = sizze_8841 + y_9267;
    int32_t w_div_group_sizze_9269 = squot32(x_9268, group_sizze_9264);
    int32_t num_groups_maybe_zzero_9270 = smin32(max_num_groups_9266,
                                                 w_div_group_sizze_9269);
    int32_t num_groups_9271 = smax32(1, num_groups_maybe_zzero_9270);
    int32_t num_threads_9272 = group_sizze_9264 * num_groups_9271;
    int32_t y_9273 = num_threads_9272 - 1;
    int32_t x_9274 = sizze_8841 + y_9273;
    int32_t per_thread_elements_9275 = squot32(x_9274, num_threads_9272);
    int64_t binop_x_9521 = sext_i32_i64(num_groups_9271);
    int64_t bytes_9520 = 4 * binop_x_9521;
    struct memblock_device mem_9522;
    
    mem_9522.references = NULL;
    memblock_alloc_device(ctx, &mem_9522, bytes_9520, "mem_9522");
    
    int64_t binop_x_9518 = sext_i32_i64(group_sizze_9264);
    int64_t bytes_9517 = 4 * binop_x_9518;
    struct memblock_local mem_9519;
    
    mem_9519.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8841);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 0,
                                  bytes_9517, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 1,
                                  sizeof(sizze_8841), &sizze_8841));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 2,
                                  sizeof(num_threads_9272), &num_threads_9272));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 3,
                                  sizeof(per_thread_elements_9275),
                                  &per_thread_elements_9275));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 6,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 7,
                                  sizeof(D_mem_9516.mem), &D_mem_9516.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9280, 8,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    if (1 * (num_groups_9271 * group_sizze_9264) != 0) {
        const size_t global_work_sizze_9704[1] = {num_groups_9271 *
                     group_sizze_9264};
        const size_t local_work_sizze_9708[1] = {group_sizze_9264};
        int64_t time_start_9705 = 0, time_end_9706 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9280");
            fprintf(stderr, "%zu", global_work_sizze_9704[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9708[0]);
            fprintf(stderr, "].\n");
            time_start_9705 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9280,
                                              1, NULL, global_work_sizze_9704,
                                              local_work_sizze_9708, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9706 = get_wall_time();
            
            long time_diff_9707 = time_end_9706 - time_start_9705;
            
            ctx->chunked_reduce_kernel_9280_total_runtime += time_diff_9707;
            ctx->chunked_reduce_kernel_9280_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9280", time_diff_9707);
        }
    }
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    
    struct memblock_device mem_9528;
    
    mem_9528.references = NULL;
    memblock_alloc_device(ctx, &mem_9528, 4, "mem_9528");
    
    int64_t binop_x_9524 = sext_i32_i64(max_num_groups_9266);
    int64_t bytes_9523 = 4 * binop_x_9524;
    struct memblock_local mem_9525;
    
    mem_9525.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9320, 0, bytes_9523,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9320, 1,
                                  sizeof(num_groups_9271), &num_groups_9271));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9320, 2,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9320, 3,
                                  sizeof(mem_9528.mem), &mem_9528.mem));
    if (1 * max_num_groups_9266 != 0) {
        const size_t global_work_sizze_9709[1] = {max_num_groups_9266};
        const size_t local_work_sizze_9713[1] = {max_num_groups_9266};
        int64_t time_start_9710 = 0, time_end_9711 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9320");
            fprintf(stderr, "%zu", global_work_sizze_9709[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9713[0]);
            fprintf(stderr, "].\n");
            time_start_9710 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9320, 1, NULL,
                                              global_work_sizze_9709,
                                              local_work_sizze_9713, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9711 = get_wall_time();
            
            long time_diff_9712 = time_end_9711 - time_start_9710;
            
            ctx->reduce_kernel_9320_total_runtime += time_diff_9712;
            ctx->reduce_kernel_9320_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9320",
                    time_diff_9712);
        }
    }
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    
    float read_res_9714;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9528.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9714, 0,
                                       NULL, NULL));
    
    float res_8868 = read_res_9714;
    
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    scalar_out_9607 = res_8868;
    *out_scalar_out_9703 = scalar_out_9607;
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
    return 0;
}
static int futrts_dot5(struct futhark_context *ctx, float *out_scalar_out_9715,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8879,
                       int32_t sizze_8880, int32_t sizze_8881,
                       int32_t sizze_8882, float a_8883, float b_8885,
                       float c_8887, float d_8889)
{
    float scalar_out_9623;
    bool dim_zzero_8891 = 0 == sizze_8880;
    bool dim_zzero_8892 = 0 == sizze_8879;
    bool both_empty_8893 = dim_zzero_8891 && dim_zzero_8892;
    bool dim_match_8894 = sizze_8879 == sizze_8880;
    bool empty_or_match_8895 = both_empty_8893 || dim_match_8894;
    bool empty_or_match_cert_8896;
    
    if (!empty_or_match_8895) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:16:1-17:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8897 = 0 == sizze_8881;
    bool both_empty_8898 = dim_zzero_8892 && dim_zzero_8897;
    bool dim_match_8899 = sizze_8879 == sizze_8881;
    bool empty_or_match_8900 = both_empty_8898 || dim_match_8899;
    bool empty_or_match_cert_8901;
    
    if (!empty_or_match_8900) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:16:1-17:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8902 = 0 == sizze_8882;
    bool both_empty_8903 = dim_zzero_8892 && dim_zzero_8902;
    bool dim_match_8904 = sizze_8879 == sizze_8882;
    bool empty_or_match_8905 = both_empty_8903 || dim_match_8904;
    bool empty_or_match_cert_8906;
    
    if (!empty_or_match_8905) {
        ctx->error = msgprintf("Error at %s:\n%s\n", "1d.fut:16:1-17:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9343;
    
    group_sizze_9343 = ctx->sizes.group_sizze_9342;
    
    int32_t max_num_groups_9345;
    
    max_num_groups_9345 = ctx->sizes.max_num_groups_9344;
    
    int32_t y_9346 = group_sizze_9343 - 1;
    int32_t x_9347 = sizze_8879 + y_9346;
    int32_t w_div_group_sizze_9348 = squot32(x_9347, group_sizze_9343);
    int32_t num_groups_maybe_zzero_9349 = smin32(max_num_groups_9345,
                                                 w_div_group_sizze_9348);
    int32_t num_groups_9350 = smax32(1, num_groups_maybe_zzero_9349);
    int32_t num_threads_9351 = group_sizze_9343 * num_groups_9350;
    int32_t y_9352 = num_threads_9351 - 1;
    int32_t x_9353 = sizze_8879 + y_9352;
    int32_t per_thread_elements_9354 = squot32(x_9353, num_threads_9351);
    int64_t binop_x_9521 = sext_i32_i64(num_groups_9350);
    int64_t bytes_9520 = 4 * binop_x_9521;
    struct memblock_device mem_9522;
    
    mem_9522.references = NULL;
    memblock_alloc_device(ctx, &mem_9522, bytes_9520, "mem_9522");
    
    int64_t binop_x_9518 = sext_i32_i64(group_sizze_9343);
    int64_t bytes_9517 = 4 * binop_x_9518;
    struct memblock_local mem_9519;
    
    mem_9519.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8879);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 0,
                                  bytes_9517, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 1,
                                  sizeof(sizze_8879), &sizze_8879));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 2,
                                  sizeof(a_8883), &a_8883));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 3,
                                  sizeof(b_8885), &b_8885));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 4,
                                  sizeof(c_8887), &c_8887));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 5,
                                  sizeof(d_8889), &d_8889));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 6,
                                  sizeof(num_threads_9351), &num_threads_9351));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 7,
                                  sizeof(per_thread_elements_9354),
                                  &per_thread_elements_9354));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 8,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 9,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 10,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 11,
                                  sizeof(D_mem_9516.mem), &D_mem_9516.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9359, 12,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    if (1 * (num_groups_9350 * group_sizze_9343) != 0) {
        const size_t global_work_sizze_9716[1] = {num_groups_9350 *
                     group_sizze_9343};
        const size_t local_work_sizze_9720[1] = {group_sizze_9343};
        int64_t time_start_9717 = 0, time_end_9718 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9359");
            fprintf(stderr, "%zu", global_work_sizze_9716[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9720[0]);
            fprintf(stderr, "].\n");
            time_start_9717 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9359,
                                              1, NULL, global_work_sizze_9716,
                                              local_work_sizze_9720, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9718 = get_wall_time();
            
            long time_diff_9719 = time_end_9718 - time_start_9717;
            
            ctx->chunked_reduce_kernel_9359_total_runtime += time_diff_9719;
            ctx->chunked_reduce_kernel_9359_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9359", time_diff_9719);
        }
    }
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    
    struct memblock_device mem_9528;
    
    mem_9528.references = NULL;
    memblock_alloc_device(ctx, &mem_9528, 4, "mem_9528");
    
    int64_t binop_x_9524 = sext_i32_i64(max_num_groups_9345);
    int64_t bytes_9523 = 4 * binop_x_9524;
    struct memblock_local mem_9525;
    
    mem_9525.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9403, 0, bytes_9523,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9403, 1,
                                  sizeof(num_groups_9350), &num_groups_9350));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9403, 2,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9403, 3,
                                  sizeof(mem_9528.mem), &mem_9528.mem));
    if (1 * max_num_groups_9345 != 0) {
        const size_t global_work_sizze_9721[1] = {max_num_groups_9345};
        const size_t local_work_sizze_9725[1] = {max_num_groups_9345};
        int64_t time_start_9722 = 0, time_end_9723 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9403");
            fprintf(stderr, "%zu", global_work_sizze_9721[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9725[0]);
            fprintf(stderr, "].\n");
            time_start_9722 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9403, 1, NULL,
                                              global_work_sizze_9721,
                                              local_work_sizze_9725, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9723 = get_wall_time();
            
            long time_diff_9724 = time_end_9723 - time_start_9722;
            
            ctx->reduce_kernel_9403_total_runtime += time_diff_9724;
            ctx->reduce_kernel_9403_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9403",
                    time_diff_9724);
        }
    }
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    
    float read_res_9726;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9528.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9726, 0,
                                       NULL, NULL));
    
    float res_8910 = read_res_9726;
    
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    scalar_out_9623 = res_8910;
    *out_scalar_out_9715 = scalar_out_9623;
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
    return 0;
}
static int futrts_dot6(struct futhark_context *ctx, float *out_scalar_out_9727,
                       int64_t A_mem_sizze_9509,
                       struct memblock_device A_mem_9510,
                       int64_t B_mem_sizze_9511,
                       struct memblock_device B_mem_9512,
                       int64_t C_mem_sizze_9513,
                       struct memblock_device C_mem_9514,
                       int64_t D_mem_sizze_9515,
                       struct memblock_device D_mem_9516, int32_t sizze_8925,
                       int32_t sizze_8926, int32_t sizze_8927,
                       int32_t sizze_8928)
{
    float scalar_out_9639;
    bool dim_zzero_8933 = 0 == sizze_8926;
    bool dim_zzero_8934 = 0 == sizze_8925;
    bool both_empty_8935 = dim_zzero_8933 && dim_zzero_8934;
    bool dim_match_8936 = sizze_8925 == sizze_8926;
    bool empty_or_match_8937 = both_empty_8935 || dim_match_8936;
    bool empty_or_match_cert_8938;
    
    if (!empty_or_match_8937) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "1d.fut:19:1-22:33 -> 1d.fut:20:14-20:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_8939 = 0 == sizze_8928;
    bool dim_zzero_8940 = 0 == sizze_8927;
    bool both_empty_8941 = dim_zzero_8939 && dim_zzero_8940;
    bool dim_match_8942 = sizze_8927 == sizze_8928;
    bool empty_or_match_8943 = both_empty_8941 || dim_match_8942;
    bool empty_or_match_cert_8944;
    
    if (!empty_or_match_8943) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "1d.fut:19:1-22:33 -> 1d.fut:21:14-21:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool both_empty_8945 = dim_zzero_8934 && dim_zzero_8940;
    bool dim_match_8946 = sizze_8925 == sizze_8927;
    bool empty_or_match_8947 = both_empty_8945 || dim_match_8946;
    bool empty_or_match_cert_8948;
    
    if (!empty_or_match_8947) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "1d.fut:19:1-22:33 -> 1d.fut:22:19-22:32",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_9426;
    
    group_sizze_9426 = ctx->sizes.group_sizze_9425;
    
    int32_t max_num_groups_9428;
    
    max_num_groups_9428 = ctx->sizes.max_num_groups_9427;
    
    int32_t y_9429 = group_sizze_9426 - 1;
    int32_t x_9430 = sizze_8925 + y_9429;
    int32_t w_div_group_sizze_9431 = squot32(x_9430, group_sizze_9426);
    int32_t num_groups_maybe_zzero_9432 = smin32(max_num_groups_9428,
                                                 w_div_group_sizze_9431);
    int32_t num_groups_9433 = smax32(1, num_groups_maybe_zzero_9432);
    int32_t num_threads_9434 = group_sizze_9426 * num_groups_9433;
    int32_t y_9435 = num_threads_9434 - 1;
    int32_t x_9436 = sizze_8925 + y_9435;
    int32_t per_thread_elements_9437 = squot32(x_9436, num_threads_9434);
    int64_t binop_x_9521 = sext_i32_i64(num_groups_9433);
    int64_t bytes_9520 = 4 * binop_x_9521;
    struct memblock_device mem_9522;
    
    mem_9522.references = NULL;
    memblock_alloc_device(ctx, &mem_9522, bytes_9520, "mem_9522");
    
    int64_t binop_x_9518 = sext_i32_i64(group_sizze_9426);
    int64_t bytes_9517 = 4 * binop_x_9518;
    struct memblock_local mem_9519;
    
    mem_9519.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_8925);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 0,
                                  bytes_9517, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 1,
                                  sizeof(sizze_8925), &sizze_8925));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 2,
                                  sizeof(num_threads_9434), &num_threads_9434));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 3,
                                  sizeof(per_thread_elements_9437),
                                  &per_thread_elements_9437));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 4,
                                  sizeof(A_mem_9510.mem), &A_mem_9510.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 5,
                                  sizeof(B_mem_9512.mem), &B_mem_9512.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 6,
                                  sizeof(C_mem_9514.mem), &C_mem_9514.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 7,
                                  sizeof(D_mem_9516.mem), &D_mem_9516.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_9442, 8,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    if (1 * (num_groups_9433 * group_sizze_9426) != 0) {
        const size_t global_work_sizze_9728[1] = {num_groups_9433 *
                     group_sizze_9426};
        const size_t local_work_sizze_9732[1] = {group_sizze_9426};
        int64_t time_start_9729 = 0, time_end_9730 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_9442");
            fprintf(stderr, "%zu", global_work_sizze_9728[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9732[0]);
            fprintf(stderr, "].\n");
            time_start_9729 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_9442,
                                              1, NULL, global_work_sizze_9728,
                                              local_work_sizze_9732, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9730 = get_wall_time();
            
            long time_diff_9731 = time_end_9730 - time_start_9729;
            
            ctx->chunked_reduce_kernel_9442_total_runtime += time_diff_9731;
            ctx->chunked_reduce_kernel_9442_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_9442", time_diff_9731);
        }
    }
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    
    struct memblock_device mem_9528;
    
    mem_9528.references = NULL;
    memblock_alloc_device(ctx, &mem_9528, 4, "mem_9528");
    
    int64_t binop_x_9524 = sext_i32_i64(max_num_groups_9428);
    int64_t bytes_9523 = 4 * binop_x_9524;
    struct memblock_local mem_9525;
    
    mem_9525.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9482, 0, bytes_9523,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9482, 1,
                                  sizeof(num_groups_9433), &num_groups_9433));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9482, 2,
                                  sizeof(mem_9522.mem), &mem_9522.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_9482, 3,
                                  sizeof(mem_9528.mem), &mem_9528.mem));
    if (1 * max_num_groups_9428 != 0) {
        const size_t global_work_sizze_9733[1] = {max_num_groups_9428};
        const size_t local_work_sizze_9737[1] = {max_num_groups_9428};
        int64_t time_start_9734 = 0, time_end_9735 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_9482");
            fprintf(stderr, "%zu", global_work_sizze_9733[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_9737[0]);
            fprintf(stderr, "].\n");
            time_start_9734 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_9482, 1, NULL,
                                              global_work_sizze_9733,
                                              local_work_sizze_9737, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_9735 = get_wall_time();
            
            long time_diff_9736 = time_end_9735 - time_start_9734;
            
            ctx->reduce_kernel_9482_total_runtime += time_diff_9736;
            ctx->reduce_kernel_9482_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_9482",
                    time_diff_9736);
        }
    }
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    
    float read_res_9738;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_9528.mem, CL_TRUE,
                                       0, sizeof(float), &read_res_9738, 0,
                                       NULL, NULL));
    
    float res_8952 = read_res_9738;
    
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    scalar_out_9639 = res_8952;
    *out_scalar_out_9727 = scalar_out_9639;
    memblock_unref_local(ctx, &mem_9525, "mem_9525");
    memblock_unref_device(ctx, &mem_9528, "mem_9528");
    memblock_unref_local(ctx, &mem_9519, "mem_9519");
    memblock_unref_device(ctx, &mem_9522, "mem_9522");
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
int futhark_entry_dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int32_t sizze_8738;
    int32_t sizze_8739;
    float scalar_out_9543;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8738 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8739 = in1->shape[0];
    
    int ret = futrts_dot(ctx, &scalar_out_9543, A_mem_sizze_9509, A_mem_9510,
                         B_mem_sizze_9511, B_mem_9512, sizze_8738, sizze_8739);
    
    if (ret == 0) {
        *out0 = scalar_out_9543;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int32_t sizze_8756;
    int32_t sizze_8757;
    int32_t sizze_8758;
    float scalar_out_9559;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8756 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8757 = in1->shape[0];
    C_mem_9514 = in2->mem;
    C_mem_sizze_9513 = in2->mem.size;
    sizze_8758 = in2->shape[0];
    
    int ret = futrts_dot1(ctx, &scalar_out_9559, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, sizze_8756, sizze_8757, sizze_8758);
    
    if (ret == 0) {
        *out0 = scalar_out_9559;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot2(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int32_t sizze_8784;
    int32_t sizze_8785;
    int32_t sizze_8786;
    float scalar_out_9575;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8784 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8785 = in1->shape[0];
    C_mem_9514 = in2->mem;
    C_mem_sizze_9513 = in2->mem.size;
    sizze_8786 = in2->shape[0];
    
    int ret = futrts_dot2(ctx, &scalar_out_9575, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, sizze_8784, sizze_8785, sizze_8786);
    
    if (ret == 0) {
        *out0 = scalar_out_9575;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot3(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int32_t sizze_8812;
    int32_t sizze_8813;
    int32_t sizze_8814;
    float scalar_out_9591;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8812 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8813 = in1->shape[0];
    C_mem_9514 = in2->mem;
    C_mem_sizze_9513 = in2->mem.size;
    sizze_8814 = in2->shape[0];
    
    int ret = futrts_dot3(ctx, &scalar_out_9591, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, sizze_8812, sizze_8813, sizze_8814);
    
    if (ret == 0) {
        *out0 = scalar_out_9591;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot4(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int64_t D_mem_sizze_9515;
    struct memblock_device D_mem_9516;
    
    D_mem_9516.references = NULL;
    
    int32_t sizze_8841;
    int32_t sizze_8842;
    int32_t sizze_8843;
    int32_t sizze_8844;
    float scalar_out_9607;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8841 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8842 = in1->shape[0];
    C_mem_9514 = in2->mem;
    C_mem_sizze_9513 = in2->mem.size;
    sizze_8843 = in2->shape[0];
    D_mem_9516 = in3->mem;
    D_mem_sizze_9515 = in3->mem.size;
    sizze_8844 = in3->shape[0];
    
    int ret = futrts_dot4(ctx, &scalar_out_9607, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, D_mem_sizze_9515, D_mem_9516, sizze_8841,
                          sizze_8842, sizze_8843, sizze_8844);
    
    if (ret == 0) {
        *out0 = scalar_out_9607;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot5(struct futhark_context *ctx, float *out0, const
                       float in0, const struct futhark_f32_1d *in1, const
                       float in2, const struct futhark_f32_1d *in3, const
                       float in4, const struct futhark_f32_1d *in5, const
                       float in6, const struct futhark_f32_1d *in7)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int64_t D_mem_sizze_9515;
    struct memblock_device D_mem_9516;
    
    D_mem_9516.references = NULL;
    
    int32_t sizze_8879;
    int32_t sizze_8880;
    int32_t sizze_8881;
    int32_t sizze_8882;
    float a_8883;
    float b_8885;
    float c_8887;
    float d_8889;
    float scalar_out_9623;
    
    lock_lock(&ctx->lock);
    a_8883 = in0;
    A_mem_9510 = in1->mem;
    A_mem_sizze_9509 = in1->mem.size;
    sizze_8879 = in1->shape[0];
    b_8885 = in2;
    B_mem_9512 = in3->mem;
    B_mem_sizze_9511 = in3->mem.size;
    sizze_8880 = in3->shape[0];
    c_8887 = in4;
    C_mem_9514 = in5->mem;
    C_mem_sizze_9513 = in5->mem.size;
    sizze_8881 = in5->shape[0];
    d_8889 = in6;
    D_mem_9516 = in7->mem;
    D_mem_sizze_9515 = in7->mem.size;
    sizze_8882 = in7->shape[0];
    
    int ret = futrts_dot5(ctx, &scalar_out_9623, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, D_mem_sizze_9515, D_mem_9516, sizze_8879,
                          sizze_8880, sizze_8881, sizze_8882, a_8883, b_8885,
                          c_8887, d_8889);
    
    if (ret == 0) {
        *out0 = scalar_out_9623;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot6(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_9509;
    struct memblock_device A_mem_9510;
    
    A_mem_9510.references = NULL;
    
    int64_t B_mem_sizze_9511;
    struct memblock_device B_mem_9512;
    
    B_mem_9512.references = NULL;
    
    int64_t C_mem_sizze_9513;
    struct memblock_device C_mem_9514;
    
    C_mem_9514.references = NULL;
    
    int64_t D_mem_sizze_9515;
    struct memblock_device D_mem_9516;
    
    D_mem_9516.references = NULL;
    
    int32_t sizze_8925;
    int32_t sizze_8926;
    int32_t sizze_8927;
    int32_t sizze_8928;
    float scalar_out_9639;
    
    lock_lock(&ctx->lock);
    A_mem_9510 = in0->mem;
    A_mem_sizze_9509 = in0->mem.size;
    sizze_8925 = in0->shape[0];
    B_mem_9512 = in1->mem;
    B_mem_sizze_9511 = in1->mem.size;
    sizze_8926 = in1->shape[0];
    C_mem_9514 = in2->mem;
    C_mem_sizze_9513 = in2->mem.size;
    sizze_8927 = in2->shape[0];
    D_mem_9516 = in3->mem;
    D_mem_sizze_9515 = in3->mem.size;
    sizze_8928 = in3->shape[0];
    
    int ret = futrts_dot6(ctx, &scalar_out_9639, A_mem_sizze_9509, A_mem_9510,
                          B_mem_sizze_9511, B_mem_9512, C_mem_sizze_9513,
                          C_mem_9514, D_mem_sizze_9515, D_mem_9516, sizze_8925,
                          sizze_8926, sizze_8927, sizze_8928);
    
    if (ret == 0) {
        *out0 = scalar_out_9639;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
