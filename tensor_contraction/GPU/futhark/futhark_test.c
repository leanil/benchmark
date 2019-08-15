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
            "urn x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\nstatic inline float futrts_log32(float x)\n{\n    return log(x);\n}\nstatic inline float futrts_log2_32(float x)\n{\n    return log2(x);\n}\nstatic inline float futrts_log10_32(float x)\n{\n    return log10(x);\n}\nstatic inline float futrts_sqrt32(float x)\n{\n    return sqrt(x);\n}\nstatic inline float futrts_exp32(float x)\n{\n    return exp(x);\n}\nstatic inline float futrts_cos32(float x)\n{\n    return cos(x);\n}\nstatic inline float futrts_sin32(float x)\n{\n    return sin(x);\n}\nstatic inline float futrts_tan32(float x)\n{\n    return tan(x);\n}\nstatic inline float futrts_acos32(float x)\n{\n    return acos(x);\n}\nstatic inline float futrts_asin32(float x)\n{\n    return asin(x);\n}\nstatic inline float futrts_atan32(float x)\n{\n    return atan(x);\n}\nstatic inline float futrts_atan2_32(float x, float y)\n{\n    return atan2(x, y);\n}\nstatic inline float futrts_round32(float x)\n{\n    return rint(x);\n}\nstatic inline char futrts_isnan32(float x)\n{\n    return isnan(x);\n}\nstatic inline char futrts_isinf32(float x)\n{\n    return isinf(x);\n}\nstatic inline int32_t futrts_to_bits32(float x)\n{\n    union {\n        float f;\n        int32_t t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\nstatic inline float futrts_from_bits32(int32_t x)\n{\n    union {\n        int32_t f;\n        float t;\n    } p;\n    \n    p.f = x;\n    return p.t;\n}\n#define group_sizze_48310 (group_size_48309)\n#define max_num_groups_48312 (max_num_groups_48311)\n#define group_sizze_48385 (group_size_48384)\n#define max_num_groups_48387 (ma",
            "x_num_groups_48386)\n#define group_sizze_48454 (group_size_48453)\n#define max_num_groups_48456 (max_num_groups_48455)\n#define group_sizze_48527 (group_size_48526)\n#define max_num_groups_48529 (max_num_groups_48528)\n#define group_sizze_48603 (group_size_48602)\n#define max_num_groups_48605 (max_num_groups_48604)\n#define group_sizze_48682 (group_size_48681)\n#define max_num_groups_48684 (max_num_groups_48683)\n#define group_sizze_48765 (group_size_48764)\n#define max_num_groups_48767 (max_num_groups_48766)\n#define group_sizze_48842 (group_size_48841)\n#define max_num_groups_48844 (max_num_groups_48843)\n#define group_sizze_48922 (group_size_48921)\n#define max_num_groups_48924 (max_num_groups_48923)\n#define group_sizze_49268 (group_size_49267)\n#define tile_sizze_50528 (tile_size_50527)\n#define tiled_group_sizze_50529 (tile_size_50527 * tile_size_50527)\n#define tile_sizze_50528 (tile_size_50527)\n#define tiled_group_sizze_50529 (tile_size_50527 * tile_size_50527)\n#define group_sizze_49945 (group_size_49944)\n#define tile_sizze_50528 (tile_size_50527)\n#define tiled_group_sizze_50529 (tile_size_50527 * tile_size_50527)\n#define tile_sizze_50528 (tile_size_50527)\n#define tiled_group_sizze_50529 (tile_size_50527 * tile_size_50527)\n#define tile_sizze_50528 (tile_size_50527)\n#define tiled_group_sizze_50529 (tile_size_50527 * tile_size_50527)\n#define group_sizze_50446 (group_size_50445)\n__kernel void chunked_reduce_kernel_48326(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_46897,\n                                          int32_t num_threads_48318,\n                                          int32_t per_thread_elements_48321,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *mem_50615)\n{\n    __local volatile char *restrict mem_50612 = mem_aligned_0;\n    int32_t wave_sizze_50764;\n    int32_t group_sizze_50",
            "765;\n    bool thread_active_50766;\n    int32_t global_tid_48326;\n    int32_t local_tid_48327;\n    int32_t group_id_48328;\n    \n    global_tid_48326 = get_global_id(0);\n    local_tid_48327 = get_local_id(0);\n    group_sizze_50765 = get_local_size(0);\n    wave_sizze_50764 = LOCKSTEP_WIDTH;\n    group_id_48328 = get_group_id(0);\n    thread_active_50766 = 1;\n    \n    int32_t chunk_sizze_48333 = smin32(per_thread_elements_48321,\n                                       squot32(sizze_46897 - global_tid_48326 +\n                                               num_threads_48318 - 1,\n                                               num_threads_48318));\n    float res_48336;\n    \n    if (thread_active_50766) {\n        float acc_48339 = 0.0F;\n        \n        for (int32_t i_48338 = 0; i_48338 < chunk_sizze_48333; i_48338++) {\n            int32_t j_t_s_50559 = num_threads_48318 * i_48338;\n            int32_t j_p_i_t_s_50560 = global_tid_48326 + j_t_s_50559;\n            float x_48341 = *(__global float *) &A_mem_50609[j_p_i_t_s_50560 *\n                                                             4];\n            float res_48344 = acc_48339 + x_48341;\n            float acc_tmp_50767 = res_48344;\n            \n            acc_48339 = acc_tmp_50767;\n        }\n        res_48336 = acc_48339;\n    }\n    \n    float final_result_48347;\n    \n    for (int32_t comb_iter_50768 = 0; comb_iter_50768 <\n         squot32(group_sizze_48310 + group_sizze_48310 - 1, group_sizze_48310);\n         comb_iter_50768++) {\n        int32_t combine_id_48331;\n        int32_t flat_comb_id_50769 = comb_iter_50768 * group_sizze_48310 +\n                local_tid_48327;\n        \n        combine_id_48331 = flat_comb_id_50769;\n        if (slt32(combine_id_48331, group_sizze_48310) && 1) {\n            *(__local float *) &mem_50612[combine_id_48331 * 4] = res_48336;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50770;\n    int32_t my_index_48348;\n    int32_t other_offset_48349;\n    float x_48350;\n ",
            "   float x_48351;\n    \n    my_index_48348 = local_tid_48327;\n    other_offset_48349 = 0;\n    if (slt32(local_tid_48327, group_sizze_48310)) {\n        x_48350 = *(__local float *) &mem_50612[(local_tid_48327 +\n                                                 other_offset_48349) * 4];\n    }\n    other_offset_48349 = 1;\n    while (slt32(other_offset_48349, wave_sizze_50764)) {\n        if (slt32(local_tid_48327 + other_offset_48349, group_sizze_48310) &&\n            ((local_tid_48327 - squot32(local_tid_48327, wave_sizze_50764) *\n              wave_sizze_50764) & (2 * other_offset_48349 - 1)) == 0) {\n            // read array element\n            {\n                x_48351 = *(volatile __local\n                            float *) &mem_50612[(local_tid_48327 +\n                                                 other_offset_48349) * 4];\n            }\n            \n            float res_48352;\n            \n            if (thread_active_50766) {\n                res_48352 = x_48350 + x_48351;\n            }\n            x_48350 = res_48352;\n            *(volatile __local float *) &mem_50612[local_tid_48327 * 4] =\n                x_48350;\n        }\n        other_offset_48349 *= 2;\n    }\n    skip_waves_50770 = 1;\n    while (slt32(skip_waves_50770, squot32(group_sizze_48310 +\n                                           wave_sizze_50764 - 1,\n                                           wave_sizze_50764))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48349 = skip_waves_50770 * wave_sizze_50764;\n        if (slt32(local_tid_48327 + other_offset_48349, group_sizze_48310) &&\n            ((local_tid_48327 - squot32(local_tid_48327, wave_sizze_50764) *\n              wave_sizze_50764) == 0 && (squot32(local_tid_48327,\n                                                 wave_sizze_50764) & (2 *\n                                                                      skip_waves_50770 -\n                                                                      1)) ==\n             0)) {\n         ",
            "   // read array element\n            {\n                x_48351 = *(__local float *) &mem_50612[(local_tid_48327 +\n                                                         other_offset_48349) *\n                                                        4];\n            }\n            \n            float res_48352;\n            \n            if (thread_active_50766) {\n                res_48352 = x_48350 + x_48351;\n            }\n            x_48350 = res_48352;\n            *(__local float *) &mem_50612[local_tid_48327 * 4] = x_48350;\n        }\n        skip_waves_50770 *= 2;\n    }\n    final_result_48347 = x_48350;\n    if (local_tid_48327 == 0) {\n        *(__global float *) &mem_50615[group_id_48328 * 4] = final_result_48347;\n    }\n}\n__kernel void chunked_reduce_kernel_48401(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_46910,\n                                          int32_t num_threads_48393,\n                                          int32_t per_thread_elements_48396,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *mem_50617)\n{\n    __local volatile char *restrict mem_50614 = mem_aligned_0;\n    int32_t wave_sizze_50786;\n    int32_t group_sizze_50787;\n    bool thread_active_50788;\n    int32_t global_tid_48401;\n    int32_t local_tid_48402;\n    int32_t group_id_48403;\n    \n    global_tid_48401 = get_global_id(0);\n    local_tid_48402 = get_local_id(0);\n    group_sizze_50787 = get_local_size(0);\n    wave_sizze_50786 = LOCKSTEP_WIDTH;\n    group_id_48403 = get_group_id(0);\n    thread_active_50788 = 1;\n    \n    int32_t chunk_sizze_48408 = smin32(per_thread_elements_48396,\n                                       squot32(sizze_46910 - global_tid_48401 +\n                                               num_threads_48393 - 1,\n  ",
            "                                             num_threads_48393));\n    float res_48412;\n    \n    if (thread_active_50788) {\n        float acc_48415 = 0.0F;\n        \n        for (int32_t i_48414 = 0; i_48414 < chunk_sizze_48408; i_48414++) {\n            int32_t j_t_s_50563 = num_threads_48393 * i_48414;\n            int32_t j_p_i_t_s_50564 = global_tid_48401 + j_t_s_50563;\n            float x_48418 = *(__global float *) &A_mem_50609[j_p_i_t_s_50564 *\n                                                             4];\n            float x_48419 = *(__global float *) &B_mem_50611[j_p_i_t_s_50564 *\n                                                             4];\n            float res_48421 = x_48418 * x_48419;\n            float res_48423 = acc_48415 + res_48421;\n            float acc_tmp_50789 = res_48423;\n            \n            acc_48415 = acc_tmp_50789;\n        }\n        res_48412 = acc_48415;\n    }\n    \n    float final_result_48426;\n    \n    for (int32_t comb_iter_50790 = 0; comb_iter_50790 <\n         squot32(group_sizze_48385 + group_sizze_48385 - 1, group_sizze_48385);\n         comb_iter_50790++) {\n        int32_t combine_id_48406;\n        int32_t flat_comb_id_50791 = comb_iter_50790 * group_sizze_48385 +\n                local_tid_48402;\n        \n        combine_id_48406 = flat_comb_id_50791;\n        if (slt32(combine_id_48406, group_sizze_48385) && 1) {\n            *(__local float *) &mem_50614[combine_id_48406 * 4] = res_48412;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50792;\n    int32_t my_index_48427;\n    int32_t other_offset_48428;\n    float x_48429;\n    float x_48430;\n    \n    my_index_48427 = local_tid_48402;\n    other_offset_48428 = 0;\n    if (slt32(local_tid_48402, group_sizze_48385)) {\n        x_48429 = *(__local float *) &mem_50614[(local_tid_48402 +\n                                                 other_offset_48428) * 4];\n    }\n    other_offset_48428 = 1;\n    while (slt32(other_offset_48428, wave_sizze_50786)) {\n        ",
            "if (slt32(local_tid_48402 + other_offset_48428, group_sizze_48385) &&\n            ((local_tid_48402 - squot32(local_tid_48402, wave_sizze_50786) *\n              wave_sizze_50786) & (2 * other_offset_48428 - 1)) == 0) {\n            // read array element\n            {\n                x_48430 = *(volatile __local\n                            float *) &mem_50614[(local_tid_48402 +\n                                                 other_offset_48428) * 4];\n            }\n            \n            float res_48431;\n            \n            if (thread_active_50788) {\n                res_48431 = x_48429 + x_48430;\n            }\n            x_48429 = res_48431;\n            *(volatile __local float *) &mem_50614[local_tid_48402 * 4] =\n                x_48429;\n        }\n        other_offset_48428 *= 2;\n    }\n    skip_waves_50792 = 1;\n    while (slt32(skip_waves_50792, squot32(group_sizze_48385 +\n                                           wave_sizze_50786 - 1,\n                                           wave_sizze_50786))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48428 = skip_waves_50792 * wave_sizze_50786;\n        if (slt32(local_tid_48402 + other_offset_48428, group_sizze_48385) &&\n            ((local_tid_48402 - squot32(local_tid_48402, wave_sizze_50786) *\n              wave_sizze_50786) == 0 && (squot32(local_tid_48402,\n                                                 wave_sizze_50786) & (2 *\n                                                                      skip_waves_50792 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48430 = *(__local float *) &mem_50614[(local_tid_48402 +\n                                                         other_offset_48428) *\n                                                        4];\n            }\n            \n            float res_48431;\n            \n            if (thread_active_50788) {\n                res_484",
            "31 = x_48429 + x_48430;\n            }\n            x_48429 = res_48431;\n            *(__local float *) &mem_50614[local_tid_48402 * 4] = x_48429;\n        }\n        skip_waves_50792 *= 2;\n    }\n    final_result_48426 = x_48429;\n    if (local_tid_48402 == 0) {\n        *(__global float *) &mem_50617[group_id_48403 * 4] = final_result_48426;\n    }\n}\n__kernel void chunked_reduce_kernel_48470(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_46928,\n                                          int32_t num_threads_48462,\n                                          int32_t per_thread_elements_48465,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *mem_50619)\n{\n    __local volatile char *restrict mem_50616 = mem_aligned_0;\n    int32_t wave_sizze_50802;\n    int32_t group_sizze_50803;\n    bool thread_active_50804;\n    int32_t global_tid_48470;\n    int32_t local_tid_48471;\n    int32_t group_id_48472;\n    \n    global_tid_48470 = get_global_id(0);\n    local_tid_48471 = get_local_id(0);\n    group_sizze_50803 = get_local_size(0);\n    wave_sizze_50802 = LOCKSTEP_WIDTH;\n    group_id_48472 = get_group_id(0);\n    thread_active_50804 = 1;\n    \n    int32_t chunk_sizze_48477 = smin32(per_thread_elements_48465,\n                                       squot32(sizze_46928 - global_tid_48470 +\n                                               num_threads_48462 - 1,\n                                               num_threads_48462));\n    float res_48482;\n    \n    if (thread_active_50804) {\n        float acc_48485 = 0.0F;\n        \n        for (int32_t i_48484 = 0; i_48484 < chunk_sizze_48477; i_48484++) {\n            int32_t j_t_s_50567 = num_threads_48462 * i_48484;\n  ",
            "          int32_t j_p_i_t_s_50568 = global_tid_48470 + j_t_s_50567;\n            float x_48489 = *(__global float *) &A_mem_50609[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48490 = *(__global float *) &B_mem_50611[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48491 = *(__global float *) &C_mem_50613[j_p_i_t_s_50568 *\n                                                             4];\n            float res_48493 = x_48489 + x_48490;\n            float res_48494 = x_48491 * res_48493;\n            float res_48496 = acc_48485 + res_48494;\n            float acc_tmp_50805 = res_48496;\n            \n            acc_48485 = acc_tmp_50805;\n        }\n        res_48482 = acc_48485;\n    }\n    \n    float final_result_48499;\n    \n    for (int32_t comb_iter_50806 = 0; comb_iter_50806 <\n         squot32(group_sizze_48454 + group_sizze_48454 - 1, group_sizze_48454);\n         comb_iter_50806++) {\n        int32_t combine_id_48475;\n        int32_t flat_comb_id_50807 = comb_iter_50806 * group_sizze_48454 +\n                local_tid_48471;\n        \n        combine_id_48475 = flat_comb_id_50807;\n        if (slt32(combine_id_48475, group_sizze_48454) && 1) {\n            *(__local float *) &mem_50616[combine_id_48475 * 4] = res_48482;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50808;\n    int32_t my_index_48500;\n    int32_t other_offset_48501;\n    float x_48502;\n    float x_48503;\n    \n    my_index_48500 = local_tid_48471;\n    other_offset_48501 = 0;\n    if (slt32(local_tid_48471, group_sizze_48454)) {\n        x_48502 = *(__local float *) &mem_50616[(local_tid_48471 +\n                                                 other_offset_48501) * 4];\n    }\n    other_offset_48501 = 1;\n    while (slt32(other_offset_48501, wave_sizze_50802)) {\n        if (slt32(local_tid_48471 + other_offset_48501, group_sizze_48454) &&\n            ((local_tid_48471 - squot32(",
            "local_tid_48471, wave_sizze_50802) *\n              wave_sizze_50802) & (2 * other_offset_48501 - 1)) == 0) {\n            // read array element\n            {\n                x_48503 = *(volatile __local\n                            float *) &mem_50616[(local_tid_48471 +\n                                                 other_offset_48501) * 4];\n            }\n            \n            float res_48504;\n            \n            if (thread_active_50804) {\n                res_48504 = x_48502 + x_48503;\n            }\n            x_48502 = res_48504;\n            *(volatile __local float *) &mem_50616[local_tid_48471 * 4] =\n                x_48502;\n        }\n        other_offset_48501 *= 2;\n    }\n    skip_waves_50808 = 1;\n    while (slt32(skip_waves_50808, squot32(group_sizze_48454 +\n                                           wave_sizze_50802 - 1,\n                                           wave_sizze_50802))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48501 = skip_waves_50808 * wave_sizze_50802;\n        if (slt32(local_tid_48471 + other_offset_48501, group_sizze_48454) &&\n            ((local_tid_48471 - squot32(local_tid_48471, wave_sizze_50802) *\n              wave_sizze_50802) == 0 && (squot32(local_tid_48471,\n                                                 wave_sizze_50802) & (2 *\n                                                                      skip_waves_50808 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48503 = *(__local float *) &mem_50616[(local_tid_48471 +\n                                                         other_offset_48501) *\n                                                        4];\n            }\n            \n            float res_48504;\n            \n            if (thread_active_50804) {\n                res_48504 = x_48502 + x_48503;\n            }\n            x_48502 = res_48504;\n            *(__local float *) &mem_506",
            "16[local_tid_48471 * 4] = x_48502;\n        }\n        skip_waves_50808 *= 2;\n    }\n    final_result_48499 = x_48502;\n    if (local_tid_48471 == 0) {\n        *(__global float *) &mem_50619[group_id_48472 * 4] = final_result_48499;\n    }\n}\n__kernel void chunked_reduce_kernel_48543(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_46956,\n                                          int32_t num_threads_48535,\n                                          int32_t per_thread_elements_48538,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *mem_50619)\n{\n    __local volatile char *restrict mem_50616 = mem_aligned_0;\n    int32_t wave_sizze_50818;\n    int32_t group_sizze_50819;\n    bool thread_active_50820;\n    int32_t global_tid_48543;\n    int32_t local_tid_48544;\n    int32_t group_id_48545;\n    \n    global_tid_48543 = get_global_id(0);\n    local_tid_48544 = get_local_id(0);\n    group_sizze_50819 = get_local_size(0);\n    wave_sizze_50818 = LOCKSTEP_WIDTH;\n    group_id_48545 = get_group_id(0);\n    thread_active_50820 = 1;\n    \n    int32_t chunk_sizze_48550 = smin32(per_thread_elements_48538,\n                                       squot32(sizze_46956 - global_tid_48543 +\n                                               num_threads_48535 - 1,\n                                               num_threads_48535));\n    float res_48555;\n    \n    if (thread_active_50820) {\n        float acc_48558 = 0.0F;\n        \n        for (int32_t i_48557 = 0; i_48557 < chunk_sizze_48550; i_48557++) {\n            int32_t j_t_s_50567 = num_threads_48535 * i_48557;\n            int32_t j_p_i_t_s_50568 = global_tid_48543 + j_t_s_50567;\n            float x_48562 = *(__global flo",
            "at *) &A_mem_50609[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48563 = *(__global float *) &B_mem_50611[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48564 = *(__global float *) &C_mem_50613[j_p_i_t_s_50568 *\n                                                             4];\n            float res_48566 = x_48562 + x_48563;\n            float res_48567 = x_48562 - x_48564;\n            float res_48568 = res_48566 * res_48567;\n            float res_48570 = acc_48558 + res_48568;\n            float acc_tmp_50821 = res_48570;\n            \n            acc_48558 = acc_tmp_50821;\n        }\n        res_48555 = acc_48558;\n    }\n    \n    float final_result_48573;\n    \n    for (int32_t comb_iter_50822 = 0; comb_iter_50822 <\n         squot32(group_sizze_48527 + group_sizze_48527 - 1, group_sizze_48527);\n         comb_iter_50822++) {\n        int32_t combine_id_48548;\n        int32_t flat_comb_id_50823 = comb_iter_50822 * group_sizze_48527 +\n                local_tid_48544;\n        \n        combine_id_48548 = flat_comb_id_50823;\n        if (slt32(combine_id_48548, group_sizze_48527) && 1) {\n            *(__local float *) &mem_50616[combine_id_48548 * 4] = res_48555;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50824;\n    int32_t my_index_48574;\n    int32_t other_offset_48575;\n    float x_48576;\n    float x_48577;\n    \n    my_index_48574 = local_tid_48544;\n    other_offset_48575 = 0;\n    if (slt32(local_tid_48544, group_sizze_48527)) {\n        x_48576 = *(__local float *) &mem_50616[(local_tid_48544 +\n                                                 other_offset_48575) * 4];\n    }\n    other_offset_48575 = 1;\n    while (slt32(other_offset_48575, wave_sizze_50818)) {\n        if (slt32(local_tid_48544 + other_offset_48575, group_sizze_48527) &&\n            ((local_tid_48544 - squot32(local_tid_48544, wave_sizze_50818) *\n              wave_siz",
            "ze_50818) & (2 * other_offset_48575 - 1)) == 0) {\n            // read array element\n            {\n                x_48577 = *(volatile __local\n                            float *) &mem_50616[(local_tid_48544 +\n                                                 other_offset_48575) * 4];\n            }\n            \n            float res_48578;\n            \n            if (thread_active_50820) {\n                res_48578 = x_48576 + x_48577;\n            }\n            x_48576 = res_48578;\n            *(volatile __local float *) &mem_50616[local_tid_48544 * 4] =\n                x_48576;\n        }\n        other_offset_48575 *= 2;\n    }\n    skip_waves_50824 = 1;\n    while (slt32(skip_waves_50824, squot32(group_sizze_48527 +\n                                           wave_sizze_50818 - 1,\n                                           wave_sizze_50818))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48575 = skip_waves_50824 * wave_sizze_50818;\n        if (slt32(local_tid_48544 + other_offset_48575, group_sizze_48527) &&\n            ((local_tid_48544 - squot32(local_tid_48544, wave_sizze_50818) *\n              wave_sizze_50818) == 0 && (squot32(local_tid_48544,\n                                                 wave_sizze_50818) & (2 *\n                                                                      skip_waves_50824 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48577 = *(__local float *) &mem_50616[(local_tid_48544 +\n                                                         other_offset_48575) *\n                                                        4];\n            }\n            \n            float res_48578;\n            \n            if (thread_active_50820) {\n                res_48578 = x_48576 + x_48577;\n            }\n            x_48576 = res_48578;\n            *(__local float *) &mem_50616[local_tid_48544 * 4] = x_48576;\n        }\n        skip_w",
            "aves_50824 *= 2;\n    }\n    final_result_48573 = x_48576;\n    if (local_tid_48544 == 0) {\n        *(__global float *) &mem_50619[group_id_48545 * 4] = final_result_48573;\n    }\n}\n__kernel void chunked_reduce_kernel_48619(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_46985,\n                                          int32_t num_threads_48611,\n                                          int32_t per_thread_elements_48614,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *D_mem_50615,\n                                          __global unsigned char *mem_50621)\n{\n    __local volatile char *restrict mem_50618 = mem_aligned_0;\n    int32_t wave_sizze_50834;\n    int32_t group_sizze_50835;\n    bool thread_active_50836;\n    int32_t global_tid_48619;\n    int32_t local_tid_48620;\n    int32_t group_id_48621;\n    \n    global_tid_48619 = get_global_id(0);\n    local_tid_48620 = get_local_id(0);\n    group_sizze_50835 = get_local_size(0);\n    wave_sizze_50834 = LOCKSTEP_WIDTH;\n    group_id_48621 = get_group_id(0);\n    thread_active_50836 = 1;\n    \n    int32_t chunk_sizze_48626 = smin32(per_thread_elements_48614,\n                                       squot32(sizze_46985 - global_tid_48619 +\n                                               num_threads_48611 - 1,\n                                               num_threads_48611));\n    float res_48632;\n    \n    if (thread_active_50836) {\n        float acc_48635 = 0.0F;\n        \n        for (int32_t i_48634 = 0; i_48634 < chunk_sizze_48626; i_48634++) {\n            int32_t j_t_s_50571 = num_threads_48611 * i_48634;\n            int32_t j_p_i_t_s_50572 = global_tid_48619 + j_t_s_50571;\n            float x_48",
            "640 = *(__global float *) &A_mem_50609[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48641 = *(__global float *) &B_mem_50611[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48642 = *(__global float *) &C_mem_50613[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48643 = *(__global float *) &D_mem_50615[j_p_i_t_s_50572 *\n                                                             4];\n            float res_48645 = x_48640 + x_48641;\n            float res_48646 = x_48642 - x_48643;\n            float res_48647 = res_48645 * res_48646;\n            float res_48649 = acc_48635 + res_48647;\n            float acc_tmp_50837 = res_48649;\n            \n            acc_48635 = acc_tmp_50837;\n        }\n        res_48632 = acc_48635;\n    }\n    \n    float final_result_48652;\n    \n    for (int32_t comb_iter_50838 = 0; comb_iter_50838 <\n         squot32(group_sizze_48603 + group_sizze_48603 - 1, group_sizze_48603);\n         comb_iter_50838++) {\n        int32_t combine_id_48624;\n        int32_t flat_comb_id_50839 = comb_iter_50838 * group_sizze_48603 +\n                local_tid_48620;\n        \n        combine_id_48624 = flat_comb_id_50839;\n        if (slt32(combine_id_48624, group_sizze_48603) && 1) {\n            *(__local float *) &mem_50618[combine_id_48624 * 4] = res_48632;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50840;\n    int32_t my_index_48653;\n    int32_t other_offset_48654;\n    float x_48655;\n    float x_48656;\n    \n    my_index_48653 = local_tid_48620;\n    other_offset_48654 = 0;\n    if (slt32(local_tid_48620, group_sizze_48603)) {\n        x_48655 = *(__local float *) &mem_50618[(local_tid_48620 +\n                                                 other_offset_48654) * 4];\n    }\n    other_offset_48654 = 1;\n    while (slt32(other_offset_48654, wave_sizze_50834)) {\n        if (s",
            "lt32(local_tid_48620 + other_offset_48654, group_sizze_48603) &&\n            ((local_tid_48620 - squot32(local_tid_48620, wave_sizze_50834) *\n              wave_sizze_50834) & (2 * other_offset_48654 - 1)) == 0) {\n            // read array element\n            {\n                x_48656 = *(volatile __local\n                            float *) &mem_50618[(local_tid_48620 +\n                                                 other_offset_48654) * 4];\n            }\n            \n            float res_48657;\n            \n            if (thread_active_50836) {\n                res_48657 = x_48655 + x_48656;\n            }\n            x_48655 = res_48657;\n            *(volatile __local float *) &mem_50618[local_tid_48620 * 4] =\n                x_48655;\n        }\n        other_offset_48654 *= 2;\n    }\n    skip_waves_50840 = 1;\n    while (slt32(skip_waves_50840, squot32(group_sizze_48603 +\n                                           wave_sizze_50834 - 1,\n                                           wave_sizze_50834))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48654 = skip_waves_50840 * wave_sizze_50834;\n        if (slt32(local_tid_48620 + other_offset_48654, group_sizze_48603) &&\n            ((local_tid_48620 - squot32(local_tid_48620, wave_sizze_50834) *\n              wave_sizze_50834) == 0 && (squot32(local_tid_48620,\n                                                 wave_sizze_50834) & (2 *\n                                                                      skip_waves_50840 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48656 = *(__local float *) &mem_50618[(local_tid_48620 +\n                                                         other_offset_48654) *\n                                                        4];\n            }\n            \n            float res_48657;\n            \n            if (thread_active_50836) {\n                res_48657 = ",
            "x_48655 + x_48656;\n            }\n            x_48655 = res_48657;\n            *(__local float *) &mem_50618[local_tid_48620 * 4] = x_48655;\n        }\n        skip_waves_50840 *= 2;\n    }\n    final_result_48652 = x_48655;\n    if (local_tid_48620 == 0) {\n        *(__global float *) &mem_50621[group_id_48621 * 4] = final_result_48652;\n    }\n}\n__kernel void chunked_reduce_kernel_48698(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_47023, float a_47027,\n                                          float b_47029, float c_47031,\n                                          float d_47033,\n                                          int32_t num_threads_48690,\n                                          int32_t per_thread_elements_48693,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *D_mem_50615,\n                                          __global unsigned char *mem_50621)\n{\n    __local volatile char *restrict mem_50618 = mem_aligned_0;\n    int32_t wave_sizze_50850;\n    int32_t group_sizze_50851;\n    bool thread_active_50852;\n    int32_t global_tid_48698;\n    int32_t local_tid_48699;\n    int32_t group_id_48700;\n    \n    global_tid_48698 = get_global_id(0);\n    local_tid_48699 = get_local_id(0);\n    group_sizze_50851 = get_local_size(0);\n    wave_sizze_50850 = LOCKSTEP_WIDTH;\n    group_id_48700 = get_group_id(0);\n    thread_active_50852 = 1;\n    \n    int32_t chunk_sizze_48705 = smin32(per_thread_elements_48693,\n                                       squot32(sizze_47023 - global_tid_48698 +\n                                               num_threads_48690 - 1,\n                                               num_threads_48690));\n    float res_48711;\n",
            "    \n    if (thread_active_50852) {\n        float acc_48714 = 0.0F;\n        \n        for (int32_t i_48713 = 0; i_48713 < chunk_sizze_48705; i_48713++) {\n            int32_t j_t_s_50571 = num_threads_48690 * i_48713;\n            int32_t j_p_i_t_s_50572 = global_tid_48698 + j_t_s_50571;\n            float x_48719 = *(__global float *) &A_mem_50609[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48720 = *(__global float *) &B_mem_50611[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48721 = *(__global float *) &C_mem_50613[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48722 = *(__global float *) &D_mem_50615[j_p_i_t_s_50572 *\n                                                             4];\n            float res_48724 = a_47027 * x_48719;\n            float res_48725 = b_47029 * x_48720;\n            float res_48726 = res_48724 + res_48725;\n            float res_48727 = c_47031 * x_48721;\n            float res_48728 = d_47033 * x_48722;\n            float res_48729 = res_48727 + res_48728;\n            float res_48730 = res_48726 * res_48729;\n            float res_48732 = acc_48714 + res_48730;\n            float acc_tmp_50853 = res_48732;\n            \n            acc_48714 = acc_tmp_50853;\n        }\n        res_48711 = acc_48714;\n    }\n    \n    float final_result_48735;\n    \n    for (int32_t comb_iter_50854 = 0; comb_iter_50854 <\n         squot32(group_sizze_48682 + group_sizze_48682 - 1, group_sizze_48682);\n         comb_iter_50854++) {\n        int32_t combine_id_48703;\n        int32_t flat_comb_id_50855 = comb_iter_50854 * group_sizze_48682 +\n                local_tid_48699;\n        \n        combine_id_48703 = flat_comb_id_50855;\n        if (slt32(combine_id_48703, group_sizze_48682) && 1) {\n            *(__local float *) &mem_50618[combine_id_48703 * 4] = res_48711;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_",
            "FENCE);\n    \n    int32_t skip_waves_50856;\n    int32_t my_index_48736;\n    int32_t other_offset_48737;\n    float x_48738;\n    float x_48739;\n    \n    my_index_48736 = local_tid_48699;\n    other_offset_48737 = 0;\n    if (slt32(local_tid_48699, group_sizze_48682)) {\n        x_48738 = *(__local float *) &mem_50618[(local_tid_48699 +\n                                                 other_offset_48737) * 4];\n    }\n    other_offset_48737 = 1;\n    while (slt32(other_offset_48737, wave_sizze_50850)) {\n        if (slt32(local_tid_48699 + other_offset_48737, group_sizze_48682) &&\n            ((local_tid_48699 - squot32(local_tid_48699, wave_sizze_50850) *\n              wave_sizze_50850) & (2 * other_offset_48737 - 1)) == 0) {\n            // read array element\n            {\n                x_48739 = *(volatile __local\n                            float *) &mem_50618[(local_tid_48699 +\n                                                 other_offset_48737) * 4];\n            }\n            \n            float res_48740;\n            \n            if (thread_active_50852) {\n                res_48740 = x_48738 + x_48739;\n            }\n            x_48738 = res_48740;\n            *(volatile __local float *) &mem_50618[local_tid_48699 * 4] =\n                x_48738;\n        }\n        other_offset_48737 *= 2;\n    }\n    skip_waves_50856 = 1;\n    while (slt32(skip_waves_50856, squot32(group_sizze_48682 +\n                                           wave_sizze_50850 - 1,\n                                           wave_sizze_50850))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48737 = skip_waves_50856 * wave_sizze_50850;\n        if (slt32(local_tid_48699 + other_offset_48737, group_sizze_48682) &&\n            ((local_tid_48699 - squot32(local_tid_48699, wave_sizze_50850) *\n              wave_sizze_50850) == 0 && (squot32(local_tid_48699,\n                                                 wave_sizze_50850) & (2 *\n                                                                      s",
            "kip_waves_50856 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48739 = *(__local float *) &mem_50618[(local_tid_48699 +\n                                                         other_offset_48737) *\n                                                        4];\n            }\n            \n            float res_48740;\n            \n            if (thread_active_50852) {\n                res_48740 = x_48738 + x_48739;\n            }\n            x_48738 = res_48740;\n            *(__local float *) &mem_50618[local_tid_48699 * 4] = x_48738;\n        }\n        skip_waves_50856 *= 2;\n    }\n    final_result_48735 = x_48738;\n    if (local_tid_48699 == 0) {\n        *(__global float *) &mem_50621[group_id_48700 * 4] = final_result_48735;\n    }\n}\n__kernel void chunked_reduce_kernel_48781(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_47069,\n                                          int32_t num_threads_48773,\n                                          int32_t per_thread_elements_48776,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *D_mem_50615,\n                                          __global unsigned char *mem_50621)\n{\n    __local volatile char *restrict mem_50618 = mem_aligned_0;\n    int32_t wave_sizze_50866;\n    int32_t group_sizze_50867;\n    bool thread_active_50868;\n    int32_t global_tid_48781;\n    int32_t local_tid_48782;\n    int32_t group_id_48783;\n    \n    global_tid_48781 = get_global_id(0);\n    local_tid_48782 = get_local_id(0);\n    group_sizze_50867 = get_local_size(0);\n    wave_sizze_50866 = LOCKSTEP_WIDTH;\n    group",
            "_id_48783 = get_group_id(0);\n    thread_active_50868 = 1;\n    \n    int32_t chunk_sizze_48788 = smin32(per_thread_elements_48776,\n                                       squot32(sizze_47069 - global_tid_48781 +\n                                               num_threads_48773 - 1,\n                                               num_threads_48773));\n    float res_48794;\n    \n    if (thread_active_50868) {\n        float acc_48797 = 0.0F;\n        \n        for (int32_t i_48796 = 0; i_48796 < chunk_sizze_48788; i_48796++) {\n            int32_t j_t_s_50571 = num_threads_48773 * i_48796;\n            int32_t j_p_i_t_s_50572 = global_tid_48781 + j_t_s_50571;\n            float x_48802 = *(__global float *) &A_mem_50609[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48803 = *(__global float *) &B_mem_50611[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48804 = *(__global float *) &C_mem_50613[j_p_i_t_s_50572 *\n                                                             4];\n            float x_48805 = *(__global float *) &D_mem_50615[j_p_i_t_s_50572 *\n                                                             4];\n            float res_48807 = x_48802 + x_48803;\n            float res_48808 = x_48804 - x_48805;\n            float res_48809 = res_48807 * res_48808;\n            float res_48811 = acc_48797 + res_48809;\n            float acc_tmp_50869 = res_48811;\n            \n            acc_48797 = acc_tmp_50869;\n        }\n        res_48794 = acc_48797;\n    }\n    \n    float final_result_48814;\n    \n    for (int32_t comb_iter_50870 = 0; comb_iter_50870 <\n         squot32(group_sizze_48765 + group_sizze_48765 - 1, group_sizze_48765);\n         comb_iter_50870++) {\n        int32_t combine_id_48786;\n        int32_t flat_comb_id_50871 = comb_iter_50870 * group_sizze_48765 +\n                local_tid_48782;\n        \n        combine_id_48786 = flat_comb_id_50871;\n        if (slt32(",
            "combine_id_48786, group_sizze_48765) && 1) {\n            *(__local float *) &mem_50618[combine_id_48786 * 4] = res_48794;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50872;\n    int32_t my_index_48815;\n    int32_t other_offset_48816;\n    float x_48817;\n    float x_48818;\n    \n    my_index_48815 = local_tid_48782;\n    other_offset_48816 = 0;\n    if (slt32(local_tid_48782, group_sizze_48765)) {\n        x_48817 = *(__local float *) &mem_50618[(local_tid_48782 +\n                                                 other_offset_48816) * 4];\n    }\n    other_offset_48816 = 1;\n    while (slt32(other_offset_48816, wave_sizze_50866)) {\n        if (slt32(local_tid_48782 + other_offset_48816, group_sizze_48765) &&\n            ((local_tid_48782 - squot32(local_tid_48782, wave_sizze_50866) *\n              wave_sizze_50866) & (2 * other_offset_48816 - 1)) == 0) {\n            // read array element\n            {\n                x_48818 = *(volatile __local\n                            float *) &mem_50618[(local_tid_48782 +\n                                                 other_offset_48816) * 4];\n            }\n            \n            float res_48819;\n            \n            if (thread_active_50868) {\n                res_48819 = x_48817 + x_48818;\n            }\n            x_48817 = res_48819;\n            *(volatile __local float *) &mem_50618[local_tid_48782 * 4] =\n                x_48817;\n        }\n        other_offset_48816 *= 2;\n    }\n    skip_waves_50872 = 1;\n    while (slt32(skip_waves_50872, squot32(group_sizze_48765 +\n                                           wave_sizze_50866 - 1,\n                                           wave_sizze_50866))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48816 = skip_waves_50872 * wave_sizze_50866;\n        if (slt32(local_tid_48782 + other_offset_48816, group_sizze_48765) &&\n            ((local_tid_48782 - squot32(local_tid_48782, wave_sizze_50866) *\n              wave_sizze_50866) == 0 && (squot3",
            "2(local_tid_48782,\n                                                 wave_sizze_50866) & (2 *\n                                                                      skip_waves_50872 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48818 = *(__local float *) &mem_50618[(local_tid_48782 +\n                                                         other_offset_48816) *\n                                                        4];\n            }\n            \n            float res_48819;\n            \n            if (thread_active_50868) {\n                res_48819 = x_48817 + x_48818;\n            }\n            x_48817 = res_48819;\n            *(__local float *) &mem_50618[local_tid_48782 * 4] = x_48817;\n        }\n        skip_waves_50872 *= 2;\n    }\n    final_result_48814 = x_48817;\n    if (local_tid_48782 == 0) {\n        *(__global float *) &mem_50621[group_id_48783 * 4] = final_result_48814;\n    }\n}\n__kernel void chunked_reduce_kernel_48858(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_47107,\n                                          int32_t num_threads_48850,\n                                          int32_t per_thread_elements_48853,\n                                          __global unsigned char *A_mem_50609,\n                                          __global unsigned char *B_mem_50611,\n                                          __global unsigned char *C_mem_50613,\n                                          __global unsigned char *mem_50619)\n{\n    __local volatile char *restrict mem_50616 = mem_aligned_0;\n    int32_t wave_sizze_50882;\n    int32_t group_sizze_50883;\n    bool thread_active_50884;\n    int32_t global_tid_48858;\n    int32_t local_tid_48859;\n    int32_t group_id_48860;\n    \n    global_tid_48858 = get_global_id(0);\n    local_tid_48859 = get_local_id(0);\n    gr",
            "oup_sizze_50883 = get_local_size(0);\n    wave_sizze_50882 = LOCKSTEP_WIDTH;\n    group_id_48860 = get_group_id(0);\n    thread_active_50884 = 1;\n    \n    int32_t chunk_sizze_48865 = smin32(per_thread_elements_48853,\n                                       squot32(sizze_47107 - global_tid_48858 +\n                                               num_threads_48850 - 1,\n                                               num_threads_48850));\n    float res_48870;\n    \n    if (thread_active_50884) {\n        float acc_48873 = 0.0F;\n        \n        for (int32_t i_48872 = 0; i_48872 < chunk_sizze_48865; i_48872++) {\n            int32_t j_t_s_50567 = num_threads_48850 * i_48872;\n            int32_t j_p_i_t_s_50568 = global_tid_48858 + j_t_s_50567;\n            float x_48877 = *(__global float *) &A_mem_50609[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48878 = *(__global float *) &B_mem_50611[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48879 = *(__global float *) &C_mem_50613[j_p_i_t_s_50568 *\n                                                             4];\n            float x_48881 = x_48877 * x_48878;\n            float res_48882 = x_48879 * x_48881;\n            float res_48884 = acc_48873 + res_48882;\n            float acc_tmp_50885 = res_48884;\n            \n            acc_48873 = acc_tmp_50885;\n        }\n        res_48870 = acc_48873;\n    }\n    \n    float final_result_48887;\n    \n    for (int32_t comb_iter_50886 = 0; comb_iter_50886 <\n         squot32(group_sizze_48842 + group_sizze_48842 - 1, group_sizze_48842);\n         comb_iter_50886++) {\n        int32_t combine_id_48863;\n        int32_t flat_comb_id_50887 = comb_iter_50886 * group_sizze_48842 +\n                local_tid_48859;\n        \n        combine_id_48863 = flat_comb_id_50887;\n        if (slt32(combine_id_48863, group_sizze_48842) && 1) {\n            *(__local float *) &mem_50616[combine_id_48863 * 4] = res",
            "_48870;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50888;\n    int32_t my_index_48888;\n    int32_t other_offset_48889;\n    float x_48890;\n    float x_48891;\n    \n    my_index_48888 = local_tid_48859;\n    other_offset_48889 = 0;\n    if (slt32(local_tid_48859, group_sizze_48842)) {\n        x_48890 = *(__local float *) &mem_50616[(local_tid_48859 +\n                                                 other_offset_48889) * 4];\n    }\n    other_offset_48889 = 1;\n    while (slt32(other_offset_48889, wave_sizze_50882)) {\n        if (slt32(local_tid_48859 + other_offset_48889, group_sizze_48842) &&\n            ((local_tid_48859 - squot32(local_tid_48859, wave_sizze_50882) *\n              wave_sizze_50882) & (2 * other_offset_48889 - 1)) == 0) {\n            // read array element\n            {\n                x_48891 = *(volatile __local\n                            float *) &mem_50616[(local_tid_48859 +\n                                                 other_offset_48889) * 4];\n            }\n            \n            float res_48892;\n            \n            if (thread_active_50884) {\n                res_48892 = x_48890 + x_48891;\n            }\n            x_48890 = res_48892;\n            *(volatile __local float *) &mem_50616[local_tid_48859 * 4] =\n                x_48890;\n        }\n        other_offset_48889 *= 2;\n    }\n    skip_waves_50888 = 1;\n    while (slt32(skip_waves_50888, squot32(group_sizze_48842 +\n                                           wave_sizze_50882 - 1,\n                                           wave_sizze_50882))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48889 = skip_waves_50888 * wave_sizze_50882;\n        if (slt32(local_tid_48859 + other_offset_48889, group_sizze_48842) &&\n            ((local_tid_48859 - squot32(local_tid_48859, wave_sizze_50882) *\n              wave_sizze_50882) == 0 && (squot32(local_tid_48859,\n                                                 wave_sizze_50882) & (2 *\n                     ",
            "                                                 skip_waves_50888 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_48891 = *(__local float *) &mem_50616[(local_tid_48859 +\n                                                         other_offset_48889) *\n                                                        4];\n            }\n            \n            float res_48892;\n            \n            if (thread_active_50884) {\n                res_48892 = x_48890 + x_48891;\n            }\n            x_48890 = res_48892;\n            *(__local float *) &mem_50616[local_tid_48859 * 4] = x_48890;\n        }\n        skip_waves_50888 *= 2;\n    }\n    final_result_48887 = x_48890;\n    if (local_tid_48859 == 0) {\n        *(__global float *) &mem_50619[group_id_48860 * 4] = final_result_48887;\n    }\n}\n__kernel void chunked_reduce_kernel_48938(__local volatile\n                                          int64_t *mem_aligned_0,\n                                          int32_t sizze_47135,\n                                          int32_t sizze_47136,\n                                          int32_t num_threads_48930,\n                                          int32_t per_thread_elements_48933,\n                                          int32_t per_chunk_50514, __global\n                                          unsigned char *mem_50614,\n                                          int32_t binop_x_50624, __global\n                                          unsigned char *mem_50627, __global\n                                          unsigned char *mem_50637, __global\n                                          unsigned char *mem_50645, __global\n                                          unsigned char *mem_50648, __global\n                                          unsigned char *mem_50657, __global\n                                          unsigned char *mem_50660, __global\n                   ",
            "                       unsigned char *mem_50664, __global\n                                          unsigned char *double_buffer_mem_50693)\n{\n    __local volatile char *restrict mem_50654 = mem_aligned_0;\n    int32_t wave_sizze_50906;\n    int32_t group_sizze_50907;\n    bool thread_active_50908;\n    int32_t global_tid_48938;\n    int32_t local_tid_48939;\n    int32_t group_id_48940;\n    \n    global_tid_48938 = get_global_id(0);\n    local_tid_48939 = get_local_id(0);\n    group_sizze_50907 = get_local_size(0);\n    wave_sizze_50906 = LOCKSTEP_WIDTH;\n    group_id_48940 = get_group_id(0);\n    thread_active_50908 = 1;\n    \n    int32_t chunk_sizze_48946;\n    int32_t starting_point_50909 = global_tid_48938 * per_thread_elements_48933;\n    int32_t remaining_elements_50910 = sizze_47135 - starting_point_50909;\n    \n    if (sle32(remaining_elements_50910, 0) || sle32(sizze_47135,\n                                                    starting_point_50909)) {\n        chunk_sizze_48946 = 0;\n    } else {\n        if (slt32(sizze_47135, (global_tid_48938 + 1) *\n                  per_thread_elements_48933)) {\n            chunk_sizze_48946 = sizze_47135 - global_tid_48938 *\n                per_thread_elements_48933;\n        } else {\n            chunk_sizze_48946 = per_thread_elements_48933;\n        }\n    }\n    \n    int32_t slice_offset_48947;\n    \n    if (thread_active_50908) {\n        slice_offset_48947 = per_thread_elements_48933 * global_tid_48938;\n        for (int32_t i_50911 = 0; i_50911 < sizze_47136; i_50911++) {\n            *(__global float *) &double_buffer_mem_50693[(group_id_48940 *\n                                                          (sizze_47136 *\n                                                           group_sizze_48922) +\n                                                          i_50911 *\n                                                          group_sizze_48922 +\n                                                          local_tid_48939) *\n                           ",
            "                              4] = *(__global\n                                                                float *) &mem_50614[i_50911 *\n                                                                                    4];\n        }\n        for (int32_t i_48954 = 0; i_48954 < chunk_sizze_48946; i_48954++) {\n            int32_t j_p_i_t_s_50568 = slice_offset_48947 + i_48954;\n            int32_t new_index_50569 = squot32(j_p_i_t_s_50568, per_chunk_50514);\n            int32_t binop_y_50571 = per_chunk_50514 * new_index_50569;\n            int32_t new_index_50572 = j_p_i_t_s_50568 - binop_y_50571;\n            float x_48959 = *(__global float *) &mem_50637[(new_index_50572 *\n                                                            num_threads_48930 +\n                                                            new_index_50569) *\n                                                           4];\n            int32_t binop_x_50573 = sizze_47136 * j_p_i_t_s_50568;\n            \n            for (int32_t i_48964 = 0; i_48964 < sizze_47136; i_48964++) {\n                int32_t binop_x_50574 = i_48964 + binop_x_50573;\n                int32_t new_index_50576 = squot32(binop_x_50574, binop_x_50624);\n                int32_t binop_y_50584 = new_index_50576 * binop_x_50624;\n                int32_t binop_x_50585 = binop_x_50574 - binop_y_50584;\n                int32_t new_index_50586 = squot32(binop_x_50585, sizze_47136);\n                int32_t binop_y_50606 = sizze_47136 * new_index_50586;\n                int32_t new_index_50607 = binop_x_50585 - binop_y_50606;\n                float x_48965 = *(__global\n                                  float *) &mem_50627[(new_index_50586 *\n                                                       (sizze_47136 *\n                                                        num_threads_48930) +\n                                                       new_index_50607 *\n                                                       num_threads_48930 +\n                 ",
            "                                      new_index_50576) * 4];\n                float res_48966 = x_48959 * x_48965;\n                \n                *(__global float *) &mem_50645[(group_id_48940 * (sizze_47136 *\n                                                                  group_sizze_48922) +\n                                                i_48964 * group_sizze_48922 +\n                                                local_tid_48939) * 4] =\n                    res_48966;\n            }\n            for (int32_t i_48972 = 0; i_48972 < sizze_47136; i_48972++) {\n                float x_48973 = *(__global\n                                  float *) &double_buffer_mem_50693[(group_id_48940 *\n                                                                     (sizze_47136 *\n                                                                      group_sizze_48922) +\n                                                                     i_48972 *\n                                                                     group_sizze_48922 +\n                                                                     local_tid_48939) *\n                                                                    4];\n                float x_48974 = *(__global float *) &mem_50645[(group_id_48940 *\n                                                                (sizze_47136 *\n                                                                 group_sizze_48922) +\n                                                                i_48972 *\n                                                                group_sizze_48922 +\n                                                                local_tid_48939) *\n                                                               4];\n                float res_48975 = x_48973 + x_48974;\n                \n                *(__global float *) &mem_50648[(group_id_48940 * (sizze_47136 *\n                                                                  group_sizze_48922) +\n          ",
            "                                      i_48972 * group_sizze_48922 +\n                                                local_tid_48939) * 4] =\n                    res_48975;\n            }\n            for (int32_t i_50915 = 0; i_50915 < sizze_47136; i_50915++) {\n                *(__global float *) &double_buffer_mem_50693[(group_id_48940 *\n                                                              (sizze_47136 *\n                                                               group_sizze_48922) +\n                                                              i_50915 *\n                                                              group_sizze_48922 +\n                                                              local_tid_48939) *\n                                                             4] = *(__global\n                                                                    float *) &mem_50648[(group_id_48940 *\n                                                                                         (sizze_47136 *\n                                                                                          group_sizze_48922) +\n                                                                                         i_50915 *\n                                                                                         group_sizze_48922 +\n                                                                                         local_tid_48939) *\n                                                                                        4];\n            }\n        }\n    }\n    for (int32_t comb_iter_50916 = 0; comb_iter_50916 <\n         squot32(group_sizze_48922 + group_sizze_48922 - 1, group_sizze_48922);\n         comb_iter_50916++) {\n        int32_t combine_id_48944;\n        int32_t flat_comb_id_50917 = comb_iter_50916 * group_sizze_48922 +\n                local_tid_48939;\n        \n        combine_id_48944 = flat_comb_id_50917;\n        if (slt32(combine_id_48944, group_sizze_48922) && 1) {\n   ",
            "         for (int32_t i_50918 = 0; i_50918 < sizze_47136; i_50918++) {\n                *(__local float *) &mem_50654[(combine_id_48944 * sizze_47136 +\n                                               i_50918) * 4] = *(__global\n                                                                 float *) &double_buffer_mem_50693[(group_id_48940 *\n                                                                                                    (sizze_47136 *\n                                                                                                     group_sizze_48922) +\n                                                                                                    i_50918 *\n                                                                                                    group_sizze_48922 +\n                                                                                                    local_tid_48939) *\n                                                                                                   4];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50919;\n    int32_t my_index_48980;\n    int32_t other_offset_48981;\n    \n    my_index_48980 = local_tid_48939;\n    other_offset_48981 = 0;\n    if (slt32(local_tid_48939, group_sizze_48922)) { }\n    other_offset_48981 = 1;\n    while (slt32(other_offset_48981, wave_sizze_50906)) {\n        if (slt32(local_tid_48939 + other_offset_48981, group_sizze_48922) &&\n            ((local_tid_48939 - squot32(local_tid_48939, wave_sizze_50906) *\n              wave_sizze_50906) & (2 * other_offset_48981 - 1)) == 0) {\n            // read array element\n            { }\n            if (thread_active_50908) {\n                for (int32_t i_48987 = 0; i_48987 < sizze_47136; i_48987++) {\n                    float x_48988 = *(volatile __local\n                                      float *) &mem_50654[(my_index_48980 *\n                                                           sizze_47136 +\n",
            "                                                           i_48987) * 4];\n                    float x_48989 = *(volatile __local\n                                      float *) &mem_50654[((my_index_48980 +\n                                                            other_offset_48981) *\n                                                           sizze_47136 +\n                                                           i_48987) * 4];\n                    float res_48990 = x_48988 + x_48989;\n                    \n                    *(volatile __global float *) &mem_50657[(group_id_48940 *\n                                                             (sizze_47136 *\n                                                              group_sizze_48922) +\n                                                             i_48987 *\n                                                             group_sizze_48922 +\n                                                             local_tid_48939) *\n                                                            4] = res_48990;\n                }\n            }\n            for (int32_t i_50921 = 0; i_50921 < sizze_47136; i_50921++) {\n                *(volatile __local float *) &mem_50654[(my_index_48980 *\n                                                        sizze_47136 + i_50921) *\n                                                       4] = *(volatile __global\n                                                              float *) &mem_50657[(group_id_48940 *\n                                                                                   (sizze_47136 *\n                                                                                    group_sizze_48922) +\n                                                                                   i_50921 *\n                                                                                   group_sizze_48922 +\n                                                                                   local_tid_48939) *\n ",
            "                                                                                 4];\n            }\n        }\n        other_offset_48981 *= 2;\n    }\n    skip_waves_50919 = 1;\n    while (slt32(skip_waves_50919, squot32(group_sizze_48922 +\n                                           wave_sizze_50906 - 1,\n                                           wave_sizze_50906))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48981 = skip_waves_50919 * wave_sizze_50906;\n        if (slt32(local_tid_48939 + other_offset_48981, group_sizze_48922) &&\n            ((local_tid_48939 - squot32(local_tid_48939, wave_sizze_50906) *\n              wave_sizze_50906) == 0 && (squot32(local_tid_48939,\n                                                 wave_sizze_50906) & (2 *\n                                                                      skip_waves_50919 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            { }\n            if (thread_active_50908) {\n                for (int32_t i_48987 = 0; i_48987 < sizze_47136; i_48987++) {\n                    float x_48988 = *(__local\n                                      float *) &mem_50654[(my_index_48980 *\n                                                           sizze_47136 +\n                                                           i_48987) * 4];\n                    float x_48989 = *(__local\n                                      float *) &mem_50654[((my_index_48980 +\n                                                            other_offset_48981) *\n                                                           sizze_47136 +\n                                                           i_48987) * 4];\n                    float res_48990 = x_48988 + x_48989;\n                    \n                    *(__global float *) &mem_50657[(group_id_48940 *\n                                                    (sizze_47136 *\n                                               ",
            "      group_sizze_48922) +\n                                                    i_48987 *\n                                                    group_sizze_48922 +\n                                                    local_tid_48939) * 4] =\n                        res_48990;\n                }\n            }\n            for (int32_t i_50923 = 0; i_50923 < sizze_47136; i_50923++) {\n                *(__local float *) &mem_50654[(my_index_48980 * sizze_47136 +\n                                               i_50923) * 4] = *(__global\n                                                                 float *) &mem_50657[(group_id_48940 *\n                                                                                      (sizze_47136 *\n                                                                                       group_sizze_48922) +\n                                                                                      i_50923 *\n                                                                                      group_sizze_48922 +\n                                                                                      local_tid_48939) *\n                                                                                     4];\n            }\n        }\n        skip_waves_50919 *= 2;\n    }\n    for (int32_t i_50924 = 0; i_50924 < sizze_47136; i_50924++) {\n        *(__global float *) &mem_50660[(group_id_48940 * (sizze_47136 *\n                                                          group_sizze_48922) +\n                                        i_50924 * group_sizze_48922 +\n                                        local_tid_48939) * 4] = *(__local\n                                                                  float *) &mem_50654[(my_index_48980 *\n                                                                                       sizze_47136 +\n                                                                                       i_50924) *\n                                     ",
            "                                                 4];\n    }\n    if (local_tid_48939 == 0) {\n        for (int32_t i_50926 = 0; i_50926 < sizze_47136; i_50926++) {\n            *(__global float *) &mem_50664[(group_id_48940 * sizze_47136 +\n                                            i_50926) * 4] = *(__global\n                                                              float *) &mem_50660[(group_id_48940 *\n                                                                                   (sizze_47136 *\n                                                                                    group_sizze_48922) +\n                                                                                   i_50926 *\n                                                                                   group_sizze_48922 +\n                                                                                   local_tid_48939) *\n                                                                                  4];\n        }\n    }\n}\n__kernel void fut_kernel_map_transpose_f32(__global float *odata,\n                                           uint odata_offset, __global\n                                           float *idata, uint idata_offset,\n                                           uint width, uint height,\n                                           uint input_size, uint output_size,\n                                           __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_global_id(0);\n    y_index = get_global_id(1);\n    \n    uint index_in = y_index * width + x_",
            "index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowheight_f32(__global float *odata,\n                                                     uint odata_offset, __global\n                                                     float *idata,\n                                                     uint idata_offset,\n                                                     uint width, uint height,\n                                                     uint input_size,\n                                                     uint output_size,\n                                                     uint mulx, __local\n                                                     float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(0) +\n        get_local_id(1) % mulx * FUT_BLOCK_DIM;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(1) / mulx;\n    \n    uint index_in = y_in",
            "dex * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0) / mulx;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM * mulx + get_local_id(1) +\n        get_local_id(0) % mulx * FUT_BLOCK_DIM;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_lowwidth_f32(__global float *odata,\n                                                    uint odata_offset, __global\n                                                    float *idata,\n                                                    uint idata_offset,\n                                                    uint width, uint height,\n                                                    uint input_size,\n                                                    uint output_size, uint muly,\n                                                    __local float *block)\n{\n    uint x_index;\n    uint y_index;\n    uint our_array_offset;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays for the third dimension.\n    our_array_offset = get_global_id(2) * width * height;\n    odata += our_array_offset;\n    idata += our_array_offset;\n    // read the matrix tile into shared memory\n    x_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(0) / muly;\n    y_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(1) +\n        get_local_id(0) % muly * FUT_BLOCK_DIM;\n    \n    ui",
            "nt index_in = y_index * width + x_index;\n    \n    if ((x_index < width && y_index < height) && index_in < input_size)\n        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =\n            idata[index_in];\n    barrier(CLK_LOCAL_MEM_FENCE);\n    // Scatter the transposed matrix tile to global memory.\n    x_index = get_group_id(1) * FUT_BLOCK_DIM * muly + get_local_id(0) +\n        get_local_id(1) % muly * FUT_BLOCK_DIM;\n    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1) / muly;\n    \n    uint index_out = y_index * height + x_index;\n    \n    if ((x_index < height && y_index < width) && index_out < output_size)\n        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +\n                                 get_local_id(1)];\n}\n__kernel void fut_kernel_map_transpose_small_f32(__global float *odata,\n                                                 uint odata_offset, __global\n                                                 float *idata,\n                                                 uint idata_offset,\n                                                 uint num_arrays, uint width,\n                                                 uint height, uint input_size,\n                                                 uint output_size)\n{\n    uint our_array_offset = get_global_id(0) / (height * width) * (height *\n                                                                   width);\n    uint x_index = get_global_id(0) % (height * width) / height;\n    uint y_index = get_global_id(0) % height;\n    \n    // Adjust the input and output arrays with the basic offset.\n    odata += odata_offset / sizeof(float);\n    idata += idata_offset / sizeof(float);\n    // Adjust the input and output arrays.\n    odata += our_array_offset;\n    idata += our_array_offset;\n    \n    uint index_in = y_index * width + x_index;\n    uint index_out = x_index * height + y_index;\n    \n    if (get_global_id(0) < input_size)\n        odata[index_out] = idata[index_in];\n}\n__kernel void ",
            "kernel_replicate_47146(int32_t sizze_47136, __global\n                                     unsigned char *mem_50614)\n{\n    const uint replicate_gtid_47146 = get_global_id(0);\n    \n    if (replicate_gtid_47146 >= sizze_47136)\n        return;\n    *(__global float *) &mem_50614[replicate_gtid_47146 * 4] = 0.0F;\n}\n__kernel void map_kernel_48370(int32_t sizze_46904, float c_46905, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *mem_50612)\n{\n    int32_t wave_sizze_50782;\n    int32_t group_sizze_50783;\n    bool thread_active_50784;\n    int32_t gtid_48363;\n    int32_t global_tid_48370;\n    int32_t local_tid_48371;\n    int32_t group_id_48372;\n    \n    global_tid_48370 = get_global_id(0);\n    local_tid_48371 = get_local_id(0);\n    group_sizze_50783 = get_local_size(0);\n    wave_sizze_50782 = LOCKSTEP_WIDTH;\n    group_id_48372 = get_group_id(0);\n    gtid_48363 = global_tid_48370;\n    thread_active_50784 = slt32(gtid_48363, sizze_46904);\n    \n    float x_48373;\n    float res_48374;\n    \n    if (thread_active_50784) {\n        x_48373 = *(__global float *) &A_mem_50609[gtid_48363 * 4];\n        res_48374 = c_46905 * x_48373;\n    }\n    if (thread_active_50784) {\n        *(__global float *) &mem_50612[gtid_48363 * 4] = res_48374;\n    }\n}\n__kernel void map_kernel_49020(int32_t sizze_47160, int32_t sizze_47161,\n                               __global unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50615, __global\n                               unsigned char *mem_50618)\n{\n    int32_t wave_sizze_50946;\n    int32_t group_sizze_50947;\n    bool thread_active_50948;\n    int32_t gtid_49013;\n    int32_t global_tid_49020;\n    int32_t local_tid_49021;\n    int32_t group_id_49022;\n    \n    global_tid_49020 = get_global_id(0);\n    local_tid_49021 = get_local_id(0);\n    group_sizze_50947 = get_local_size(0);\n    wave_sizze_50946 = LOCKSTEP_WIDTH;\n    group_id_49022 = get_group_id(0);\n   ",
            " gtid_49013 = global_tid_49020;\n    thread_active_50948 = slt32(gtid_49013, sizze_47160);\n    \n    float res_49024;\n    \n    if (thread_active_50948) {\n        float x_49027 = 0.0F;\n        \n        for (int32_t chunk_offset_49026 = 0; chunk_offset_49026 < sizze_47161;\n             chunk_offset_49026++) {\n            float x_49036 = *(__global float *) &mem_50615[(chunk_offset_49026 *\n                                                            sizze_47160 +\n                                                            gtid_49013) * 4];\n            float x_49037 = *(__global\n                              float *) &B_mem_50611[chunk_offset_49026 * 4];\n            float res_49039 = x_49036 * x_49037;\n            float res_49041 = x_49027 + res_49039;\n            float x_tmp_50949 = res_49041;\n            \n            x_49027 = x_tmp_50949;\n        }\n        res_49024 = x_49027;\n    }\n    if (thread_active_50948) {\n        *(__global float *) &mem_50618[gtid_49013 * 4] = res_49024;\n    }\n}\n__kernel void map_kernel_49060(int32_t sizze_47181, int32_t sizze_47182,\n                               __global unsigned char *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50617, __global\n                               unsigned char *mem_50620)\n{\n    int32_t wave_sizze_50953;\n    int32_t group_sizze_50954;\n    bool thread_active_50955;\n    int32_t gtid_49053;\n    int32_t global_tid_49060;\n    int32_t local_tid_49061;\n    int32_t group_id_49062;\n    \n    global_tid_49060 = get_global_id(0);\n    local_tid_49061 = get_local_id(0);\n    group_sizze_50954 = get_local_size(0);\n    wave_sizze_50953 = LOCKSTEP_WIDTH;\n    group_id_49062 = get_group_id(0);\n    gtid_49053 = global_tid_49060;\n    thread_active_50955 = slt32(gtid_49053, sizze_47181);\n    \n    float x_49064;\n    float res_49065;\n    \n    if (thread_active_50955) {\n        x_49064 = *(__global float *) &C_mem_50613[gtid_49053 * 4];\n        \n",
            "        float x_49068 = 0.0F;\n        \n        for (int32_t chunk_offset_49067 = 0; chunk_offset_49067 < sizze_47182;\n             chunk_offset_49067++) {\n            float x_49077 = *(__global float *) &mem_50617[(chunk_offset_49067 *\n                                                            sizze_47181 +\n                                                            gtid_49053) * 4];\n            float x_49078 = *(__global\n                              float *) &B_mem_50611[chunk_offset_49067 * 4];\n            float x_49080 = x_49077 * x_49078;\n            float res_49081 = x_49064 * x_49080;\n            float res_49083 = x_49068 + res_49081;\n            float x_tmp_50956 = res_49083;\n            \n            x_49068 = x_tmp_50956;\n        }\n        res_49065 = x_49068;\n    }\n    if (thread_active_50955) {\n        *(__global float *) &mem_50620[gtid_49053 * 4] = res_49065;\n    }\n}\n__kernel void map_kernel_49104(int32_t sizze_47213, int32_t sizze_47214,\n                               int32_t sizze_47215, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50617, __global\n                               unsigned char *mem_50621, __global\n                               unsigned char *mem_50624)\n{\n    int32_t wave_sizze_50960;\n    int32_t group_sizze_50961;\n    bool thread_active_50962;\n    int32_t gtid_49097;\n    int32_t global_tid_49104;\n    int32_t local_tid_49105;\n    int32_t group_id_49106;\n    \n    global_tid_49104 = get_global_id(0);\n    local_tid_49105 = get_local_id(0);\n    group_sizze_50961 = get_local_size(0);\n    wave_sizze_50960 = LOCKSTEP_WIDTH;\n    group_id_49106 = get_group_id(0);\n    gtid_49097 = global_tid_49104;\n    thread_active_50962 = slt32(gtid_49097, sizze_47213);\n    \n    float res_49109;\n    \n    if (thread_active_50962) {\n        float x_49112 = 0.0F;\n        \n        for (int32_t chunk_offset_49111 = 0; chunk_offset_49111 < sizze_47214;\n             chunk_offset_49111++",
            ") {\n            float x_49123 = *(__global float *) &mem_50617[(chunk_offset_49111 *\n                                                            sizze_47213 +\n                                                            gtid_49097) * 4];\n            float x_49124 = *(__global float *) &mem_50621[(chunk_offset_49111 *\n                                                            sizze_47215 +\n                                                            gtid_49097) * 4];\n            float x_49125 = *(__global\n                              float *) &C_mem_50613[chunk_offset_49111 * 4];\n            float res_49127 = x_49123 + x_49124;\n            float res_49128 = x_49125 * res_49127;\n            float res_49130 = x_49112 + res_49128;\n            float x_tmp_50963 = res_49130;\n            \n            x_49112 = x_tmp_50963;\n        }\n        res_49109 = x_49112;\n    }\n    if (thread_active_50962) {\n        *(__global float *) &mem_50624[gtid_49097 * 4] = res_49109;\n    }\n}\n__kernel void map_kernel_49138(int32_t sizze_47253, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50618)\n{\n    int32_t wave_sizze_50967;\n    int32_t group_sizze_50968;\n    bool thread_active_50969;\n    int32_t gtid_49131;\n    int32_t global_tid_49138;\n    int32_t local_tid_49139;\n    int32_t group_id_49140;\n    \n    global_tid_49138 = get_global_id(0);\n    local_tid_49139 = get_local_id(0);\n    group_sizze_50968 = get_local_size(0);\n    wave_sizze_50967 = LOCKSTEP_WIDTH;\n    group_id_49140 = get_group_id(0);\n    gtid_49131 = global_tid_49138;\n    thread_active_50969 = slt32(gtid_49131, sizze_47253);\n    \n    float x_49141;\n    float x_49142;\n    float res_49143;\n    \n    if (thread_active_50969) {\n        x_49141 = *(__global float *) &C_mem_50613[gtid_49131 * 4];\n        x_49142 = *(__global float *) &D_mem_50615[gtid_49131 * 4];\n        res_49143 = x_49141 ",
            "+ x_49142;\n    }\n    if (thread_active_50969) {\n        *(__global float *) &mem_50618[gtid_49131 * 4] = res_49143;\n    }\n}\n__kernel void map_kernel_49164(int32_t sizze_47252, int32_t sizze_47253,\n                               int32_t sizze_47254, __global\n                               unsigned char *mem_50618, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50626, __global\n                               unsigned char *mem_50629)\n{\n    int32_t wave_sizze_50970;\n    int32_t group_sizze_50971;\n    bool thread_active_50972;\n    int32_t gtid_49157;\n    int32_t global_tid_49164;\n    int32_t local_tid_49165;\n    int32_t group_id_49166;\n    \n    global_tid_49164 = get_global_id(0);\n    local_tid_49165 = get_local_id(0);\n    group_sizze_50971 = get_local_size(0);\n    wave_sizze_50970 = LOCKSTEP_WIDTH;\n    group_id_49166 = get_group_id(0);\n    gtid_49157 = global_tid_49164;\n    thread_active_50972 = slt32(gtid_49157, sizze_47252);\n    \n    float res_49169;\n    \n    if (thread_active_50972) {\n        float x_49172 = 0.0F;\n        \n        for (int32_t chunk_offset_49171 = 0; chunk_offset_49171 < sizze_47253;\n             chunk_offset_49171++) {\n            float x_49183 = *(__global float *) &mem_50622[(chunk_offset_49171 *\n                                                            sizze_47252 +\n                                                            gtid_49157) * 4];\n            float x_49184 = *(__global float *) &mem_50626[(chunk_offset_49171 *\n                                                            sizze_47254 +\n                                                            gtid_49157) * 4];\n            float x_49185 = *(__global float *) &mem_50618[chunk_offset_49171 *\n                                                           4];\n            float res_49187 = x_49183 + x_49184;\n            float res_49188 = x_49185 * res_49187;\n            float res_49190 = x_49172 + res_49188;\n        ",
            "    float x_tmp_50973 = res_49190;\n            \n            x_49172 = x_tmp_50973;\n        }\n        res_49169 = x_49172;\n    }\n    if (thread_active_50972) {\n        *(__global float *) &mem_50629[gtid_49157 * 4] = res_49169;\n    }\n}\n__kernel void map_kernel_49198(int32_t sizze_47304, float c_47313,\n                               float d_47315, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50618)\n{\n    int32_t wave_sizze_50977;\n    int32_t group_sizze_50978;\n    bool thread_active_50979;\n    int32_t gtid_49191;\n    int32_t global_tid_49198;\n    int32_t local_tid_49199;\n    int32_t group_id_49200;\n    \n    global_tid_49198 = get_global_id(0);\n    local_tid_49199 = get_local_id(0);\n    group_sizze_50978 = get_local_size(0);\n    wave_sizze_50977 = LOCKSTEP_WIDTH;\n    group_id_49200 = get_group_id(0);\n    gtid_49191 = global_tid_49198;\n    thread_active_50979 = slt32(gtid_49191, sizze_47304);\n    \n    float x_49201;\n    float x_49202;\n    float res_49203;\n    float res_49204;\n    float res_49205;\n    \n    if (thread_active_50979) {\n        x_49201 = *(__global float *) &C_mem_50613[gtid_49191 * 4];\n        x_49202 = *(__global float *) &D_mem_50615[gtid_49191 * 4];\n        res_49203 = c_47313 * x_49201;\n        res_49204 = d_47315 * x_49202;\n        res_49205 = res_49203 + res_49204;\n    }\n    if (thread_active_50979) {\n        *(__global float *) &mem_50618[gtid_49191 * 4] = res_49205;\n    }\n}\n__kernel void map_kernel_49226(int32_t sizze_47303, int32_t sizze_47304,\n                               int32_t sizze_47305, float a_47309,\n                               float b_47311, __global unsigned char *mem_50618,\n                               __global unsigned char *mem_50622, __global\n                               unsigned char *mem_50626, __global\n                               unsigned char *mem_50629)\n{\n    int32_t wav",
            "e_sizze_50980;\n    int32_t group_sizze_50981;\n    bool thread_active_50982;\n    int32_t gtid_49219;\n    int32_t global_tid_49226;\n    int32_t local_tid_49227;\n    int32_t group_id_49228;\n    \n    global_tid_49226 = get_global_id(0);\n    local_tid_49227 = get_local_id(0);\n    group_sizze_50981 = get_local_size(0);\n    wave_sizze_50980 = LOCKSTEP_WIDTH;\n    group_id_49228 = get_group_id(0);\n    gtid_49219 = global_tid_49226;\n    thread_active_50982 = slt32(gtid_49219, sizze_47303);\n    \n    float res_49231;\n    \n    if (thread_active_50982) {\n        float x_49234 = 0.0F;\n        \n        for (int32_t chunk_offset_49233 = 0; chunk_offset_49233 < sizze_47304;\n             chunk_offset_49233++) {\n            float x_49245 = *(__global float *) &mem_50622[(chunk_offset_49233 *\n                                                            sizze_47303 +\n                                                            gtid_49219) * 4];\n            float x_49246 = *(__global float *) &mem_50626[(chunk_offset_49233 *\n                                                            sizze_47305 +\n                                                            gtid_49219) * 4];\n            float x_49247 = *(__global float *) &mem_50618[chunk_offset_49233 *\n                                                           4];\n            float res_49249 = a_47309 * x_49245;\n            float res_49250 = b_47311 * x_49246;\n            float res_49251 = res_49249 + res_49250;\n            float res_49252 = x_49247 * res_49251;\n            float res_49254 = x_49234 + res_49252;\n            float x_tmp_50983 = res_49254;\n            \n            x_49234 = x_tmp_50983;\n        }\n        res_49231 = x_49234;\n    }\n    if (thread_active_50982) {\n        *(__global float *) &mem_50629[gtid_49219 * 4] = res_49231;\n    }\n}\n__kernel void map_kernel_49273(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_4736",
            "2, int32_t sizze_47363,\n                               __global unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50624)\n{\n    __local volatile char *restrict mem_50618 = mem_aligned_0;\n    __local volatile char *restrict mem_50621 = mem_aligned_1;\n    int32_t wave_sizze_50987;\n    int32_t group_sizze_50988;\n    bool thread_active_50989;\n    int32_t gtid_49266;\n    int32_t global_tid_49273;\n    int32_t local_tid_49274;\n    int32_t group_id_49275;\n    \n    global_tid_49273 = get_global_id(0);\n    local_tid_49274 = get_local_id(0);\n    group_sizze_50988 = get_local_size(0);\n    wave_sizze_50987 = LOCKSTEP_WIDTH;\n    group_id_49275 = get_group_id(0);\n    gtid_49266 = global_tid_49273;\n    thread_active_50989 = slt32(gtid_49266, sizze_47362);\n    \n    float x_49276;\n    float x_49277;\n    \n    if (thread_active_50989) {\n        x_49276 = *(__global float *) &A_mem_50609[gtid_49266 * 4];\n        x_49277 = *(__global float *) &C_mem_50613[gtid_49266 * 4];\n    }\n    \n    float res_49278;\n    float x_49281 = 0.0F;\n    int32_t chunk_sizze_49279;\n    int32_t chunk_offset_49280 = 0;\n    \n    while (slt32(chunk_offset_49280, sizze_47363)) {\n        if (slt32(sizze_47363 - chunk_offset_49280, group_sizze_49268)) {\n            chunk_sizze_49279 = sizze_47363 - chunk_offset_49280;\n        } else {\n            chunk_sizze_49279 = group_sizze_49268;\n        }\n        \n        float res_49284;\n        float sync_50533;\n        \n        for (int32_t comb_iter_50990 = 0; comb_iter_50990 <\n             squot32(group_sizze_49268 + group_sizze_49268 - 1,\n                     group_sizze_49268); comb_iter_50990++) {\n            int32_t cid_50529;\n            int32_t flat_comb_id_50991 = comb_iter_50990 * group_sizze_49268 +\n                    local_tid",
            "_49274;\n            \n            cid_50529 = flat_comb_id_50991;\n            if (slt32(cid_50529, chunk_sizze_49279) && 1) {\n                float x_chunk_outer_elem_50528 = *(__global\n                                                   float *) &B_mem_50611[(chunk_offset_49280 +\n                                                                          local_tid_49274) *\n                                                                         4];\n                \n                *(__local float *) &mem_50618[cid_50529 * 4] =\n                    x_chunk_outer_elem_50528;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t comb_iter_50992 = 0; comb_iter_50992 <\n             squot32(group_sizze_49268 + group_sizze_49268 - 1,\n                     group_sizze_49268); comb_iter_50992++) {\n            int32_t cid_50532;\n            int32_t flat_comb_id_50993 = comb_iter_50992 * group_sizze_49268 +\n                    local_tid_49274;\n            \n            cid_50532 = flat_comb_id_50993;\n            if (slt32(cid_50532, chunk_sizze_49279) && 1) {\n                float x_chunk_outer_elem_50531 = *(__global\n                                                   float *) &D_mem_50615[(chunk_offset_49280 +\n                                                                          local_tid_49274) *\n                                                                         4];\n                \n                *(__local float *) &mem_50621[cid_50532 * 4] =\n                    x_chunk_outer_elem_50531;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        float acc_49287 = x_49281;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_49285 = 1;\n        \n        if (thread_active_50989) {\n            if (chunk_sizze_49279 == group_sizze_49268) {\n                for (int32_t i_49286 = 0; i_49286 < group_sizze_49268;\n                     i_49286++) {\n                    float x_49290 = *(__local float *) &mem_50618[i_49286 * 4];\n  ",
            "                  float x_49291 = *(__local float *) &mem_50621[i_49286 * 4];\n                    float x_49293 = x_49276 * x_49290;\n                    float x_49294 = x_49277 * x_49293;\n                    float res_49295 = x_49291 * x_49294;\n                    float res_49297 = acc_49287 + res_49295;\n                    float acc_tmp_50994 = res_49297;\n                    \n                    acc_49287 = acc_tmp_50994;\n                }\n            } else {\n                for (int32_t i_49286 = 0; i_49286 < chunk_sizze_49279;\n                     i_49286++) {\n                    float x_49290 = *(__local float *) &mem_50618[i_49286 * 4];\n                    float x_49291 = *(__local float *) &mem_50621[i_49286 * 4];\n                    float x_49293 = x_49276 * x_49290;\n                    float x_49294 = x_49277 * x_49293;\n                    float res_49295 = x_49291 * x_49294;\n                    float res_49297 = acc_49287 + res_49295;\n                    float acc_tmp_50994 = res_49297;\n                    \n                    acc_49287 = acc_tmp_50994;\n                }\n            }\n        }\n        res_49284 = acc_49287;\n        sync_50533 = res_49284;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_49281 = sync_50533;\n        chunk_offset_49280 += group_sizze_49268;\n    }\n    res_49278 = x_49281;\n    if (thread_active_50989) {\n        *(__global float *) &mem_50624[gtid_49266 * 4] = res_49278;\n    }\n}\n__kernel void map_kernel_49316(int32_t sizze_47397, int32_t sizze_47398,\n                               int32_t sizze_47399, __global\n                               unsigned char *D_mem_50613, __global\n                               unsigned char *mem_50617, __global\n                               unsigned char *mem_50620)\n{\n    int32_t wave_sizze_50998;\n    int32_t group_sizze_50999;\n    bool thread_active_51000;\n    int32_t gtid_49309;\n    int32_t global_tid_49316;\n    int32_t local_tid_49317;\n    int32_t group_id_49318;\n    \n    global_tid_49316 = get",
            "_global_id(0);\n    local_tid_49317 = get_local_id(0);\n    group_sizze_50999 = get_local_size(0);\n    wave_sizze_50998 = LOCKSTEP_WIDTH;\n    group_id_49318 = get_group_id(0);\n    gtid_49309 = global_tid_49316;\n    thread_active_51000 = slt32(gtid_49309, sizze_47397);\n    \n    float res_49320;\n    \n    if (thread_active_51000) {\n        float x_49323 = 0.0F;\n        \n        for (int32_t chunk_offset_49322 = 0; chunk_offset_49322 < sizze_47399;\n             chunk_offset_49322++) {\n            float x_49332 = *(__global float *) &mem_50617[(chunk_offset_49322 *\n                                                            sizze_47398 +\n                                                            gtid_49309) * 4];\n            float x_49333 = *(__global\n                              float *) &D_mem_50613[chunk_offset_49322 * 4];\n            float res_49335 = x_49332 * x_49333;\n            float res_49337 = x_49323 + res_49335;\n            float x_tmp_51001 = res_49337;\n            \n            x_49323 = x_tmp_51001;\n        }\n        res_49320 = x_49323;\n    }\n    if (thread_active_51000) {\n        *(__global float *) &mem_50620[gtid_49309 * 4] = res_49320;\n    }\n}\n__kernel void map_kernel_49356(int32_t sizze_47396, int32_t sizze_47397,\n                               __global unsigned char *mem_50620, __global\n                               unsigned char *mem_50624, __global\n                               unsigned char *mem_50627)\n{\n    int32_t wave_sizze_51002;\n    int32_t group_sizze_51003;\n    bool thread_active_51004;\n    int32_t gtid_49349;\n    int32_t global_tid_49356;\n    int32_t local_tid_49357;\n    int32_t group_id_49358;\n    \n    global_tid_49356 = get_global_id(0);\n    local_tid_49357 = get_local_id(0);\n    group_sizze_51003 = get_local_size(0);\n    wave_sizze_51002 = LOCKSTEP_WIDTH;\n    group_id_49358 = get_group_id(0);\n    gtid_49349 = global_tid_49356;\n    thread_active_51004 = slt32(gtid_49349, sizze_47396);\n    \n    float res_49360;\n    \n    if (thread_activ",
            "e_51004) {\n        float x_49363 = 0.0F;\n        \n        for (int32_t chunk_offset_49362 = 0; chunk_offset_49362 < sizze_47397;\n             chunk_offset_49362++) {\n            float x_49372 = *(__global float *) &mem_50624[(chunk_offset_49362 *\n                                                            sizze_47396 +\n                                                            gtid_49349) * 4];\n            float x_49373 = *(__global float *) &mem_50620[chunk_offset_49362 *\n                                                           4];\n            float res_49375 = x_49372 * x_49373;\n            float res_49377 = x_49363 + res_49375;\n            float x_tmp_51005 = res_49377;\n            \n            x_49363 = x_tmp_51005;\n        }\n        res_49360 = x_49363;\n    }\n    if (thread_active_51004) {\n        *(__global float *) &mem_50627[gtid_49349 * 4] = res_49360;\n    }\n}\n__kernel void map_kernel_49385(int32_t sizze_47441, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50618)\n{\n    int32_t wave_sizze_51009;\n    int32_t group_sizze_51010;\n    bool thread_active_51011;\n    int32_t gtid_49378;\n    int32_t global_tid_49385;\n    int32_t local_tid_49386;\n    int32_t group_id_49387;\n    \n    global_tid_49385 = get_global_id(0);\n    local_tid_49386 = get_local_id(0);\n    group_sizze_51010 = get_local_size(0);\n    wave_sizze_51009 = LOCKSTEP_WIDTH;\n    group_id_49387 = get_group_id(0);\n    gtid_49378 = global_tid_49385;\n    thread_active_51011 = slt32(gtid_49378, sizze_47441);\n    \n    float x_49388;\n    float x_49389;\n    float res_49390;\n    \n    if (thread_active_51011) {\n        x_49388 = *(__global float *) &C_mem_50613[gtid_49378 * 4];\n        x_49389 = *(__global float *) &D_mem_50615[gtid_49378 * 4];\n        res_49390 = x_49388 + x_49389;\n    }\n    if (thread_active_51011) {\n        *(__global float *) &mem_50618[gtid_493",
            "78 * 4] = res_49390;\n    }\n}\n__kernel void map_kernel_49420(int32_t sizze_47438, int32_t sizze_47439,\n                               int32_t sizze_47441, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50618, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50625)\n{\n    int32_t wave_sizze_51012;\n    int32_t group_sizze_51013;\n    bool thread_active_51014;\n    int32_t gtid_49413;\n    int32_t global_tid_49420;\n    int32_t local_tid_49421;\n    int32_t group_id_49422;\n    \n    global_tid_49420 = get_global_id(0);\n    local_tid_49421 = get_local_id(0);\n    group_sizze_51013 = get_local_size(0);\n    wave_sizze_51012 = LOCKSTEP_WIDTH;\n    group_id_49422 = get_group_id(0);\n    gtid_49413 = global_tid_49420;\n    thread_active_51014 = slt32(gtid_49413, sizze_47438);\n    \n    float res_49424;\n    \n    if (thread_active_51014) {\n        float x_49427 = 0.0F;\n        \n        for (int32_t chunk_offset_49426 = 0; chunk_offset_49426 < sizze_47441;\n             chunk_offset_49426++) {\n            float x_49437 = *(__global float *) &mem_50618[chunk_offset_49426 *\n                                                           4];\n            float res_49439;\n            float x_49442 = 0.0F;\n            \n            for (int32_t chunk_offset_49441 = 0; chunk_offset_49441 <\n                 sizze_47439; chunk_offset_49441++) {\n                float x_49451 = *(__global\n                                  float *) &mem_50622[(chunk_offset_49441 *\n                                                       sizze_47438 +\n                                                       gtid_49413) * 4];\n                float x_49452 = *(__global\n                                  float *) &B_mem_50611[(chunk_offset_49441 *\n                                                         sizze_47441 +\n                                                         chunk_of",
            "fset_49426) *\n                                                        4];\n                float res_49454 = x_49451 * x_49452;\n                float res_49456 = x_49442 + res_49454;\n                float x_tmp_51016 = res_49456;\n                \n                x_49442 = x_tmp_51016;\n            }\n            res_49439 = x_49442;\n            \n            float res_49457 = x_49437 * res_49439;\n            float res_49459 = x_49427 + res_49457;\n            float x_tmp_51015 = res_49459;\n            \n            x_49427 = x_tmp_51015;\n        }\n        res_49424 = x_49427;\n    }\n    if (thread_active_51014) {\n        *(__global float *) &mem_50625[gtid_49413 * 4] = res_49424;\n    }\n}\n__kernel void map_kernel_49478(int32_t sizze_47492, int32_t sizze_47494,\n                               int32_t sizze_47495, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50619, __global\n                               unsigned char *mem_50622)\n{\n    int32_t wave_sizze_51020;\n    int32_t group_sizze_51021;\n    bool thread_active_51022;\n    int32_t gtid_49471;\n    int32_t global_tid_49478;\n    int32_t local_tid_49479;\n    int32_t group_id_49480;\n    \n    global_tid_49478 = get_global_id(0);\n    local_tid_49479 = get_local_id(0);\n    group_sizze_51021 = get_local_size(0);\n    wave_sizze_51020 = LOCKSTEP_WIDTH;\n    group_id_49480 = get_group_id(0);\n    gtid_49471 = global_tid_49478;\n    thread_active_51022 = slt32(gtid_49471, sizze_47494);\n    \n    float res_49482;\n    \n    if (thread_active_51022) {\n        float x_49485 = 0.0F;\n        \n        for (int32_t chunk_offset_49484 = 0; chunk_offset_49484 < sizze_47492;\n             chunk_offset_49484++) {\n            float x_49494 = *(__global float *) &mem_50619[(chunk_offset_49484 *\n                                                            sizze_47495 +\n                                                            gtid_49471) * 4];\n            float x_49495 = *(__g",
            "lobal\n                              float *) &D_mem_50615[chunk_offset_49484 * 4];\n            float res_49497 = x_49494 * x_49495;\n            float res_49499 = x_49485 + res_49497;\n            float x_tmp_51023 = res_49499;\n            \n            x_49485 = x_tmp_51023;\n        }\n        res_49482 = x_49485;\n    }\n    if (thread_active_51022) {\n        *(__global float *) &mem_50622[gtid_49471 * 4] = res_49482;\n    }\n}\n__kernel void map_kernel_49529(int32_t sizze_47491, int32_t sizze_47492,\n                               int32_t sizze_47494, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50626, __global\n                               unsigned char *mem_50629)\n{\n    int32_t wave_sizze_51024;\n    int32_t group_sizze_51025;\n    bool thread_active_51026;\n    int32_t gtid_49522;\n    int32_t global_tid_49529;\n    int32_t local_tid_49530;\n    int32_t group_id_49531;\n    \n    global_tid_49529 = get_global_id(0);\n    local_tid_49530 = get_local_id(0);\n    group_sizze_51025 = get_local_size(0);\n    wave_sizze_51024 = LOCKSTEP_WIDTH;\n    group_id_49531 = get_group_id(0);\n    gtid_49522 = global_tid_49529;\n    thread_active_51026 = slt32(gtid_49522, sizze_47491);\n    \n    float res_49533;\n    \n    if (thread_active_51026) {\n        float x_49536 = 0.0F;\n        \n        for (int32_t chunk_offset_49535 = 0; chunk_offset_49535 < sizze_47494;\n             chunk_offset_49535++) {\n            float x_49546 = *(__global float *) &mem_50622[chunk_offset_49535 *\n                                                           4];\n            float res_49548;\n            float x_49551 = 0.0F;\n            \n            for (int32_t chunk_offset_49550 = 0; chunk_offset_49550 <\n                 sizze_47492; chunk_offset_49550++) {\n                float x_49560 = *(__global\n                                  float *) &mem_50626[(chunk_offset_49550 *\n  ",
            "                                                     sizze_47491 +\n                                                       gtid_49522) * 4];\n                float x_49561 = *(__global\n                                  float *) &B_mem_50611[(chunk_offset_49550 *\n                                                         sizze_47494 +\n                                                         chunk_offset_49535) *\n                                                        4];\n                float res_49563 = x_49560 * x_49561;\n                float res_49565 = x_49551 + res_49563;\n                float x_tmp_51028 = res_49565;\n                \n                x_49551 = x_tmp_51028;\n            }\n            res_49548 = x_49551;\n            \n            float res_49566 = x_49546 * res_49548;\n            float res_49568 = x_49536 + res_49566;\n            float x_tmp_51027 = res_49568;\n            \n            x_49536 = x_tmp_51027;\n        }\n        res_49533 = x_49536;\n    }\n    if (thread_active_51026) {\n        *(__global float *) &mem_50629[gtid_49522 * 4] = res_49533;\n    }\n}\n__kernel void map_kernel_49587(int32_t sizze_47556, int32_t sizze_47557,\n                               int32_t sizze_47558, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50617, __global\n                               unsigned char *mem_50620)\n{\n    int32_t wave_sizze_51033;\n    int32_t group_sizze_51034;\n    bool thread_active_51035;\n    int32_t gtid_49580;\n    int32_t global_tid_49587;\n    int32_t local_tid_49588;\n    int32_t group_id_49589;\n    \n    global_tid_49587 = get_global_id(0);\n    local_tid_49588 = get_local_id(0);\n    group_sizze_51034 = get_local_size(0);\n    wave_sizze_51033 = LOCKSTEP_WIDTH;\n    group_id_49589 = get_group_id(0);\n    gtid_49580 = global_tid_49587;\n    thread_active_51035 = slt32(gtid_49580, sizze_47556);\n    \n    float res_49591;\n    \n    if (thread_active_51035) {\n        float x_49594 = ",
            "0.0F;\n        \n        for (int32_t chunk_offset_49593 = 0; chunk_offset_49593 < sizze_47558;\n             chunk_offset_49593++) {\n            float x_49603 = *(__global float *) &mem_50617[(chunk_offset_49593 *\n                                                            sizze_47557 +\n                                                            gtid_49580) * 4];\n            float x_49604 = *(__global\n                              float *) &C_mem_50613[chunk_offset_49593 * 4];\n            float res_49606 = x_49603 * x_49604;\n            float res_49608 = x_49594 + res_49606;\n            float x_tmp_51036 = res_49608;\n            \n            x_49594 = x_tmp_51036;\n        }\n        res_49591 = x_49594;\n    }\n    if (thread_active_51035) {\n        *(__global float *) &mem_50620[gtid_49580 * 4] = res_49591;\n    }\n}\n__kernel void map_kernel_49629(int32_t sizze_47554, int32_t sizze_47555,\n                               int32_t sizze_47556, __global\n                               unsigned char *mem_50620, __global\n                               unsigned char *mem_50625, __global\n                               unsigned char *mem_50629)\n{\n    int32_t wave_sizze_51037;\n    int32_t group_sizze_51038;\n    bool thread_active_51039;\n    int32_t gtid_49620;\n    int32_t gtid_49621;\n    int32_t global_tid_49629;\n    int32_t local_tid_49630;\n    int32_t group_id_49631;\n    \n    global_tid_49629 = get_global_id(0);\n    local_tid_49630 = get_local_id(0);\n    group_sizze_51038 = get_local_size(0);\n    wave_sizze_51037 = LOCKSTEP_WIDTH;\n    group_id_49631 = get_group_id(0);\n    gtid_49620 = squot32(global_tid_49629, sizze_47555);\n    gtid_49621 = global_tid_49629 - squot32(global_tid_49629, sizze_47555) *\n        sizze_47555;\n    thread_active_51039 = slt32(gtid_49620, sizze_47554) && slt32(gtid_49621,\n                                                                  sizze_47555);\n    \n    float res_49633;\n    \n    if (thread_active_51039) {\n        float x_49636 = 0.0F;\n        \n       ",
            " for (int32_t chunk_offset_49635 = 0; chunk_offset_49635 < sizze_47556;\n             chunk_offset_49635++) {\n            float x_49645 = *(__global float *) &mem_50625[(chunk_offset_49635 *\n                                                            (sizze_47554 *\n                                                             sizze_47555) +\n                                                            gtid_49620 *\n                                                            sizze_47555 +\n                                                            gtid_49621) * 4];\n            float x_49646 = *(__global float *) &mem_50620[chunk_offset_49635 *\n                                                           4];\n            float res_49648 = x_49645 * x_49646;\n            float res_49650 = x_49636 + res_49648;\n            float x_tmp_51040 = res_49650;\n            \n            x_49636 = x_tmp_51040;\n        }\n        res_49633 = x_49636;\n    }\n    if (thread_active_51039) {\n        *(__global float *) &mem_50629[(gtid_49620 * sizze_47555 + gtid_49621) *\n                                       4] = res_49633;\n    }\n}\n__kernel void map_kernel_49669(int32_t sizze_47600, int32_t sizze_47603,\n                               int32_t sizze_47604, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50619, __global\n                               unsigned char *mem_50622)\n{\n    int32_t wave_sizze_51044;\n    int32_t group_sizze_51045;\n    bool thread_active_51046;\n    int32_t gtid_49662;\n    int32_t global_tid_49669;\n    int32_t local_tid_49670;\n    int32_t group_id_49671;\n    \n    global_tid_49669 = get_global_id(0);\n    local_tid_49670 = get_local_id(0);\n    group_sizze_51045 = get_local_size(0);\n    wave_sizze_51044 = LOCKSTEP_WIDTH;\n    group_id_49671 = get_group_id(0);\n    gtid_49662 = global_tid_49669;\n    thread_active_51046 = slt32(gtid_49662, sizze_47600);\n    \n    float res_49673;\n    \n    if (thread_active_",
            "51046) {\n        float x_49676 = 0.0F;\n        \n        for (int32_t chunk_offset_49675 = 0; chunk_offset_49675 < sizze_47604;\n             chunk_offset_49675++) {\n            float x_49685 = *(__global float *) &mem_50619[(chunk_offset_49675 *\n                                                            sizze_47603 +\n                                                            gtid_49662) * 4];\n            float x_49686 = *(__global\n                              float *) &D_mem_50615[chunk_offset_49675 * 4];\n            float res_49688 = x_49685 * x_49686;\n            float res_49690 = x_49676 + res_49688;\n            float x_tmp_51047 = res_49690;\n            \n            x_49676 = x_tmp_51047;\n        }\n        res_49673 = x_49676;\n    }\n    if (thread_active_51046) {\n        *(__global float *) &mem_50622[gtid_49662 * 4] = res_49673;\n    }\n}\n__kernel void map_kernel_49711(int32_t sizze_47599, int32_t sizze_47600,\n                               int32_t sizze_47601, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50626, __global\n                               unsigned char *mem_50630, __global\n                               unsigned char *mem_50633)\n{\n    int32_t wave_sizze_51048;\n    int32_t group_sizze_51049;\n    bool thread_active_51050;\n    int32_t gtid_49704;\n    int32_t global_tid_49711;\n    int32_t local_tid_49712;\n    int32_t group_id_49713;\n    \n    global_tid_49711 = get_global_id(0);\n    local_tid_49712 = get_local_id(0);\n    group_sizze_51049 = get_local_size(0);\n    wave_sizze_51048 = LOCKSTEP_WIDTH;\n    group_id_49713 = get_group_id(0);\n    gtid_49704 = global_tid_49711;\n    thread_active_51050 = slt32(gtid_49704, sizze_47599);\n    \n    float res_49716;\n    \n    if (thread_active_51050) {\n        float x_49719 = 0.0F;\n        \n        for (int32_t chunk_offset_49718 = 0; chunk_offset_49718 < sizze_47600;\n             chunk_offset_49718++) {\n            float x_49730 = *(__glob",
            "al float *) &mem_50626[(chunk_offset_49718 *\n                                                            sizze_47599 +\n                                                            gtid_49704) * 4];\n            float x_49731 = *(__global float *) &mem_50630[(chunk_offset_49718 *\n                                                            sizze_47601 +\n                                                            gtid_49704) * 4];\n            float x_49732 = *(__global float *) &mem_50622[chunk_offset_49718 *\n                                                           4];\n            float res_49734 = x_49730 + x_49731;\n            float res_49735 = x_49732 * res_49734;\n            float res_49737 = x_49719 + res_49735;\n            float x_tmp_51051 = res_49737;\n            \n            x_49719 = x_tmp_51051;\n        }\n        res_49716 = x_49719;\n    }\n    if (thread_active_51050) {\n        *(__global float *) &mem_50633[gtid_49704 * 4] = res_49716;\n    }\n}\n__kernel void map_kernel_49760(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_47659, int32_t sizze_47660,\n                               int32_t sizze_47661, int32_t sizze_47663,\n                               __global unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50624)\n{\n    __local volatile char *restrict mem_50615 = mem_aligned_0;\n    __local volatile char *restrict mem_50619 = mem_aligned_1;\n    int32_t wave_sizze_51057;\n    int32_t group_sizze_51058;\n    bool thread_active_51059;\n    int32_t gtid_49749;\n    int32_t gtid_49750;\n    int32_t gtid_49751;\n    int32_t global_tid_49760;\n    int32_t local_tid_49761;\n    int32_t group_id_49762;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t ltid_50532;\n    \n    global_tid_49760 = get_global_id(0);\n    local_tid_49761 = get_local_id(0);\n    gr",
            "oup_sizze_51058 = get_local_size(0);\n    wave_sizze_51057 = LOCKSTEP_WIDTH;\n    group_id_49762 = get_group_id(0);\n    gtid_49749 = squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528) + squot32(squot32(global_tid_49760,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47660 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47663 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528));\n    gtid_49750 = squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528) + squot32(squot32(global_tid_49760,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528) -\n                                                     squot32(squot32(global_tid_49760,\n                                                                     tile_sizze_50528 *\n                                                                     ti",
            "le_sizze_50528),\n                                                             squot32(sizze_47660 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528) *\n                                                             squot32(sizze_47663 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528)) *\n                                                     (squot32(sizze_47660 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528) *\n                                                      squot32(sizze_47663 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528)),\n                                                     squot32(sizze_47663 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n        tile_sizze_50528;\n    gtid_49751 = srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_5",
            "0528) -\n                squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528 +\n        (squot32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528) -\n         squot32(squot32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528),\n                 squot32(sizze_47660 + tile_sizze_50528 - 1, tile_sizze_50528) *\n                 squot32(sizze_47663 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * (squot32(sizze_47660 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528) *\n                                               squot32(sizze_47663 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528)) -\n         squot32(squot32(global_tid_49760, tile_sizze_50528 *\n                         tile_sizze_50528) - squot32(squot32(global_tid_49760,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47660 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47663 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n                 (squot32(sizz",
            "e_47660 + tile_sizze_50528 - 1,\n                          tile_sizze_50528) * squot32(sizze_47663 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)),\n                 squot32(sizze_47663 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * squot32(sizze_47663 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)) *\n        tile_sizze_50528;\n    ltid_50530 = squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528);\n    ltid_50531 = squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528);\n    ltid_50532 = srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_49760, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_49760, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528;\n    thread_active_51059 = (slt32(gtid_49749, sizze_47659) && slt32(gtid_49750,\n      ",
            "                                                             sizze_47660)) &&\n        slt32(gtid_49751, sizze_47663);\n    if (thread_active_51059) { }\n    \n    float res_49765;\n    float x_49768 = 0.0F;\n    int32_t chunk_sizze_49766;\n    int32_t chunk_offset_49767 = 0;\n    \n    while (slt32(chunk_offset_49767, sizze_47661)) {\n        if (slt32(sizze_47661 - chunk_offset_49767, tile_sizze_50528)) {\n            chunk_sizze_49766 = sizze_47661 - chunk_offset_49767;\n        } else {\n            chunk_sizze_49766 = tile_sizze_50528;\n        }\n        for (int32_t comb_iter_51060 = 0; comb_iter_51060 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51060++) {\n            int32_t cid_50548;\n            int32_t cid_50549;\n            int32_t flat_comb_id_51061 = comb_iter_51060 *\n                    tiled_group_sizze_50529 + local_tid_49761;\n            \n            cid_50548 = squot32(flat_comb_id_51061, tile_sizze_50528);\n            cid_50549 = flat_comb_id_51061 - squot32(flat_comb_id_51061,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50548, tile_sizze_50528) && slt32(cid_50549,\n                                                             chunk_sizze_49766)) &&\n                slt32(gtid_49750, sizze_47660)) {\n                float x_chunk_outer_elem_50547 = *(__global\n                                                   float *) &A_mem_50609[(gtid_49749 *\n                                                                          (sizze_47660 *\n                                                                           sizze_47661) +\n                                                                          gtid_49750 *\n                                                                          sizze_47661 +\n                                                                        ",
            "  (chunk_offset_49767 +\n                                                                           ltid_50532)) *\n                                                                         4];\n                \n                *(__local float *) &mem_50615[(cid_50548 * tile_sizze_50528 +\n                                               cid_50549) * 4] =\n                    x_chunk_outer_elem_50547;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51059) { }\n        for (int32_t comb_iter_51062 = 0; comb_iter_51062 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51062++) {\n            int32_t cid_50553;\n            int32_t cid_50554;\n            int32_t flat_comb_id_51063 = comb_iter_51062 *\n                    tiled_group_sizze_50529 + local_tid_49761;\n            \n            cid_50553 = squot32(flat_comb_id_51063, tile_sizze_50528);\n            cid_50554 = flat_comb_id_51063 - squot32(flat_comb_id_51063,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50553, chunk_sizze_49766) && slt32(cid_50554,\n                                                              tile_sizze_50528)) &&\n                slt32(gtid_49751, sizze_47663)) {\n                float x_chunk_outer_elem_50552 = *(__global\n                                                   float *) &B_mem_50611[((chunk_offset_49767 +\n                                                                           ltid_50531) *\n                                                                          sizze_47663 +\n                                                                          gtid_49751) *\n                                                                         4];\n                \n                *(__local float *) &mem_50619[(cid_50553 * tile_sizze_50528 +\n                           ",
            "                    cid_50554) * 4] =\n                    x_chunk_outer_elem_50552;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51059) { }\n        \n        float res_49771;\n        float sync_50556;\n        float acc_49774 = x_49768;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_49772 = 1;\n        \n        if (thread_active_51059) {\n            if (chunk_sizze_49766 == tile_sizze_50528) {\n                for (int32_t i_49773 = 0; i_49773 < tile_sizze_50528;\n                     i_49773++) {\n                    float x_49777 = *(__local float *) &mem_50615[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_49773) *\n                                                                  4];\n                    float x_49778 = *(__local float *) &mem_50619[(i_49773 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_49780 = x_49777 * x_49778;\n                    float res_49782 = acc_49774 + res_49780;\n                    float acc_tmp_51064 = res_49782;\n                    \n                    acc_49774 = acc_tmp_51064;\n                }\n            } else {\n                for (int32_t i_49773 = 0; i_49773 < chunk_sizze_49766;\n                     i_49773++) {\n                    float x_49777 = *(__local float *) &mem_50615[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_49773) *\n                                                                  4];\n                    float x_49778 = *(__local float *) &mem_50619[(i_49773 *\n                                                 ",
            "                  tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_49780 = x_49777 * x_49778;\n                    float res_49782 = acc_49774 + res_49780;\n                    float acc_tmp_51064 = res_49782;\n                    \n                    acc_49774 = acc_tmp_51064;\n                }\n            }\n        }\n        res_49771 = acc_49774;\n        sync_50556 = res_49771;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_49768 = sync_50556;\n        chunk_offset_49767 += tile_sizze_50528;\n    }\n    res_49765 = x_49768;\n    if (thread_active_51059) {\n        *(__global float *) &mem_50624[(gtid_49749 * (sizze_47660 *\n                                                      sizze_47663) +\n                                        gtid_49750 * sizze_47663 + gtid_49751) *\n                                       4] = res_49765;\n    }\n}\n__kernel void map_kernel_49792(int32_t sizze_47692, int32_t sizze_47694,\n                               __global unsigned char *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50617)\n{\n    int32_t wave_sizze_51070;\n    int32_t group_sizze_51071;\n    bool thread_active_51072;\n    int32_t gtid_49783;\n    int32_t gtid_49784;\n    int32_t global_tid_49792;\n    int32_t local_tid_49793;\n    int32_t group_id_49794;\n    \n    global_tid_49792 = get_global_id(0);\n    local_tid_49793 = get_local_id(0);\n    group_sizze_51071 = get_local_size(0);\n    wave_sizze_51070 = LOCKSTEP_WIDTH;\n    group_id_49794 = get_group_id(0);\n    gtid_49783 = squot32(global_tid_49792, sizze_47694);\n    gtid_49784 = global_tid_49792 - squot32(global_tid_49792, sizze_47694) *\n        sizze_47694;\n    thread_active_51072 = slt32(gtid_49783, sizze_47692) && slt32(gtid_49784,\n                                                                  si",
            "zze_47694);\n    \n    float x_49795;\n    float x_49796;\n    float res_49797;\n    \n    if (thread_active_51072) {\n        x_49795 = *(__global float *) &B_mem_50611[gtid_49783 * 4];\n        x_49796 = *(__global float *) &C_mem_50613[gtid_49784 * 4];\n        res_49797 = x_49795 * x_49796;\n    }\n    if (thread_active_51072) {\n        *(__global float *) &mem_50617[(gtid_49783 * sizze_47694 + gtid_49784) *\n                                       4] = res_49797;\n    }\n}\n__kernel void map_kernel_49820(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_47690, int32_t sizze_47691,\n                               int32_t sizze_47692, int32_t sizze_47694,\n                               __global unsigned char *A_mem_50609, __global\n                               unsigned char *mem_50617, __global\n                               unsigned char *mem_50630)\n{\n    __local volatile char *restrict mem_50621 = mem_aligned_0;\n    __local volatile char *restrict mem_50625 = mem_aligned_1;\n    int32_t wave_sizze_51073;\n    int32_t group_sizze_51074;\n    bool thread_active_51075;\n    int32_t gtid_49809;\n    int32_t gtid_49810;\n    int32_t gtid_49811;\n    int32_t global_tid_49820;\n    int32_t local_tid_49821;\n    int32_t group_id_49822;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t ltid_50532;\n    \n    global_tid_49820 = get_global_id(0);\n    local_tid_49821 = get_local_id(0);\n    group_sizze_51074 = get_local_size(0);\n    wave_sizze_51073 = LOCKSTEP_WIDTH;\n    group_id_49822 = get_group_id(0);\n    gtid_49809 = squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528) + squot32(squot32(global_tid_49820,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n         ",
            "                                            squot32(sizze_47691 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47694 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528));\n    gtid_49810 = squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528) + squot32(squot32(global_tid_49820,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528) -\n                                                     squot32(squot32(global_tid_49820,\n                                                                     tile_sizze_50528 *\n                                                                     tile_sizze_50528),\n                                                             squot32(sizze_47691 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528) *\n                                                             squot32(sizze_47694 +\n                                                                     ",
            "tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528)) *\n                                                     (squot32(sizze_47691 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528) *\n                                                      squot32(sizze_47694 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528)),\n                                                     squot32(sizze_47694 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n        tile_sizze_50528;\n    gtid_49811 = srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528 +\n        (squot32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528) -\n         squot32(squot32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528),\n                 s",
            "quot32(sizze_47691 + tile_sizze_50528 - 1, tile_sizze_50528) *\n                 squot32(sizze_47694 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * (squot32(sizze_47691 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528) *\n                                               squot32(sizze_47694 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528)) -\n         squot32(squot32(global_tid_49820, tile_sizze_50528 *\n                         tile_sizze_50528) - squot32(squot32(global_tid_49820,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47691 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47694 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n                 (squot32(sizze_47691 + tile_sizze_50528 - 1,\n                          tile_sizze_50528) * squot32(sizze_47694 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)),\n                 squot32(sizze_47694 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * squot32(sizze_47694 +\n                                                      tile_sizze_50528 - 1,\n                                              ",
            "        tile_sizze_50528)) *\n        tile_sizze_50528;\n    ltid_50530 = squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528);\n    ltid_50531 = squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528);\n    ltid_50532 = srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_49820, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_49820, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528;\n    thread_active_51075 = (slt32(gtid_49809, sizze_47690) && slt32(gtid_49810,\n                                                                   sizze_47691)) &&\n        slt32(gtid_49811, sizze_47694);\n    if (thread_active_51075) { }\n    \n    float res_49825;\n    float x_49828 = 0.0F;\n    int32_t chunk_sizze_49826;\n    int32_t chunk_offset_49827 = 0;\n    \n    while (slt32(chunk_offset_49827, sizze_47692)) {\n        if (slt32(sizze_47692 - chunk_offset_49827, tile_sizze_50528)) {\n            chunk_sizze_49826 = sizze_47692 - chunk_offset_49827;\n        } else {\n            chunk",
            "_sizze_49826 = tile_sizze_50528;\n        }\n        for (int32_t comb_iter_51076 = 0; comb_iter_51076 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51076++) {\n            int32_t cid_50548;\n            int32_t cid_50549;\n            int32_t flat_comb_id_51077 = comb_iter_51076 *\n                    tiled_group_sizze_50529 + local_tid_49821;\n            \n            cid_50548 = squot32(flat_comb_id_51077, tile_sizze_50528);\n            cid_50549 = flat_comb_id_51077 - squot32(flat_comb_id_51077,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50548, tile_sizze_50528) && slt32(cid_50549,\n                                                             chunk_sizze_49826)) &&\n                slt32(gtid_49810, sizze_47691)) {\n                float x_chunk_outer_elem_50547 = *(__global\n                                                   float *) &A_mem_50609[(gtid_49809 *\n                                                                          (sizze_47691 *\n                                                                           sizze_47692) +\n                                                                          gtid_49810 *\n                                                                          sizze_47692 +\n                                                                          (chunk_offset_49827 +\n                                                                           ltid_50532)) *\n                                                                         4];\n                \n                *(__local float *) &mem_50621[(cid_50548 * tile_sizze_50528 +\n                                               cid_50549) * 4] =\n                    x_chunk_outer_elem_50547;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51075) { }\n     ",
            "   for (int32_t comb_iter_51078 = 0; comb_iter_51078 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51078++) {\n            int32_t cid_50553;\n            int32_t cid_50554;\n            int32_t flat_comb_id_51079 = comb_iter_51078 *\n                    tiled_group_sizze_50529 + local_tid_49821;\n            \n            cid_50553 = squot32(flat_comb_id_51079, tile_sizze_50528);\n            cid_50554 = flat_comb_id_51079 - squot32(flat_comb_id_51079,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50553, chunk_sizze_49826) && slt32(cid_50554,\n                                                              tile_sizze_50528)) &&\n                slt32(gtid_49811, sizze_47694)) {\n                float x_chunk_outer_elem_50552 = *(__global\n                                                   float *) &mem_50617[((chunk_offset_49827 +\n                                                                         ltid_50531) *\n                                                                        sizze_47694 +\n                                                                        gtid_49811) *\n                                                                       4];\n                \n                *(__local float *) &mem_50625[(cid_50553 * tile_sizze_50528 +\n                                               cid_50554) * 4] =\n                    x_chunk_outer_elem_50552;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51075) { }\n        \n        float res_49831;\n        float sync_50556;\n        float acc_49834 = x_49828;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_49832 = 1;\n        \n        if (thread_active_51075) {\n            if (chunk_sizze_49826 == tile_sizze_50528) {\n                for (int32_t i_49833 = 0; i_49833 < tile_sizze_505",
            "28;\n                     i_49833++) {\n                    float x_49837 = *(__local float *) &mem_50621[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_49833) *\n                                                                  4];\n                    float x_49838 = *(__local float *) &mem_50625[(i_49833 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_49840 = x_49837 * x_49838;\n                    float res_49842 = acc_49834 + res_49840;\n                    float acc_tmp_51080 = res_49842;\n                    \n                    acc_49834 = acc_tmp_51080;\n                }\n            } else {\n                for (int32_t i_49833 = 0; i_49833 < chunk_sizze_49826;\n                     i_49833++) {\n                    float x_49837 = *(__local float *) &mem_50621[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_49833) *\n                                                                  4];\n                    float x_49838 = *(__local float *) &mem_50625[(i_49833 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_49840 = x_49837 * x_49838;\n                    float res_49842 = acc_49834 + res_49840;\n                    float acc_tmp_51080 = res_49842;\n                    \n                    acc_49834 = acc_tmp_51080;\n                }\n            }\n        }\n        res_49831 = acc_49834;\n        syn",
            "c_50556 = res_49831;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_49828 = sync_50556;\n        chunk_offset_49827 += tile_sizze_50528;\n    }\n    res_49825 = x_49828;\n    if (thread_active_51075) {\n        *(__global float *) &mem_50630[(gtid_49809 * (sizze_47691 *\n                                                      sizze_47694) +\n                                        gtid_49810 * sizze_47694 + gtid_49811) *\n                                       4] = res_49825;\n    }\n}\n__kernel void map_kernel_49864(int32_t sizze_47724, int32_t sizze_47725,\n                               int32_t sizze_47726, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50627, __global\n                               unsigned char *mem_50631)\n{\n    int32_t wave_sizze_51089;\n    int32_t group_sizze_51090;\n    bool thread_active_51091;\n    int32_t gtid_49855;\n    int32_t gtid_49856;\n    int32_t global_tid_49864;\n    int32_t local_tid_49865;\n    int32_t group_id_49866;\n    \n    global_tid_49864 = get_global_id(0);\n    local_tid_49865 = get_local_id(0);\n    group_sizze_51090 = get_local_size(0);\n    wave_sizze_51089 = LOCKSTEP_WIDTH;\n    group_id_49866 = get_group_id(0);\n    gtid_49855 = squot32(global_tid_49864, sizze_47725);\n    gtid_49856 = global_tid_49864 - squot32(global_tid_49864, sizze_47725) *\n        sizze_47725;\n    thread_active_51091 = slt32(gtid_49855, sizze_47724) && slt32(gtid_49856,\n                                                                  sizze_47725);\n    \n    float res_49869;\n    \n    if (thread_active_51091) {\n        float x_49872 = 0.0F;\n        \n        for (int32_t chunk_offset_49871 = 0; chunk_offset_49871 < sizze_47726;\n             chunk_offset_49871++) {\n            float x_49881 = *(__global float *) &mem_50627[(chunk_offset_49871 *\n                                                            (sizze_47724 *\n                                                             sizze_47725) +\n  ",
            "                                                          gtid_49855 *\n                                                            sizze_47725 +\n                                                            gtid_49856) * 4];\n            float x_49882 = *(__global float *) &mem_50622[(gtid_49855 *\n                                                            sizze_47726 +\n                                                            chunk_offset_49871) *\n                                                           4];\n            float res_49884 = x_49881 * x_49882;\n            float res_49886 = x_49872 + res_49884;\n            float x_tmp_51092 = res_49886;\n            \n            x_49872 = x_tmp_51092;\n        }\n        res_49869 = x_49872;\n    }\n    if (thread_active_51091) {\n        *(__global float *) &mem_50631[(gtid_49855 * sizze_47725 + gtid_49856) *\n                                       4] = res_49869;\n    }\n}\n__kernel void map_kernel_49907(int32_t sizze_47724, int32_t sizze_47726,\n                               int32_t sizze_47727, int32_t sizze_47728,\n                               int32_t sizze_47729, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50618, __global\n                               unsigned char *mem_50622)\n{\n    int32_t wave_sizze_51085;\n    int32_t group_sizze_51086;\n    bool thread_active_51087;\n    int32_t gtid_49898;\n    int32_t gtid_49899;\n    int32_t global_tid_49907;\n    int32_t local_tid_49908;\n    int32_t group_id_49909;\n    \n    global_tid_49907 = get_global_id(0);\n    local_tid_49908 = get_local_id(0);\n    group_sizze_51086 = get_local_size(0);\n    wave_sizze_51085 = LOCKSTEP_WIDTH;\n    group_id_49909 = get_group_id(0);\n    gtid_49898 = squot32(global_tid_49907, sizze_47726);\n    gtid_49899 = global_tid_49907 - squot32(global_tid_49907, sizze_47726) *\n        sizze_47726;\n    thread_active_51087 = slt32(gtid_49898, sizze_47724) && slt32(gtid_49899,\n          ",
            "                                                        sizze_47726);\n    \n    float res_49911;\n    \n    if (thread_active_51087) {\n        float x_49914 = 0.0F;\n        \n        for (int32_t chunk_offset_49913 = 0; chunk_offset_49913 < sizze_47729;\n             chunk_offset_49913++) {\n            float x_49923 = *(__global float *) &mem_50618[(chunk_offset_49913 *\n                                                            (sizze_47727 *\n                                                             sizze_47728) +\n                                                            gtid_49898 *\n                                                            sizze_47728 +\n                                                            gtid_49899) * 4];\n            float x_49924 = *(__global\n                              float *) &C_mem_50613[chunk_offset_49913 * 4];\n            float res_49926 = x_49923 * x_49924;\n            float res_49928 = x_49914 + res_49926;\n            float x_tmp_51088 = res_49928;\n            \n            x_49914 = x_tmp_51088;\n        }\n        res_49911 = x_49914;\n    }\n    if (thread_active_51087) {\n        *(__global float *) &mem_50622[(gtid_49898 * sizze_47726 + gtid_49899) *\n                                       4] = res_49911;\n    }\n}\n__kernel void map_kernel_49950(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_47777, int32_t sizze_47778,\n                               int32_t sizze_47779, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50626, __global\n                               unsigned char *mem_50636)\n{\n    __local volatile char *restrict mem_50629 = mem_aligned_0;\n    __local volatile char *restrict mem_50632 = mem_aligned",
            "_1;\n    int32_t wave_sizze_51101;\n    int32_t group_sizze_51102;\n    bool thread_active_51103;\n    int32_t gtid_49941;\n    int32_t gtid_49942;\n    int32_t global_tid_49950;\n    int32_t local_tid_49951;\n    int32_t group_id_49952;\n    \n    global_tid_49950 = get_global_id(0);\n    local_tid_49951 = get_local_id(0);\n    group_sizze_51102 = get_local_size(0);\n    wave_sizze_51101 = LOCKSTEP_WIDTH;\n    group_id_49952 = get_group_id(0);\n    gtid_49941 = squot32(global_tid_49950, sizze_47778);\n    gtid_49942 = global_tid_49950 - squot32(global_tid_49950, sizze_47778) *\n        sizze_47778;\n    thread_active_51103 = slt32(gtid_49941, sizze_47777) && slt32(gtid_49942,\n                                                                  sizze_47778);\n    \n    float x_49953;\n    float x_49955;\n    float x_49956;\n    \n    if (thread_active_51103) {\n        x_49953 = *(__global float *) &A_mem_50609[gtid_49941 * 4];\n        x_49955 = *(__global float *) &B_mem_50611[gtid_49942 * 4];\n        x_49956 = x_49953 * x_49955;\n    }\n    \n    float res_49957;\n    float x_49960 = 0.0F;\n    int32_t chunk_sizze_49958;\n    int32_t chunk_offset_49959 = 0;\n    \n    while (slt32(chunk_offset_49959, sizze_47779)) {\n        if (slt32(sizze_47779 - chunk_offset_49959, group_sizze_49945)) {\n            chunk_sizze_49958 = sizze_47779 - chunk_offset_49959;\n        } else {\n            chunk_sizze_49958 = group_sizze_49945;\n        }\n        \n        float res_49963;\n        float sync_50542;\n        \n        for (int32_t comb_iter_51104 = 0; comb_iter_51104 <\n             squot32(group_sizze_49945 + group_sizze_49945 - 1,\n                     group_sizze_49945); comb_iter_51104++) {\n            int32_t cid_50529;\n            int32_t flat_comb_id_51105 = comb_iter_51104 * group_sizze_49945 +\n                    local_tid_49951;\n            \n            cid_50529 = flat_comb_id_51105;\n            if (slt32(cid_50529, chunk_sizze_49958) && 1) {\n                float x_chunk_outer_elem_50528 = *(__global\n ",
            "                                                  float *) &C_mem_50613[(chunk_offset_49959 +\n                                                                          local_tid_49951) *\n                                                                         4];\n                \n                *(__local float *) &mem_50629[cid_50529 * 4] =\n                    x_chunk_outer_elem_50528;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        for (int32_t comb_iter_51106 = 0; comb_iter_51106 <\n             squot32(group_sizze_49945 + group_sizze_49945 - 1,\n                     group_sizze_49945); comb_iter_51106++) {\n            int32_t cid_50532;\n            int32_t flat_comb_id_51107 = comb_iter_51106 * group_sizze_49945 +\n                    local_tid_49951;\n            \n            cid_50532 = flat_comb_id_51107;\n            if (slt32(cid_50532, chunk_sizze_49958) && 1) {\n                float x_chunk_outer_elem_50531 = *(__global\n                                                   float *) &mem_50626[(gtid_49941 *\n                                                                        sizze_47779 +\n                                                                        (chunk_offset_49959 +\n                                                                         local_tid_49951)) *\n                                                                       4];\n                \n                *(__local float *) &mem_50632[cid_50532 * 4] =\n                    x_chunk_outer_elem_50531;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        \n        float acc_49966 = x_49960;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_49964 = 1;\n        \n        if (thread_active_51103) {\n            if (chunk_sizze_49958 == group_sizze_49945) {\n                for (int32_t i_49965 = 0; i_49965 < group_sizze_49945;\n                     i_49965++) {\n                    float x_49969 = *(__local float *) &mem_50629[i_49965 * 4];\n                    ",
            "float x_49970 = *(__local float *) &mem_50632[i_49965 * 4];\n                    float res_49972 = x_49956 * x_49969;\n                    float res_49973 = x_49970 * res_49972;\n                    float res_49975 = acc_49966 + res_49973;\n                    float acc_tmp_51108 = res_49975;\n                    \n                    acc_49966 = acc_tmp_51108;\n                }\n            } else {\n                for (int32_t i_49965 = 0; i_49965 < chunk_sizze_49958;\n                     i_49965++) {\n                    float x_49969 = *(__local float *) &mem_50629[i_49965 * 4];\n                    float x_49970 = *(__local float *) &mem_50632[i_49965 * 4];\n                    float res_49972 = x_49956 * x_49969;\n                    float res_49973 = x_49970 * res_49972;\n                    float res_49975 = acc_49966 + res_49973;\n                    float acc_tmp_51108 = res_49975;\n                    \n                    acc_49966 = acc_tmp_51108;\n                }\n            }\n        }\n        res_49963 = acc_49966;\n        sync_50542 = res_49963;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_49960 = sync_50542;\n        chunk_offset_49959 += group_sizze_49945;\n    }\n    res_49957 = x_49960;\n    if (thread_active_51103) {\n        *(__global float *) &mem_50636[(gtid_49941 * sizze_47778 + gtid_49942) *\n                                       4] = res_49957;\n    }\n}\n__kernel void map_kernel_49996(int32_t sizze_47777, int32_t sizze_47779,\n                               int32_t sizze_47780, int32_t sizze_47781,\n                               int32_t sizze_47782, __global\n                               unsigned char *E_mem_50617, __global\n                               unsigned char *mem_50622, __global\n                               unsigned char *mem_50626)\n{\n    int32_t wave_sizze_51097;\n    int32_t group_sizze_51098;\n    bool thread_active_51099;\n    int32_t gtid_49987;\n    int32_t gtid_49988;\n    int32_t global_tid_49996;\n    int32_t local_tid_49997;\n    int32_t gr",
            "oup_id_49998;\n    \n    global_tid_49996 = get_global_id(0);\n    local_tid_49997 = get_local_id(0);\n    group_sizze_51098 = get_local_size(0);\n    wave_sizze_51097 = LOCKSTEP_WIDTH;\n    group_id_49998 = get_group_id(0);\n    gtid_49987 = squot32(global_tid_49996, sizze_47779);\n    gtid_49988 = global_tid_49996 - squot32(global_tid_49996, sizze_47779) *\n        sizze_47779;\n    thread_active_51099 = slt32(gtid_49987, sizze_47777) && slt32(gtid_49988,\n                                                                  sizze_47779);\n    \n    float res_50000;\n    \n    if (thread_active_51099) {\n        float x_50003 = 0.0F;\n        \n        for (int32_t chunk_offset_50002 = 0; chunk_offset_50002 < sizze_47782;\n             chunk_offset_50002++) {\n            float x_50012 = *(__global float *) &mem_50622[(chunk_offset_50002 *\n                                                            (sizze_47780 *\n                                                             sizze_47781) +\n                                                            gtid_49987 *\n                                                            sizze_47781 +\n                                                            gtid_49988) * 4];\n            float x_50013 = *(__global\n                              float *) &E_mem_50617[chunk_offset_50002 * 4];\n            float res_50015 = x_50012 * x_50013;\n            float res_50017 = x_50003 + res_50015;\n            float x_tmp_51100 = res_50017;\n            \n            x_50003 = x_tmp_51100;\n        }\n        res_50000 = x_50003;\n    }\n    if (thread_active_51099) {\n        *(__global float *) &mem_50626[(gtid_49987 * sizze_47779 + gtid_49988) *\n                                       4] = res_50000;\n    }\n}\n__kernel void map_kernel_50042(int32_t sizze_47834, int32_t sizze_47835,\n                               int32_t sizze_47836, int32_t sizze_47838,\n                               __global unsigned char *A_mem_50609, __global\n                               unsigned char",
            " *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50620)\n{\n    int32_t wave_sizze_51114;\n    int32_t group_sizze_51115;\n    bool thread_active_51116;\n    int32_t gtid_50031;\n    int32_t gtid_50032;\n    int32_t gtid_50033;\n    int32_t global_tid_50042;\n    int32_t local_tid_50043;\n    int32_t group_id_50044;\n    \n    global_tid_50042 = get_global_id(0);\n    local_tid_50043 = get_local_id(0);\n    group_sizze_51115 = get_local_size(0);\n    wave_sizze_51114 = LOCKSTEP_WIDTH;\n    group_id_50044 = get_group_id(0);\n    gtid_50031 = squot32(global_tid_50042, sizze_47835 * sizze_47838);\n    gtid_50032 = squot32(global_tid_50042 - squot32(global_tid_50042,\n                                                    sizze_47835 * sizze_47838) *\n                         (sizze_47835 * sizze_47838), sizze_47838);\n    gtid_50033 = global_tid_50042 - squot32(global_tid_50042, sizze_47835 *\n                                            sizze_47838) * (sizze_47835 *\n                                                            sizze_47838) -\n        squot32(global_tid_50042 - squot32(global_tid_50042, sizze_47835 *\n                                           sizze_47838) * (sizze_47835 *\n                                                           sizze_47838),\n                sizze_47838) * sizze_47838;\n    thread_active_51116 = (slt32(gtid_50031, sizze_47834) && slt32(gtid_50032,\n                                                                   sizze_47835)) &&\n        slt32(gtid_50033, sizze_47838);\n    \n    float x_50046;\n    float res_50048;\n    \n    if (thread_active_51116) {\n        x_50046 = *(__global float *) &D_mem_50615[gtid_50032 * 4];\n        \n        float x_50051 = 0.0F;\n        \n        for (int32_t chunk_offset_50050 = 0; chunk_offset_50050 < sizze_47836;\n             chunk_offset_50050++) {\n            float x_50062 = *(",
            "__global float *) &A_mem_50609[(gtid_50031 *\n                                                              (sizze_47835 *\n                                                               sizze_47836) +\n                                                              gtid_50032 *\n                                                              sizze_47836 +\n                                                              chunk_offset_50050) *\n                                                             4];\n            float x_50063 = *(__global\n                              float *) &B_mem_50611[(chunk_offset_50050 *\n                                                     sizze_47838 + gtid_50033) *\n                                                    4];\n            float x_50064 = *(__global\n                              float *) &C_mem_50613[chunk_offset_50050 * 4];\n            float x_50066 = x_50062 * x_50063;\n            float x_50067 = x_50064 * x_50066;\n            float res_50068 = x_50046 * x_50067;\n            float res_50070 = x_50051 + res_50068;\n            float x_tmp_51117 = res_50070;\n            \n            x_50051 = x_tmp_51117;\n        }\n        res_50048 = x_50051;\n    }\n    if (thread_active_51116) {\n        *(__global float *) &mem_50620[(gtid_50031 * (sizze_47835 *\n                                                      sizze_47838) +\n                                        gtid_50032 * sizze_47838 + gtid_50033) *\n                                       4] = res_50048;\n    }\n}\n__kernel void map_kernel_50095(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_47886, int32_t sizze_47887,\n                               int32_t sizze_47888, int32_t sizze_47893,\n                               __global unsigned char *C_mem_50613, __global\n                               unsigned char *mem_50618, __global\n                               unsigned char *mem_50631)\n{\n",
            "    __local volatile char *restrict mem_50622 = mem_aligned_0;\n    __local volatile char *restrict mem_50626 = mem_aligned_1;\n    int32_t wave_sizze_51126;\n    int32_t group_sizze_51127;\n    bool thread_active_51128;\n    int32_t gtid_50084;\n    int32_t gtid_50085;\n    int32_t gtid_50086;\n    int32_t global_tid_50095;\n    int32_t local_tid_50096;\n    int32_t group_id_50097;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t ltid_50532;\n    \n    global_tid_50095 = get_global_id(0);\n    local_tid_50096 = get_local_id(0);\n    group_sizze_51127 = get_local_size(0);\n    wave_sizze_51126 = LOCKSTEP_WIDTH;\n    group_id_50097 = get_group_id(0);\n    gtid_50084 = squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528) + squot32(squot32(global_tid_50095,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47887 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47893 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528));\n    gtid_50085 = squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                    ",
            "                                  tile_sizze_50528),\n                         tile_sizze_50528) + squot32(squot32(global_tid_50095,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528) -\n                                                     squot32(squot32(global_tid_50095,\n                                                                     tile_sizze_50528 *\n                                                                     tile_sizze_50528),\n                                                             squot32(sizze_47887 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528) *\n                                                             squot32(sizze_47893 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528)) *\n                                                     (squot32(sizze_47887 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528) *\n                                                      squot32(sizze_47893 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528)),\n                                                     squot32(sizze_47893 +\n                                                             tile_sizze_50528 -\n                ",
            "                                             1,\n                                                             tile_sizze_50528)) *\n        tile_sizze_50528;\n    gtid_50086 = srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528 +\n        (squot32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528) -\n         squot32(squot32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528),\n                 squot32(sizze_47887 + tile_sizze_50528 - 1, tile_sizze_50528) *\n                 squot32(sizze_47893 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * (squot32(sizze_47887 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528) *\n                                               squot32(sizze_47893 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528)) -\n         squot32(squot32(global_tid_50095, tile_sizze_50528 *\n                         tile_sizze_50528) - squot32(squot32(global_tid_50095,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47887 +\n                 ",
            "                                            tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47893 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n                 (squot32(sizze_47887 + tile_sizze_50528 - 1,\n                          tile_sizze_50528) * squot32(sizze_47893 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)),\n                 squot32(sizze_47893 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * squot32(sizze_47893 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)) *\n        tile_sizze_50528;\n    ltid_50530 = squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528);\n    ltid_50531 = squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528);\n    ltid_50532 = srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                     ",
            "                                   tile_sizze_50528) -\n        squot32(srem32(global_tid_50095, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50095, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528;\n    thread_active_51128 = (slt32(gtid_50084, sizze_47886) && slt32(gtid_50085,\n                                                                   sizze_47887)) &&\n        slt32(gtid_50086, sizze_47893);\n    if (thread_active_51128) { }\n    \n    float res_50100;\n    float x_50103 = 0.0F;\n    int32_t chunk_sizze_50101;\n    int32_t chunk_offset_50102 = 0;\n    \n    while (slt32(chunk_offset_50102, sizze_47888)) {\n        if (slt32(sizze_47888 - chunk_offset_50102, tile_sizze_50528)) {\n            chunk_sizze_50101 = sizze_47888 - chunk_offset_50102;\n        } else {\n            chunk_sizze_50101 = tile_sizze_50528;\n        }\n        for (int32_t comb_iter_51129 = 0; comb_iter_51129 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51129++) {\n            int32_t cid_50548;\n            int32_t cid_50549;\n            int32_t flat_comb_id_51130 = comb_iter_51129 *\n                    tiled_group_sizze_50529 + local_tid_50096;\n            \n            cid_50548 = squot32(flat_comb_id_51130, tile_sizze_50528);\n            cid_50549 = flat_comb_id_51130 - squot32(flat_comb_id_51130,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50548, tile_sizze_50528) && slt32(cid_50549,\n                                                             chunk_sizze_50101)) &&\n                slt32(gtid_50085, sizze_47887)) {\n                float x_chunk_oute",
            "r_elem_50547 = *(__global\n                                                   float *) &mem_50618[(gtid_50084 *\n                                                                        (sizze_47887 *\n                                                                         sizze_47888) +\n                                                                        gtid_50085 *\n                                                                        sizze_47888 +\n                                                                        (chunk_offset_50102 +\n                                                                         ltid_50532)) *\n                                                                       4];\n                \n                *(__local float *) &mem_50622[(cid_50548 * tile_sizze_50528 +\n                                               cid_50549) * 4] =\n                    x_chunk_outer_elem_50547;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51128) { }\n        for (int32_t comb_iter_51131 = 0; comb_iter_51131 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51131++) {\n            int32_t cid_50553;\n            int32_t cid_50554;\n            int32_t flat_comb_id_51132 = comb_iter_51131 *\n                    tiled_group_sizze_50529 + local_tid_50096;\n            \n            cid_50553 = squot32(flat_comb_id_51132, tile_sizze_50528);\n            cid_50554 = flat_comb_id_51132 - squot32(flat_comb_id_51132,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50553, chunk_sizze_50101) && slt32(cid_50554,\n                                                              tile_sizze_50528)) &&\n                slt32(gtid_50086, sizze_47893)) {\n                float x_chunk_outer_elem_50552 = *(__global\n                                     ",
            "              float *) &C_mem_50613[((chunk_offset_50102 +\n                                                                           ltid_50531) *\n                                                                          sizze_47893 +\n                                                                          gtid_50086) *\n                                                                         4];\n                \n                *(__local float *) &mem_50626[(cid_50553 * tile_sizze_50528 +\n                                               cid_50554) * 4] =\n                    x_chunk_outer_elem_50552;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51128) { }\n        \n        float res_50106;\n        float sync_50556;\n        float acc_50109 = x_50103;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_50107 = 1;\n        \n        if (thread_active_51128) {\n            if (chunk_sizze_50101 == tile_sizze_50528) {\n                for (int32_t i_50108 = 0; i_50108 < tile_sizze_50528;\n                     i_50108++) {\n                    float x_50112 = *(__local float *) &mem_50622[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50108) *\n                                                                  4];\n                    float x_50113 = *(__local float *) &mem_50626[(i_50108 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_50115 = x_50112 * x_50113;\n                    float res_50117 = acc_50109 + res_50115;\n                    float acc_tmp_51133 = res_50117;\n                    \n                    acc_50109 = acc_tmp_51133;\n                }\n            } else {\n                for (i",
            "nt32_t i_50108 = 0; i_50108 < chunk_sizze_50101;\n                     i_50108++) {\n                    float x_50112 = *(__local float *) &mem_50622[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50108) *\n                                                                  4];\n                    float x_50113 = *(__local float *) &mem_50626[(i_50108 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_50115 = x_50112 * x_50113;\n                    float res_50117 = acc_50109 + res_50115;\n                    float acc_tmp_51133 = res_50117;\n                    \n                    acc_50109 = acc_tmp_51133;\n                }\n            }\n        }\n        res_50106 = acc_50109;\n        sync_50556 = res_50106;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_50103 = sync_50556;\n        chunk_offset_50102 += tile_sizze_50528;\n    }\n    res_50100 = x_50103;\n    if (thread_active_51128) {\n        *(__global float *) &mem_50631[(gtid_50084 * (sizze_47887 *\n                                                      sizze_47893) +\n                                        gtid_50085 * sizze_47893 + gtid_50086) *\n                                       4] = res_50100;\n    }\n}\n__kernel void map_kernel_50129(int32_t sizze_47886, int32_t sizze_47887,\n                               int32_t sizze_47888, int32_t sizze_47890,\n                               int32_t sizze_47891, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50618)\n{\n    int32_t wave_sizze_51123;\n    int32_t group_sizze_51124;\n    bool thread_active_",
            "51125;\n    int32_t gtid_50118;\n    int32_t gtid_50119;\n    int32_t gtid_50120;\n    int32_t global_tid_50129;\n    int32_t local_tid_50130;\n    int32_t group_id_50131;\n    \n    global_tid_50129 = get_global_id(0);\n    local_tid_50130 = get_local_id(0);\n    group_sizze_51124 = get_local_size(0);\n    wave_sizze_51123 = LOCKSTEP_WIDTH;\n    group_id_50131 = get_group_id(0);\n    gtid_50118 = squot32(global_tid_50129, sizze_47887 * sizze_47888);\n    gtid_50119 = squot32(global_tid_50129 - squot32(global_tid_50129,\n                                                    sizze_47887 * sizze_47888) *\n                         (sizze_47887 * sizze_47888), sizze_47888);\n    gtid_50120 = global_tid_50129 - squot32(global_tid_50129, sizze_47887 *\n                                            sizze_47888) * (sizze_47887 *\n                                                            sizze_47888) -\n        squot32(global_tid_50129 - squot32(global_tid_50129, sizze_47887 *\n                                           sizze_47888) * (sizze_47887 *\n                                                           sizze_47888),\n                sizze_47888) * sizze_47888;\n    thread_active_51125 = (slt32(gtid_50118, sizze_47886) && slt32(gtid_50119,\n                                                                   sizze_47887)) &&\n        slt32(gtid_50120, sizze_47888);\n    \n    float x_50132;\n    float x_50133;\n    float res_50134;\n    \n    if (thread_active_51125) {\n        x_50132 = *(__global float *) &A_mem_50609[(gtid_50118 * (sizze_47887 *\n                                                                  sizze_47888) +\n                                                    gtid_50119 * sizze_47888 +\n                                                    gtid_50120) * 4];\n        x_50133 = *(__global float *) &B_mem_50611[(gtid_50118 * (sizze_47890 *\n                                                                  sizze_47891) +\n                                                    gtid_50119 * sizze_4789",
            "1 +\n                                                    gtid_50120) * 4];\n        res_50134 = x_50132 + x_50133;\n    }\n    if (thread_active_51125) {\n        *(__global float *) &mem_50618[(gtid_50118 * (sizze_47887 *\n                                                      sizze_47888) +\n                                        gtid_50119 * sizze_47888 + gtid_50120) *\n                                       4] = res_50134;\n    }\n}\n__kernel void map_kernel_50144(int32_t sizze_47947, int32_t sizze_47952,\n                               int32_t sizze_47954, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50619)\n{\n    int32_t wave_sizze_51139;\n    int32_t group_sizze_51140;\n    bool thread_active_51141;\n    int32_t gtid_50135;\n    int32_t gtid_50136;\n    int32_t global_tid_50144;\n    int32_t local_tid_50145;\n    int32_t group_id_50146;\n    \n    global_tid_50144 = get_global_id(0);\n    local_tid_50145 = get_local_id(0);\n    group_sizze_51140 = get_local_size(0);\n    wave_sizze_51139 = LOCKSTEP_WIDTH;\n    group_id_50146 = get_group_id(0);\n    gtid_50135 = squot32(global_tid_50144, sizze_47952);\n    gtid_50136 = global_tid_50144 - squot32(global_tid_50144, sizze_47952) *\n        sizze_47952;\n    thread_active_51141 = slt32(gtid_50135, sizze_47947) && slt32(gtid_50136,\n                                                                  sizze_47952);\n    \n    float x_50147;\n    float x_50148;\n    float res_50149;\n    \n    if (thread_active_51141) {\n        x_50147 = *(__global float *) &C_mem_50613[(gtid_50135 * sizze_47952 +\n                                                    gtid_50136) * 4];\n        x_50148 = *(__global float *) &D_mem_50615[(gtid_50135 * sizze_47954 +\n                                                    gtid_50136) * 4];\n        res_50149 = x_50147 + x_50148;\n    }\n    if (thread_active_51141) {\n        *(__global flo",
            "at *) &mem_50619[(gtid_50135 * sizze_47952 + gtid_50136) *\n                                       4] = res_50149;\n    }\n}\n__kernel void map_kernel_50174(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_47945, int32_t sizze_47946,\n                               int32_t sizze_47947, int32_t sizze_47952,\n                               __global unsigned char *mem_50619, __global\n                               unsigned char *mem_50624, __global\n                               unsigned char *mem_50637)\n{\n    __local volatile char *restrict mem_50628 = mem_aligned_0;\n    __local volatile char *restrict mem_50632 = mem_aligned_1;\n    int32_t wave_sizze_51145;\n    int32_t group_sizze_51146;\n    bool thread_active_51147;\n    int32_t gtid_50163;\n    int32_t gtid_50164;\n    int32_t gtid_50165;\n    int32_t global_tid_50174;\n    int32_t local_tid_50175;\n    int32_t group_id_50176;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t ltid_50532;\n    \n    global_tid_50174 = get_global_id(0);\n    local_tid_50175 = get_local_id(0);\n    group_sizze_51146 = get_local_size(0);\n    wave_sizze_51145 = LOCKSTEP_WIDTH;\n    group_id_50176 = get_group_id(0);\n    gtid_50163 = squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528) + squot32(squot32(global_tid_50174,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47946 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squo",
            "t32(sizze_47952 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528));\n    gtid_50164 = squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528) + squot32(squot32(global_tid_50174,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528) -\n                                                     squot32(squot32(global_tid_50174,\n                                                                     tile_sizze_50528 *\n                                                                     tile_sizze_50528),\n                                                             squot32(sizze_47946 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528) *\n                                                             squot32(sizze_47952 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528)) *\n                                                     (squot32(sizze_47946 +\n                                                              tile_sizze_50528 -\n          ",
            "                                                    1,\n                                                              tile_sizze_50528) *\n                                                      squot32(sizze_47952 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528)),\n                                                     squot32(sizze_47952 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n        tile_sizze_50528;\n    gtid_50165 = srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528 +\n        (squot32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528) -\n         squot32(squot32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528),\n                 squot32(sizze_47946 + tile_sizze_50528 - 1, tile_sizze_50528) *\n                 squot32(sizze_47952 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * (squot32(sizze_47946 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528) *\n   ",
            "                                            squot32(sizze_47952 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528)) -\n         squot32(squot32(global_tid_50174, tile_sizze_50528 *\n                         tile_sizze_50528) - squot32(squot32(global_tid_50174,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_47946 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_47952 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n                 (squot32(sizze_47946 + tile_sizze_50528 - 1,\n                          tile_sizze_50528) * squot32(sizze_47952 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)),\n                 squot32(sizze_47952 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * squot32(sizze_47952 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)) *\n        tile_sizze_50528;\n    ltid_50530 = squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528);\n    ltid_50531 = squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                tile_sizz",
            "e_50528) -\n                         squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528);\n    ltid_50532 = srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_50174, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50174, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528;\n    thread_active_51147 = (slt32(gtid_50163, sizze_47945) && slt32(gtid_50164,\n                                                                   sizze_47946)) &&\n        slt32(gtid_50165, sizze_47952);\n    if (thread_active_51147) { }\n    \n    float res_50179;\n    float x_50182 = 0.0F;\n    int32_t chunk_sizze_50180;\n    int32_t chunk_offset_50181 = 0;\n    \n    while (slt32(chunk_offset_50181, sizze_47947)) {\n        if (slt32(sizze_47947 - chunk_offset_50181, tile_sizze_50528)) {\n            chunk_sizze_50180 = sizze_47947 - chunk_offset_50181;\n        } else {\n            chunk_sizze_50180 = tile_sizze_50528;\n        }\n        for (int32_t comb_iter_51148 = 0; comb_iter_51148 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51148++) {\n            int32_t cid_50548;\n            int32_t cid_50549;\n            in",
            "t32_t flat_comb_id_51149 = comb_iter_51148 *\n                    tiled_group_sizze_50529 + local_tid_50175;\n            \n            cid_50548 = squot32(flat_comb_id_51149, tile_sizze_50528);\n            cid_50549 = flat_comb_id_51149 - squot32(flat_comb_id_51149,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50548, tile_sizze_50528) && slt32(cid_50549,\n                                                             chunk_sizze_50180)) &&\n                slt32(gtid_50164, sizze_47946)) {\n                float x_chunk_outer_elem_50547 = *(__global\n                                                   float *) &mem_50624[(gtid_50163 *\n                                                                        (sizze_47946 *\n                                                                         sizze_47947) +\n                                                                        gtid_50164 *\n                                                                        sizze_47947 +\n                                                                        (chunk_offset_50181 +\n                                                                         ltid_50532)) *\n                                                                       4];\n                \n                *(__local float *) &mem_50628[(cid_50548 * tile_sizze_50528 +\n                                               cid_50549) * 4] =\n                    x_chunk_outer_elem_50547;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51147) { }\n        for (int32_t comb_iter_51150 = 0; comb_iter_51150 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51150++) {\n            int32_t cid_50553;\n            int32_t cid_50554;\n            int32_t flat_comb_id_51151 = comb_iter_51150 *\n                   ",
            " tiled_group_sizze_50529 + local_tid_50175;\n            \n            cid_50553 = squot32(flat_comb_id_51151, tile_sizze_50528);\n            cid_50554 = flat_comb_id_51151 - squot32(flat_comb_id_51151,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50553, chunk_sizze_50180) && slt32(cid_50554,\n                                                              tile_sizze_50528)) &&\n                slt32(gtid_50165, sizze_47952)) {\n                float x_chunk_outer_elem_50552 = *(__global\n                                                   float *) &mem_50619[((chunk_offset_50181 +\n                                                                         ltid_50531) *\n                                                                        sizze_47952 +\n                                                                        gtid_50165) *\n                                                                       4];\n                \n                *(__local float *) &mem_50632[(cid_50553 * tile_sizze_50528 +\n                                               cid_50554) * 4] =\n                    x_chunk_outer_elem_50552;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51147) { }\n        \n        float res_50185;\n        float sync_50556;\n        float acc_50188 = x_50182;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_50186 = 1;\n        \n        if (thread_active_51147) {\n            if (chunk_sizze_50180 == tile_sizze_50528) {\n                for (int32_t i_50187 = 0; i_50187 < tile_sizze_50528;\n                     i_50187++) {\n                    float x_50191 = *(__local float *) &mem_50628[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50187) *\n                                                                  4];\n            ",
            "        float x_50192 = *(__local float *) &mem_50632[(i_50187 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_50194 = x_50191 * x_50192;\n                    float res_50196 = acc_50188 + res_50194;\n                    float acc_tmp_51152 = res_50196;\n                    \n                    acc_50188 = acc_tmp_51152;\n                }\n            } else {\n                for (int32_t i_50187 = 0; i_50187 < chunk_sizze_50180;\n                     i_50187++) {\n                    float x_50191 = *(__local float *) &mem_50628[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50187) *\n                                                                  4];\n                    float x_50192 = *(__local float *) &mem_50632[(i_50187 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_50194 = x_50191 * x_50192;\n                    float res_50196 = acc_50188 + res_50194;\n                    float acc_tmp_51152 = res_50196;\n                    \n                    acc_50188 = acc_tmp_51152;\n                }\n            }\n        }\n        res_50185 = acc_50188;\n        sync_50556 = res_50185;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_50182 = sync_50556;\n        chunk_offset_50181 += tile_sizze_50528;\n    }\n    res_50179 = x_50182;\n    if (thread_active_51147) {\n        *(__global float *) &mem_50637[(gtid_50163 * (sizze_47946 *\n                                                      sizze_47952) +\n                            ",
            "            gtid_50164 * sizze_47952 + gtid_50165) *\n                                       4] = res_50179;\n    }\n}\n__kernel void map_kernel_50208(int32_t sizze_47945, int32_t sizze_47946,\n                               int32_t sizze_47947, int32_t sizze_47949,\n                               int32_t sizze_47950, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50624)\n{\n    int32_t wave_sizze_51142;\n    int32_t group_sizze_51143;\n    bool thread_active_51144;\n    int32_t gtid_50197;\n    int32_t gtid_50198;\n    int32_t gtid_50199;\n    int32_t global_tid_50208;\n    int32_t local_tid_50209;\n    int32_t group_id_50210;\n    \n    global_tid_50208 = get_global_id(0);\n    local_tid_50209 = get_local_id(0);\n    group_sizze_51143 = get_local_size(0);\n    wave_sizze_51142 = LOCKSTEP_WIDTH;\n    group_id_50210 = get_group_id(0);\n    gtid_50197 = squot32(global_tid_50208, sizze_47946 * sizze_47947);\n    gtid_50198 = squot32(global_tid_50208 - squot32(global_tid_50208,\n                                                    sizze_47946 * sizze_47947) *\n                         (sizze_47946 * sizze_47947), sizze_47947);\n    gtid_50199 = global_tid_50208 - squot32(global_tid_50208, sizze_47946 *\n                                            sizze_47947) * (sizze_47946 *\n                                                            sizze_47947) -\n        squot32(global_tid_50208 - squot32(global_tid_50208, sizze_47946 *\n                                           sizze_47947) * (sizze_47946 *\n                                                           sizze_47947),\n                sizze_47947) * sizze_47947;\n    thread_active_51144 = (slt32(gtid_50197, sizze_47945) && slt32(gtid_50198,\n                                                                   sizze_47946)) &&\n        slt32(gtid_50199, sizze_47947);\n    \n    float x_50211;\n    float x_50212;\n ",
            "   float res_50213;\n    \n    if (thread_active_51144) {\n        x_50211 = *(__global float *) &A_mem_50609[(gtid_50197 * (sizze_47946 *\n                                                                  sizze_47947) +\n                                                    gtid_50198 * sizze_47947 +\n                                                    gtid_50199) * 4];\n        x_50212 = *(__global float *) &B_mem_50611[(gtid_50197 * (sizze_47949 *\n                                                                  sizze_47950) +\n                                                    gtid_50198 * sizze_47950 +\n                                                    gtid_50199) * 4];\n        res_50213 = x_50211 + x_50212;\n    }\n    if (thread_active_51144) {\n        *(__global float *) &mem_50624[(gtid_50197 * (sizze_47946 *\n                                                      sizze_47947) +\n                                        gtid_50198 * sizze_47947 + gtid_50199) *\n                                       4] = res_50213;\n    }\n}\n__kernel void map_kernel_50223(int32_t sizze_48026, int32_t sizze_48031,\n                               int32_t sizze_48033, float c_48038,\n                               float d_48040, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50619)\n{\n    int32_t wave_sizze_51158;\n    int32_t group_sizze_51159;\n    bool thread_active_51160;\n    int32_t gtid_50214;\n    int32_t gtid_50215;\n    int32_t global_tid_50223;\n    int32_t local_tid_50224;\n    int32_t group_id_50225;\n    \n    global_tid_50223 = get_global_id(0);\n    local_tid_50224 = get_local_id(0);\n    group_sizze_51159 = get_local_size(0);\n    wave_sizze_51158 = LOCKSTEP_WIDTH;\n    group_id_50225 = get_group_id(0);\n    gtid_50214 = squot32(global_tid_50223, sizze_48031);\n    gtid_50215 = global_tid_50223 - squot32(global_tid_50223, sizze_48031) *\n        sizz",
            "e_48031;\n    thread_active_51160 = slt32(gtid_50214, sizze_48026) && slt32(gtid_50215,\n                                                                  sizze_48031);\n    \n    float x_50226;\n    float x_50227;\n    float res_50228;\n    float res_50229;\n    float res_50230;\n    \n    if (thread_active_51160) {\n        x_50226 = *(__global float *) &C_mem_50613[(gtid_50214 * sizze_48031 +\n                                                    gtid_50215) * 4];\n        x_50227 = *(__global float *) &D_mem_50615[(gtid_50214 * sizze_48033 +\n                                                    gtid_50215) * 4];\n        res_50228 = c_48038 * x_50226;\n        res_50229 = d_48040 * x_50227;\n        res_50230 = res_50228 + res_50229;\n    }\n    if (thread_active_51160) {\n        *(__global float *) &mem_50619[(gtid_50214 * sizze_48031 + gtid_50215) *\n                                       4] = res_50230;\n    }\n}\n__kernel void map_kernel_50255(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_48024, int32_t sizze_48025,\n                               int32_t sizze_48026, int32_t sizze_48031,\n                               __global unsigned char *mem_50619, __global\n                               unsigned char *mem_50624, __global\n                               unsigned char *mem_50637)\n{\n    __local volatile char *restrict mem_50628 = mem_aligned_0;\n    __local volatile char *restrict mem_50632 = mem_aligned_1;\n    int32_t wave_sizze_51164;\n    int32_t group_sizze_51165;\n    bool thread_active_51166;\n    int32_t gtid_50244;\n    int32_t gtid_50245;\n    int32_t gtid_50246;\n    int32_t global_tid_50255;\n    int32_t local_tid_50256;\n    int32_t group_id_50257;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t ltid_50532;\n    \n    global_tid_50255 = get_global_id(0);\n    local_tid_50256 = get_local_id(0);\n    group_sizze_51165 = get_local_size(0);\n    wave_sizze_51164 = LO",
            "CKSTEP_WIDTH;\n    group_id_50257 = get_group_id(0);\n    gtid_50244 = squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528) + squot32(squot32(global_tid_50255,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_48025 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_48031 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528));\n    gtid_50245 = squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528) + squot32(squot32(global_tid_50255,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528) -\n                                                     squot32(squot32(global_tid_50255,\n                                                                     tile_sizze_50528 *\n                                                                     tile_sizze_50528),\n                                             ",
            "                squot32(sizze_48025 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528) *\n                                                             squot32(sizze_48031 +\n                                                                     tile_sizze_50528 -\n                                                                     1,\n                                                                     tile_sizze_50528)) *\n                                                     (squot32(sizze_48025 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528) *\n                                                      squot32(sizze_48031 +\n                                                              tile_sizze_50528 -\n                                                              1,\n                                                              tile_sizze_50528)),\n                                                     squot32(sizze_48031 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n        tile_sizze_50528;\n    gtid_50246 = srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50255, tile_",
            "sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528 +\n        (squot32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528) -\n         squot32(squot32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528),\n                 squot32(sizze_48025 + tile_sizze_50528 - 1, tile_sizze_50528) *\n                 squot32(sizze_48031 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * (squot32(sizze_48025 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528) *\n                                               squot32(sizze_48031 +\n                                                       tile_sizze_50528 - 1,\n                                                       tile_sizze_50528)) -\n         squot32(squot32(global_tid_50255, tile_sizze_50528 *\n                         tile_sizze_50528) - squot32(squot32(global_tid_50255,\n                                                             tile_sizze_50528 *\n                                                             tile_sizze_50528),\n                                                     squot32(sizze_48025 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528) *\n                                                     squot32(sizze_48031 +\n                                                             tile_sizze_50528 -\n                                                             1,\n                                                             tile_sizze_50528)) *\n                 (squot32(sizze_48025 + tile_sizze_50528 - 1,\n                          tile",
            "_sizze_50528) * squot32(sizze_48031 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)),\n                 squot32(sizze_48031 + tile_sizze_50528 - 1,\n                         tile_sizze_50528)) * squot32(sizze_48031 +\n                                                      tile_sizze_50528 - 1,\n                                                      tile_sizze_50528)) *\n        tile_sizze_50528;\n    ltid_50530 = squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                tile_sizze_50528), tile_sizze_50528 *\n                         tile_sizze_50528);\n    ltid_50531 = squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                tile_sizze_50528) -\n                         squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                                        tile_sizze_50528), tile_sizze_50528 *\n                                 tile_sizze_50528) * (tile_sizze_50528 *\n                                                      tile_sizze_50528),\n                         tile_sizze_50528);\n    ltid_50532 = srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528) -\n        squot32(srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528),\n                tile_sizze_50528 * tile_sizze_50528) * (tile_sizze_50528 *\n                                                        tile_sizze_50528) -\n        squot32(srem32(global_tid_50255, tile_sizze_50528 * tile_sizze_50528) -\n                squot32(srem32(global_tid_50255, tile_sizze_50528 *\n                               tile_sizze_50528), tile_sizze_50528 *\n                        tile_sizze_50528) * (tile_sizze_50528 *\n                                             tile_sizze_50528),\n                tile_sizze_50528) * tile_sizze_50528;\n    thread_active_51166 = (slt32(gtid_50244, sizze_48024) && slt32(gtid_50245,\n                                                                   s",
            "izze_48025)) &&\n        slt32(gtid_50246, sizze_48031);\n    if (thread_active_51166) { }\n    \n    float res_50260;\n    float x_50263 = 0.0F;\n    int32_t chunk_sizze_50261;\n    int32_t chunk_offset_50262 = 0;\n    \n    while (slt32(chunk_offset_50262, sizze_48026)) {\n        if (slt32(sizze_48026 - chunk_offset_50262, tile_sizze_50528)) {\n            chunk_sizze_50261 = sizze_48026 - chunk_offset_50262;\n        } else {\n            chunk_sizze_50261 = tile_sizze_50528;\n        }\n        for (int32_t comb_iter_51167 = 0; comb_iter_51167 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51167++) {\n            int32_t cid_50548;\n            int32_t cid_50549;\n            int32_t flat_comb_id_51168 = comb_iter_51167 *\n                    tiled_group_sizze_50529 + local_tid_50256;\n            \n            cid_50548 = squot32(flat_comb_id_51168, tile_sizze_50528);\n            cid_50549 = flat_comb_id_51168 - squot32(flat_comb_id_51168,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50548, tile_sizze_50528) && slt32(cid_50549,\n                                                             chunk_sizze_50261)) &&\n                slt32(gtid_50245, sizze_48025)) {\n                float x_chunk_outer_elem_50547 = *(__global\n                                                   float *) &mem_50624[(gtid_50244 *\n                                                                        (sizze_48025 *\n                                                                         sizze_48026) +\n                                                                        gtid_50245 *\n                                                                        sizze_48026 +\n                                                                        (chunk_offset_50262 +\n                                                  ",
            "                       ltid_50532)) *\n                                                                       4];\n                \n                *(__local float *) &mem_50628[(cid_50548 * tile_sizze_50528 +\n                                               cid_50549) * 4] =\n                    x_chunk_outer_elem_50547;\n            }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51166) { }\n        for (int32_t comb_iter_51169 = 0; comb_iter_51169 <\n             squot32(tile_sizze_50528 * tile_sizze_50528 +\n                     tiled_group_sizze_50529 - 1, tiled_group_sizze_50529);\n             comb_iter_51169++) {\n            int32_t cid_50553;\n            int32_t cid_50554;\n            int32_t flat_comb_id_51170 = comb_iter_51169 *\n                    tiled_group_sizze_50529 + local_tid_50256;\n            \n            cid_50553 = squot32(flat_comb_id_51170, tile_sizze_50528);\n            cid_50554 = flat_comb_id_51170 - squot32(flat_comb_id_51170,\n                                                     tile_sizze_50528) *\n                tile_sizze_50528;\n            if ((slt32(cid_50553, chunk_sizze_50261) && slt32(cid_50554,\n                                                              tile_sizze_50528)) &&\n                slt32(gtid_50246, sizze_48031)) {\n                float x_chunk_outer_elem_50552 = *(__global\n                                                   float *) &mem_50619[((chunk_offset_50262 +\n                                                                         ltid_50531) *\n                                                                        sizze_48031 +\n                                                                        gtid_50246) *\n                                                                       4];\n                \n                *(__local float *) &mem_50632[(cid_50553 * tile_sizze_50528 +\n                                               cid_50554) * 4] =\n                    x_chunk_outer_elem_50552;\n    ",
            "        }\n        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n        if (thread_active_51166) { }\n        \n        float res_50266;\n        float sync_50556;\n        float acc_50269 = x_50263;\n        int32_t groupstream_mapaccum_dummy_chunk_sizze_50267 = 1;\n        \n        if (thread_active_51166) {\n            if (chunk_sizze_50261 == tile_sizze_50528) {\n                for (int32_t i_50268 = 0; i_50268 < tile_sizze_50528;\n                     i_50268++) {\n                    float x_50272 = *(__local float *) &mem_50628[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50268) *\n                                                                  4];\n                    float x_50273 = *(__local float *) &mem_50632[(i_50268 *\n                                                                   tile_sizze_50528 +\n                                                                   ltid_50532) *\n                                                                  4];\n                    float res_50275 = x_50272 * x_50273;\n                    float res_50277 = acc_50269 + res_50275;\n                    float acc_tmp_51171 = res_50277;\n                    \n                    acc_50269 = acc_tmp_51171;\n                }\n            } else {\n                for (int32_t i_50268 = 0; i_50268 < chunk_sizze_50261;\n                     i_50268++) {\n                    float x_50272 = *(__local float *) &mem_50628[(ltid_50531 *\n                                                                   tile_sizze_50528 +\n                                                                   i_50268) *\n                                                                  4];\n                    float x_50273 = *(__local float *) &mem_50632[(i_50268 *\n                                                                   tile_sizze_50528 +\n                                                   ",
            "                ltid_50532) *\n                                                                  4];\n                    float res_50275 = x_50272 * x_50273;\n                    float res_50277 = acc_50269 + res_50275;\n                    float acc_tmp_51171 = res_50277;\n                    \n                    acc_50269 = acc_tmp_51171;\n                }\n            }\n        }\n        res_50266 = acc_50269;\n        sync_50556 = res_50266;\n        barrier(CLK_LOCAL_MEM_FENCE);\n        x_50263 = sync_50556;\n        chunk_offset_50262 += tile_sizze_50528;\n    }\n    res_50260 = x_50263;\n    if (thread_active_51166) {\n        *(__global float *) &mem_50637[(gtid_50244 * (sizze_48025 *\n                                                      sizze_48031) +\n                                        gtid_50245 * sizze_48031 + gtid_50246) *\n                                       4] = res_50260;\n    }\n}\n__kernel void map_kernel_50289(int32_t sizze_48024, int32_t sizze_48025,\n                               int32_t sizze_48026, int32_t sizze_48028,\n                               int32_t sizze_48029, float a_48034,\n                               float b_48036, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50624)\n{\n    int32_t wave_sizze_51161;\n    int32_t group_sizze_51162;\n    bool thread_active_51163;\n    int32_t gtid_50278;\n    int32_t gtid_50279;\n    int32_t gtid_50280;\n    int32_t global_tid_50289;\n    int32_t local_tid_50290;\n    int32_t group_id_50291;\n    \n    global_tid_50289 = get_global_id(0);\n    local_tid_50290 = get_local_id(0);\n    group_sizze_51162 = get_local_size(0);\n    wave_sizze_51161 = LOCKSTEP_WIDTH;\n    group_id_50291 = get_group_id(0);\n    gtid_50278 = squot32(global_tid_50289, sizze_48025 * sizze_48026);\n    gtid_50279 = squot32(global_tid_50289 - squot32(global_tid_50289,\n                                      ",
            "              sizze_48025 * sizze_48026) *\n                         (sizze_48025 * sizze_48026), sizze_48026);\n    gtid_50280 = global_tid_50289 - squot32(global_tid_50289, sizze_48025 *\n                                            sizze_48026) * (sizze_48025 *\n                                                            sizze_48026) -\n        squot32(global_tid_50289 - squot32(global_tid_50289, sizze_48025 *\n                                           sizze_48026) * (sizze_48025 *\n                                                           sizze_48026),\n                sizze_48026) * sizze_48026;\n    thread_active_51163 = (slt32(gtid_50278, sizze_48024) && slt32(gtid_50279,\n                                                                   sizze_48025)) &&\n        slt32(gtid_50280, sizze_48026);\n    \n    float x_50292;\n    float x_50293;\n    float res_50294;\n    float res_50295;\n    float res_50296;\n    \n    if (thread_active_51163) {\n        x_50292 = *(__global float *) &A_mem_50609[(gtid_50278 * (sizze_48025 *\n                                                                  sizze_48026) +\n                                                    gtid_50279 * sizze_48026 +\n                                                    gtid_50280) * 4];\n        x_50293 = *(__global float *) &B_mem_50611[(gtid_50278 * (sizze_48028 *\n                                                                  sizze_48029) +\n                                                    gtid_50279 * sizze_48029 +\n                                                    gtid_50280) * 4];\n        res_50294 = a_48034 * x_50292;\n        res_50295 = b_48036 * x_50293;\n        res_50296 = res_50294 + res_50295;\n    }\n    if (thread_active_51163) {\n        *(__global float *) &mem_50624[(gtid_50278 * (sizze_48025 *\n                                                      sizze_48026) +\n                                        gtid_50279 * sizze_48026 + gtid_50280) *\n                                       4] = res_50296;\n   ",
            " }\n}\n__kernel void map_kernel_50304(int32_t sizze_48113, float e_48133,\n                               float f_48135, __global\n                               unsigned char *E_mem_50617, __global\n                               unsigned char *F_mem_50619, __global\n                               unsigned char *mem_50630)\n{\n    int32_t wave_sizze_51180;\n    int32_t group_sizze_51181;\n    bool thread_active_51182;\n    int32_t gtid_50297;\n    int32_t global_tid_50304;\n    int32_t local_tid_50305;\n    int32_t group_id_50306;\n    \n    global_tid_50304 = get_global_id(0);\n    local_tid_50305 = get_local_id(0);\n    group_sizze_51181 = get_local_size(0);\n    wave_sizze_51180 = LOCKSTEP_WIDTH;\n    group_id_50306 = get_group_id(0);\n    gtid_50297 = global_tid_50304;\n    thread_active_51182 = slt32(gtid_50297, sizze_48113);\n    \n    float x_50307;\n    float x_50308;\n    float res_50309;\n    float res_50310;\n    float res_50311;\n    \n    if (thread_active_51182) {\n        x_50307 = *(__global float *) &E_mem_50617[gtid_50297 * 4];\n        x_50308 = *(__global float *) &F_mem_50619[gtid_50297 * 4];\n        res_50309 = e_48133 * x_50307;\n        res_50310 = f_48135 * x_50308;\n        res_50311 = res_50309 + res_50310;\n    }\n    if (thread_active_51182) {\n        *(__global float *) &mem_50630[gtid_50297 * 4] = res_50311;\n    }\n}\n__kernel void map_kernel_50321(int32_t sizze_48113, int32_t sizze_48118,\n                               int32_t sizze_48120, float c_48129,\n                               float d_48131, __global\n                               unsigned char *C_mem_50613, __global\n                               unsigned char *D_mem_50615, __global\n                               unsigned char *mem_50627)\n{\n    int32_t wave_sizze_51177;\n    int32_t group_sizze_51178;\n    bool thread_active_51179;\n    int32_t gtid_50312;\n    int32_t gtid_50313;\n    int32_t global_tid_50321;\n    int32_t local_tid_50322;\n    int32_t group_id_50323;\n    \n    global_tid_50321 = get_global_id(0);\n    ",
            "local_tid_50322 = get_local_id(0);\n    group_sizze_51178 = get_local_size(0);\n    wave_sizze_51177 = LOCKSTEP_WIDTH;\n    group_id_50323 = get_group_id(0);\n    gtid_50312 = squot32(global_tid_50321, sizze_48118);\n    gtid_50313 = global_tid_50321 - squot32(global_tid_50321, sizze_48118) *\n        sizze_48118;\n    thread_active_51179 = slt32(gtid_50312, sizze_48113) && slt32(gtid_50313,\n                                                                  sizze_48118);\n    \n    float x_50324;\n    float x_50325;\n    float res_50326;\n    float res_50327;\n    float res_50328;\n    \n    if (thread_active_51179) {\n        x_50324 = *(__global float *) &C_mem_50613[(gtid_50312 * sizze_48118 +\n                                                    gtid_50313) * 4];\n        x_50325 = *(__global float *) &D_mem_50615[(gtid_50312 * sizze_48120 +\n                                                    gtid_50313) * 4];\n        res_50326 = c_48129 * x_50324;\n        res_50327 = d_48131 * x_50325;\n        res_50328 = res_50326 + res_50327;\n    }\n    if (thread_active_51179) {\n        *(__global float *) &mem_50627[(gtid_50312 * sizze_48118 + gtid_50313) *\n                                       4] = res_50328;\n    }\n}\n__kernel void map_kernel_50336(int32_t sizze_48112, float g_48137,\n                               float h_48139, __global\n                               unsigned char *G_mem_50621, __global\n                               unsigned char *H_mem_50623, __global\n                               unsigned char *mem_50633)\n{\n    int32_t wave_sizze_51183;\n    int32_t group_sizze_51184;\n    bool thread_active_51185;\n    int32_t gtid_50329;\n    int32_t global_tid_50336;\n    int32_t local_tid_50337;\n    int32_t group_id_50338;\n    \n    global_tid_50336 = get_global_id(0);\n    local_tid_50337 = get_local_id(0);\n    group_sizze_51184 = get_local_size(0);\n    wave_sizze_51183 = LOCKSTEP_WIDTH;\n    group_id_50338 = get_group_id(0);\n    gtid_50329 = global_tid_50336;\n    thread_active_51185 = slt32",
            "(gtid_50329, sizze_48112);\n    \n    float x_50339;\n    float x_50340;\n    float res_50341;\n    float res_50342;\n    float res_50343;\n    \n    if (thread_active_51185) {\n        x_50339 = *(__global float *) &G_mem_50621[gtid_50329 * 4];\n        x_50340 = *(__global float *) &H_mem_50623[gtid_50329 * 4];\n        res_50341 = g_48137 * x_50339;\n        res_50342 = h_48139 * x_50340;\n        res_50343 = res_50341 + res_50342;\n    }\n    if (thread_active_51185) {\n        *(__global float *) &mem_50633[gtid_50329 * 4] = res_50343;\n    }\n}\n__kernel void map_kernel_50370(int32_t sizze_48111, int32_t sizze_48112,\n                               int32_t sizze_48113, int32_t sizze_48118,\n                               __global unsigned char *mem_50627, __global\n                               unsigned char *mem_50630, __global\n                               unsigned char *mem_50633, __global\n                               unsigned char *mem_50638, __global\n                               unsigned char *mem_50643)\n{\n    int32_t wave_sizze_51189;\n    int32_t group_sizze_51190;\n    bool thread_active_51191;\n    int32_t gtid_50359;\n    int32_t gtid_50360;\n    int32_t gtid_50361;\n    int32_t global_tid_50370;\n    int32_t local_tid_50371;\n    int32_t group_id_50372;\n    \n    global_tid_50370 = get_global_id(0);\n    local_tid_50371 = get_local_id(0);\n    group_sizze_51190 = get_local_size(0);\n    wave_sizze_51189 = LOCKSTEP_WIDTH;\n    group_id_50372 = get_group_id(0);\n    gtid_50359 = squot32(global_tid_50370, sizze_48112 * sizze_48118);\n    gtid_50360 = squot32(global_tid_50370 - squot32(global_tid_50370,\n                                                    sizze_48112 * sizze_48118) *\n                         (sizze_48112 * sizze_48118), sizze_48118);\n    gtid_50361 = global_tid_50370 - squot32(global_tid_50370, sizze_48112 *\n                                            sizze_48118) * (sizze_48112 *\n                                                            sizze_48118) -\n        squot",
            "32(global_tid_50370 - squot32(global_tid_50370, sizze_48112 *\n                                           sizze_48118) * (sizze_48112 *\n                                                           sizze_48118),\n                sizze_48118) * sizze_48118;\n    thread_active_51191 = (slt32(gtid_50359, sizze_48111) && slt32(gtid_50360,\n                                                                   sizze_48112)) &&\n        slt32(gtid_50361, sizze_48118);\n    \n    float x_50373;\n    float res_50376;\n    \n    if (thread_active_51191) {\n        x_50373 = *(__global float *) &mem_50633[gtid_50360 * 4];\n        \n        float x_50379 = 0.0F;\n        \n        for (int32_t chunk_offset_50378 = 0; chunk_offset_50378 < sizze_48113;\n             chunk_offset_50378++) {\n            float x_50390 = *(__global float *) &mem_50638[(gtid_50359 *\n                                                            (sizze_48112 *\n                                                             sizze_48113) +\n                                                            gtid_50360 *\n                                                            sizze_48113 +\n                                                            chunk_offset_50378) *\n                                                           4];\n            float x_50391 = *(__global float *) &mem_50627[(chunk_offset_50378 *\n                                                            sizze_48118 +\n                                                            gtid_50361) * 4];\n            float x_50392 = *(__global float *) &mem_50630[chunk_offset_50378 *\n                                                           4];\n            float x_50394 = x_50390 * x_50391;\n            float x_50395 = x_50392 * x_50394;\n            float res_50396 = x_50373 * x_50395;\n            float res_50398 = x_50379 + res_50396;\n            float x_tmp_51192 = res_50398;\n            \n            x_50379 = x_tmp_51192;\n        }\n        res_50376 = x_50379;\n    }\n    if (thre",
            "ad_active_51191) {\n        *(__global float *) &mem_50643[(gtid_50359 * (sizze_48112 *\n                                                      sizze_48118) +\n                                        gtid_50360 * sizze_48118 + gtid_50361) *\n                                       4] = res_50376;\n    }\n}\n__kernel void map_kernel_50410(int32_t sizze_48111, int32_t sizze_48112,\n                               int32_t sizze_48113, int32_t sizze_48115,\n                               int32_t sizze_48116, float a_48125,\n                               float b_48127, __global\n                               unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *mem_50638)\n{\n    int32_t wave_sizze_51186;\n    int32_t group_sizze_51187;\n    bool thread_active_51188;\n    int32_t gtid_50399;\n    int32_t gtid_50400;\n    int32_t gtid_50401;\n    int32_t global_tid_50410;\n    int32_t local_tid_50411;\n    int32_t group_id_50412;\n    \n    global_tid_50410 = get_global_id(0);\n    local_tid_50411 = get_local_id(0);\n    group_sizze_51187 = get_local_size(0);\n    wave_sizze_51186 = LOCKSTEP_WIDTH;\n    group_id_50412 = get_group_id(0);\n    gtid_50399 = squot32(global_tid_50410, sizze_48112 * sizze_48113);\n    gtid_50400 = squot32(global_tid_50410 - squot32(global_tid_50410,\n                                                    sizze_48112 * sizze_48113) *\n                         (sizze_48112 * sizze_48113), sizze_48113);\n    gtid_50401 = global_tid_50410 - squot32(global_tid_50410, sizze_48112 *\n                                            sizze_48113) * (sizze_48112 *\n                                                            sizze_48113) -\n        squot32(global_tid_50410 - squot32(global_tid_50410, sizze_48112 *\n                                           sizze_48113) * (sizze_48112 *\n                                                           sizze_48113),\n                sizze_48113) * sizze_48113;\n  ",
            "  thread_active_51188 = (slt32(gtid_50399, sizze_48111) && slt32(gtid_50400,\n                                                                   sizze_48112)) &&\n        slt32(gtid_50401, sizze_48113);\n    \n    float x_50413;\n    float x_50414;\n    float res_50415;\n    float res_50416;\n    float res_50417;\n    \n    if (thread_active_51188) {\n        x_50413 = *(__global float *) &A_mem_50609[(gtid_50399 * (sizze_48112 *\n                                                                  sizze_48113) +\n                                                    gtid_50400 * sizze_48113 +\n                                                    gtid_50401) * 4];\n        x_50414 = *(__global float *) &B_mem_50611[(gtid_50399 * (sizze_48115 *\n                                                                  sizze_48116) +\n                                                    gtid_50400 * sizze_48116 +\n                                                    gtid_50401) * 4];\n        res_50415 = a_48125 * x_50413;\n        res_50416 = b_48127 * x_50414;\n        res_50417 = res_50415 + res_50416;\n    }\n    if (thread_active_51188) {\n        *(__global float *) &mem_50638[(gtid_50399 * (sizze_48112 *\n                                                      sizze_48113) +\n                                        gtid_50400 * sizze_48113 + gtid_50401) *\n                                       4] = res_50417;\n    }\n}\n__kernel void map_kernel_50451(__local volatile int64_t *mem_aligned_0,\n                               __local volatile int64_t *mem_aligned_1,\n                               int32_t sizze_48250, int32_t sizze_48251,\n                               int32_t sizze_48252, int32_t sizze_48254,\n                               int32_t sizze_48256, int32_t inner_ldim_50533,\n                               __global unsigned char *A_mem_50609, __global\n                               unsigned char *B_mem_50611, __global\n                               unsigned char *C_mem_50613, __global\n                 ",
            "              unsigned char *mem_50624)\n{\n    __local volatile char *restrict mem_50616 = mem_aligned_0;\n    __local volatile char *restrict mem_50619 = mem_aligned_1;\n    int32_t wave_sizze_51198;\n    int32_t group_sizze_51199;\n    bool thread_active_51200;\n    int32_t gtid_50440;\n    int32_t gtid_50441;\n    int32_t gtid_50442;\n    int32_t global_tid_50451;\n    int32_t local_tid_50452;\n    int32_t group_id_50453;\n    int32_t ltid_50530;\n    int32_t ltid_50531;\n    int32_t inner_ltid_50532;\n    \n    global_tid_50451 = get_global_id(0);\n    local_tid_50452 = get_local_id(0);\n    group_sizze_51199 = get_local_size(0);\n    wave_sizze_51198 = LOCKSTEP_WIDTH;\n    group_id_50453 = get_group_id(0);\n    gtid_50440 = squot32(srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                                inner_ldim_50533), sizze_48254 *\n                         inner_ldim_50533) + squot32(squot32(global_tid_50451,\n                                                             sizze_48250 *\n                                                             sizze_48254 *\n                                                             inner_ldim_50533),\n                                                     squot32(sizze_48254 +\n                                                             sizze_48254 - 1,\n                                                             sizze_48254) *\n                                                     squot32(sizze_48256 +\n                                                             inner_ldim_50533 -\n                                                             1,\n                                                             inner_ldim_50533)) *\n        sizze_48250;\n    gtid_50441 = squot32(srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                                inner_ldim_50533) -\n                         squot32(srem32(global_tid_50451, sizze_48250 *\n                                        sizze_48254 * inner_ldim_50533),\n                     ",
            "            sizze_48254 * inner_ldim_50533) *\n                         (sizze_48254 * inner_ldim_50533), inner_ldim_50533) +\n        squot32(squot32(global_tid_50451, sizze_48250 * sizze_48254 *\n                        inner_ldim_50533) - squot32(squot32(global_tid_50451,\n                                                            sizze_48250 *\n                                                            sizze_48254 *\n                                                            inner_ldim_50533),\n                                                    squot32(sizze_48254 +\n                                                            sizze_48254 - 1,\n                                                            sizze_48254) *\n                                                    squot32(sizze_48256 +\n                                                            inner_ldim_50533 -\n                                                            1,\n                                                            inner_ldim_50533)) *\n                (squot32(sizze_48254 + sizze_48254 - 1, sizze_48254) *\n                 squot32(sizze_48256 + inner_ldim_50533 - 1, inner_ldim_50533)),\n                squot32(sizze_48256 + inner_ldim_50533 - 1, inner_ldim_50533)) *\n        sizze_48254;\n    gtid_50442 = srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                        inner_ldim_50533) - squot32(srem32(global_tid_50451,\n                                                           sizze_48250 *\n                                                           sizze_48254 *\n                                                           inner_ldim_50533),\n                                                    sizze_48254 *\n                                                    inner_ldim_50533) *\n        (sizze_48254 * inner_ldim_50533) - squot32(srem32(global_tid_50451,\n                                                          sizze_48250 *\n                                                          sizze_48",
            "254 *\n                                                          inner_ldim_50533) -\n                                                   squot32(srem32(global_tid_50451,\n                                                                  sizze_48250 *\n                                                                  sizze_48254 *\n                                                                  inner_ldim_50533),\n                                                           sizze_48254 *\n                                                           inner_ldim_50533) *\n                                                   (sizze_48254 *\n                                                    inner_ldim_50533),\n                                                   inner_ldim_50533) *\n        inner_ldim_50533 + (squot32(global_tid_50451, sizze_48250 *\n                                    sizze_48254 * inner_ldim_50533) -\n                            squot32(squot32(global_tid_50451, sizze_48250 *\n                                            sizze_48254 * inner_ldim_50533),\n                                    squot32(sizze_48254 + sizze_48254 - 1,\n                                            sizze_48254) * squot32(sizze_48256 +\n                                                                   inner_ldim_50533 -\n                                                                   1,\n                                                                   inner_ldim_50533)) *\n                            (squot32(sizze_48254 + sizze_48254 - 1,\n                                     sizze_48254) * squot32(sizze_48256 +\n                                                            inner_ldim_50533 -\n                                                            1,\n                                                            inner_ldim_50533)) -\n                            squot32(squot32(global_tid_50451, sizze_48250 *\n                                            sizze_48254 * inner_ldim_50533) -\n                 ",
            "                   squot32(squot32(global_tid_50451,\n                                                    sizze_48250 * sizze_48254 *\n                                                    inner_ldim_50533),\n                                            squot32(sizze_48254 + sizze_48254 -\n                                                    1, sizze_48254) *\n                                            squot32(sizze_48256 +\n                                                    inner_ldim_50533 - 1,\n                                                    inner_ldim_50533)) *\n                                    (squot32(sizze_48254 + sizze_48254 - 1,\n                                             sizze_48254) *\n                                     squot32(sizze_48256 + inner_ldim_50533 - 1,\n                                             inner_ldim_50533)),\n                                    squot32(sizze_48256 + inner_ldim_50533 - 1,\n                                            inner_ldim_50533)) *\n                            squot32(sizze_48256 + inner_ldim_50533 - 1,\n                                    inner_ldim_50533)) * inner_ldim_50533;\n    ltid_50530 = squot32(srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                                inner_ldim_50533), sizze_48254 *\n                         inner_ldim_50533);\n    ltid_50531 = squot32(srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                                inner_ldim_50533) -\n                         squot32(srem32(global_tid_50451, sizze_48250 *\n                                        sizze_48254 * inner_ldim_50533),\n                                 sizze_48254 * inner_ldim_50533) *\n                         (sizze_48254 * inner_ldim_50533), inner_ldim_50533);\n    inner_ltid_50532 = srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                              inner_ldim_50533) -\n        squot32(srem32(global_tid_50451, sizze_48250 * sizze_48254 *\n                       inner_ldim_50533), sizze_48254 * i",
            "nner_ldim_50533) *\n        (sizze_48254 * inner_ldim_50533) - squot32(srem32(global_tid_50451,\n                                                          sizze_48250 *\n                                                          sizze_48254 *\n                                                          inner_ldim_50533) -\n                                                   squot32(srem32(global_tid_50451,\n                                                                  sizze_48250 *\n                                                                  sizze_48254 *\n                                                                  inner_ldim_50533),\n                                                           sizze_48254 *\n                                                           inner_ldim_50533) *\n                                                   (sizze_48254 *\n                                                    inner_ldim_50533),\n                                                   inner_ldim_50533) *\n        inner_ldim_50533;\n    thread_active_51200 = (slt32(gtid_50440, sizze_48250) && slt32(gtid_50441,\n                                                                   sizze_48254)) &&\n        slt32(gtid_50442, sizze_48256);\n    \n    float res_50457;\n    \n    if (thread_active_51200) {\n        float x_50460 = 0.0F;\n        \n        for (int32_t chunk_offset_50459 = 0; chunk_offset_50459 < sizze_48251;\n             chunk_offset_50459++) {\n            float x_50470 = *(__global\n                              float *) &C_mem_50613[(chunk_offset_50459 *\n                                                     sizze_48256 + gtid_50442) *\n                                                    4];\n            float res_50472;\n            float x_50475 = 0.0F;\n            int32_t chunk_sizze_50473;\n            int32_t chunk_offset_50474 = 0;\n            \n            while (slt32(chunk_offset_50474, sizze_48252)) {\n                if (slt32(sizze_48252 - chunk_offset_50474,\n                  ",
            "        group_sizze_50446)) {\n                    chunk_sizze_50473 = sizze_48252 - chunk_offset_50474;\n                } else {\n                    chunk_sizze_50473 = group_sizze_50446;\n                }\n                \n                float res_50478;\n                float sync_50555;\n                \n                for (int32_t comb_iter_51202 = 0; comb_iter_51202 <\n                     squot32(group_sizze_50446 + inner_ldim_50533 - 1,\n                             inner_ldim_50533); comb_iter_51202++) {\n                    int32_t cid_50529;\n                    int32_t flat_comb_id_51203 = comb_iter_51202 *\n                            inner_ldim_50533 + local_tid_50452;\n                    \n                    cid_50529 = flat_comb_id_51203;\n                    if (slt32(cid_50529, chunk_sizze_50473) && 1) {\n                        float x_chunk_outer_elem_50528 = *(__global\n                                                           float *) &A_mem_50609[(gtid_50440 *\n                                                                                  (sizze_48251 *\n                                                                                   sizze_48252) +\n                                                                                  chunk_offset_50459 *\n                                                                                  sizze_48252 +\n                                                                                  (chunk_offset_50474 +\n                                                                                   local_tid_50452)) *\n                                                                                 4];\n                        \n                        *(__local float *) &mem_50616[cid_50529 * 4] =\n                            x_chunk_outer_elem_50528;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                for (int32_t comb_iter_51204 = 0; comb_iter_51204 <\n                     ",
            "squot32(group_sizze_50446 + inner_ldim_50533 - 1,\n                             inner_ldim_50533); comb_iter_51204++) {\n                    int32_t cid_50543;\n                    int32_t flat_comb_id_51205 = comb_iter_51204 *\n                            inner_ldim_50533 + local_tid_50452;\n                    \n                    cid_50543 = flat_comb_id_51205;\n                    if (slt32(cid_50543, chunk_sizze_50473) && 1) {\n                        float x_chunk_outer_elem_50542 = *(__global\n                                                           float *) &B_mem_50611[((chunk_offset_50474 +\n                                                                                   local_tid_50452) *\n                                                                                  sizze_48254 +\n                                                                                  gtid_50441) *\n                                                                                 4];\n                        \n                        *(__local float *) &mem_50619[cid_50543 * 4] =\n                            x_chunk_outer_elem_50542;\n                    }\n                }\n                barrier(CLK_LOCAL_MEM_FENCE);\n                \n                float acc_50481 = x_50475;\n                int32_t groupstream_mapaccum_dummy_chunk_sizze_50479 = 1;\n                \n                if (chunk_sizze_50473 == group_sizze_50446) {\n                    for (int32_t i_50480 = 0; i_50480 < group_sizze_50446;\n                         i_50480++) {\n                        float x_50484 = *(__local float *) &mem_50616[i_50480 *\n                                                                      4];\n                        float x_50485 = *(__local float *) &mem_50619[i_50480 *\n                                                                      4];\n                        float res_50487 = x_50484 * x_50485;\n                        float res_50489 = acc_50481 + res_50487;\n                       ",
            " float acc_tmp_51206 = res_50489;\n                        \n                        acc_50481 = acc_tmp_51206;\n                    }\n                } else {\n                    for (int32_t i_50480 = 0; i_50480 < chunk_sizze_50473;\n                         i_50480++) {\n                        float x_50484 = *(__local float *) &mem_50616[i_50480 *\n                                                                      4];\n                        float x_50485 = *(__local float *) &mem_50619[i_50480 *\n                                                                      4];\n                        float res_50487 = x_50484 * x_50485;\n                        float res_50489 = acc_50481 + res_50487;\n                        float acc_tmp_51206 = res_50489;\n                        \n                        acc_50481 = acc_tmp_51206;\n                    }\n                }\n                res_50478 = acc_50481;\n                sync_50555 = res_50478;\n                barrier(CLK_LOCAL_MEM_FENCE);\n                x_50475 = sync_50555;\n                chunk_offset_50474 += group_sizze_50446;\n            }\n            res_50472 = x_50475;\n            \n            float res_50490 = x_50470 * res_50472;\n            float res_50492 = x_50460 + res_50490;\n            float x_tmp_51201 = res_50492;\n            \n            x_50460 = x_tmp_51201;\n        }\n        res_50457 = x_50460;\n    }\n    if (thread_active_51200) {\n        *(__global float *) &mem_50624[(gtid_50440 * (sizze_48254 *\n                                                      sizze_48256) +\n                                        gtid_50441 * sizze_48256 + gtid_50442) *\n                                       4] = res_50457;\n    }\n}\n__kernel void reduce_kernel_48354(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48317, __global\n                                  unsigned char *mem_50615, __global\n                                  unsigned char *mem_50621)\n{\n    __local volati",
            "le char *restrict mem_50618 = mem_aligned_0;\n    int32_t wave_sizze_50772;\n    int32_t group_sizze_50773;\n    bool thread_active_50774;\n    int32_t global_tid_48354;\n    int32_t local_tid_48355;\n    int32_t group_id_48356;\n    \n    global_tid_48354 = get_global_id(0);\n    local_tid_48355 = get_local_id(0);\n    group_sizze_50773 = get_local_size(0);\n    wave_sizze_50772 = LOCKSTEP_WIDTH;\n    group_id_48356 = get_group_id(0);\n    thread_active_50774 = 1;\n    \n    bool in_bounds_48357;\n    float x_50493;\n    \n    if (thread_active_50774) {\n        in_bounds_48357 = slt32(local_tid_48355, num_groups_48317);\n        if (in_bounds_48357) {\n            float x_48358 = *(__global float *) &mem_50615[global_tid_48354 *\n                                                           4];\n            \n            x_50493 = x_48358;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48362;\n    \n    for (int32_t comb_iter_50775 = 0; comb_iter_50775 <\n         squot32(max_num_groups_48312 + max_num_groups_48312 - 1,\n                 max_num_groups_48312); comb_iter_50775++) {\n        int32_t combine_id_48361;\n        int32_t flat_comb_id_50776 = comb_iter_50775 * max_num_groups_48312 +\n                local_tid_48355;\n        \n        combine_id_48361 = flat_comb_id_50776;\n        if (slt32(combine_id_48361, max_num_groups_48312) && 1) {\n            *(__local float *) &mem_50618[combine_id_48361 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50777;\n    float x_46900;\n    float x_46901;\n    int32_t my_index_48324;\n    int32_t other_offset_48325;\n    \n    my_index_48324 = local_tid_48355;\n    other_offset_48325 = 0;\n    if (slt32(local_tid_48355, max_num_groups_48312)) {\n        x_46900 = *(__local float *) &mem_50618[(local_tid_48355 +\n                                                 other_offset_48325) * 4];\n    }\n    other_offset_48325 = 1;\n    while (slt32(other_offset_48325, wave_sizze_50772)) {\n   ",
            "     if (slt32(local_tid_48355 + other_offset_48325, max_num_groups_48312) &&\n            ((local_tid_48355 - squot32(local_tid_48355, wave_sizze_50772) *\n              wave_sizze_50772) & (2 * other_offset_48325 - 1)) == 0) {\n            // read array element\n            {\n                x_46901 = *(volatile __local\n                            float *) &mem_50618[(local_tid_48355 +\n                                                 other_offset_48325) * 4];\n            }\n            \n            float res_46902;\n            \n            if (thread_active_50774) {\n                res_46902 = x_46900 + x_46901;\n            }\n            x_46900 = res_46902;\n            *(volatile __local float *) &mem_50618[local_tid_48355 * 4] =\n                x_46900;\n        }\n        other_offset_48325 *= 2;\n    }\n    skip_waves_50777 = 1;\n    while (slt32(skip_waves_50777, squot32(max_num_groups_48312 +\n                                           wave_sizze_50772 - 1,\n                                           wave_sizze_50772))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48325 = skip_waves_50777 * wave_sizze_50772;\n        if (slt32(local_tid_48355 + other_offset_48325, max_num_groups_48312) &&\n            ((local_tid_48355 - squot32(local_tid_48355, wave_sizze_50772) *\n              wave_sizze_50772) == 0 && (squot32(local_tid_48355,\n                                                 wave_sizze_50772) & (2 *\n                                                                      skip_waves_50777 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_46901 = *(__local float *) &mem_50618[(local_tid_48355 +\n                                                         other_offset_48325) *\n                                                        4];\n            }\n            \n            float res_46902;\n            \n            if (thread_active_50774) {\n         ",
            "       res_46902 = x_46900 + x_46901;\n            }\n            x_46900 = res_46902;\n            *(__local float *) &mem_50618[local_tid_48355 * 4] = x_46900;\n        }\n        skip_waves_50777 *= 2;\n    }\n    final_result_48362 = x_46900;\n    if (local_tid_48355 == 0) {\n        *(__global float *) &mem_50621[group_id_48356 * 4] = final_result_48362;\n    }\n}\n__kernel void reduce_kernel_48433(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48392, __global\n                                  unsigned char *mem_50617, __global\n                                  unsigned char *mem_50623)\n{\n    __local volatile char *restrict mem_50620 = mem_aligned_0;\n    int32_t wave_sizze_50794;\n    int32_t group_sizze_50795;\n    bool thread_active_50796;\n    int32_t global_tid_48433;\n    int32_t local_tid_48434;\n    int32_t group_id_48435;\n    \n    global_tid_48433 = get_global_id(0);\n    local_tid_48434 = get_local_id(0);\n    group_sizze_50795 = get_local_size(0);\n    wave_sizze_50794 = LOCKSTEP_WIDTH;\n    group_id_48435 = get_group_id(0);\n    thread_active_50796 = 1;\n    \n    bool in_bounds_48436;\n    float x_50493;\n    \n    if (thread_active_50796) {\n        in_bounds_48436 = slt32(local_tid_48434, num_groups_48392);\n        if (in_bounds_48436) {\n            float x_48437 = *(__global float *) &mem_50617[global_tid_48433 *\n                                                           4];\n            \n            x_50493 = x_48437;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48441;\n    \n    for (int32_t comb_iter_50797 = 0; comb_iter_50797 <\n         squot32(max_num_groups_48387 + max_num_groups_48387 - 1,\n                 max_num_groups_48387); comb_iter_50797++) {\n        int32_t combine_id_48440;\n        int32_t flat_comb_id_50798 = comb_iter_50797 * max_num_groups_48387 +\n                local_tid_48434;\n        \n        combine_id_48440 = flat_comb_id_50798;\n        if (slt32(combine_id_48440",
            ", max_num_groups_48387) && 1) {\n            *(__local float *) &mem_50620[combine_id_48440 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50799;\n    float x_46922;\n    float x_46923;\n    int32_t my_index_48399;\n    int32_t other_offset_48400;\n    \n    my_index_48399 = local_tid_48434;\n    other_offset_48400 = 0;\n    if (slt32(local_tid_48434, max_num_groups_48387)) {\n        x_46922 = *(__local float *) &mem_50620[(local_tid_48434 +\n                                                 other_offset_48400) * 4];\n    }\n    other_offset_48400 = 1;\n    while (slt32(other_offset_48400, wave_sizze_50794)) {\n        if (slt32(local_tid_48434 + other_offset_48400, max_num_groups_48387) &&\n            ((local_tid_48434 - squot32(local_tid_48434, wave_sizze_50794) *\n              wave_sizze_50794) & (2 * other_offset_48400 - 1)) == 0) {\n            // read array element\n            {\n                x_46923 = *(volatile __local\n                            float *) &mem_50620[(local_tid_48434 +\n                                                 other_offset_48400) * 4];\n            }\n            \n            float res_46924;\n            \n            if (thread_active_50796) {\n                res_46924 = x_46922 + x_46923;\n            }\n            x_46922 = res_46924;\n            *(volatile __local float *) &mem_50620[local_tid_48434 * 4] =\n                x_46922;\n        }\n        other_offset_48400 *= 2;\n    }\n    skip_waves_50799 = 1;\n    while (slt32(skip_waves_50799, squot32(max_num_groups_48387 +\n                                           wave_sizze_50794 - 1,\n                                           wave_sizze_50794))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48400 = skip_waves_50799 * wave_sizze_50794;\n        if (slt32(local_tid_48434 + other_offset_48400, max_num_groups_48387) &&\n            ((local_tid_48434 - squot32(local_tid_48434, wave_sizze_50794) *\n              wave_sizze_50794) == 0 && (squot32(l",
            "ocal_tid_48434,\n                                                 wave_sizze_50794) & (2 *\n                                                                      skip_waves_50799 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_46923 = *(__local float *) &mem_50620[(local_tid_48434 +\n                                                         other_offset_48400) *\n                                                        4];\n            }\n            \n            float res_46924;\n            \n            if (thread_active_50796) {\n                res_46924 = x_46922 + x_46923;\n            }\n            x_46922 = res_46924;\n            *(__local float *) &mem_50620[local_tid_48434 * 4] = x_46922;\n        }\n        skip_waves_50799 *= 2;\n    }\n    final_result_48441 = x_46922;\n    if (local_tid_48434 == 0) {\n        *(__global float *) &mem_50623[group_id_48435 * 4] = final_result_48441;\n    }\n}\n__kernel void reduce_kernel_48506(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48461, __global\n                                  unsigned char *mem_50619, __global\n                                  unsigned char *mem_50625)\n{\n    __local volatile char *restrict mem_50622 = mem_aligned_0;\n    int32_t wave_sizze_50810;\n    int32_t group_sizze_50811;\n    bool thread_active_50812;\n    int32_t global_tid_48506;\n    int32_t local_tid_48507;\n    int32_t group_id_48508;\n    \n    global_tid_48506 = get_global_id(0);\n    local_tid_48507 = get_local_id(0);\n    group_sizze_50811 = get_local_size(0);\n    wave_sizze_50810 = LOCKSTEP_WIDTH;\n    group_id_48508 = get_group_id(0);\n    thread_active_50812 = 1;\n    \n    bool in_bounds_48509;\n    float x_50493;\n    \n    if (thread_active_50812) {\n        in_bounds_48509 = slt32(local_tid_48507, num_groups_48461);\n        if (in_bounds_48509) {\n            float x_48510 = *(__global float *) &m",
            "em_50619[global_tid_48506 *\n                                                           4];\n            \n            x_50493 = x_48510;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48514;\n    \n    for (int32_t comb_iter_50813 = 0; comb_iter_50813 <\n         squot32(max_num_groups_48456 + max_num_groups_48456 - 1,\n                 max_num_groups_48456); comb_iter_50813++) {\n        int32_t combine_id_48513;\n        int32_t flat_comb_id_50814 = comb_iter_50813 * max_num_groups_48456 +\n                local_tid_48507;\n        \n        combine_id_48513 = flat_comb_id_50814;\n        if (slt32(combine_id_48513, max_num_groups_48456) && 1) {\n            *(__local float *) &mem_50622[combine_id_48513 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50815;\n    float x_46948;\n    float x_46949;\n    int32_t my_index_48468;\n    int32_t other_offset_48469;\n    \n    my_index_48468 = local_tid_48507;\n    other_offset_48469 = 0;\n    if (slt32(local_tid_48507, max_num_groups_48456)) {\n        x_46948 = *(__local float *) &mem_50622[(local_tid_48507 +\n                                                 other_offset_48469) * 4];\n    }\n    other_offset_48469 = 1;\n    while (slt32(other_offset_48469, wave_sizze_50810)) {\n        if (slt32(local_tid_48507 + other_offset_48469, max_num_groups_48456) &&\n            ((local_tid_48507 - squot32(local_tid_48507, wave_sizze_50810) *\n              wave_sizze_50810) & (2 * other_offset_48469 - 1)) == 0) {\n            // read array element\n            {\n                x_46949 = *(volatile __local\n                            float *) &mem_50622[(local_tid_48507 +\n                                                 other_offset_48469) * 4];\n            }\n            \n            float res_46950;\n            \n            if (thread_active_50812) {\n                res_46950 = x_46948 + x_46949;\n            }\n            x_46948 = res_46950;\n            *(volatile __loc",
            "al float *) &mem_50622[local_tid_48507 * 4] =\n                x_46948;\n        }\n        other_offset_48469 *= 2;\n    }\n    skip_waves_50815 = 1;\n    while (slt32(skip_waves_50815, squot32(max_num_groups_48456 +\n                                           wave_sizze_50810 - 1,\n                                           wave_sizze_50810))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48469 = skip_waves_50815 * wave_sizze_50810;\n        if (slt32(local_tid_48507 + other_offset_48469, max_num_groups_48456) &&\n            ((local_tid_48507 - squot32(local_tid_48507, wave_sizze_50810) *\n              wave_sizze_50810) == 0 && (squot32(local_tid_48507,\n                                                 wave_sizze_50810) & (2 *\n                                                                      skip_waves_50815 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_46949 = *(__local float *) &mem_50622[(local_tid_48507 +\n                                                         other_offset_48469) *\n                                                        4];\n            }\n            \n            float res_46950;\n            \n            if (thread_active_50812) {\n                res_46950 = x_46948 + x_46949;\n            }\n            x_46948 = res_46950;\n            *(__local float *) &mem_50622[local_tid_48507 * 4] = x_46948;\n        }\n        skip_waves_50815 *= 2;\n    }\n    final_result_48514 = x_46948;\n    if (local_tid_48507 == 0) {\n        *(__global float *) &mem_50625[group_id_48508 * 4] = final_result_48514;\n    }\n}\n__kernel void reduce_kernel_48580(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48534, __global\n                                  unsigned char *mem_50619, __global\n                                  unsigned char *mem_50625)\n{\n    __local volatile char *restrict mem_50622 = mem_alig",
            "ned_0;\n    int32_t wave_sizze_50826;\n    int32_t group_sizze_50827;\n    bool thread_active_50828;\n    int32_t global_tid_48580;\n    int32_t local_tid_48581;\n    int32_t group_id_48582;\n    \n    global_tid_48580 = get_global_id(0);\n    local_tid_48581 = get_local_id(0);\n    group_sizze_50827 = get_local_size(0);\n    wave_sizze_50826 = LOCKSTEP_WIDTH;\n    group_id_48582 = get_group_id(0);\n    thread_active_50828 = 1;\n    \n    bool in_bounds_48583;\n    float x_50493;\n    \n    if (thread_active_50828) {\n        in_bounds_48583 = slt32(local_tid_48581, num_groups_48534);\n        if (in_bounds_48583) {\n            float x_48584 = *(__global float *) &mem_50619[global_tid_48580 *\n                                                           4];\n            \n            x_50493 = x_48584;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48588;\n    \n    for (int32_t comb_iter_50829 = 0; comb_iter_50829 <\n         squot32(max_num_groups_48529 + max_num_groups_48529 - 1,\n                 max_num_groups_48529); comb_iter_50829++) {\n        int32_t combine_id_48587;\n        int32_t flat_comb_id_50830 = comb_iter_50829 * max_num_groups_48529 +\n                local_tid_48581;\n        \n        combine_id_48587 = flat_comb_id_50830;\n        if (slt32(combine_id_48587, max_num_groups_48529) && 1) {\n            *(__local float *) &mem_50622[combine_id_48587 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50831;\n    float x_46976;\n    float x_46977;\n    int32_t my_index_48541;\n    int32_t other_offset_48542;\n    \n    my_index_48541 = local_tid_48581;\n    other_offset_48542 = 0;\n    if (slt32(local_tid_48581, max_num_groups_48529)) {\n        x_46976 = *(__local float *) &mem_50622[(local_tid_48581 +\n                                                 other_offset_48542) * 4];\n    }\n    other_offset_48542 = 1;\n    while (slt32(other_offset_48542, wave_sizze_50826)) {\n        if (slt32(local_tid_48581 + other",
            "_offset_48542, max_num_groups_48529) &&\n            ((local_tid_48581 - squot32(local_tid_48581, wave_sizze_50826) *\n              wave_sizze_50826) & (2 * other_offset_48542 - 1)) == 0) {\n            // read array element\n            {\n                x_46977 = *(volatile __local\n                            float *) &mem_50622[(local_tid_48581 +\n                                                 other_offset_48542) * 4];\n            }\n            \n            float res_46978;\n            \n            if (thread_active_50828) {\n                res_46978 = x_46976 + x_46977;\n            }\n            x_46976 = res_46978;\n            *(volatile __local float *) &mem_50622[local_tid_48581 * 4] =\n                x_46976;\n        }\n        other_offset_48542 *= 2;\n    }\n    skip_waves_50831 = 1;\n    while (slt32(skip_waves_50831, squot32(max_num_groups_48529 +\n                                           wave_sizze_50826 - 1,\n                                           wave_sizze_50826))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48542 = skip_waves_50831 * wave_sizze_50826;\n        if (slt32(local_tid_48581 + other_offset_48542, max_num_groups_48529) &&\n            ((local_tid_48581 - squot32(local_tid_48581, wave_sizze_50826) *\n              wave_sizze_50826) == 0 && (squot32(local_tid_48581,\n                                                 wave_sizze_50826) & (2 *\n                                                                      skip_waves_50831 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_46977 = *(__local float *) &mem_50622[(local_tid_48581 +\n                                                         other_offset_48542) *\n                                                        4];\n            }\n            \n            float res_46978;\n            \n            if (thread_active_50828) {\n                res_46978 = x_46976 + x_46977;\n",
            "            }\n            x_46976 = res_46978;\n            *(__local float *) &mem_50622[local_tid_48581 * 4] = x_46976;\n        }\n        skip_waves_50831 *= 2;\n    }\n    final_result_48588 = x_46976;\n    if (local_tid_48581 == 0) {\n        *(__global float *) &mem_50625[group_id_48582 * 4] = final_result_48588;\n    }\n}\n__kernel void reduce_kernel_48659(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48610, __global\n                                  unsigned char *mem_50621, __global\n                                  unsigned char *mem_50627)\n{\n    __local volatile char *restrict mem_50624 = mem_aligned_0;\n    int32_t wave_sizze_50842;\n    int32_t group_sizze_50843;\n    bool thread_active_50844;\n    int32_t global_tid_48659;\n    int32_t local_tid_48660;\n    int32_t group_id_48661;\n    \n    global_tid_48659 = get_global_id(0);\n    local_tid_48660 = get_local_id(0);\n    group_sizze_50843 = get_local_size(0);\n    wave_sizze_50842 = LOCKSTEP_WIDTH;\n    group_id_48661 = get_group_id(0);\n    thread_active_50844 = 1;\n    \n    bool in_bounds_48662;\n    float x_50493;\n    \n    if (thread_active_50844) {\n        in_bounds_48662 = slt32(local_tid_48660, num_groups_48610);\n        if (in_bounds_48662) {\n            float x_48663 = *(__global float *) &mem_50621[global_tid_48659 *\n                                                           4];\n            \n            x_50493 = x_48663;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48667;\n    \n    for (int32_t comb_iter_50845 = 0; comb_iter_50845 <\n         squot32(max_num_groups_48605 + max_num_groups_48605 - 1,\n                 max_num_groups_48605); comb_iter_50845++) {\n        int32_t combine_id_48666;\n        int32_t flat_comb_id_50846 = comb_iter_50845 * max_num_groups_48605 +\n                local_tid_48660;\n        \n        combine_id_48666 = flat_comb_id_50846;\n        if (slt32(combine_id_48666, max_num_groups_48605) && 1) {\n      ",
            "      *(__local float *) &mem_50624[combine_id_48666 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50847;\n    float x_47013;\n    float x_47014;\n    int32_t my_index_48617;\n    int32_t other_offset_48618;\n    \n    my_index_48617 = local_tid_48660;\n    other_offset_48618 = 0;\n    if (slt32(local_tid_48660, max_num_groups_48605)) {\n        x_47013 = *(__local float *) &mem_50624[(local_tid_48660 +\n                                                 other_offset_48618) * 4];\n    }\n    other_offset_48618 = 1;\n    while (slt32(other_offset_48618, wave_sizze_50842)) {\n        if (slt32(local_tid_48660 + other_offset_48618, max_num_groups_48605) &&\n            ((local_tid_48660 - squot32(local_tid_48660, wave_sizze_50842) *\n              wave_sizze_50842) & (2 * other_offset_48618 - 1)) == 0) {\n            // read array element\n            {\n                x_47014 = *(volatile __local\n                            float *) &mem_50624[(local_tid_48660 +\n                                                 other_offset_48618) * 4];\n            }\n            \n            float res_47015;\n            \n            if (thread_active_50844) {\n                res_47015 = x_47013 + x_47014;\n            }\n            x_47013 = res_47015;\n            *(volatile __local float *) &mem_50624[local_tid_48660 * 4] =\n                x_47013;\n        }\n        other_offset_48618 *= 2;\n    }\n    skip_waves_50847 = 1;\n    while (slt32(skip_waves_50847, squot32(max_num_groups_48605 +\n                                           wave_sizze_50842 - 1,\n                                           wave_sizze_50842))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48618 = skip_waves_50847 * wave_sizze_50842;\n        if (slt32(local_tid_48660 + other_offset_48618, max_num_groups_48605) &&\n            ((local_tid_48660 - squot32(local_tid_48660, wave_sizze_50842) *\n              wave_sizze_50842) == 0 && (squot32(local_tid_48660,\n                      ",
            "                           wave_sizze_50842) & (2 *\n                                                                      skip_waves_50847 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_47014 = *(__local float *) &mem_50624[(local_tid_48660 +\n                                                         other_offset_48618) *\n                                                        4];\n            }\n            \n            float res_47015;\n            \n            if (thread_active_50844) {\n                res_47015 = x_47013 + x_47014;\n            }\n            x_47013 = res_47015;\n            *(__local float *) &mem_50624[local_tid_48660 * 4] = x_47013;\n        }\n        skip_waves_50847 *= 2;\n    }\n    final_result_48667 = x_47013;\n    if (local_tid_48660 == 0) {\n        *(__global float *) &mem_50627[group_id_48661 * 4] = final_result_48667;\n    }\n}\n__kernel void reduce_kernel_48742(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48689, __global\n                                  unsigned char *mem_50621, __global\n                                  unsigned char *mem_50627)\n{\n    __local volatile char *restrict mem_50624 = mem_aligned_0;\n    int32_t wave_sizze_50858;\n    int32_t group_sizze_50859;\n    bool thread_active_50860;\n    int32_t global_tid_48742;\n    int32_t local_tid_48743;\n    int32_t group_id_48744;\n    \n    global_tid_48742 = get_global_id(0);\n    local_tid_48743 = get_local_id(0);\n    group_sizze_50859 = get_local_size(0);\n    wave_sizze_50858 = LOCKSTEP_WIDTH;\n    group_id_48744 = get_group_id(0);\n    thread_active_50860 = 1;\n    \n    bool in_bounds_48745;\n    float x_50493;\n    \n    if (thread_active_50860) {\n        in_bounds_48745 = slt32(local_tid_48743, num_groups_48689);\n        if (in_bounds_48745) {\n            float x_48746 = *(__global float *) &mem_50621[global_tid_48742 *\n          ",
            "                                                 4];\n            \n            x_50493 = x_48746;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48750;\n    \n    for (int32_t comb_iter_50861 = 0; comb_iter_50861 <\n         squot32(max_num_groups_48684 + max_num_groups_48684 - 1,\n                 max_num_groups_48684); comb_iter_50861++) {\n        int32_t combine_id_48749;\n        int32_t flat_comb_id_50862 = comb_iter_50861 * max_num_groups_48684 +\n                local_tid_48743;\n        \n        combine_id_48749 = flat_comb_id_50862;\n        if (slt32(combine_id_48749, max_num_groups_48684) && 1) {\n            *(__local float *) &mem_50624[combine_id_48749 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50863;\n    float x_47055;\n    float x_47056;\n    int32_t my_index_48696;\n    int32_t other_offset_48697;\n    \n    my_index_48696 = local_tid_48743;\n    other_offset_48697 = 0;\n    if (slt32(local_tid_48743, max_num_groups_48684)) {\n        x_47055 = *(__local float *) &mem_50624[(local_tid_48743 +\n                                                 other_offset_48697) * 4];\n    }\n    other_offset_48697 = 1;\n    while (slt32(other_offset_48697, wave_sizze_50858)) {\n        if (slt32(local_tid_48743 + other_offset_48697, max_num_groups_48684) &&\n            ((local_tid_48743 - squot32(local_tid_48743, wave_sizze_50858) *\n              wave_sizze_50858) & (2 * other_offset_48697 - 1)) == 0) {\n            // read array element\n            {\n                x_47056 = *(volatile __local\n                            float *) &mem_50624[(local_tid_48743 +\n                                                 other_offset_48697) * 4];\n            }\n            \n            float res_47057;\n            \n            if (thread_active_50860) {\n                res_47057 = x_47055 + x_47056;\n            }\n            x_47055 = res_47057;\n            *(volatile __local float *) &mem_50624[local_tid_48743",
            " * 4] =\n                x_47055;\n        }\n        other_offset_48697 *= 2;\n    }\n    skip_waves_50863 = 1;\n    while (slt32(skip_waves_50863, squot32(max_num_groups_48684 +\n                                           wave_sizze_50858 - 1,\n                                           wave_sizze_50858))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48697 = skip_waves_50863 * wave_sizze_50858;\n        if (slt32(local_tid_48743 + other_offset_48697, max_num_groups_48684) &&\n            ((local_tid_48743 - squot32(local_tid_48743, wave_sizze_50858) *\n              wave_sizze_50858) == 0 && (squot32(local_tid_48743,\n                                                 wave_sizze_50858) & (2 *\n                                                                      skip_waves_50863 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_47056 = *(__local float *) &mem_50624[(local_tid_48743 +\n                                                         other_offset_48697) *\n                                                        4];\n            }\n            \n            float res_47057;\n            \n            if (thread_active_50860) {\n                res_47057 = x_47055 + x_47056;\n            }\n            x_47055 = res_47057;\n            *(__local float *) &mem_50624[local_tid_48743 * 4] = x_47055;\n        }\n        skip_waves_50863 *= 2;\n    }\n    final_result_48750 = x_47055;\n    if (local_tid_48743 == 0) {\n        *(__global float *) &mem_50627[group_id_48744 * 4] = final_result_48750;\n    }\n}\n__kernel void reduce_kernel_48821(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48772, __global\n                                  unsigned char *mem_50621, __global\n                                  unsigned char *mem_50627)\n{\n    __local volatile char *restrict mem_50624 = mem_aligned_0;\n    int32_t wave_sizze_50874;\n ",
            "   int32_t group_sizze_50875;\n    bool thread_active_50876;\n    int32_t global_tid_48821;\n    int32_t local_tid_48822;\n    int32_t group_id_48823;\n    \n    global_tid_48821 = get_global_id(0);\n    local_tid_48822 = get_local_id(0);\n    group_sizze_50875 = get_local_size(0);\n    wave_sizze_50874 = LOCKSTEP_WIDTH;\n    group_id_48823 = get_group_id(0);\n    thread_active_50876 = 1;\n    \n    bool in_bounds_48824;\n    float x_50493;\n    \n    if (thread_active_50876) {\n        in_bounds_48824 = slt32(local_tid_48822, num_groups_48772);\n        if (in_bounds_48824) {\n            float x_48825 = *(__global float *) &mem_50621[global_tid_48821 *\n                                                           4];\n            \n            x_50493 = x_48825;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48829;\n    \n    for (int32_t comb_iter_50877 = 0; comb_iter_50877 <\n         squot32(max_num_groups_48767 + max_num_groups_48767 - 1,\n                 max_num_groups_48767); comb_iter_50877++) {\n        int32_t combine_id_48828;\n        int32_t flat_comb_id_50878 = comb_iter_50877 * max_num_groups_48767 +\n                local_tid_48822;\n        \n        combine_id_48828 = flat_comb_id_50878;\n        if (slt32(combine_id_48828, max_num_groups_48767) && 1) {\n            *(__local float *) &mem_50624[combine_id_48828 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50879;\n    float x_47097;\n    float x_47098;\n    int32_t my_index_48779;\n    int32_t other_offset_48780;\n    \n    my_index_48779 = local_tid_48822;\n    other_offset_48780 = 0;\n    if (slt32(local_tid_48822, max_num_groups_48767)) {\n        x_47097 = *(__local float *) &mem_50624[(local_tid_48822 +\n                                                 other_offset_48780) * 4];\n    }\n    other_offset_48780 = 1;\n    while (slt32(other_offset_48780, wave_sizze_50874)) {\n        if (slt32(local_tid_48822 + other_offset_48780, max_num_groups_48767) &",
            "&\n            ((local_tid_48822 - squot32(local_tid_48822, wave_sizze_50874) *\n              wave_sizze_50874) & (2 * other_offset_48780 - 1)) == 0) {\n            // read array element\n            {\n                x_47098 = *(volatile __local\n                            float *) &mem_50624[(local_tid_48822 +\n                                                 other_offset_48780) * 4];\n            }\n            \n            float res_47099;\n            \n            if (thread_active_50876) {\n                res_47099 = x_47097 + x_47098;\n            }\n            x_47097 = res_47099;\n            *(volatile __local float *) &mem_50624[local_tid_48822 * 4] =\n                x_47097;\n        }\n        other_offset_48780 *= 2;\n    }\n    skip_waves_50879 = 1;\n    while (slt32(skip_waves_50879, squot32(max_num_groups_48767 +\n                                           wave_sizze_50874 - 1,\n                                           wave_sizze_50874))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48780 = skip_waves_50879 * wave_sizze_50874;\n        if (slt32(local_tid_48822 + other_offset_48780, max_num_groups_48767) &&\n            ((local_tid_48822 - squot32(local_tid_48822, wave_sizze_50874) *\n              wave_sizze_50874) == 0 && (squot32(local_tid_48822,\n                                                 wave_sizze_50874) & (2 *\n                                                                      skip_waves_50879 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_47098 = *(__local float *) &mem_50624[(local_tid_48822 +\n                                                         other_offset_48780) *\n                                                        4];\n            }\n            \n            float res_47099;\n            \n            if (thread_active_50876) {\n                res_47099 = x_47097 + x_47098;\n            }\n            x_47097 = re",
            "s_47099;\n            *(__local float *) &mem_50624[local_tid_48822 * 4] = x_47097;\n        }\n        skip_waves_50879 *= 2;\n    }\n    final_result_48829 = x_47097;\n    if (local_tid_48822 == 0) {\n        *(__global float *) &mem_50627[group_id_48823 * 4] = final_result_48829;\n    }\n}\n__kernel void reduce_kernel_48894(__local volatile int64_t *mem_aligned_0,\n                                  int32_t num_groups_48849, __global\n                                  unsigned char *mem_50619, __global\n                                  unsigned char *mem_50625)\n{\n    __local volatile char *restrict mem_50622 = mem_aligned_0;\n    int32_t wave_sizze_50890;\n    int32_t group_sizze_50891;\n    bool thread_active_50892;\n    int32_t global_tid_48894;\n    int32_t local_tid_48895;\n    int32_t group_id_48896;\n    \n    global_tid_48894 = get_global_id(0);\n    local_tid_48895 = get_local_id(0);\n    group_sizze_50891 = get_local_size(0);\n    wave_sizze_50890 = LOCKSTEP_WIDTH;\n    group_id_48896 = get_group_id(0);\n    thread_active_50892 = 1;\n    \n    bool in_bounds_48897;\n    float x_50493;\n    \n    if (thread_active_50892) {\n        in_bounds_48897 = slt32(local_tid_48895, num_groups_48849);\n        if (in_bounds_48897) {\n            float x_48898 = *(__global float *) &mem_50619[global_tid_48894 *\n                                                           4];\n            \n            x_50493 = x_48898;\n        } else {\n            x_50493 = 0.0F;\n        }\n    }\n    \n    float final_result_48902;\n    \n    for (int32_t comb_iter_50893 = 0; comb_iter_50893 <\n         squot32(max_num_groups_48844 + max_num_groups_48844 - 1,\n                 max_num_groups_48844); comb_iter_50893++) {\n        int32_t combine_id_48901;\n        int32_t flat_comb_id_50894 = comb_iter_50893 * max_num_groups_48844 +\n                local_tid_48895;\n        \n        combine_id_48901 = flat_comb_id_50894;\n        if (slt32(combine_id_48901, max_num_groups_48844) && 1) {\n            *(__local float *) &mem_50622[co",
            "mbine_id_48901 * 4] = x_50493;\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50895;\n    float x_47127;\n    float x_47128;\n    int32_t my_index_48856;\n    int32_t other_offset_48857;\n    \n    my_index_48856 = local_tid_48895;\n    other_offset_48857 = 0;\n    if (slt32(local_tid_48895, max_num_groups_48844)) {\n        x_47127 = *(__local float *) &mem_50622[(local_tid_48895 +\n                                                 other_offset_48857) * 4];\n    }\n    other_offset_48857 = 1;\n    while (slt32(other_offset_48857, wave_sizze_50890)) {\n        if (slt32(local_tid_48895 + other_offset_48857, max_num_groups_48844) &&\n            ((local_tid_48895 - squot32(local_tid_48895, wave_sizze_50890) *\n              wave_sizze_50890) & (2 * other_offset_48857 - 1)) == 0) {\n            // read array element\n            {\n                x_47128 = *(volatile __local\n                            float *) &mem_50622[(local_tid_48895 +\n                                                 other_offset_48857) * 4];\n            }\n            \n            float res_47129;\n            \n            if (thread_active_50892) {\n                res_47129 = x_47127 + x_47128;\n            }\n            x_47127 = res_47129;\n            *(volatile __local float *) &mem_50622[local_tid_48895 * 4] =\n                x_47127;\n        }\n        other_offset_48857 *= 2;\n    }\n    skip_waves_50895 = 1;\n    while (slt32(skip_waves_50895, squot32(max_num_groups_48844 +\n                                           wave_sizze_50890 - 1,\n                                           wave_sizze_50890))) {\n        barrier(CLK_LOCAL_MEM_FENCE);\n        other_offset_48857 = skip_waves_50895 * wave_sizze_50890;\n        if (slt32(local_tid_48895 + other_offset_48857, max_num_groups_48844) &&\n            ((local_tid_48895 - squot32(local_tid_48895, wave_sizze_50890) *\n              wave_sizze_50890) == 0 && (squot32(local_tid_48895,\n                                                 wave_sizze_",
            "50890) & (2 *\n                                                                      skip_waves_50895 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            {\n                x_47128 = *(__local float *) &mem_50622[(local_tid_48895 +\n                                                         other_offset_48857) *\n                                                        4];\n            }\n            \n            float res_47129;\n            \n            if (thread_active_50892) {\n                res_47129 = x_47127 + x_47128;\n            }\n            x_47127 = res_47129;\n            *(__local float *) &mem_50622[local_tid_48895 * 4] = x_47127;\n        }\n        skip_waves_50895 *= 2;\n    }\n    final_result_48902 = x_47127;\n    if (local_tid_48895 == 0) {\n        *(__global float *) &mem_50625[group_id_48896 * 4] = final_result_48902;\n    }\n}\n__kernel void reduce_kernel_48993(__local volatile int64_t *mem_aligned_0,\n                                  int32_t sizze_47136, int32_t num_groups_48929,\n                                  __global unsigned char *mem_50614, __global\n                                  unsigned char *mem_50664, __global\n                                  unsigned char *mem_50677, __global\n                                  unsigned char *mem_50680, __global\n                                  unsigned char *mem_50683, __global\n                                  unsigned char *mem_50690)\n{\n    __local volatile char *restrict mem_50674 = mem_aligned_0;\n    int32_t wave_sizze_50927;\n    int32_t group_sizze_50928;\n    bool thread_active_50929;\n    int32_t global_tid_48993;\n    int32_t local_tid_48994;\n    int32_t group_id_48995;\n    \n    global_tid_48993 = get_global_id(0);\n    local_tid_48994 = get_local_id(0);\n    group_sizze_50928 = get_local_size(0);\n    wave_sizze_50927 = LOCKSTEP_WIDTH;\n    group_id_48995 = get_group_id(0);\n    thread_active_50929 = 1;\n    \n    bool ",
            "in_bounds_48996;\n    \n    if (thread_active_50929) {\n        in_bounds_48996 = slt32(local_tid_48994, num_groups_48929);\n        if (in_bounds_48996) {\n            for (int32_t i_50930 = 0; i_50930 < sizze_47136; i_50930++) {\n                *(__global float *) &mem_50690[(group_id_48995 * (sizze_47136 *\n                                                                  max_num_groups_48924) +\n                                                i_50930 * max_num_groups_48924 +\n                                                local_tid_48994) * 4] =\n                    *(__global float *) &mem_50664[(global_tid_48993 *\n                                                    sizze_47136 + i_50930) * 4];\n            }\n        } else {\n            for (int32_t i_50931 = 0; i_50931 < sizze_47136; i_50931++) {\n                *(__global float *) &mem_50690[(group_id_48995 * (sizze_47136 *\n                                                                  max_num_groups_48924) +\n                                                i_50931 * max_num_groups_48924 +\n                                                local_tid_48994) * 4] =\n                    *(__global float *) &mem_50614[i_50931 * 4];\n            }\n        }\n    }\n    for (int32_t comb_iter_50932 = 0; comb_iter_50932 <\n         squot32(max_num_groups_48924 + max_num_groups_48924 - 1,\n                 max_num_groups_48924); comb_iter_50932++) {\n        int32_t combine_id_49000;\n        int32_t flat_comb_id_50933 = comb_iter_50932 * max_num_groups_48924 +\n                local_tid_48994;\n        \n        combine_id_49000 = flat_comb_id_50933;\n        if (slt32(combine_id_49000, max_num_groups_48924) && 1) {\n            for (int32_t i_50934 = 0; i_50934 < sizze_47136; i_50934++) {\n                *(__local float *) &mem_50674[(combine_id_49000 * sizze_47136 +\n                                               i_50934) * 4] = *(__global\n                                                                 float *) &mem_50690[(group_id_489",
            "95 *\n                                                                                      (sizze_47136 *\n                                                                                       max_num_groups_48924) +\n                                                                                      i_50934 *\n                                                                                      max_num_groups_48924 +\n                                                                                      local_tid_48994) *\n                                                                                     4];\n            }\n        }\n    }\n    barrier(CLK_LOCAL_MEM_FENCE);\n    \n    int32_t skip_waves_50935;\n    int32_t my_index_48936;\n    int32_t other_offset_48937;\n    \n    my_index_48936 = local_tid_48994;\n    other_offset_48937 = 0;\n    if (slt32(local_tid_48994, max_num_groups_48924)) { }\n    other_offset_48937 = 1;\n    while (slt32(other_offset_48937, wave_sizze_50927)) {\n        if (slt32(local_tid_48994 + other_offset_48937, max_num_groups_48924) &&\n            ((local_tid_48994 - squot32(local_tid_48994, wave_sizze_50927) *\n              wave_sizze_50927) & (2 * other_offset_48937 - 1)) == 0) {\n            // read array element\n            { }\n            if (thread_active_50929) {\n                for (int32_t i_48905 = 0; i_48905 < sizze_47136; i_48905++) {\n                    float x_47152 = *(volatile __local\n                                      float *) &mem_50674[(my_index_48936 *\n                                                           sizze_47136 +\n                                                           i_48905) * 4];\n                    float x_47153 = *(volatile __local\n                                      float *) &mem_50674[((my_index_48936 +\n                                                            other_offset_48937) *\n                                                           sizze_47136 +\n                                                ",
            "           i_48905) * 4];\n                    float res_47154 = x_47152 + x_47153;\n                    \n                    *(volatile __global float *) &mem_50677[(group_id_48995 *\n                                                             (sizze_47136 *\n                                                              max_num_groups_48924) +\n                                                             i_48905 *\n                                                             max_num_groups_48924 +\n                                                             local_tid_48994) *\n                                                            4] = res_47154;\n                }\n            }\n            for (int32_t i_50937 = 0; i_50937 < sizze_47136; i_50937++) {\n                *(volatile __local float *) &mem_50674[(my_index_48936 *\n                                                        sizze_47136 + i_50937) *\n                                                       4] = *(volatile __global\n                                                              float *) &mem_50677[(group_id_48995 *\n                                                                                   (sizze_47136 *\n                                                                                    max_num_groups_48924) +\n                                                                                   i_50937 *\n                                                                                   max_num_groups_48924 +\n                                                                                   local_tid_48994) *\n                                                                                  4];\n            }\n        }\n        other_offset_48937 *= 2;\n    }\n    skip_waves_50935 = 1;\n    while (slt32(skip_waves_50935, squot32(max_num_groups_48924 +\n                                           wave_sizze_50927 - 1,\n                                           wave_sizze_50927))) {\n        barrier(CLK_LOCAL_M",
            "EM_FENCE);\n        other_offset_48937 = skip_waves_50935 * wave_sizze_50927;\n        if (slt32(local_tid_48994 + other_offset_48937, max_num_groups_48924) &&\n            ((local_tid_48994 - squot32(local_tid_48994, wave_sizze_50927) *\n              wave_sizze_50927) == 0 && (squot32(local_tid_48994,\n                                                 wave_sizze_50927) & (2 *\n                                                                      skip_waves_50935 -\n                                                                      1)) ==\n             0)) {\n            // read array element\n            { }\n            if (thread_active_50929) {\n                for (int32_t i_48905 = 0; i_48905 < sizze_47136; i_48905++) {\n                    float x_47152 = *(__local\n                                      float *) &mem_50674[(my_index_48936 *\n                                                           sizze_47136 +\n                                                           i_48905) * 4];\n                    float x_47153 = *(__local\n                                      float *) &mem_50674[((my_index_48936 +\n                                                            other_offset_48937) *\n                                                           sizze_47136 +\n                                                           i_48905) * 4];\n                    float res_47154 = x_47152 + x_47153;\n                    \n                    *(__global float *) &mem_50677[(group_id_48995 *\n                                                    (sizze_47136 *\n                                                     max_num_groups_48924) +\n                                                    i_48905 *\n                                                    max_num_groups_48924 +\n                                                    local_tid_48994) * 4] =\n                        res_47154;\n                }\n            }\n            for (int32_t i_50939 = 0; i_50939 < sizze_47136; i_50939++) {\n       ",
            "         *(__local float *) &mem_50674[(my_index_48936 * sizze_47136 +\n                                               i_50939) * 4] = *(__global\n                                                                 float *) &mem_50677[(group_id_48995 *\n                                                                                      (sizze_47136 *\n                                                                                       max_num_groups_48924) +\n                                                                                      i_50939 *\n                                                                                      max_num_groups_48924 +\n                                                                                      local_tid_48994) *\n                                                                                     4];\n            }\n        }\n        skip_waves_50935 *= 2;\n    }\n    for (int32_t i_50940 = 0; i_50940 < sizze_47136; i_50940++) {\n        *(__global float *) &mem_50680[(group_id_48995 * (sizze_47136 *\n                                                          max_num_groups_48924) +\n                                        i_50940 * max_num_groups_48924 +\n                                        local_tid_48994) * 4] = *(__local\n                                                                  float *) &mem_50674[(my_index_48936 *\n                                                                                       sizze_47136 +\n                                                                                       i_50940) *\n                                                                                      4];\n    }\n    if (local_tid_48994 == 0) {\n        for (int32_t i_50942 = 0; i_50942 < sizze_47136; i_50942++) {\n            *(__global float *) &mem_50683[(group_id_48995 * sizze_47136 +\n                                            i_50942) * 4] = *(__global\n                                                              floa",
            "t *) &mem_50680[(group_id_48995 *\n                                                                                   (sizze_47136 *\n                                                                                    max_num_groups_48924) +\n                                                                                   i_50942 *\n                                                                                   max_num_groups_48924 +\n                                                                                   local_tid_48994) *\n                                                                                  4];\n        }\n    }\n}\n",
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
static const char *size_names[] = {"group_size_48309", "max_num_groups_48311",
                                   "group_size_48364", "group_size_48384",
                                   "max_num_groups_48386", "group_size_48453",
                                   "max_num_groups_48455", "group_size_48526",
                                   "max_num_groups_48528", "group_size_48602",
                                   "max_num_groups_48604", "group_size_48681",
                                   "max_num_groups_48683", "group_size_48764",
                                   "max_num_groups_48766", "group_size_48841",
                                   "max_num_groups_48843", "group_size_48921",
                                   "max_num_groups_48923", "group_size_49014",
                                   "group_size_49054", "group_size_49098",
                                   "group_size_49132", "group_size_49158",
                                   "group_size_49192", "group_size_49220",
                                   "group_size_49267", "group_size_49310",
                                   "group_size_49350", "group_size_49379",
                                   "group_size_49414", "group_size_49472",
                                   "group_size_49523", "group_size_49581",
                                   "group_size_49623", "group_size_49663",
                                   "group_size_49705", "group_size_49786",
                                   "group_size_49858", "group_size_49901",
                                   "group_size_49944", "group_size_49990",
                                   "group_size_50036", "group_size_50123",
                                   "group_size_50138", "group_size_50202",
                                   "group_size_50217", "group_size_50283",
                                   "group_size_50298", "group_size_50315",
                                   "group_size_50330", "group_size_50364",
                                   "group_size_50404", "group_size_50445",
                                   "tile_size_50527", "group_size_50902"};
static const char *size_classes[] = {"group_size", "num_groups", "group_size",
                                     "group_size", "num_groups", "group_size",
                                     "num_groups", "group_size", "num_groups",
                                     "group_size", "num_groups", "group_size",
                                     "num_groups", "group_size", "num_groups",
                                     "group_size", "num_groups", "group_size",
                                     "num_groups", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "group_size", "group_size", "group_size",
                                     "tile_size", "group_size"};
static const char *size_entry_points[] = {"sum", "sum", "prod", "dot", "dot",
                                          "dot2", "dot2", "dot3", "dot3",
                                          "dot4", "dot4", "dot5", "dot5",
                                          "dot6", "dot6", "dot1", "dot1",
                                          "t1b_2d", "t1b_2d", "t1_2d", "t2_2d",
                                          "t3_2d", "t4_2d", "t4_2d", "t5_2d",
                                          "t5_2d", "t6_2d", "t7_2d", "t7_2d",
                                          "t9_2d", "t9_2d", "t10_2d", "t10_2d",
                                          "t9_3d", "t9_3d", "t8_2d", "t8_2d",
                                          "t2_3d", "t10_3d", "t10_3d", "t11_3d",
                                          "t11_3d", "t3_3d", "t4_3d", "t5_3d",
                                          "t5_3d", "t6_3d", "t6_3d", "t7_3d",
                                          "t7_3d", "t7_3d", "t7_3d", "t7_3d",
                                          "t8_3d", "t1_3d", "t1b_2d"};
int futhark_get_num_sizes(void)
{
    return 56;
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
    size_t group_sizze_48309;
    size_t max_num_groups_48311;
    size_t group_sizze_48364;
    size_t group_sizze_48384;
    size_t max_num_groups_48386;
    size_t group_sizze_48453;
    size_t max_num_groups_48455;
    size_t group_sizze_48526;
    size_t max_num_groups_48528;
    size_t group_sizze_48602;
    size_t max_num_groups_48604;
    size_t group_sizze_48681;
    size_t max_num_groups_48683;
    size_t group_sizze_48764;
    size_t max_num_groups_48766;
    size_t group_sizze_48841;
    size_t max_num_groups_48843;
    size_t group_sizze_48921;
    size_t max_num_groups_48923;
    size_t group_sizze_49014;
    size_t group_sizze_49054;
    size_t group_sizze_49098;
    size_t group_sizze_49132;
    size_t group_sizze_49158;
    size_t group_sizze_49192;
    size_t group_sizze_49220;
    size_t group_sizze_49267;
    size_t group_sizze_49310;
    size_t group_sizze_49350;
    size_t group_sizze_49379;
    size_t group_sizze_49414;
    size_t group_sizze_49472;
    size_t group_sizze_49523;
    size_t group_sizze_49581;
    size_t group_sizze_49623;
    size_t group_sizze_49663;
    size_t group_sizze_49705;
    size_t group_sizze_49786;
    size_t group_sizze_49858;
    size_t group_sizze_49901;
    size_t group_sizze_49944;
    size_t group_sizze_49990;
    size_t group_sizze_50036;
    size_t group_sizze_50123;
    size_t group_sizze_50138;
    size_t group_sizze_50202;
    size_t group_sizze_50217;
    size_t group_sizze_50283;
    size_t group_sizze_50298;
    size_t group_sizze_50315;
    size_t group_sizze_50330;
    size_t group_sizze_50364;
    size_t group_sizze_50404;
    size_t group_sizze_50445;
    size_t tile_sizze_50527;
    size_t group_sizze_50902;
} ;
struct futhark_context_config {
    struct opencl_config opencl;
    size_t sizes[56];
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
    cfg->sizes[20] = 0;
    cfg->sizes[21] = 0;
    cfg->sizes[22] = 0;
    cfg->sizes[23] = 0;
    cfg->sizes[24] = 0;
    cfg->sizes[25] = 0;
    cfg->sizes[26] = 0;
    cfg->sizes[27] = 0;
    cfg->sizes[28] = 0;
    cfg->sizes[29] = 0;
    cfg->sizes[30] = 0;
    cfg->sizes[31] = 0;
    cfg->sizes[32] = 0;
    cfg->sizes[33] = 0;
    cfg->sizes[34] = 0;
    cfg->sizes[35] = 0;
    cfg->sizes[36] = 0;
    cfg->sizes[37] = 0;
    cfg->sizes[38] = 0;
    cfg->sizes[39] = 0;
    cfg->sizes[40] = 0;
    cfg->sizes[41] = 0;
    cfg->sizes[42] = 0;
    cfg->sizes[43] = 0;
    cfg->sizes[44] = 0;
    cfg->sizes[45] = 0;
    cfg->sizes[46] = 0;
    cfg->sizes[47] = 0;
    cfg->sizes[48] = 0;
    cfg->sizes[49] = 0;
    cfg->sizes[50] = 0;
    cfg->sizes[51] = 0;
    cfg->sizes[52] = 0;
    cfg->sizes[53] = 0;
    cfg->sizes[54] = 0;
    cfg->sizes[55] = 0;
    opencl_config_init(&cfg->opencl, 56, size_names, cfg->sizes, size_classes,
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
    for (int i = 0; i < 56; i++) {
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
    cl_kernel chunked_reduce_kernel_48326;
    int chunked_reduce_kernel_48326_total_runtime;
    int chunked_reduce_kernel_48326_runs;
    cl_kernel chunked_reduce_kernel_48401;
    int chunked_reduce_kernel_48401_total_runtime;
    int chunked_reduce_kernel_48401_runs;
    cl_kernel chunked_reduce_kernel_48470;
    int chunked_reduce_kernel_48470_total_runtime;
    int chunked_reduce_kernel_48470_runs;
    cl_kernel chunked_reduce_kernel_48543;
    int chunked_reduce_kernel_48543_total_runtime;
    int chunked_reduce_kernel_48543_runs;
    cl_kernel chunked_reduce_kernel_48619;
    int chunked_reduce_kernel_48619_total_runtime;
    int chunked_reduce_kernel_48619_runs;
    cl_kernel chunked_reduce_kernel_48698;
    int chunked_reduce_kernel_48698_total_runtime;
    int chunked_reduce_kernel_48698_runs;
    cl_kernel chunked_reduce_kernel_48781;
    int chunked_reduce_kernel_48781_total_runtime;
    int chunked_reduce_kernel_48781_runs;
    cl_kernel chunked_reduce_kernel_48858;
    int chunked_reduce_kernel_48858_total_runtime;
    int chunked_reduce_kernel_48858_runs;
    cl_kernel chunked_reduce_kernel_48938;
    int chunked_reduce_kernel_48938_total_runtime;
    int chunked_reduce_kernel_48938_runs;
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
    cl_kernel kernel_replicate_47146;
    int kernel_replicate_47146_total_runtime;
    int kernel_replicate_47146_runs;
    cl_kernel map_kernel_48370;
    int map_kernel_48370_total_runtime;
    int map_kernel_48370_runs;
    cl_kernel map_kernel_49020;
    int map_kernel_49020_total_runtime;
    int map_kernel_49020_runs;
    cl_kernel map_kernel_49060;
    int map_kernel_49060_total_runtime;
    int map_kernel_49060_runs;
    cl_kernel map_kernel_49104;
    int map_kernel_49104_total_runtime;
    int map_kernel_49104_runs;
    cl_kernel map_kernel_49138;
    int map_kernel_49138_total_runtime;
    int map_kernel_49138_runs;
    cl_kernel map_kernel_49164;
    int map_kernel_49164_total_runtime;
    int map_kernel_49164_runs;
    cl_kernel map_kernel_49198;
    int map_kernel_49198_total_runtime;
    int map_kernel_49198_runs;
    cl_kernel map_kernel_49226;
    int map_kernel_49226_total_runtime;
    int map_kernel_49226_runs;
    cl_kernel map_kernel_49273;
    int map_kernel_49273_total_runtime;
    int map_kernel_49273_runs;
    cl_kernel map_kernel_49316;
    int map_kernel_49316_total_runtime;
    int map_kernel_49316_runs;
    cl_kernel map_kernel_49356;
    int map_kernel_49356_total_runtime;
    int map_kernel_49356_runs;
    cl_kernel map_kernel_49385;
    int map_kernel_49385_total_runtime;
    int map_kernel_49385_runs;
    cl_kernel map_kernel_49420;
    int map_kernel_49420_total_runtime;
    int map_kernel_49420_runs;
    cl_kernel map_kernel_49478;
    int map_kernel_49478_total_runtime;
    int map_kernel_49478_runs;
    cl_kernel map_kernel_49529;
    int map_kernel_49529_total_runtime;
    int map_kernel_49529_runs;
    cl_kernel map_kernel_49587;
    int map_kernel_49587_total_runtime;
    int map_kernel_49587_runs;
    cl_kernel map_kernel_49629;
    int map_kernel_49629_total_runtime;
    int map_kernel_49629_runs;
    cl_kernel map_kernel_49669;
    int map_kernel_49669_total_runtime;
    int map_kernel_49669_runs;
    cl_kernel map_kernel_49711;
    int map_kernel_49711_total_runtime;
    int map_kernel_49711_runs;
    cl_kernel map_kernel_49760;
    int map_kernel_49760_total_runtime;
    int map_kernel_49760_runs;
    cl_kernel map_kernel_49792;
    int map_kernel_49792_total_runtime;
    int map_kernel_49792_runs;
    cl_kernel map_kernel_49820;
    int map_kernel_49820_total_runtime;
    int map_kernel_49820_runs;
    cl_kernel map_kernel_49864;
    int map_kernel_49864_total_runtime;
    int map_kernel_49864_runs;
    cl_kernel map_kernel_49907;
    int map_kernel_49907_total_runtime;
    int map_kernel_49907_runs;
    cl_kernel map_kernel_49950;
    int map_kernel_49950_total_runtime;
    int map_kernel_49950_runs;
    cl_kernel map_kernel_49996;
    int map_kernel_49996_total_runtime;
    int map_kernel_49996_runs;
    cl_kernel map_kernel_50042;
    int map_kernel_50042_total_runtime;
    int map_kernel_50042_runs;
    cl_kernel map_kernel_50095;
    int map_kernel_50095_total_runtime;
    int map_kernel_50095_runs;
    cl_kernel map_kernel_50129;
    int map_kernel_50129_total_runtime;
    int map_kernel_50129_runs;
    cl_kernel map_kernel_50144;
    int map_kernel_50144_total_runtime;
    int map_kernel_50144_runs;
    cl_kernel map_kernel_50174;
    int map_kernel_50174_total_runtime;
    int map_kernel_50174_runs;
    cl_kernel map_kernel_50208;
    int map_kernel_50208_total_runtime;
    int map_kernel_50208_runs;
    cl_kernel map_kernel_50223;
    int map_kernel_50223_total_runtime;
    int map_kernel_50223_runs;
    cl_kernel map_kernel_50255;
    int map_kernel_50255_total_runtime;
    int map_kernel_50255_runs;
    cl_kernel map_kernel_50289;
    int map_kernel_50289_total_runtime;
    int map_kernel_50289_runs;
    cl_kernel map_kernel_50304;
    int map_kernel_50304_total_runtime;
    int map_kernel_50304_runs;
    cl_kernel map_kernel_50321;
    int map_kernel_50321_total_runtime;
    int map_kernel_50321_runs;
    cl_kernel map_kernel_50336;
    int map_kernel_50336_total_runtime;
    int map_kernel_50336_runs;
    cl_kernel map_kernel_50370;
    int map_kernel_50370_total_runtime;
    int map_kernel_50370_runs;
    cl_kernel map_kernel_50410;
    int map_kernel_50410_total_runtime;
    int map_kernel_50410_runs;
    cl_kernel map_kernel_50451;
    int map_kernel_50451_total_runtime;
    int map_kernel_50451_runs;
    cl_kernel reduce_kernel_48354;
    int reduce_kernel_48354_total_runtime;
    int reduce_kernel_48354_runs;
    cl_kernel reduce_kernel_48433;
    int reduce_kernel_48433_total_runtime;
    int reduce_kernel_48433_runs;
    cl_kernel reduce_kernel_48506;
    int reduce_kernel_48506_total_runtime;
    int reduce_kernel_48506_runs;
    cl_kernel reduce_kernel_48580;
    int reduce_kernel_48580_total_runtime;
    int reduce_kernel_48580_runs;
    cl_kernel reduce_kernel_48659;
    int reduce_kernel_48659_total_runtime;
    int reduce_kernel_48659_runs;
    cl_kernel reduce_kernel_48742;
    int reduce_kernel_48742_total_runtime;
    int reduce_kernel_48742_runs;
    cl_kernel reduce_kernel_48821;
    int reduce_kernel_48821_total_runtime;
    int reduce_kernel_48821_runs;
    cl_kernel reduce_kernel_48894;
    int reduce_kernel_48894_total_runtime;
    int reduce_kernel_48894_runs;
    cl_kernel reduce_kernel_48993;
    int reduce_kernel_48993_total_runtime;
    int reduce_kernel_48993_runs;
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
    ctx->chunked_reduce_kernel_48326_total_runtime = 0;
    ctx->chunked_reduce_kernel_48326_runs = 0;
    ctx->chunked_reduce_kernel_48401_total_runtime = 0;
    ctx->chunked_reduce_kernel_48401_runs = 0;
    ctx->chunked_reduce_kernel_48470_total_runtime = 0;
    ctx->chunked_reduce_kernel_48470_runs = 0;
    ctx->chunked_reduce_kernel_48543_total_runtime = 0;
    ctx->chunked_reduce_kernel_48543_runs = 0;
    ctx->chunked_reduce_kernel_48619_total_runtime = 0;
    ctx->chunked_reduce_kernel_48619_runs = 0;
    ctx->chunked_reduce_kernel_48698_total_runtime = 0;
    ctx->chunked_reduce_kernel_48698_runs = 0;
    ctx->chunked_reduce_kernel_48781_total_runtime = 0;
    ctx->chunked_reduce_kernel_48781_runs = 0;
    ctx->chunked_reduce_kernel_48858_total_runtime = 0;
    ctx->chunked_reduce_kernel_48858_runs = 0;
    ctx->chunked_reduce_kernel_48938_total_runtime = 0;
    ctx->chunked_reduce_kernel_48938_runs = 0;
    ctx->fut_kernel_map_transpose_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowheight_f32_runs = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_lowwidth_f32_runs = 0;
    ctx->fut_kernel_map_transpose_small_f32_total_runtime = 0;
    ctx->fut_kernel_map_transpose_small_f32_runs = 0;
    ctx->kernel_replicate_47146_total_runtime = 0;
    ctx->kernel_replicate_47146_runs = 0;
    ctx->map_kernel_48370_total_runtime = 0;
    ctx->map_kernel_48370_runs = 0;
    ctx->map_kernel_49020_total_runtime = 0;
    ctx->map_kernel_49020_runs = 0;
    ctx->map_kernel_49060_total_runtime = 0;
    ctx->map_kernel_49060_runs = 0;
    ctx->map_kernel_49104_total_runtime = 0;
    ctx->map_kernel_49104_runs = 0;
    ctx->map_kernel_49138_total_runtime = 0;
    ctx->map_kernel_49138_runs = 0;
    ctx->map_kernel_49164_total_runtime = 0;
    ctx->map_kernel_49164_runs = 0;
    ctx->map_kernel_49198_total_runtime = 0;
    ctx->map_kernel_49198_runs = 0;
    ctx->map_kernel_49226_total_runtime = 0;
    ctx->map_kernel_49226_runs = 0;
    ctx->map_kernel_49273_total_runtime = 0;
    ctx->map_kernel_49273_runs = 0;
    ctx->map_kernel_49316_total_runtime = 0;
    ctx->map_kernel_49316_runs = 0;
    ctx->map_kernel_49356_total_runtime = 0;
    ctx->map_kernel_49356_runs = 0;
    ctx->map_kernel_49385_total_runtime = 0;
    ctx->map_kernel_49385_runs = 0;
    ctx->map_kernel_49420_total_runtime = 0;
    ctx->map_kernel_49420_runs = 0;
    ctx->map_kernel_49478_total_runtime = 0;
    ctx->map_kernel_49478_runs = 0;
    ctx->map_kernel_49529_total_runtime = 0;
    ctx->map_kernel_49529_runs = 0;
    ctx->map_kernel_49587_total_runtime = 0;
    ctx->map_kernel_49587_runs = 0;
    ctx->map_kernel_49629_total_runtime = 0;
    ctx->map_kernel_49629_runs = 0;
    ctx->map_kernel_49669_total_runtime = 0;
    ctx->map_kernel_49669_runs = 0;
    ctx->map_kernel_49711_total_runtime = 0;
    ctx->map_kernel_49711_runs = 0;
    ctx->map_kernel_49760_total_runtime = 0;
    ctx->map_kernel_49760_runs = 0;
    ctx->map_kernel_49792_total_runtime = 0;
    ctx->map_kernel_49792_runs = 0;
    ctx->map_kernel_49820_total_runtime = 0;
    ctx->map_kernel_49820_runs = 0;
    ctx->map_kernel_49864_total_runtime = 0;
    ctx->map_kernel_49864_runs = 0;
    ctx->map_kernel_49907_total_runtime = 0;
    ctx->map_kernel_49907_runs = 0;
    ctx->map_kernel_49950_total_runtime = 0;
    ctx->map_kernel_49950_runs = 0;
    ctx->map_kernel_49996_total_runtime = 0;
    ctx->map_kernel_49996_runs = 0;
    ctx->map_kernel_50042_total_runtime = 0;
    ctx->map_kernel_50042_runs = 0;
    ctx->map_kernel_50095_total_runtime = 0;
    ctx->map_kernel_50095_runs = 0;
    ctx->map_kernel_50129_total_runtime = 0;
    ctx->map_kernel_50129_runs = 0;
    ctx->map_kernel_50144_total_runtime = 0;
    ctx->map_kernel_50144_runs = 0;
    ctx->map_kernel_50174_total_runtime = 0;
    ctx->map_kernel_50174_runs = 0;
    ctx->map_kernel_50208_total_runtime = 0;
    ctx->map_kernel_50208_runs = 0;
    ctx->map_kernel_50223_total_runtime = 0;
    ctx->map_kernel_50223_runs = 0;
    ctx->map_kernel_50255_total_runtime = 0;
    ctx->map_kernel_50255_runs = 0;
    ctx->map_kernel_50289_total_runtime = 0;
    ctx->map_kernel_50289_runs = 0;
    ctx->map_kernel_50304_total_runtime = 0;
    ctx->map_kernel_50304_runs = 0;
    ctx->map_kernel_50321_total_runtime = 0;
    ctx->map_kernel_50321_runs = 0;
    ctx->map_kernel_50336_total_runtime = 0;
    ctx->map_kernel_50336_runs = 0;
    ctx->map_kernel_50370_total_runtime = 0;
    ctx->map_kernel_50370_runs = 0;
    ctx->map_kernel_50410_total_runtime = 0;
    ctx->map_kernel_50410_runs = 0;
    ctx->map_kernel_50451_total_runtime = 0;
    ctx->map_kernel_50451_runs = 0;
    ctx->reduce_kernel_48354_total_runtime = 0;
    ctx->reduce_kernel_48354_runs = 0;
    ctx->reduce_kernel_48433_total_runtime = 0;
    ctx->reduce_kernel_48433_runs = 0;
    ctx->reduce_kernel_48506_total_runtime = 0;
    ctx->reduce_kernel_48506_runs = 0;
    ctx->reduce_kernel_48580_total_runtime = 0;
    ctx->reduce_kernel_48580_runs = 0;
    ctx->reduce_kernel_48659_total_runtime = 0;
    ctx->reduce_kernel_48659_runs = 0;
    ctx->reduce_kernel_48742_total_runtime = 0;
    ctx->reduce_kernel_48742_runs = 0;
    ctx->reduce_kernel_48821_total_runtime = 0;
    ctx->reduce_kernel_48821_runs = 0;
    ctx->reduce_kernel_48894_total_runtime = 0;
    ctx->reduce_kernel_48894_runs = 0;
    ctx->reduce_kernel_48993_total_runtime = 0;
    ctx->reduce_kernel_48993_runs = 0;
    
    int required_types = 0;
    cl_int error;
    cl_program prog = setup_opencl(&ctx->opencl, opencl_program,
                                   required_types);
    
    {
        ctx->chunked_reduce_kernel_48326 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48326",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48326");
    }
    {
        ctx->chunked_reduce_kernel_48401 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48401",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48401");
    }
    {
        ctx->chunked_reduce_kernel_48470 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48470",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48470");
    }
    {
        ctx->chunked_reduce_kernel_48543 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48543",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48543");
    }
    {
        ctx->chunked_reduce_kernel_48619 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48619",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48619");
    }
    {
        ctx->chunked_reduce_kernel_48698 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48698",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48698");
    }
    {
        ctx->chunked_reduce_kernel_48781 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48781",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48781");
    }
    {
        ctx->chunked_reduce_kernel_48858 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48858",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48858");
    }
    {
        ctx->chunked_reduce_kernel_48938 = clCreateKernel(prog,
                                                          "chunked_reduce_kernel_48938",
                                                          &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n",
                    "chunked_reduce_kernel_48938");
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
        ctx->kernel_replicate_47146 = clCreateKernel(prog,
                                                     "kernel_replicate_47146",
                                                     &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "kernel_replicate_47146");
    }
    {
        ctx->map_kernel_48370 = clCreateKernel(prog, "map_kernel_48370",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_48370");
    }
    {
        ctx->map_kernel_49020 = clCreateKernel(prog, "map_kernel_49020",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49020");
    }
    {
        ctx->map_kernel_49060 = clCreateKernel(prog, "map_kernel_49060",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49060");
    }
    {
        ctx->map_kernel_49104 = clCreateKernel(prog, "map_kernel_49104",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49104");
    }
    {
        ctx->map_kernel_49138 = clCreateKernel(prog, "map_kernel_49138",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49138");
    }
    {
        ctx->map_kernel_49164 = clCreateKernel(prog, "map_kernel_49164",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49164");
    }
    {
        ctx->map_kernel_49198 = clCreateKernel(prog, "map_kernel_49198",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49198");
    }
    {
        ctx->map_kernel_49226 = clCreateKernel(prog, "map_kernel_49226",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49226");
    }
    {
        ctx->map_kernel_49273 = clCreateKernel(prog, "map_kernel_49273",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49273");
    }
    {
        ctx->map_kernel_49316 = clCreateKernel(prog, "map_kernel_49316",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49316");
    }
    {
        ctx->map_kernel_49356 = clCreateKernel(prog, "map_kernel_49356",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49356");
    }
    {
        ctx->map_kernel_49385 = clCreateKernel(prog, "map_kernel_49385",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49385");
    }
    {
        ctx->map_kernel_49420 = clCreateKernel(prog, "map_kernel_49420",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49420");
    }
    {
        ctx->map_kernel_49478 = clCreateKernel(prog, "map_kernel_49478",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49478");
    }
    {
        ctx->map_kernel_49529 = clCreateKernel(prog, "map_kernel_49529",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49529");
    }
    {
        ctx->map_kernel_49587 = clCreateKernel(prog, "map_kernel_49587",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49587");
    }
    {
        ctx->map_kernel_49629 = clCreateKernel(prog, "map_kernel_49629",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49629");
    }
    {
        ctx->map_kernel_49669 = clCreateKernel(prog, "map_kernel_49669",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49669");
    }
    {
        ctx->map_kernel_49711 = clCreateKernel(prog, "map_kernel_49711",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49711");
    }
    {
        ctx->map_kernel_49760 = clCreateKernel(prog, "map_kernel_49760",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49760");
    }
    {
        ctx->map_kernel_49792 = clCreateKernel(prog, "map_kernel_49792",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49792");
    }
    {
        ctx->map_kernel_49820 = clCreateKernel(prog, "map_kernel_49820",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49820");
    }
    {
        ctx->map_kernel_49864 = clCreateKernel(prog, "map_kernel_49864",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49864");
    }
    {
        ctx->map_kernel_49907 = clCreateKernel(prog, "map_kernel_49907",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49907");
    }
    {
        ctx->map_kernel_49950 = clCreateKernel(prog, "map_kernel_49950",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49950");
    }
    {
        ctx->map_kernel_49996 = clCreateKernel(prog, "map_kernel_49996",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_49996");
    }
    {
        ctx->map_kernel_50042 = clCreateKernel(prog, "map_kernel_50042",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50042");
    }
    {
        ctx->map_kernel_50095 = clCreateKernel(prog, "map_kernel_50095",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50095");
    }
    {
        ctx->map_kernel_50129 = clCreateKernel(prog, "map_kernel_50129",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50129");
    }
    {
        ctx->map_kernel_50144 = clCreateKernel(prog, "map_kernel_50144",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50144");
    }
    {
        ctx->map_kernel_50174 = clCreateKernel(prog, "map_kernel_50174",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50174");
    }
    {
        ctx->map_kernel_50208 = clCreateKernel(prog, "map_kernel_50208",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50208");
    }
    {
        ctx->map_kernel_50223 = clCreateKernel(prog, "map_kernel_50223",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50223");
    }
    {
        ctx->map_kernel_50255 = clCreateKernel(prog, "map_kernel_50255",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50255");
    }
    {
        ctx->map_kernel_50289 = clCreateKernel(prog, "map_kernel_50289",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50289");
    }
    {
        ctx->map_kernel_50304 = clCreateKernel(prog, "map_kernel_50304",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50304");
    }
    {
        ctx->map_kernel_50321 = clCreateKernel(prog, "map_kernel_50321",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50321");
    }
    {
        ctx->map_kernel_50336 = clCreateKernel(prog, "map_kernel_50336",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50336");
    }
    {
        ctx->map_kernel_50370 = clCreateKernel(prog, "map_kernel_50370",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50370");
    }
    {
        ctx->map_kernel_50410 = clCreateKernel(prog, "map_kernel_50410",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50410");
    }
    {
        ctx->map_kernel_50451 = clCreateKernel(prog, "map_kernel_50451",
                                               &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_50451");
    }
    {
        ctx->reduce_kernel_48354 = clCreateKernel(prog, "reduce_kernel_48354",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48354");
    }
    {
        ctx->reduce_kernel_48433 = clCreateKernel(prog, "reduce_kernel_48433",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48433");
    }
    {
        ctx->reduce_kernel_48506 = clCreateKernel(prog, "reduce_kernel_48506",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48506");
    }
    {
        ctx->reduce_kernel_48580 = clCreateKernel(prog, "reduce_kernel_48580",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48580");
    }
    {
        ctx->reduce_kernel_48659 = clCreateKernel(prog, "reduce_kernel_48659",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48659");
    }
    {
        ctx->reduce_kernel_48742 = clCreateKernel(prog, "reduce_kernel_48742",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48742");
    }
    {
        ctx->reduce_kernel_48821 = clCreateKernel(prog, "reduce_kernel_48821",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48821");
    }
    {
        ctx->reduce_kernel_48894 = clCreateKernel(prog, "reduce_kernel_48894",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48894");
    }
    {
        ctx->reduce_kernel_48993 = clCreateKernel(prog, "reduce_kernel_48993",
                                                  &error);
        assert(error == 0);
        if (ctx->debugging)
            fprintf(stderr, "Created kernel %s.\n", "reduce_kernel_48993");
    }
    ctx->sizes.group_sizze_48309 = cfg->sizes[0];
    ctx->sizes.max_num_groups_48311 = cfg->sizes[1];
    ctx->sizes.group_sizze_48364 = cfg->sizes[2];
    ctx->sizes.group_sizze_48384 = cfg->sizes[3];
    ctx->sizes.max_num_groups_48386 = cfg->sizes[4];
    ctx->sizes.group_sizze_48453 = cfg->sizes[5];
    ctx->sizes.max_num_groups_48455 = cfg->sizes[6];
    ctx->sizes.group_sizze_48526 = cfg->sizes[7];
    ctx->sizes.max_num_groups_48528 = cfg->sizes[8];
    ctx->sizes.group_sizze_48602 = cfg->sizes[9];
    ctx->sizes.max_num_groups_48604 = cfg->sizes[10];
    ctx->sizes.group_sizze_48681 = cfg->sizes[11];
    ctx->sizes.max_num_groups_48683 = cfg->sizes[12];
    ctx->sizes.group_sizze_48764 = cfg->sizes[13];
    ctx->sizes.max_num_groups_48766 = cfg->sizes[14];
    ctx->sizes.group_sizze_48841 = cfg->sizes[15];
    ctx->sizes.max_num_groups_48843 = cfg->sizes[16];
    ctx->sizes.group_sizze_48921 = cfg->sizes[17];
    ctx->sizes.max_num_groups_48923 = cfg->sizes[18];
    ctx->sizes.group_sizze_49014 = cfg->sizes[19];
    ctx->sizes.group_sizze_49054 = cfg->sizes[20];
    ctx->sizes.group_sizze_49098 = cfg->sizes[21];
    ctx->sizes.group_sizze_49132 = cfg->sizes[22];
    ctx->sizes.group_sizze_49158 = cfg->sizes[23];
    ctx->sizes.group_sizze_49192 = cfg->sizes[24];
    ctx->sizes.group_sizze_49220 = cfg->sizes[25];
    ctx->sizes.group_sizze_49267 = cfg->sizes[26];
    ctx->sizes.group_sizze_49310 = cfg->sizes[27];
    ctx->sizes.group_sizze_49350 = cfg->sizes[28];
    ctx->sizes.group_sizze_49379 = cfg->sizes[29];
    ctx->sizes.group_sizze_49414 = cfg->sizes[30];
    ctx->sizes.group_sizze_49472 = cfg->sizes[31];
    ctx->sizes.group_sizze_49523 = cfg->sizes[32];
    ctx->sizes.group_sizze_49581 = cfg->sizes[33];
    ctx->sizes.group_sizze_49623 = cfg->sizes[34];
    ctx->sizes.group_sizze_49663 = cfg->sizes[35];
    ctx->sizes.group_sizze_49705 = cfg->sizes[36];
    ctx->sizes.group_sizze_49786 = cfg->sizes[37];
    ctx->sizes.group_sizze_49858 = cfg->sizes[38];
    ctx->sizes.group_sizze_49901 = cfg->sizes[39];
    ctx->sizes.group_sizze_49944 = cfg->sizes[40];
    ctx->sizes.group_sizze_49990 = cfg->sizes[41];
    ctx->sizes.group_sizze_50036 = cfg->sizes[42];
    ctx->sizes.group_sizze_50123 = cfg->sizes[43];
    ctx->sizes.group_sizze_50138 = cfg->sizes[44];
    ctx->sizes.group_sizze_50202 = cfg->sizes[45];
    ctx->sizes.group_sizze_50217 = cfg->sizes[46];
    ctx->sizes.group_sizze_50283 = cfg->sizes[47];
    ctx->sizes.group_sizze_50298 = cfg->sizes[48];
    ctx->sizes.group_sizze_50315 = cfg->sizes[49];
    ctx->sizes.group_sizze_50330 = cfg->sizes[50];
    ctx->sizes.group_sizze_50364 = cfg->sizes[51];
    ctx->sizes.group_sizze_50404 = cfg->sizes[52];
    ctx->sizes.group_sizze_50445 = cfg->sizes[53];
    ctx->sizes.tile_sizze_50527 = cfg->sizes[54];
    ctx->sizes.group_sizze_50902 = cfg->sizes[55];
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
                "Kernel chunked_reduce_kernel_48326            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48326_runs,
                (long) ctx->chunked_reduce_kernel_48326_total_runtime /
                (ctx->chunked_reduce_kernel_48326_runs !=
                 0 ? ctx->chunked_reduce_kernel_48326_runs : 1),
                (long) ctx->chunked_reduce_kernel_48326_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48326_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48326_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48401            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48401_runs,
                (long) ctx->chunked_reduce_kernel_48401_total_runtime /
                (ctx->chunked_reduce_kernel_48401_runs !=
                 0 ? ctx->chunked_reduce_kernel_48401_runs : 1),
                (long) ctx->chunked_reduce_kernel_48401_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48401_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48401_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48470            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48470_runs,
                (long) ctx->chunked_reduce_kernel_48470_total_runtime /
                (ctx->chunked_reduce_kernel_48470_runs !=
                 0 ? ctx->chunked_reduce_kernel_48470_runs : 1),
                (long) ctx->chunked_reduce_kernel_48470_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48470_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48470_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48543            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48543_runs,
                (long) ctx->chunked_reduce_kernel_48543_total_runtime /
                (ctx->chunked_reduce_kernel_48543_runs !=
                 0 ? ctx->chunked_reduce_kernel_48543_runs : 1),
                (long) ctx->chunked_reduce_kernel_48543_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48543_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48543_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48619            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48619_runs,
                (long) ctx->chunked_reduce_kernel_48619_total_runtime /
                (ctx->chunked_reduce_kernel_48619_runs !=
                 0 ? ctx->chunked_reduce_kernel_48619_runs : 1),
                (long) ctx->chunked_reduce_kernel_48619_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48619_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48619_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48698            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48698_runs,
                (long) ctx->chunked_reduce_kernel_48698_total_runtime /
                (ctx->chunked_reduce_kernel_48698_runs !=
                 0 ? ctx->chunked_reduce_kernel_48698_runs : 1),
                (long) ctx->chunked_reduce_kernel_48698_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48698_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48698_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48781            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48781_runs,
                (long) ctx->chunked_reduce_kernel_48781_total_runtime /
                (ctx->chunked_reduce_kernel_48781_runs !=
                 0 ? ctx->chunked_reduce_kernel_48781_runs : 1),
                (long) ctx->chunked_reduce_kernel_48781_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48781_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48781_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48858            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48858_runs,
                (long) ctx->chunked_reduce_kernel_48858_total_runtime /
                (ctx->chunked_reduce_kernel_48858_runs !=
                 0 ? ctx->chunked_reduce_kernel_48858_runs : 1),
                (long) ctx->chunked_reduce_kernel_48858_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48858_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48858_runs;
        fprintf(stderr,
                "Kernel chunked_reduce_kernel_48938            executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->chunked_reduce_kernel_48938_runs,
                (long) ctx->chunked_reduce_kernel_48938_total_runtime /
                (ctx->chunked_reduce_kernel_48938_runs !=
                 0 ? ctx->chunked_reduce_kernel_48938_runs : 1),
                (long) ctx->chunked_reduce_kernel_48938_total_runtime);
        ctx->total_runtime += ctx->chunked_reduce_kernel_48938_total_runtime;
        ctx->total_runs += ctx->chunked_reduce_kernel_48938_runs;
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
                "Kernel kernel_replicate_47146                 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->kernel_replicate_47146_runs,
                (long) ctx->kernel_replicate_47146_total_runtime /
                (ctx->kernel_replicate_47146_runs !=
                 0 ? ctx->kernel_replicate_47146_runs : 1),
                (long) ctx->kernel_replicate_47146_total_runtime);
        ctx->total_runtime += ctx->kernel_replicate_47146_total_runtime;
        ctx->total_runs += ctx->kernel_replicate_47146_runs;
        fprintf(stderr,
                "Kernel map_kernel_48370                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_48370_runs,
                (long) ctx->map_kernel_48370_total_runtime /
                (ctx->map_kernel_48370_runs !=
                 0 ? ctx->map_kernel_48370_runs : 1),
                (long) ctx->map_kernel_48370_total_runtime);
        ctx->total_runtime += ctx->map_kernel_48370_total_runtime;
        ctx->total_runs += ctx->map_kernel_48370_runs;
        fprintf(stderr,
                "Kernel map_kernel_49020                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49020_runs,
                (long) ctx->map_kernel_49020_total_runtime /
                (ctx->map_kernel_49020_runs !=
                 0 ? ctx->map_kernel_49020_runs : 1),
                (long) ctx->map_kernel_49020_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49020_total_runtime;
        ctx->total_runs += ctx->map_kernel_49020_runs;
        fprintf(stderr,
                "Kernel map_kernel_49060                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49060_runs,
                (long) ctx->map_kernel_49060_total_runtime /
                (ctx->map_kernel_49060_runs !=
                 0 ? ctx->map_kernel_49060_runs : 1),
                (long) ctx->map_kernel_49060_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49060_total_runtime;
        ctx->total_runs += ctx->map_kernel_49060_runs;
        fprintf(stderr,
                "Kernel map_kernel_49104                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49104_runs,
                (long) ctx->map_kernel_49104_total_runtime /
                (ctx->map_kernel_49104_runs !=
                 0 ? ctx->map_kernel_49104_runs : 1),
                (long) ctx->map_kernel_49104_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49104_total_runtime;
        ctx->total_runs += ctx->map_kernel_49104_runs;
        fprintf(stderr,
                "Kernel map_kernel_49138                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49138_runs,
                (long) ctx->map_kernel_49138_total_runtime /
                (ctx->map_kernel_49138_runs !=
                 0 ? ctx->map_kernel_49138_runs : 1),
                (long) ctx->map_kernel_49138_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49138_total_runtime;
        ctx->total_runs += ctx->map_kernel_49138_runs;
        fprintf(stderr,
                "Kernel map_kernel_49164                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49164_runs,
                (long) ctx->map_kernel_49164_total_runtime /
                (ctx->map_kernel_49164_runs !=
                 0 ? ctx->map_kernel_49164_runs : 1),
                (long) ctx->map_kernel_49164_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49164_total_runtime;
        ctx->total_runs += ctx->map_kernel_49164_runs;
        fprintf(stderr,
                "Kernel map_kernel_49198                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49198_runs,
                (long) ctx->map_kernel_49198_total_runtime /
                (ctx->map_kernel_49198_runs !=
                 0 ? ctx->map_kernel_49198_runs : 1),
                (long) ctx->map_kernel_49198_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49198_total_runtime;
        ctx->total_runs += ctx->map_kernel_49198_runs;
        fprintf(stderr,
                "Kernel map_kernel_49226                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49226_runs,
                (long) ctx->map_kernel_49226_total_runtime /
                (ctx->map_kernel_49226_runs !=
                 0 ? ctx->map_kernel_49226_runs : 1),
                (long) ctx->map_kernel_49226_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49226_total_runtime;
        ctx->total_runs += ctx->map_kernel_49226_runs;
        fprintf(stderr,
                "Kernel map_kernel_49273                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49273_runs,
                (long) ctx->map_kernel_49273_total_runtime /
                (ctx->map_kernel_49273_runs !=
                 0 ? ctx->map_kernel_49273_runs : 1),
                (long) ctx->map_kernel_49273_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49273_total_runtime;
        ctx->total_runs += ctx->map_kernel_49273_runs;
        fprintf(stderr,
                "Kernel map_kernel_49316                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49316_runs,
                (long) ctx->map_kernel_49316_total_runtime /
                (ctx->map_kernel_49316_runs !=
                 0 ? ctx->map_kernel_49316_runs : 1),
                (long) ctx->map_kernel_49316_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49316_total_runtime;
        ctx->total_runs += ctx->map_kernel_49316_runs;
        fprintf(stderr,
                "Kernel map_kernel_49356                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49356_runs,
                (long) ctx->map_kernel_49356_total_runtime /
                (ctx->map_kernel_49356_runs !=
                 0 ? ctx->map_kernel_49356_runs : 1),
                (long) ctx->map_kernel_49356_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49356_total_runtime;
        ctx->total_runs += ctx->map_kernel_49356_runs;
        fprintf(stderr,
                "Kernel map_kernel_49385                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49385_runs,
                (long) ctx->map_kernel_49385_total_runtime /
                (ctx->map_kernel_49385_runs !=
                 0 ? ctx->map_kernel_49385_runs : 1),
                (long) ctx->map_kernel_49385_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49385_total_runtime;
        ctx->total_runs += ctx->map_kernel_49385_runs;
        fprintf(stderr,
                "Kernel map_kernel_49420                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49420_runs,
                (long) ctx->map_kernel_49420_total_runtime /
                (ctx->map_kernel_49420_runs !=
                 0 ? ctx->map_kernel_49420_runs : 1),
                (long) ctx->map_kernel_49420_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49420_total_runtime;
        ctx->total_runs += ctx->map_kernel_49420_runs;
        fprintf(stderr,
                "Kernel map_kernel_49478                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49478_runs,
                (long) ctx->map_kernel_49478_total_runtime /
                (ctx->map_kernel_49478_runs !=
                 0 ? ctx->map_kernel_49478_runs : 1),
                (long) ctx->map_kernel_49478_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49478_total_runtime;
        ctx->total_runs += ctx->map_kernel_49478_runs;
        fprintf(stderr,
                "Kernel map_kernel_49529                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49529_runs,
                (long) ctx->map_kernel_49529_total_runtime /
                (ctx->map_kernel_49529_runs !=
                 0 ? ctx->map_kernel_49529_runs : 1),
                (long) ctx->map_kernel_49529_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49529_total_runtime;
        ctx->total_runs += ctx->map_kernel_49529_runs;
        fprintf(stderr,
                "Kernel map_kernel_49587                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49587_runs,
                (long) ctx->map_kernel_49587_total_runtime /
                (ctx->map_kernel_49587_runs !=
                 0 ? ctx->map_kernel_49587_runs : 1),
                (long) ctx->map_kernel_49587_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49587_total_runtime;
        ctx->total_runs += ctx->map_kernel_49587_runs;
        fprintf(stderr,
                "Kernel map_kernel_49629                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49629_runs,
                (long) ctx->map_kernel_49629_total_runtime /
                (ctx->map_kernel_49629_runs !=
                 0 ? ctx->map_kernel_49629_runs : 1),
                (long) ctx->map_kernel_49629_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49629_total_runtime;
        ctx->total_runs += ctx->map_kernel_49629_runs;
        fprintf(stderr,
                "Kernel map_kernel_49669                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49669_runs,
                (long) ctx->map_kernel_49669_total_runtime /
                (ctx->map_kernel_49669_runs !=
                 0 ? ctx->map_kernel_49669_runs : 1),
                (long) ctx->map_kernel_49669_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49669_total_runtime;
        ctx->total_runs += ctx->map_kernel_49669_runs;
        fprintf(stderr,
                "Kernel map_kernel_49711                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49711_runs,
                (long) ctx->map_kernel_49711_total_runtime /
                (ctx->map_kernel_49711_runs !=
                 0 ? ctx->map_kernel_49711_runs : 1),
                (long) ctx->map_kernel_49711_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49711_total_runtime;
        ctx->total_runs += ctx->map_kernel_49711_runs;
        fprintf(stderr,
                "Kernel map_kernel_49760                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49760_runs,
                (long) ctx->map_kernel_49760_total_runtime /
                (ctx->map_kernel_49760_runs !=
                 0 ? ctx->map_kernel_49760_runs : 1),
                (long) ctx->map_kernel_49760_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49760_total_runtime;
        ctx->total_runs += ctx->map_kernel_49760_runs;
        fprintf(stderr,
                "Kernel map_kernel_49792                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49792_runs,
                (long) ctx->map_kernel_49792_total_runtime /
                (ctx->map_kernel_49792_runs !=
                 0 ? ctx->map_kernel_49792_runs : 1),
                (long) ctx->map_kernel_49792_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49792_total_runtime;
        ctx->total_runs += ctx->map_kernel_49792_runs;
        fprintf(stderr,
                "Kernel map_kernel_49820                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49820_runs,
                (long) ctx->map_kernel_49820_total_runtime /
                (ctx->map_kernel_49820_runs !=
                 0 ? ctx->map_kernel_49820_runs : 1),
                (long) ctx->map_kernel_49820_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49820_total_runtime;
        ctx->total_runs += ctx->map_kernel_49820_runs;
        fprintf(stderr,
                "Kernel map_kernel_49864                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49864_runs,
                (long) ctx->map_kernel_49864_total_runtime /
                (ctx->map_kernel_49864_runs !=
                 0 ? ctx->map_kernel_49864_runs : 1),
                (long) ctx->map_kernel_49864_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49864_total_runtime;
        ctx->total_runs += ctx->map_kernel_49864_runs;
        fprintf(stderr,
                "Kernel map_kernel_49907                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49907_runs,
                (long) ctx->map_kernel_49907_total_runtime /
                (ctx->map_kernel_49907_runs !=
                 0 ? ctx->map_kernel_49907_runs : 1),
                (long) ctx->map_kernel_49907_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49907_total_runtime;
        ctx->total_runs += ctx->map_kernel_49907_runs;
        fprintf(stderr,
                "Kernel map_kernel_49950                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49950_runs,
                (long) ctx->map_kernel_49950_total_runtime /
                (ctx->map_kernel_49950_runs !=
                 0 ? ctx->map_kernel_49950_runs : 1),
                (long) ctx->map_kernel_49950_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49950_total_runtime;
        ctx->total_runs += ctx->map_kernel_49950_runs;
        fprintf(stderr,
                "Kernel map_kernel_49996                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_49996_runs,
                (long) ctx->map_kernel_49996_total_runtime /
                (ctx->map_kernel_49996_runs !=
                 0 ? ctx->map_kernel_49996_runs : 1),
                (long) ctx->map_kernel_49996_total_runtime);
        ctx->total_runtime += ctx->map_kernel_49996_total_runtime;
        ctx->total_runs += ctx->map_kernel_49996_runs;
        fprintf(stderr,
                "Kernel map_kernel_50042                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50042_runs,
                (long) ctx->map_kernel_50042_total_runtime /
                (ctx->map_kernel_50042_runs !=
                 0 ? ctx->map_kernel_50042_runs : 1),
                (long) ctx->map_kernel_50042_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50042_total_runtime;
        ctx->total_runs += ctx->map_kernel_50042_runs;
        fprintf(stderr,
                "Kernel map_kernel_50095                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50095_runs,
                (long) ctx->map_kernel_50095_total_runtime /
                (ctx->map_kernel_50095_runs !=
                 0 ? ctx->map_kernel_50095_runs : 1),
                (long) ctx->map_kernel_50095_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50095_total_runtime;
        ctx->total_runs += ctx->map_kernel_50095_runs;
        fprintf(stderr,
                "Kernel map_kernel_50129                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50129_runs,
                (long) ctx->map_kernel_50129_total_runtime /
                (ctx->map_kernel_50129_runs !=
                 0 ? ctx->map_kernel_50129_runs : 1),
                (long) ctx->map_kernel_50129_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50129_total_runtime;
        ctx->total_runs += ctx->map_kernel_50129_runs;
        fprintf(stderr,
                "Kernel map_kernel_50144                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50144_runs,
                (long) ctx->map_kernel_50144_total_runtime /
                (ctx->map_kernel_50144_runs !=
                 0 ? ctx->map_kernel_50144_runs : 1),
                (long) ctx->map_kernel_50144_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50144_total_runtime;
        ctx->total_runs += ctx->map_kernel_50144_runs;
        fprintf(stderr,
                "Kernel map_kernel_50174                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50174_runs,
                (long) ctx->map_kernel_50174_total_runtime /
                (ctx->map_kernel_50174_runs !=
                 0 ? ctx->map_kernel_50174_runs : 1),
                (long) ctx->map_kernel_50174_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50174_total_runtime;
        ctx->total_runs += ctx->map_kernel_50174_runs;
        fprintf(stderr,
                "Kernel map_kernel_50208                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50208_runs,
                (long) ctx->map_kernel_50208_total_runtime /
                (ctx->map_kernel_50208_runs !=
                 0 ? ctx->map_kernel_50208_runs : 1),
                (long) ctx->map_kernel_50208_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50208_total_runtime;
        ctx->total_runs += ctx->map_kernel_50208_runs;
        fprintf(stderr,
                "Kernel map_kernel_50223                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50223_runs,
                (long) ctx->map_kernel_50223_total_runtime /
                (ctx->map_kernel_50223_runs !=
                 0 ? ctx->map_kernel_50223_runs : 1),
                (long) ctx->map_kernel_50223_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50223_total_runtime;
        ctx->total_runs += ctx->map_kernel_50223_runs;
        fprintf(stderr,
                "Kernel map_kernel_50255                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50255_runs,
                (long) ctx->map_kernel_50255_total_runtime /
                (ctx->map_kernel_50255_runs !=
                 0 ? ctx->map_kernel_50255_runs : 1),
                (long) ctx->map_kernel_50255_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50255_total_runtime;
        ctx->total_runs += ctx->map_kernel_50255_runs;
        fprintf(stderr,
                "Kernel map_kernel_50289                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50289_runs,
                (long) ctx->map_kernel_50289_total_runtime /
                (ctx->map_kernel_50289_runs !=
                 0 ? ctx->map_kernel_50289_runs : 1),
                (long) ctx->map_kernel_50289_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50289_total_runtime;
        ctx->total_runs += ctx->map_kernel_50289_runs;
        fprintf(stderr,
                "Kernel map_kernel_50304                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50304_runs,
                (long) ctx->map_kernel_50304_total_runtime /
                (ctx->map_kernel_50304_runs !=
                 0 ? ctx->map_kernel_50304_runs : 1),
                (long) ctx->map_kernel_50304_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50304_total_runtime;
        ctx->total_runs += ctx->map_kernel_50304_runs;
        fprintf(stderr,
                "Kernel map_kernel_50321                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50321_runs,
                (long) ctx->map_kernel_50321_total_runtime /
                (ctx->map_kernel_50321_runs !=
                 0 ? ctx->map_kernel_50321_runs : 1),
                (long) ctx->map_kernel_50321_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50321_total_runtime;
        ctx->total_runs += ctx->map_kernel_50321_runs;
        fprintf(stderr,
                "Kernel map_kernel_50336                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50336_runs,
                (long) ctx->map_kernel_50336_total_runtime /
                (ctx->map_kernel_50336_runs !=
                 0 ? ctx->map_kernel_50336_runs : 1),
                (long) ctx->map_kernel_50336_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50336_total_runtime;
        ctx->total_runs += ctx->map_kernel_50336_runs;
        fprintf(stderr,
                "Kernel map_kernel_50370                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50370_runs,
                (long) ctx->map_kernel_50370_total_runtime /
                (ctx->map_kernel_50370_runs !=
                 0 ? ctx->map_kernel_50370_runs : 1),
                (long) ctx->map_kernel_50370_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50370_total_runtime;
        ctx->total_runs += ctx->map_kernel_50370_runs;
        fprintf(stderr,
                "Kernel map_kernel_50410                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50410_runs,
                (long) ctx->map_kernel_50410_total_runtime /
                (ctx->map_kernel_50410_runs !=
                 0 ? ctx->map_kernel_50410_runs : 1),
                (long) ctx->map_kernel_50410_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50410_total_runtime;
        ctx->total_runs += ctx->map_kernel_50410_runs;
        fprintf(stderr,
                "Kernel map_kernel_50451                       executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->map_kernel_50451_runs,
                (long) ctx->map_kernel_50451_total_runtime /
                (ctx->map_kernel_50451_runs !=
                 0 ? ctx->map_kernel_50451_runs : 1),
                (long) ctx->map_kernel_50451_total_runtime);
        ctx->total_runtime += ctx->map_kernel_50451_total_runtime;
        ctx->total_runs += ctx->map_kernel_50451_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48354                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48354_runs,
                (long) ctx->reduce_kernel_48354_total_runtime /
                (ctx->reduce_kernel_48354_runs !=
                 0 ? ctx->reduce_kernel_48354_runs : 1),
                (long) ctx->reduce_kernel_48354_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48354_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48354_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48433                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48433_runs,
                (long) ctx->reduce_kernel_48433_total_runtime /
                (ctx->reduce_kernel_48433_runs !=
                 0 ? ctx->reduce_kernel_48433_runs : 1),
                (long) ctx->reduce_kernel_48433_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48433_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48433_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48506                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48506_runs,
                (long) ctx->reduce_kernel_48506_total_runtime /
                (ctx->reduce_kernel_48506_runs !=
                 0 ? ctx->reduce_kernel_48506_runs : 1),
                (long) ctx->reduce_kernel_48506_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48506_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48506_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48580                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48580_runs,
                (long) ctx->reduce_kernel_48580_total_runtime /
                (ctx->reduce_kernel_48580_runs !=
                 0 ? ctx->reduce_kernel_48580_runs : 1),
                (long) ctx->reduce_kernel_48580_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48580_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48580_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48659                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48659_runs,
                (long) ctx->reduce_kernel_48659_total_runtime /
                (ctx->reduce_kernel_48659_runs !=
                 0 ? ctx->reduce_kernel_48659_runs : 1),
                (long) ctx->reduce_kernel_48659_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48659_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48659_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48742                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48742_runs,
                (long) ctx->reduce_kernel_48742_total_runtime /
                (ctx->reduce_kernel_48742_runs !=
                 0 ? ctx->reduce_kernel_48742_runs : 1),
                (long) ctx->reduce_kernel_48742_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48742_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48742_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48821                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48821_runs,
                (long) ctx->reduce_kernel_48821_total_runtime /
                (ctx->reduce_kernel_48821_runs !=
                 0 ? ctx->reduce_kernel_48821_runs : 1),
                (long) ctx->reduce_kernel_48821_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48821_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48821_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48894                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48894_runs,
                (long) ctx->reduce_kernel_48894_total_runtime /
                (ctx->reduce_kernel_48894_runs !=
                 0 ? ctx->reduce_kernel_48894_runs : 1),
                (long) ctx->reduce_kernel_48894_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48894_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48894_runs;
        fprintf(stderr,
                "Kernel reduce_kernel_48993                    executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                ctx->reduce_kernel_48993_runs,
                (long) ctx->reduce_kernel_48993_total_runtime /
                (ctx->reduce_kernel_48993_runs !=
                 0 ? ctx->reduce_kernel_48993_runs : 1),
                (long) ctx->reduce_kernel_48993_total_runtime);
        ctx->total_runtime += ctx->reduce_kernel_48993_total_runtime;
        ctx->total_runs += ctx->reduce_kernel_48993_runs;
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
static int futrts_sum(struct futhark_context *ctx, float *out_scalar_out_51227,
                      int64_t A_mem_sizze_50608,
                      struct memblock_device A_mem_50609, int32_t sizze_46897);
static int futrts_prod(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_51239,
                       struct memblock_device *out_mem_p_51240,
                       int32_t *out_out_arrsizze_51241,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609, int32_t sizze_46904,
                       float c_46905);
static int futrts_dot(struct futhark_context *ctx, float *out_scalar_out_51247,
                      int64_t A_mem_sizze_50608,
                      struct memblock_device A_mem_50609,
                      int64_t B_mem_sizze_50610,
                      struct memblock_device B_mem_50611, int32_t sizze_46910,
                      int32_t sizze_46911);
static int futrts_dot2(struct futhark_context *ctx, float *out_scalar_out_51259,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_46928,
                       int32_t sizze_46929, int32_t sizze_46930);
static int futrts_dot3(struct futhark_context *ctx, float *out_scalar_out_51271,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_46956,
                       int32_t sizze_46957, int32_t sizze_46958);
static int futrts_dot4(struct futhark_context *ctx, float *out_scalar_out_51283,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_46985,
                       int32_t sizze_46986, int32_t sizze_46987,
                       int32_t sizze_46988);
static int futrts_dot5(struct futhark_context *ctx, float *out_scalar_out_51295,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_47023,
                       int32_t sizze_47024, int32_t sizze_47025,
                       int32_t sizze_47026, float a_47027, float b_47029,
                       float c_47031, float d_47033);
static int futrts_dot6(struct futhark_context *ctx, float *out_scalar_out_51307,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_47069,
                       int32_t sizze_47070, int32_t sizze_47071,
                       int32_t sizze_47072);
static int futrts_dot1(struct futhark_context *ctx, float *out_scalar_out_51319,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_47107,
                       int32_t sizze_47108, int32_t sizze_47109);
static int futrts_t1b_2d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51331,
                         struct memblock_device *out_mem_p_51332,
                         int32_t *out_out_arrsizze_51333,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int32_t sizze_47135, int32_t sizze_47136,
                         int32_t sizze_47137);
static int futrts_t1_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51351,
                        struct memblock_device *out_mem_p_51352,
                        int32_t *out_out_arrsizze_51353,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611, int32_t sizze_47160,
                        int32_t sizze_47161, int32_t sizze_47162);
static int futrts_t2_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51360,
                        struct memblock_device *out_mem_p_51361,
                        int32_t *out_out_arrsizze_51362,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47181,
                        int32_t sizze_47182, int32_t sizze_47183,
                        int32_t sizze_47184);
static int futrts_t3_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51369,
                        struct memblock_device *out_mem_p_51370,
                        int32_t *out_out_arrsizze_51371,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47213,
                        int32_t sizze_47214, int32_t sizze_47215,
                        int32_t sizze_47216, int32_t sizze_47217);
static int futrts_t4_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51379,
                        struct memblock_device *out_mem_p_51380,
                        int32_t *out_out_arrsizze_51381,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47252,
                        int32_t sizze_47253, int32_t sizze_47254,
                        int32_t sizze_47255, int32_t sizze_47256,
                        int32_t sizze_47257);
static int futrts_t5_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51394,
                        struct memblock_device *out_mem_p_51395,
                        int32_t *out_out_arrsizze_51396,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47303,
                        int32_t sizze_47304, int32_t sizze_47305,
                        int32_t sizze_47306, int32_t sizze_47307,
                        int32_t sizze_47308, float a_47309, float b_47311,
                        float c_47313, float d_47315);
static int futrts_t6_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51409,
                        struct memblock_device *out_mem_p_51410,
                        int32_t *out_out_arrsizze_51411,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47362,
                        int32_t sizze_47363, int32_t sizze_47364,
                        int32_t sizze_47365);
static int futrts_t7_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51417,
                        struct memblock_device *out_mem_p_51418,
                        int32_t *out_out_arrsizze_51419,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t C_mem_sizze_50610,
                        struct memblock_device C_mem_50611,
                        int64_t D_mem_sizze_50612,
                        struct memblock_device D_mem_50613, int32_t sizze_47396,
                        int32_t sizze_47397, int32_t sizze_47398,
                        int32_t sizze_47399, int32_t sizze_47400);
static int futrts_t9_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51432,
                        struct memblock_device *out_mem_p_51433,
                        int32_t *out_out_arrsizze_51434,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47438,
                        int32_t sizze_47439, int32_t sizze_47440,
                        int32_t sizze_47441, int32_t sizze_47442,
                        int32_t sizze_47443);
static int futrts_t10_2d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51446,
                         struct memblock_device *out_mem_p_51447,
                         int32_t *out_out_arrsizze_51448,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int64_t D_mem_sizze_50614,
                         struct memblock_device D_mem_50615,
                         int32_t sizze_47491, int32_t sizze_47492,
                         int32_t sizze_47493, int32_t sizze_47494,
                         int32_t sizze_47495, int32_t sizze_47496,
                         int32_t sizze_47497);
static int futrts_t9_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51461,
                        struct memblock_device *out_mem_p_51462,
                        int32_t *out_out_arrsizze_51463,
                        int32_t *out_out_arrsizze_51464,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47554,
                        int32_t sizze_47555, int32_t sizze_47556,
                        int32_t sizze_47557, int32_t sizze_47558,
                        int32_t sizze_47559);
static int futrts_t8_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51477,
                        struct memblock_device *out_mem_p_51478,
                        int32_t *out_out_arrsizze_51479,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47599,
                        int32_t sizze_47600, int32_t sizze_47601,
                        int32_t sizze_47602, int32_t sizze_47603,
                        int32_t sizze_47604, int32_t sizze_47605);
static int futrts_t1_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51493,
                        struct memblock_device *out_mem_p_51494,
                        int32_t *out_out_arrsizze_51495,
                        int32_t *out_out_arrsizze_51496,
                        int32_t *out_out_arrsizze_51497,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611, int32_t sizze_47659,
                        int32_t sizze_47660, int32_t sizze_47661,
                        int32_t sizze_47662, int32_t sizze_47663);
static int futrts_t2_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51503,
                        struct memblock_device *out_mem_p_51504,
                        int32_t *out_out_arrsizze_51505,
                        int32_t *out_out_arrsizze_51506,
                        int32_t *out_out_arrsizze_51507,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47690,
                        int32_t sizze_47691, int32_t sizze_47692,
                        int32_t sizze_47693, int32_t sizze_47694);
static int futrts_t10_3d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51518,
                         struct memblock_device *out_mem_p_51519,
                         int32_t *out_out_arrsizze_51520,
                         int32_t *out_out_arrsizze_51521,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int32_t sizze_47724, int32_t sizze_47725,
                         int32_t sizze_47726, int32_t sizze_47727,
                         int32_t sizze_47728, int32_t sizze_47729,
                         int32_t sizze_47730);
static int futrts_t11_3d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51534,
                         struct memblock_device *out_mem_p_51535,
                         int32_t *out_out_arrsizze_51536,
                         int32_t *out_out_arrsizze_51537,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int64_t D_mem_sizze_50614,
                         struct memblock_device D_mem_50615,
                         int64_t E_mem_sizze_50616,
                         struct memblock_device E_mem_50617,
                         int32_t sizze_47777, int32_t sizze_47778,
                         int32_t sizze_47779, int32_t sizze_47780,
                         int32_t sizze_47781, int32_t sizze_47782,
                         int32_t sizze_47783);
static int futrts_t3_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51549,
                        struct memblock_device *out_mem_p_51550,
                        int32_t *out_out_arrsizze_51551,
                        int32_t *out_out_arrsizze_51552,
                        int32_t *out_out_arrsizze_51553,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47834,
                        int32_t sizze_47835, int32_t sizze_47836,
                        int32_t sizze_47837, int32_t sizze_47838,
                        int32_t sizze_47839, int32_t sizze_47840);
static int futrts_t4_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51559,
                        struct memblock_device *out_mem_p_51560,
                        int32_t *out_out_arrsizze_51561,
                        int32_t *out_out_arrsizze_51562,
                        int32_t *out_out_arrsizze_51563,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47886,
                        int32_t sizze_47887, int32_t sizze_47888,
                        int32_t sizze_47889, int32_t sizze_47890,
                        int32_t sizze_47891, int32_t sizze_47892,
                        int32_t sizze_47893);
static int futrts_t5_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51574,
                        struct memblock_device *out_mem_p_51575,
                        int32_t *out_out_arrsizze_51576,
                        int32_t *out_out_arrsizze_51577,
                        int32_t *out_out_arrsizze_51578,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47945,
                        int32_t sizze_47946, int32_t sizze_47947,
                        int32_t sizze_47948, int32_t sizze_47949,
                        int32_t sizze_47950, int32_t sizze_47951,
                        int32_t sizze_47952, int32_t sizze_47953,
                        int32_t sizze_47954);
static int futrts_t6_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51594,
                        struct memblock_device *out_mem_p_51595,
                        int32_t *out_out_arrsizze_51596,
                        int32_t *out_out_arrsizze_51597,
                        int32_t *out_out_arrsizze_51598,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_48024,
                        int32_t sizze_48025, int32_t sizze_48026,
                        int32_t sizze_48027, int32_t sizze_48028,
                        int32_t sizze_48029, int32_t sizze_48030,
                        int32_t sizze_48031, int32_t sizze_48032,
                        int32_t sizze_48033, float a_48034, float b_48036,
                        float c_48038, float d_48040);
static int futrts_t7_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51614,
                        struct memblock_device *out_mem_p_51615,
                        int32_t *out_out_arrsizze_51616,
                        int32_t *out_out_arrsizze_51617,
                        int32_t *out_out_arrsizze_51618,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615,
                        int64_t E_mem_sizze_50616,
                        struct memblock_device E_mem_50617,
                        int64_t F_mem_sizze_50618,
                        struct memblock_device F_mem_50619,
                        int64_t G_mem_sizze_50620,
                        struct memblock_device G_mem_50621,
                        int64_t H_mem_sizze_50622,
                        struct memblock_device H_mem_50623, int32_t sizze_48111,
                        int32_t sizze_48112, int32_t sizze_48113,
                        int32_t sizze_48114, int32_t sizze_48115,
                        int32_t sizze_48116, int32_t sizze_48117,
                        int32_t sizze_48118, int32_t sizze_48119,
                        int32_t sizze_48120, int32_t sizze_48121,
                        int32_t sizze_48122, int32_t sizze_48123,
                        int32_t sizze_48124, float a_48125, float b_48127,
                        float c_48129, float d_48131, float e_48133,
                        float f_48135, float g_48137, float h_48139);
static int futrts_t8_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51644,
                        struct memblock_device *out_mem_p_51645,
                        int32_t *out_out_arrsizze_51646,
                        int32_t *out_out_arrsizze_51647,
                        int32_t *out_out_arrsizze_51648,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_48250,
                        int32_t sizze_48251, int32_t sizze_48252,
                        int32_t sizze_48253, int32_t sizze_48254,
                        int32_t sizze_48255, int32_t sizze_48256);
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
                    const size_t global_work_sizze_51207[3] = {x_elems_5 +
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
                    const size_t local_work_sizze_51211[3] = {16, 16, 1};
                    int64_t time_start_51208 = 0, time_end_51209 = 0;
                    
                    if (ctx->debugging) {
                        fprintf(stderr, "Launching %s with global work size [",
                                "fut_kernel_map_transpose_lowwidth_f32");
                        fprintf(stderr, "%zu", global_work_sizze_51207[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_51207[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", global_work_sizze_51207[2]);
                        fprintf(stderr, "] and local work size [");
                        fprintf(stderr, "%zu", local_work_sizze_51211[0]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_51211[1]);
                        fprintf(stderr, ", ");
                        fprintf(stderr, "%zu", local_work_sizze_51211[2]);
                        fprintf(stderr, "].\n");
                        time_start_51208 = get_wall_time();
                    }
                    OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                          ctx->fut_kernel_map_transpose_lowwidth_f32,
                                                          3, NULL,
                                                          global_work_sizze_51207,
                                                          local_work_sizze_51211,
                                                          0, NULL, NULL));
                    if (ctx->debugging) {
                        OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                        time_end_51209 = get_wall_time();
                        
                        long time_diff_51210 = time_end_51209 -
                             time_start_51208;
                        
                        ctx->fut_kernel_map_transpose_lowwidth_f32_total_runtime +=
                            time_diff_51210;
                        ctx->fut_kernel_map_transpose_lowwidth_f32_runs++;
                        fprintf(stderr, "kernel %s runtime: %ldus\n",
                                "fut_kernel_map_transpose_lowwidth_f32",
                                time_diff_51210);
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
                        const size_t global_work_sizze_51212[3] =
                                     {new_width_12 + srem32(16 -
                                                            srem32(new_width_12,
                                                                   16), 16),
                                      y_elems_6 + srem32(16 - srem32(y_elems_6,
                                                                     16), 16),
                                      num_arrays_4};
                        const size_t local_work_sizze_51216[3] = {16, 16, 1};
                        int64_t time_start_51213 = 0, time_end_51214 = 0;
                        
                        if (ctx->debugging) {
                            fprintf(stderr,
                                    "Launching %s with global work size [",
                                    "fut_kernel_map_transpose_lowheight_f32");
                            fprintf(stderr, "%zu", global_work_sizze_51212[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_51212[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", global_work_sizze_51212[2]);
                            fprintf(stderr, "] and local work size [");
                            fprintf(stderr, "%zu", local_work_sizze_51216[0]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_51216[1]);
                            fprintf(stderr, ", ");
                            fprintf(stderr, "%zu", local_work_sizze_51216[2]);
                            fprintf(stderr, "].\n");
                            time_start_51213 = get_wall_time();
                        }
                        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                              ctx->fut_kernel_map_transpose_lowheight_f32,
                                                              3, NULL,
                                                              global_work_sizze_51212,
                                                              local_work_sizze_51216,
                                                              0, NULL, NULL));
                        if (ctx->debugging) {
                            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                            time_end_51214 = get_wall_time();
                            
                            long time_diff_51215 = time_end_51214 -
                                 time_start_51213;
                            
                            ctx->fut_kernel_map_transpose_lowheight_f32_total_runtime +=
                                time_diff_51215;
                            ctx->fut_kernel_map_transpose_lowheight_f32_runs++;
                            fprintf(stderr, "kernel %s runtime: %ldus\n",
                                    "fut_kernel_map_transpose_lowheight_f32",
                                    time_diff_51215);
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
                            const size_t global_work_sizze_51217[1] =
                                         {num_arrays_4 * x_elems_5 * y_elems_6 +
                                         srem32(256 - srem32(num_arrays_4 *
                                                             x_elems_5 *
                                                             y_elems_6, 256),
                                                256)};
                            const size_t local_work_sizze_51221[1] = {256};
                            int64_t time_start_51218 = 0, time_end_51219 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_small_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_51217[0]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_51221[0]);
                                fprintf(stderr, "].\n");
                                time_start_51218 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_small_f32,
                                                                  1, NULL,
                                                                  global_work_sizze_51217,
                                                                  local_work_sizze_51221,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_51219 = get_wall_time();
                                
                                long time_diff_51220 = time_end_51219 -
                                     time_start_51218;
                                
                                ctx->fut_kernel_map_transpose_small_f32_total_runtime +=
                                    time_diff_51220;
                                ctx->fut_kernel_map_transpose_small_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_small_f32",
                                        time_diff_51220);
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
                            const size_t global_work_sizze_51222[3] =
                                         {x_elems_5 + srem32(16 -
                                                             srem32(x_elems_5,
                                                                    16), 16),
                                          y_elems_6 + srem32(16 -
                                                             srem32(y_elems_6,
                                                                    16), 16),
                                          num_arrays_4};
                            const size_t local_work_sizze_51226[3] = {16, 16,
                                                                      1};
                            int64_t time_start_51223 = 0, time_end_51224 = 0;
                            
                            if (ctx->debugging) {
                                fprintf(stderr,
                                        "Launching %s with global work size [",
                                        "fut_kernel_map_transpose_f32");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_51222[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_51222[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        global_work_sizze_51222[2]);
                                fprintf(stderr, "] and local work size [");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_51226[0]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_51226[1]);
                                fprintf(stderr, ", ");
                                fprintf(stderr, "%zu",
                                        local_work_sizze_51226[2]);
                                fprintf(stderr, "].\n");
                                time_start_51223 = get_wall_time();
                            }
                            OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                                                  ctx->fut_kernel_map_transpose_f32,
                                                                  3, NULL,
                                                                  global_work_sizze_51222,
                                                                  local_work_sizze_51226,
                                                                  0, NULL,
                                                                  NULL));
                            if (ctx->debugging) {
                                OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
                                time_end_51224 = get_wall_time();
                                
                                long time_diff_51225 = time_end_51224 -
                                     time_start_51223;
                                
                                ctx->fut_kernel_map_transpose_f32_total_runtime +=
                                    time_diff_51225;
                                ctx->fut_kernel_map_transpose_f32_runs++;
                                fprintf(stderr, "kernel %s runtime: %ldus\n",
                                        "fut_kernel_map_transpose_f32",
                                        time_diff_51225);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
static int futrts_sum(struct futhark_context *ctx, float *out_scalar_out_51227,
                      int64_t A_mem_sizze_50608,
                      struct memblock_device A_mem_50609, int32_t sizze_46897)
{
    float scalar_out_50763;
    int32_t group_sizze_48310;
    
    group_sizze_48310 = ctx->sizes.group_sizze_48309;
    
    int32_t max_num_groups_48312;
    
    max_num_groups_48312 = ctx->sizes.max_num_groups_48311;
    
    int32_t y_48313 = group_sizze_48310 - 1;
    int32_t x_48314 = sizze_46897 + y_48313;
    int32_t w_div_group_sizze_48315 = squot32(x_48314, group_sizze_48310);
    int32_t num_groups_maybe_zzero_48316 = smin32(max_num_groups_48312,
                                                  w_div_group_sizze_48315);
    int32_t num_groups_48317 = smax32(1, num_groups_maybe_zzero_48316);
    int32_t num_threads_48318 = group_sizze_48310 * num_groups_48317;
    int32_t y_48319 = num_threads_48318 - 1;
    int32_t x_48320 = sizze_46897 + y_48319;
    int32_t per_thread_elements_48321 = squot32(x_48320, num_threads_48318);
    int64_t binop_x_50614 = sext_i32_i64(num_groups_48317);
    int64_t bytes_50613 = 4 * binop_x_50614;
    struct memblock_device mem_50615;
    
    mem_50615.references = NULL;
    memblock_alloc_device(ctx, &mem_50615, bytes_50613, "mem_50615");
    
    int64_t binop_x_50611 = sext_i32_i64(group_sizze_48310);
    int64_t bytes_50610 = 4 * binop_x_50611;
    struct memblock_local mem_50612;
    
    mem_50612.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_46897);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 0,
                                  bytes_50610, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 1,
                                  sizeof(sizze_46897), &sizze_46897));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 2,
                                  sizeof(num_threads_48318),
                                  &num_threads_48318));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 3,
                                  sizeof(per_thread_elements_48321),
                                  &per_thread_elements_48321));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48326, 5,
                                  sizeof(mem_50615.mem), &mem_50615.mem));
    if (1 * (num_groups_48317 * group_sizze_48310) != 0) {
        const size_t global_work_sizze_51228[1] = {num_groups_48317 *
                     group_sizze_48310};
        const size_t local_work_sizze_51232[1] = {group_sizze_48310};
        int64_t time_start_51229 = 0, time_end_51230 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48326");
            fprintf(stderr, "%zu", global_work_sizze_51228[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51232[0]);
            fprintf(stderr, "].\n");
            time_start_51229 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48326,
                                              1, NULL, global_work_sizze_51228,
                                              local_work_sizze_51232, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51230 = get_wall_time();
            
            long time_diff_51231 = time_end_51230 - time_start_51229;
            
            ctx->chunked_reduce_kernel_48326_total_runtime += time_diff_51231;
            ctx->chunked_reduce_kernel_48326_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48326", time_diff_51231);
        }
    }
    memblock_unref_local(ctx, &mem_50612, "mem_50612");
    
    struct memblock_device mem_50621;
    
    mem_50621.references = NULL;
    memblock_alloc_device(ctx, &mem_50621, 4, "mem_50621");
    
    int64_t binop_x_50617 = sext_i32_i64(max_num_groups_48312);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_local mem_50618;
    
    mem_50618.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48354, 0, bytes_50616,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48354, 1,
                                  sizeof(num_groups_48317), &num_groups_48317));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48354, 2,
                                  sizeof(mem_50615.mem), &mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48354, 3,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    if (1 * max_num_groups_48312 != 0) {
        const size_t global_work_sizze_51233[1] = {max_num_groups_48312};
        const size_t local_work_sizze_51237[1] = {max_num_groups_48312};
        int64_t time_start_51234 = 0, time_end_51235 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48354");
            fprintf(stderr, "%zu", global_work_sizze_51233[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51237[0]);
            fprintf(stderr, "].\n");
            time_start_51234 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48354, 1, NULL,
                                              global_work_sizze_51233,
                                              local_work_sizze_51237, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51235 = get_wall_time();
            
            long time_diff_51236 = time_end_51235 - time_start_51234;
            
            ctx->reduce_kernel_48354_total_runtime += time_diff_51236;
            ctx->reduce_kernel_48354_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48354",
                    time_diff_51236);
        }
    }
    memblock_unref_device(ctx, &mem_50615, "mem_50615");
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    
    float read_res_51238;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50621.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51238, 0, NULL, NULL));
    
    float res_46899 = read_res_51238;
    
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    scalar_out_50763 = res_46899;
    *out_scalar_out_51227 = scalar_out_50763;
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50612, "mem_50612");
    memblock_unref_device(ctx, &mem_50615, "mem_50615");
    return 0;
}
static int futrts_prod(struct futhark_context *ctx,
                       int64_t *out_out_memsizze_51239,
                       struct memblock_device *out_mem_p_51240,
                       int32_t *out_out_arrsizze_51241,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609, int32_t sizze_46904,
                       float c_46905)
{
    int64_t out_memsizze_50780;
    struct memblock_device out_mem_50779;
    
    out_mem_50779.references = NULL;
    
    int32_t out_arrsizze_50781;
    int32_t group_sizze_48365;
    
    group_sizze_48365 = ctx->sizes.group_sizze_48364;
    
    int32_t y_48366 = group_sizze_48365 - 1;
    int32_t x_48367 = sizze_46904 + y_48366;
    int32_t num_groups_48368 = squot32(x_48367, group_sizze_48365);
    int32_t num_threads_48369 = group_sizze_48365 * num_groups_48368;
    int64_t binop_x_50611 = sext_i32_i64(sizze_46904);
    int64_t bytes_50610 = 4 * binop_x_50611;
    struct memblock_device mem_50612;
    
    mem_50612.references = NULL;
    memblock_alloc_device(ctx, &mem_50612, bytes_50610, "mem_50612");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_48370, 0, sizeof(sizze_46904),
                                  &sizze_46904));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_48370, 1, sizeof(c_46905),
                                  &c_46905));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_48370, 2,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_48370, 3,
                                  sizeof(mem_50612.mem), &mem_50612.mem));
    if (1 * (num_groups_48368 * group_sizze_48365) != 0) {
        const size_t global_work_sizze_51242[1] = {num_groups_48368 *
                     group_sizze_48365};
        const size_t local_work_sizze_51246[1] = {group_sizze_48365};
        int64_t time_start_51243 = 0, time_end_51244 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_48370");
            fprintf(stderr, "%zu", global_work_sizze_51242[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51246[0]);
            fprintf(stderr, "].\n");
            time_start_51243 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_48370, 1, NULL,
                                              global_work_sizze_51242,
                                              local_work_sizze_51246, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51244 = get_wall_time();
            
            long time_diff_51245 = time_end_51244 - time_start_51243;
            
            ctx->map_kernel_48370_total_runtime += time_diff_51245;
            ctx->map_kernel_48370_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_48370",
                    time_diff_51245);
        }
    }
    out_arrsizze_50781 = sizze_46904;
    out_memsizze_50780 = bytes_50610;
    memblock_set_device(ctx, &out_mem_50779, &mem_50612, "mem_50612");
    *out_out_memsizze_51239 = out_memsizze_50780;
    (*out_mem_p_51240).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51240, &out_mem_50779,
                        "out_mem_50779");
    *out_out_arrsizze_51241 = out_arrsizze_50781;
    memblock_unref_device(ctx, &mem_50612, "mem_50612");
    memblock_unref_device(ctx, &out_mem_50779, "out_mem_50779");
    return 0;
}
static int futrts_dot(struct futhark_context *ctx, float *out_scalar_out_51247,
                      int64_t A_mem_sizze_50608,
                      struct memblock_device A_mem_50609,
                      int64_t B_mem_sizze_50610,
                      struct memblock_device B_mem_50611, int32_t sizze_46910,
                      int32_t sizze_46911)
{
    float scalar_out_50785;
    bool dim_zzero_46914 = 0 == sizze_46911;
    bool dim_zzero_46915 = 0 == sizze_46910;
    bool both_empty_46916 = dim_zzero_46914 && dim_zzero_46915;
    bool dim_match_46917 = sizze_46910 == sizze_46911;
    bool empty_or_match_46918 = both_empty_46916 || dim_match_46917;
    bool empty_or_match_cert_46919;
    
    if (!empty_or_match_46918) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:11:1-12:31",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48385;
    
    group_sizze_48385 = ctx->sizes.group_sizze_48384;
    
    int32_t max_num_groups_48387;
    
    max_num_groups_48387 = ctx->sizes.max_num_groups_48386;
    
    int32_t y_48388 = group_sizze_48385 - 1;
    int32_t x_48389 = sizze_46910 + y_48388;
    int32_t w_div_group_sizze_48390 = squot32(x_48389, group_sizze_48385);
    int32_t num_groups_maybe_zzero_48391 = smin32(max_num_groups_48387,
                                                  w_div_group_sizze_48390);
    int32_t num_groups_48392 = smax32(1, num_groups_maybe_zzero_48391);
    int32_t num_threads_48393 = group_sizze_48385 * num_groups_48392;
    int32_t y_48394 = num_threads_48393 - 1;
    int32_t x_48395 = sizze_46910 + y_48394;
    int32_t per_thread_elements_48396 = squot32(x_48395, num_threads_48393);
    int64_t binop_x_50616 = sext_i32_i64(num_groups_48392);
    int64_t bytes_50615 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50615, "mem_50617");
    
    int64_t binop_x_50613 = sext_i32_i64(group_sizze_48385);
    int64_t bytes_50612 = 4 * binop_x_50613;
    struct memblock_local mem_50614;
    
    mem_50614.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_46910);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 0,
                                  bytes_50612, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 1,
                                  sizeof(sizze_46910), &sizze_46910));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 2,
                                  sizeof(num_threads_48393),
                                  &num_threads_48393));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 3,
                                  sizeof(per_thread_elements_48396),
                                  &per_thread_elements_48396));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48401, 6,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    if (1 * (num_groups_48392 * group_sizze_48385) != 0) {
        const size_t global_work_sizze_51248[1] = {num_groups_48392 *
                     group_sizze_48385};
        const size_t local_work_sizze_51252[1] = {group_sizze_48385};
        int64_t time_start_51249 = 0, time_end_51250 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48401");
            fprintf(stderr, "%zu", global_work_sizze_51248[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51252[0]);
            fprintf(stderr, "].\n");
            time_start_51249 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48401,
                                              1, NULL, global_work_sizze_51248,
                                              local_work_sizze_51252, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51250 = get_wall_time();
            
            long time_diff_51251 = time_end_51250 - time_start_51249;
            
            ctx->chunked_reduce_kernel_48401_total_runtime += time_diff_51251;
            ctx->chunked_reduce_kernel_48401_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48401", time_diff_51251);
        }
    }
    memblock_unref_local(ctx, &mem_50614, "mem_50614");
    
    struct memblock_device mem_50623;
    
    mem_50623.references = NULL;
    memblock_alloc_device(ctx, &mem_50623, 4, "mem_50623");
    
    int64_t binop_x_50619 = sext_i32_i64(max_num_groups_48387);
    int64_t bytes_50618 = 4 * binop_x_50619;
    struct memblock_local mem_50620;
    
    mem_50620.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48433, 0, bytes_50618,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48433, 1,
                                  sizeof(num_groups_48392), &num_groups_48392));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48433, 2,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48433, 3,
                                  sizeof(mem_50623.mem), &mem_50623.mem));
    if (1 * max_num_groups_48387 != 0) {
        const size_t global_work_sizze_51253[1] = {max_num_groups_48387};
        const size_t local_work_sizze_51257[1] = {max_num_groups_48387};
        int64_t time_start_51254 = 0, time_end_51255 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48433");
            fprintf(stderr, "%zu", global_work_sizze_51253[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51257[0]);
            fprintf(stderr, "].\n");
            time_start_51254 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48433, 1, NULL,
                                              global_work_sizze_51253,
                                              local_work_sizze_51257, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51255 = get_wall_time();
            
            long time_diff_51256 = time_end_51255 - time_start_51254;
            
            ctx->reduce_kernel_48433_total_runtime += time_diff_51256;
            ctx->reduce_kernel_48433_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48433",
                    time_diff_51256);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_local(ctx, &mem_50620, "mem_50620");
    
    float read_res_51258;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50623.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51258, 0, NULL, NULL));
    
    float res_46921 = read_res_51258;
    
    memblock_unref_device(ctx, &mem_50623, "mem_50623");
    scalar_out_50785 = res_46921;
    *out_scalar_out_51247 = scalar_out_50785;
    memblock_unref_local(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50623, "mem_50623");
    memblock_unref_local(ctx, &mem_50614, "mem_50614");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    return 0;
}
static int futrts_dot2(struct futhark_context *ctx, float *out_scalar_out_51259,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_46928,
                       int32_t sizze_46929, int32_t sizze_46930)
{
    float scalar_out_50801;
    bool dim_zzero_46934 = 0 == sizze_46929;
    bool dim_zzero_46935 = 0 == sizze_46928;
    bool both_empty_46936 = dim_zzero_46934 && dim_zzero_46935;
    bool dim_match_46937 = sizze_46928 == sizze_46929;
    bool empty_or_match_46938 = both_empty_46936 || dim_match_46937;
    bool empty_or_match_cert_46939;
    
    if (!empty_or_match_46938) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:17:1-18:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_46940 = 0 == sizze_46930;
    bool both_empty_46941 = dim_zzero_46935 && dim_zzero_46940;
    bool dim_match_46942 = sizze_46928 == sizze_46930;
    bool empty_or_match_46943 = both_empty_46941 || dim_match_46942;
    bool empty_or_match_cert_46944;
    
    if (!empty_or_match_46943) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:17:1-18:44",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48454;
    
    group_sizze_48454 = ctx->sizes.group_sizze_48453;
    
    int32_t max_num_groups_48456;
    
    max_num_groups_48456 = ctx->sizes.max_num_groups_48455;
    
    int32_t y_48457 = group_sizze_48454 - 1;
    int32_t x_48458 = sizze_46928 + y_48457;
    int32_t w_div_group_sizze_48459 = squot32(x_48458, group_sizze_48454);
    int32_t num_groups_maybe_zzero_48460 = smin32(max_num_groups_48456,
                                                  w_div_group_sizze_48459);
    int32_t num_groups_48461 = smax32(1, num_groups_maybe_zzero_48460);
    int32_t num_threads_48462 = group_sizze_48454 * num_groups_48461;
    int32_t y_48463 = num_threads_48462 - 1;
    int32_t x_48464 = sizze_46928 + y_48463;
    int32_t per_thread_elements_48465 = squot32(x_48464, num_threads_48462);
    int64_t binop_x_50618 = sext_i32_i64(num_groups_48461);
    int64_t bytes_50617 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50617, "mem_50619");
    
    int64_t binop_x_50615 = sext_i32_i64(group_sizze_48454);
    int64_t bytes_50614 = 4 * binop_x_50615;
    struct memblock_local mem_50616;
    
    mem_50616.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_46928);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 0,
                                  bytes_50614, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 1,
                                  sizeof(sizze_46928), &sizze_46928));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 2,
                                  sizeof(num_threads_48462),
                                  &num_threads_48462));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 3,
                                  sizeof(per_thread_elements_48465),
                                  &per_thread_elements_48465));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48470, 7,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    if (1 * (num_groups_48461 * group_sizze_48454) != 0) {
        const size_t global_work_sizze_51260[1] = {num_groups_48461 *
                     group_sizze_48454};
        const size_t local_work_sizze_51264[1] = {group_sizze_48454};
        int64_t time_start_51261 = 0, time_end_51262 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48470");
            fprintf(stderr, "%zu", global_work_sizze_51260[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51264[0]);
            fprintf(stderr, "].\n");
            time_start_51261 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48470,
                                              1, NULL, global_work_sizze_51260,
                                              local_work_sizze_51264, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51262 = get_wall_time();
            
            long time_diff_51263 = time_end_51262 - time_start_51261;
            
            ctx->chunked_reduce_kernel_48470_total_runtime += time_diff_51263;
            ctx->chunked_reduce_kernel_48470_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48470", time_diff_51263);
        }
    }
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    
    struct memblock_device mem_50625;
    
    mem_50625.references = NULL;
    memblock_alloc_device(ctx, &mem_50625, 4, "mem_50625");
    
    int64_t binop_x_50621 = sext_i32_i64(max_num_groups_48456);
    int64_t bytes_50620 = 4 * binop_x_50621;
    struct memblock_local mem_50622;
    
    mem_50622.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48506, 0, bytes_50620,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48506, 1,
                                  sizeof(num_groups_48461), &num_groups_48461));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48506, 2,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48506, 3,
                                  sizeof(mem_50625.mem), &mem_50625.mem));
    if (1 * max_num_groups_48456 != 0) {
        const size_t global_work_sizze_51265[1] = {max_num_groups_48456};
        const size_t local_work_sizze_51269[1] = {max_num_groups_48456};
        int64_t time_start_51266 = 0, time_end_51267 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48506");
            fprintf(stderr, "%zu", global_work_sizze_51265[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51269[0]);
            fprintf(stderr, "].\n");
            time_start_51266 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48506, 1, NULL,
                                              global_work_sizze_51265,
                                              local_work_sizze_51269, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51267 = get_wall_time();
            
            long time_diff_51268 = time_end_51267 - time_start_51266;
            
            ctx->reduce_kernel_48506_total_runtime += time_diff_51268;
            ctx->reduce_kernel_48506_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48506",
                    time_diff_51268);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    
    float read_res_51270;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50625.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51270, 0, NULL, NULL));
    
    float res_46947 = read_res_51270;
    
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    scalar_out_50801 = res_46947;
    *out_scalar_out_51259 = scalar_out_50801;
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    return 0;
}
static int futrts_dot3(struct futhark_context *ctx, float *out_scalar_out_51271,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_46956,
                       int32_t sizze_46957, int32_t sizze_46958)
{
    float scalar_out_50817;
    bool dim_zzero_46962 = 0 == sizze_46957;
    bool dim_zzero_46963 = 0 == sizze_46956;
    bool both_empty_46964 = dim_zzero_46962 && dim_zzero_46963;
    bool dim_match_46965 = sizze_46956 == sizze_46957;
    bool empty_or_match_46966 = both_empty_46964 || dim_match_46965;
    bool empty_or_match_cert_46967;
    
    if (!empty_or_match_46966) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:20:1-21:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_46968 = 0 == sizze_46958;
    bool both_empty_46969 = dim_zzero_46963 && dim_zzero_46968;
    bool dim_match_46970 = sizze_46956 == sizze_46958;
    bool empty_or_match_46971 = both_empty_46969 || dim_match_46970;
    bool empty_or_match_cert_46972;
    
    if (!empty_or_match_46971) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:20:1-21:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48527;
    
    group_sizze_48527 = ctx->sizes.group_sizze_48526;
    
    int32_t max_num_groups_48529;
    
    max_num_groups_48529 = ctx->sizes.max_num_groups_48528;
    
    int32_t y_48530 = group_sizze_48527 - 1;
    int32_t x_48531 = sizze_46956 + y_48530;
    int32_t w_div_group_sizze_48532 = squot32(x_48531, group_sizze_48527);
    int32_t num_groups_maybe_zzero_48533 = smin32(max_num_groups_48529,
                                                  w_div_group_sizze_48532);
    int32_t num_groups_48534 = smax32(1, num_groups_maybe_zzero_48533);
    int32_t num_threads_48535 = group_sizze_48527 * num_groups_48534;
    int32_t y_48536 = num_threads_48535 - 1;
    int32_t x_48537 = sizze_46956 + y_48536;
    int32_t per_thread_elements_48538 = squot32(x_48537, num_threads_48535);
    int64_t binop_x_50618 = sext_i32_i64(num_groups_48534);
    int64_t bytes_50617 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50617, "mem_50619");
    
    int64_t binop_x_50615 = sext_i32_i64(group_sizze_48527);
    int64_t bytes_50614 = 4 * binop_x_50615;
    struct memblock_local mem_50616;
    
    mem_50616.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_46956);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 0,
                                  bytes_50614, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 1,
                                  sizeof(sizze_46956), &sizze_46956));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 2,
                                  sizeof(num_threads_48535),
                                  &num_threads_48535));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 3,
                                  sizeof(per_thread_elements_48538),
                                  &per_thread_elements_48538));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48543, 7,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    if (1 * (num_groups_48534 * group_sizze_48527) != 0) {
        const size_t global_work_sizze_51272[1] = {num_groups_48534 *
                     group_sizze_48527};
        const size_t local_work_sizze_51276[1] = {group_sizze_48527};
        int64_t time_start_51273 = 0, time_end_51274 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48543");
            fprintf(stderr, "%zu", global_work_sizze_51272[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51276[0]);
            fprintf(stderr, "].\n");
            time_start_51273 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48543,
                                              1, NULL, global_work_sizze_51272,
                                              local_work_sizze_51276, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51274 = get_wall_time();
            
            long time_diff_51275 = time_end_51274 - time_start_51273;
            
            ctx->chunked_reduce_kernel_48543_total_runtime += time_diff_51275;
            ctx->chunked_reduce_kernel_48543_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48543", time_diff_51275);
        }
    }
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    
    struct memblock_device mem_50625;
    
    mem_50625.references = NULL;
    memblock_alloc_device(ctx, &mem_50625, 4, "mem_50625");
    
    int64_t binop_x_50621 = sext_i32_i64(max_num_groups_48529);
    int64_t bytes_50620 = 4 * binop_x_50621;
    struct memblock_local mem_50622;
    
    mem_50622.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48580, 0, bytes_50620,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48580, 1,
                                  sizeof(num_groups_48534), &num_groups_48534));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48580, 2,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48580, 3,
                                  sizeof(mem_50625.mem), &mem_50625.mem));
    if (1 * max_num_groups_48529 != 0) {
        const size_t global_work_sizze_51277[1] = {max_num_groups_48529};
        const size_t local_work_sizze_51281[1] = {max_num_groups_48529};
        int64_t time_start_51278 = 0, time_end_51279 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48580");
            fprintf(stderr, "%zu", global_work_sizze_51277[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51281[0]);
            fprintf(stderr, "].\n");
            time_start_51278 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48580, 1, NULL,
                                              global_work_sizze_51277,
                                              local_work_sizze_51281, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51279 = get_wall_time();
            
            long time_diff_51280 = time_end_51279 - time_start_51278;
            
            ctx->reduce_kernel_48580_total_runtime += time_diff_51280;
            ctx->reduce_kernel_48580_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48580",
                    time_diff_51280);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    
    float read_res_51282;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50625.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51282, 0, NULL, NULL));
    
    float res_46975 = read_res_51282;
    
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    scalar_out_50817 = res_46975;
    *out_scalar_out_51271 = scalar_out_50817;
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    return 0;
}
static int futrts_dot4(struct futhark_context *ctx, float *out_scalar_out_51283,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_46985,
                       int32_t sizze_46986, int32_t sizze_46987,
                       int32_t sizze_46988)
{
    float scalar_out_50833;
    bool dim_zzero_46993 = 0 == sizze_46986;
    bool dim_zzero_46994 = 0 == sizze_46985;
    bool both_empty_46995 = dim_zzero_46993 && dim_zzero_46994;
    bool dim_match_46996 = sizze_46985 == sizze_46986;
    bool empty_or_match_46997 = both_empty_46995 || dim_match_46996;
    bool empty_or_match_cert_46998;
    
    if (!empty_or_match_46997) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:23:1-24:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_46999 = 0 == sizze_46987;
    bool both_empty_47000 = dim_zzero_46994 && dim_zzero_46999;
    bool dim_match_47001 = sizze_46985 == sizze_46987;
    bool empty_or_match_47002 = both_empty_47000 || dim_match_47001;
    bool empty_or_match_cert_47003;
    
    if (!empty_or_match_47002) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:23:1-24:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_47004 = 0 == sizze_46988;
    bool both_empty_47005 = dim_zzero_46994 && dim_zzero_47004;
    bool dim_match_47006 = sizze_46985 == sizze_46988;
    bool empty_or_match_47007 = both_empty_47005 || dim_match_47006;
    bool empty_or_match_cert_47008;
    
    if (!empty_or_match_47007) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:23:1-24:57",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48603;
    
    group_sizze_48603 = ctx->sizes.group_sizze_48602;
    
    int32_t max_num_groups_48605;
    
    max_num_groups_48605 = ctx->sizes.max_num_groups_48604;
    
    int32_t y_48606 = group_sizze_48603 - 1;
    int32_t x_48607 = sizze_46985 + y_48606;
    int32_t w_div_group_sizze_48608 = squot32(x_48607, group_sizze_48603);
    int32_t num_groups_maybe_zzero_48609 = smin32(max_num_groups_48605,
                                                  w_div_group_sizze_48608);
    int32_t num_groups_48610 = smax32(1, num_groups_maybe_zzero_48609);
    int32_t num_threads_48611 = group_sizze_48603 * num_groups_48610;
    int32_t y_48612 = num_threads_48611 - 1;
    int32_t x_48613 = sizze_46985 + y_48612;
    int32_t per_thread_elements_48614 = squot32(x_48613, num_threads_48611);
    int64_t binop_x_50620 = sext_i32_i64(num_groups_48610);
    int64_t bytes_50619 = 4 * binop_x_50620;
    struct memblock_device mem_50621;
    
    mem_50621.references = NULL;
    memblock_alloc_device(ctx, &mem_50621, bytes_50619, "mem_50621");
    
    int64_t binop_x_50617 = sext_i32_i64(group_sizze_48603);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_local mem_50618;
    
    mem_50618.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_46985);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 0,
                                  bytes_50616, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 1,
                                  sizeof(sizze_46985), &sizze_46985));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 2,
                                  sizeof(num_threads_48611),
                                  &num_threads_48611));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 3,
                                  sizeof(per_thread_elements_48614),
                                  &per_thread_elements_48614));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 7,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48619, 8,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    if (1 * (num_groups_48610 * group_sizze_48603) != 0) {
        const size_t global_work_sizze_51284[1] = {num_groups_48610 *
                     group_sizze_48603};
        const size_t local_work_sizze_51288[1] = {group_sizze_48603};
        int64_t time_start_51285 = 0, time_end_51286 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48619");
            fprintf(stderr, "%zu", global_work_sizze_51284[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51288[0]);
            fprintf(stderr, "].\n");
            time_start_51285 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48619,
                                              1, NULL, global_work_sizze_51284,
                                              local_work_sizze_51288, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51286 = get_wall_time();
            
            long time_diff_51287 = time_end_51286 - time_start_51285;
            
            ctx->chunked_reduce_kernel_48619_total_runtime += time_diff_51287;
            ctx->chunked_reduce_kernel_48619_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48619", time_diff_51287);
        }
    }
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, 4, "mem_50627");
    
    int64_t binop_x_50623 = sext_i32_i64(max_num_groups_48605);
    int64_t bytes_50622 = 4 * binop_x_50623;
    struct memblock_local mem_50624;
    
    mem_50624.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48659, 0, bytes_50622,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48659, 1,
                                  sizeof(num_groups_48610), &num_groups_48610));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48659, 2,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48659, 3,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    if (1 * max_num_groups_48605 != 0) {
        const size_t global_work_sizze_51289[1] = {max_num_groups_48605};
        const size_t local_work_sizze_51293[1] = {max_num_groups_48605};
        int64_t time_start_51290 = 0, time_end_51291 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48659");
            fprintf(stderr, "%zu", global_work_sizze_51289[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51293[0]);
            fprintf(stderr, "].\n");
            time_start_51290 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48659, 1, NULL,
                                              global_work_sizze_51289,
                                              local_work_sizze_51293, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51291 = get_wall_time();
            
            long time_diff_51292 = time_end_51291 - time_start_51290;
            
            ctx->reduce_kernel_48659_total_runtime += time_diff_51292;
            ctx->reduce_kernel_48659_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48659",
                    time_diff_51292);
        }
    }
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    
    float read_res_51294;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50627.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51294, 0, NULL, NULL));
    
    float res_47012 = read_res_51294;
    
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    scalar_out_50833 = res_47012;
    *out_scalar_out_51283 = scalar_out_50833;
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    return 0;
}
static int futrts_dot5(struct futhark_context *ctx, float *out_scalar_out_51295,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_47023,
                       int32_t sizze_47024, int32_t sizze_47025,
                       int32_t sizze_47026, float a_47027, float b_47029,
                       float c_47031, float d_47033)
{
    float scalar_out_50849;
    bool dim_zzero_47035 = 0 == sizze_47024;
    bool dim_zzero_47036 = 0 == sizze_47023;
    bool both_empty_47037 = dim_zzero_47035 && dim_zzero_47036;
    bool dim_match_47038 = sizze_47023 == sizze_47024;
    bool empty_or_match_47039 = both_empty_47037 || dim_match_47038;
    bool empty_or_match_cert_47040;
    
    if (!empty_or_match_47039) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:26:1-27:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_47041 = 0 == sizze_47025;
    bool both_empty_47042 = dim_zzero_47036 && dim_zzero_47041;
    bool dim_match_47043 = sizze_47023 == sizze_47025;
    bool empty_or_match_47044 = both_empty_47042 || dim_match_47043;
    bool empty_or_match_cert_47045;
    
    if (!empty_or_match_47044) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:26:1-27:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_47046 = 0 == sizze_47026;
    bool both_empty_47047 = dim_zzero_47036 && dim_zzero_47046;
    bool dim_match_47048 = sizze_47023 == sizze_47026;
    bool empty_or_match_47049 = both_empty_47047 || dim_match_47048;
    bool empty_or_match_cert_47050;
    
    if (!empty_or_match_47049) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:26:1-27:101",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48682;
    
    group_sizze_48682 = ctx->sizes.group_sizze_48681;
    
    int32_t max_num_groups_48684;
    
    max_num_groups_48684 = ctx->sizes.max_num_groups_48683;
    
    int32_t y_48685 = group_sizze_48682 - 1;
    int32_t x_48686 = sizze_47023 + y_48685;
    int32_t w_div_group_sizze_48687 = squot32(x_48686, group_sizze_48682);
    int32_t num_groups_maybe_zzero_48688 = smin32(max_num_groups_48684,
                                                  w_div_group_sizze_48687);
    int32_t num_groups_48689 = smax32(1, num_groups_maybe_zzero_48688);
    int32_t num_threads_48690 = group_sizze_48682 * num_groups_48689;
    int32_t y_48691 = num_threads_48690 - 1;
    int32_t x_48692 = sizze_47023 + y_48691;
    int32_t per_thread_elements_48693 = squot32(x_48692, num_threads_48690);
    int64_t binop_x_50620 = sext_i32_i64(num_groups_48689);
    int64_t bytes_50619 = 4 * binop_x_50620;
    struct memblock_device mem_50621;
    
    mem_50621.references = NULL;
    memblock_alloc_device(ctx, &mem_50621, bytes_50619, "mem_50621");
    
    int64_t binop_x_50617 = sext_i32_i64(group_sizze_48682);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_local mem_50618;
    
    mem_50618.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_47023);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 0,
                                  bytes_50616, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 1,
                                  sizeof(sizze_47023), &sizze_47023));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 2,
                                  sizeof(a_47027), &a_47027));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 3,
                                  sizeof(b_47029), &b_47029));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 4,
                                  sizeof(c_47031), &c_47031));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 5,
                                  sizeof(d_47033), &d_47033));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 6,
                                  sizeof(num_threads_48690),
                                  &num_threads_48690));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 7,
                                  sizeof(per_thread_elements_48693),
                                  &per_thread_elements_48693));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 8,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 9,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 10,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 11,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48698, 12,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    if (1 * (num_groups_48689 * group_sizze_48682) != 0) {
        const size_t global_work_sizze_51296[1] = {num_groups_48689 *
                     group_sizze_48682};
        const size_t local_work_sizze_51300[1] = {group_sizze_48682};
        int64_t time_start_51297 = 0, time_end_51298 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48698");
            fprintf(stderr, "%zu", global_work_sizze_51296[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51300[0]);
            fprintf(stderr, "].\n");
            time_start_51297 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48698,
                                              1, NULL, global_work_sizze_51296,
                                              local_work_sizze_51300, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51298 = get_wall_time();
            
            long time_diff_51299 = time_end_51298 - time_start_51297;
            
            ctx->chunked_reduce_kernel_48698_total_runtime += time_diff_51299;
            ctx->chunked_reduce_kernel_48698_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48698", time_diff_51299);
        }
    }
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, 4, "mem_50627");
    
    int64_t binop_x_50623 = sext_i32_i64(max_num_groups_48684);
    int64_t bytes_50622 = 4 * binop_x_50623;
    struct memblock_local mem_50624;
    
    mem_50624.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48742, 0, bytes_50622,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48742, 1,
                                  sizeof(num_groups_48689), &num_groups_48689));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48742, 2,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48742, 3,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    if (1 * max_num_groups_48684 != 0) {
        const size_t global_work_sizze_51301[1] = {max_num_groups_48684};
        const size_t local_work_sizze_51305[1] = {max_num_groups_48684};
        int64_t time_start_51302 = 0, time_end_51303 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48742");
            fprintf(stderr, "%zu", global_work_sizze_51301[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51305[0]);
            fprintf(stderr, "].\n");
            time_start_51302 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48742, 1, NULL,
                                              global_work_sizze_51301,
                                              local_work_sizze_51305, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51303 = get_wall_time();
            
            long time_diff_51304 = time_end_51303 - time_start_51302;
            
            ctx->reduce_kernel_48742_total_runtime += time_diff_51304;
            ctx->reduce_kernel_48742_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48742",
                    time_diff_51304);
        }
    }
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    
    float read_res_51306;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50627.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51306, 0, NULL, NULL));
    
    float res_47054 = read_res_51306;
    
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    scalar_out_50849 = res_47054;
    *out_scalar_out_51295 = scalar_out_50849;
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    return 0;
}
static int futrts_dot6(struct futhark_context *ctx, float *out_scalar_out_51307,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613,
                       int64_t D_mem_sizze_50614,
                       struct memblock_device D_mem_50615, int32_t sizze_47069,
                       int32_t sizze_47070, int32_t sizze_47071,
                       int32_t sizze_47072)
{
    float scalar_out_50865;
    bool dim_zzero_47077 = 0 == sizze_47070;
    bool dim_zzero_47078 = 0 == sizze_47069;
    bool both_empty_47079 = dim_zzero_47077 && dim_zzero_47078;
    bool dim_match_47080 = sizze_47069 == sizze_47070;
    bool empty_or_match_47081 = both_empty_47079 || dim_match_47080;
    bool empty_or_match_cert_47082;
    
    if (!empty_or_match_47081) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:29:1-32:33 -> futhark_test.fut:30:14-30:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_47083 = 0 == sizze_47072;
    bool dim_zzero_47084 = 0 == sizze_47071;
    bool both_empty_47085 = dim_zzero_47083 && dim_zzero_47084;
    bool dim_match_47086 = sizze_47071 == sizze_47072;
    bool empty_or_match_47087 = both_empty_47085 || dim_match_47086;
    bool empty_or_match_cert_47088;
    
    if (!empty_or_match_47087) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:29:1-32:33 -> futhark_test.fut:31:14-31:25",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool both_empty_47089 = dim_zzero_47078 && dim_zzero_47084;
    bool dim_match_47090 = sizze_47069 == sizze_47071;
    bool empty_or_match_47091 = both_empty_47089 || dim_match_47090;
    bool empty_or_match_cert_47092;
    
    if (!empty_or_match_47091) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:29:1-32:33 -> futhark_test.fut:32:19-32:32",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48765;
    
    group_sizze_48765 = ctx->sizes.group_sizze_48764;
    
    int32_t max_num_groups_48767;
    
    max_num_groups_48767 = ctx->sizes.max_num_groups_48766;
    
    int32_t y_48768 = group_sizze_48765 - 1;
    int32_t x_48769 = sizze_47069 + y_48768;
    int32_t w_div_group_sizze_48770 = squot32(x_48769, group_sizze_48765);
    int32_t num_groups_maybe_zzero_48771 = smin32(max_num_groups_48767,
                                                  w_div_group_sizze_48770);
    int32_t num_groups_48772 = smax32(1, num_groups_maybe_zzero_48771);
    int32_t num_threads_48773 = group_sizze_48765 * num_groups_48772;
    int32_t y_48774 = num_threads_48773 - 1;
    int32_t x_48775 = sizze_47069 + y_48774;
    int32_t per_thread_elements_48776 = squot32(x_48775, num_threads_48773);
    int64_t binop_x_50620 = sext_i32_i64(num_groups_48772);
    int64_t bytes_50619 = 4 * binop_x_50620;
    struct memblock_device mem_50621;
    
    mem_50621.references = NULL;
    memblock_alloc_device(ctx, &mem_50621, bytes_50619, "mem_50621");
    
    int64_t binop_x_50617 = sext_i32_i64(group_sizze_48765);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_local mem_50618;
    
    mem_50618.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_47069);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 0,
                                  bytes_50616, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 1,
                                  sizeof(sizze_47069), &sizze_47069));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 2,
                                  sizeof(num_threads_48773),
                                  &num_threads_48773));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 3,
                                  sizeof(per_thread_elements_48776),
                                  &per_thread_elements_48776));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 7,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48781, 8,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    if (1 * (num_groups_48772 * group_sizze_48765) != 0) {
        const size_t global_work_sizze_51308[1] = {num_groups_48772 *
                     group_sizze_48765};
        const size_t local_work_sizze_51312[1] = {group_sizze_48765};
        int64_t time_start_51309 = 0, time_end_51310 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48781");
            fprintf(stderr, "%zu", global_work_sizze_51308[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51312[0]);
            fprintf(stderr, "].\n");
            time_start_51309 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48781,
                                              1, NULL, global_work_sizze_51308,
                                              local_work_sizze_51312, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51310 = get_wall_time();
            
            long time_diff_51311 = time_end_51310 - time_start_51309;
            
            ctx->chunked_reduce_kernel_48781_total_runtime += time_diff_51311;
            ctx->chunked_reduce_kernel_48781_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48781", time_diff_51311);
        }
    }
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, 4, "mem_50627");
    
    int64_t binop_x_50623 = sext_i32_i64(max_num_groups_48767);
    int64_t bytes_50622 = 4 * binop_x_50623;
    struct memblock_local mem_50624;
    
    mem_50624.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48821, 0, bytes_50622,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48821, 1,
                                  sizeof(num_groups_48772), &num_groups_48772));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48821, 2,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48821, 3,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    if (1 * max_num_groups_48767 != 0) {
        const size_t global_work_sizze_51313[1] = {max_num_groups_48767};
        const size_t local_work_sizze_51317[1] = {max_num_groups_48767};
        int64_t time_start_51314 = 0, time_end_51315 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48821");
            fprintf(stderr, "%zu", global_work_sizze_51313[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51317[0]);
            fprintf(stderr, "].\n");
            time_start_51314 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48821, 1, NULL,
                                              global_work_sizze_51313,
                                              local_work_sizze_51317, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51315 = get_wall_time();
            
            long time_diff_51316 = time_end_51315 - time_start_51314;
            
            ctx->reduce_kernel_48821_total_runtime += time_diff_51316;
            ctx->reduce_kernel_48821_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48821",
                    time_diff_51316);
        }
    }
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    
    float read_res_51318;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50627.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51318, 0, NULL, NULL));
    
    float res_47096 = read_res_51318;
    
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    scalar_out_50865 = res_47096;
    *out_scalar_out_51307 = scalar_out_50865;
    memblock_unref_local(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    return 0;
}
static int futrts_dot1(struct futhark_context *ctx, float *out_scalar_out_51319,
                       int64_t A_mem_sizze_50608,
                       struct memblock_device A_mem_50609,
                       int64_t B_mem_sizze_50610,
                       struct memblock_device B_mem_50611,
                       int64_t C_mem_sizze_50612,
                       struct memblock_device C_mem_50613, int32_t sizze_47107,
                       int32_t sizze_47108, int32_t sizze_47109)
{
    float scalar_out_50881;
    bool dim_zzero_47113 = 0 == sizze_47108;
    bool dim_zzero_47114 = 0 == sizze_47107;
    bool both_empty_47115 = dim_zzero_47113 && dim_zzero_47114;
    bool dim_match_47116 = sizze_47107 == sizze_47108;
    bool empty_or_match_47117 = both_empty_47115 || dim_match_47116;
    bool empty_or_match_cert_47118;
    
    if (!empty_or_match_47117) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:14:1-15:47",
                               "function arguments of wrong shape");
        return 1;
    }
    
    bool dim_zzero_47119 = 0 == sizze_47109;
    bool both_empty_47120 = dim_zzero_47114 && dim_zzero_47119;
    bool dim_match_47121 = sizze_47107 == sizze_47109;
    bool empty_or_match_47122 = both_empty_47120 || dim_match_47121;
    bool empty_or_match_cert_47123;
    
    if (!empty_or_match_47122) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:14:1-15:47",
                               "function arguments of wrong shape");
        return 1;
    }
    
    int32_t group_sizze_48842;
    
    group_sizze_48842 = ctx->sizes.group_sizze_48841;
    
    int32_t max_num_groups_48844;
    
    max_num_groups_48844 = ctx->sizes.max_num_groups_48843;
    
    int32_t y_48845 = group_sizze_48842 - 1;
    int32_t x_48846 = sizze_47107 + y_48845;
    int32_t w_div_group_sizze_48847 = squot32(x_48846, group_sizze_48842);
    int32_t num_groups_maybe_zzero_48848 = smin32(max_num_groups_48844,
                                                  w_div_group_sizze_48847);
    int32_t num_groups_48849 = smax32(1, num_groups_maybe_zzero_48848);
    int32_t num_threads_48850 = group_sizze_48842 * num_groups_48849;
    int32_t y_48851 = num_threads_48850 - 1;
    int32_t x_48852 = sizze_47107 + y_48851;
    int32_t per_thread_elements_48853 = squot32(x_48852, num_threads_48850);
    int64_t binop_x_50618 = sext_i32_i64(num_groups_48849);
    int64_t bytes_50617 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50617, "mem_50619");
    
    int64_t binop_x_50615 = sext_i32_i64(group_sizze_48842);
    int64_t bytes_50614 = 4 * binop_x_50615;
    struct memblock_local mem_50616;
    
    mem_50616.references = NULL;
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_47107);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 0,
                                  bytes_50614, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 1,
                                  sizeof(sizze_47107), &sizze_47107));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 2,
                                  sizeof(num_threads_48850),
                                  &num_threads_48850));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 3,
                                  sizeof(per_thread_elements_48853),
                                  &per_thread_elements_48853));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48858, 7,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    if (1 * (num_groups_48849 * group_sizze_48842) != 0) {
        const size_t global_work_sizze_51320[1] = {num_groups_48849 *
                     group_sizze_48842};
        const size_t local_work_sizze_51324[1] = {group_sizze_48842};
        int64_t time_start_51321 = 0, time_end_51322 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48858");
            fprintf(stderr, "%zu", global_work_sizze_51320[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51324[0]);
            fprintf(stderr, "].\n");
            time_start_51321 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48858,
                                              1, NULL, global_work_sizze_51320,
                                              local_work_sizze_51324, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51322 = get_wall_time();
            
            long time_diff_51323 = time_end_51322 - time_start_51321;
            
            ctx->chunked_reduce_kernel_48858_total_runtime += time_diff_51323;
            ctx->chunked_reduce_kernel_48858_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48858", time_diff_51323);
        }
    }
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    
    struct memblock_device mem_50625;
    
    mem_50625.references = NULL;
    memblock_alloc_device(ctx, &mem_50625, 4, "mem_50625");
    
    int64_t binop_x_50621 = sext_i32_i64(max_num_groups_48844);
    int64_t bytes_50620 = 4 * binop_x_50621;
    struct memblock_local mem_50622;
    
    mem_50622.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48894, 0, bytes_50620,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48894, 1,
                                  sizeof(num_groups_48849), &num_groups_48849));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48894, 2,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48894, 3,
                                  sizeof(mem_50625.mem), &mem_50625.mem));
    if (1 * max_num_groups_48844 != 0) {
        const size_t global_work_sizze_51325[1] = {max_num_groups_48844};
        const size_t local_work_sizze_51329[1] = {max_num_groups_48844};
        int64_t time_start_51326 = 0, time_end_51327 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48894");
            fprintf(stderr, "%zu", global_work_sizze_51325[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51329[0]);
            fprintf(stderr, "].\n");
            time_start_51326 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48894, 1, NULL,
                                              global_work_sizze_51325,
                                              local_work_sizze_51329, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51327 = get_wall_time();
            
            long time_diff_51328 = time_end_51327 - time_start_51326;
            
            ctx->reduce_kernel_48894_total_runtime += time_diff_51328;
            ctx->reduce_kernel_48894_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48894",
                    time_diff_51328);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    
    float read_res_51330;
    
    OPENCL_SUCCEED(clEnqueueReadBuffer(ctx->opencl.queue, mem_50625.mem,
                                       CL_TRUE, 0, sizeof(float),
                                       &read_res_51330, 0, NULL, NULL));
    
    float res_47126 = read_res_51330;
    
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    scalar_out_50881 = res_47126;
    *out_scalar_out_51319 = scalar_out_50881;
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    return 0;
}
static int futrts_t1b_2d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51331,
                         struct memblock_device *out_mem_p_51332,
                         int32_t *out_out_arrsizze_51333,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int32_t sizze_47135, int32_t sizze_47136,
                         int32_t sizze_47137)
{
    int64_t out_memsizze_50898;
    struct memblock_device out_mem_50897;
    
    out_mem_50897.references = NULL;
    
    int32_t out_arrsizze_50899;
    bool dim_zzero_47140 = 0 == sizze_47137;
    bool dim_zzero_47141 = 0 == sizze_47135;
    bool both_empty_47142 = dim_zzero_47140 && dim_zzero_47141;
    bool dim_match_47143 = sizze_47135 == sizze_47137;
    bool empty_or_match_47144 = both_empty_47142 || dim_match_47143;
    bool empty_or_match_cert_47145;
    
    if (!empty_or_match_47144) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:36:1-39:44",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50897, "out_mem_50897");
        return 1;
    }
    
    int64_t binop_x_50613 = sext_i32_i64(sizze_47136);
    int64_t bytes_50612 = 4 * binop_x_50613;
    struct memblock_device mem_50614;
    
    mem_50614.references = NULL;
    memblock_alloc_device(ctx, &mem_50614, bytes_50612, "mem_50614");
    
    int32_t group_sizze_50902;
    int32_t num_groups_50903;
    
    group_sizze_50902 = ctx->sizes.group_sizze_50902;
    num_groups_50903 = squot32(sizze_47136 + sext_i32_i32(group_sizze_50902) -
                               1, sext_i32_i32(group_sizze_50902));
    OPENCL_SUCCEED(clSetKernelArg(ctx->kernel_replicate_47146, 0,
                                  sizeof(sizze_47136), &sizze_47136));
    OPENCL_SUCCEED(clSetKernelArg(ctx->kernel_replicate_47146, 1,
                                  sizeof(mem_50614.mem), &mem_50614.mem));
    if (1 * (num_groups_50903 * group_sizze_50902) != 0) {
        const size_t global_work_sizze_51334[1] = {num_groups_50903 *
                     group_sizze_50902};
        const size_t local_work_sizze_51338[1] = {group_sizze_50902};
        int64_t time_start_51335 = 0, time_end_51336 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "kernel_replicate_47146");
            fprintf(stderr, "%zu", global_work_sizze_51334[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51338[0]);
            fprintf(stderr, "].\n");
            time_start_51335 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->kernel_replicate_47146, 1,
                                              NULL, global_work_sizze_51334,
                                              local_work_sizze_51338, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51336 = get_wall_time();
            
            long time_diff_51337 = time_end_51336 - time_start_51335;
            
            ctx->kernel_replicate_47146_total_runtime += time_diff_51337;
            ctx->kernel_replicate_47146_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "kernel_replicate_47146", time_diff_51337);
        }
    }
    
    int32_t group_sizze_48922;
    
    group_sizze_48922 = ctx->sizes.group_sizze_48921;
    
    int32_t max_num_groups_48924;
    
    max_num_groups_48924 = ctx->sizes.max_num_groups_48923;
    
    int32_t y_48925 = group_sizze_48922 - 1;
    int32_t x_48926 = sizze_47135 + y_48925;
    int32_t w_div_group_sizze_48927 = squot32(x_48926, group_sizze_48922);
    int32_t num_groups_maybe_zzero_48928 = smin32(max_num_groups_48924,
                                                  w_div_group_sizze_48927);
    int32_t num_groups_48929 = smax32(1, num_groups_maybe_zzero_48928);
    int32_t num_threads_48930 = group_sizze_48922 * num_groups_48929;
    int32_t y_48931 = num_threads_48930 - 1;
    int32_t x_48932 = sizze_47135 + y_48931;
    int32_t per_thread_elements_48933 = squot32(x_48932, num_threads_48930);
    int32_t y_50497 = smod32(sizze_47135, num_threads_48930);
    int32_t x_50498 = num_threads_48930 - y_50497;
    int32_t y_50499 = smod32(x_50498, num_threads_48930);
    int32_t padded_sizze_50500 = sizze_47135 + y_50499;
    int32_t per_chunk_50502 = squot32(padded_sizze_50500, num_threads_48930);
    int32_t convop_x_50616 = sizze_47136 * y_50499;
    int64_t binop_x_50617 = sext_i32_i64(convop_x_50616);
    int64_t bytes_50615 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50615, "mem_50618");
    
    int32_t convop_x_50620 = sizze_47136 * padded_sizze_50500;
    int64_t binop_x_50621 = sext_i32_i64(convop_x_50620);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50619, "mem_50622");
    
    int32_t tmp_offs_50904 = 0;
    
    if (sizze_47135 * sizze_47136 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, A_mem_50609.mem,
                                           mem_50622.mem, 0, sizze_47136 *
                                           tmp_offs_50904 * 4, sizze_47135 *
                                           sizze_47136 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_50904 += sizze_47135;
    if (y_50499 * sizze_47136 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_50618.mem,
                                           mem_50622.mem, 0, sizze_47136 *
                                           tmp_offs_50904 * 4, y_50499 *
                                           sizze_47136 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_50904 += y_50499;
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    
    int32_t binop_x_50624 = sizze_47136 * per_chunk_50502;
    int32_t convop_x_50625 = num_threads_48930 * binop_x_50624;
    int64_t binop_x_50626 = sext_i32_i64(convop_x_50625);
    int64_t bytes_50623 = 4 * binop_x_50626;
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, bytes_50623, "mem_50627");
    
    int call_ret_51339 = futrts_map_transpose_opencl_f32(ctx, mem_50627, 0,
                                                         mem_50622, 0, 1,
                                                         per_chunk_50502 *
                                                         sizze_47136,
                                                         num_threads_48930,
                                                         num_threads_48930 *
                                                         per_chunk_50502 *
                                                         sizze_47136,
                                                         num_threads_48930 *
                                                         per_chunk_50502 *
                                                         sizze_47136);
    
    assert(call_ret_51339 == 0);
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    
    int32_t y_50509 = smod32(sizze_47137, num_threads_48930);
    int32_t x_50510 = num_threads_48930 - y_50509;
    int32_t y_50511 = smod32(x_50510, num_threads_48930);
    int32_t padded_sizze_50512 = sizze_47137 + y_50511;
    int32_t per_chunk_50514 = squot32(padded_sizze_50512, num_threads_48930);
    int64_t binop_x_50629 = sext_i32_i64(y_50511);
    int64_t bytes_50628 = 4 * binop_x_50629;
    struct memblock_device mem_50630;
    
    mem_50630.references = NULL;
    memblock_alloc_device(ctx, &mem_50630, bytes_50628, "mem_50630");
    
    int64_t binop_x_50632 = sext_i32_i64(padded_sizze_50512);
    int64_t bytes_50631 = 4 * binop_x_50632;
    struct memblock_device mem_50633;
    
    mem_50633.references = NULL;
    memblock_alloc_device(ctx, &mem_50633, bytes_50631, "mem_50633");
    
    int32_t tmp_offs_50905 = 0;
    
    if (sizze_47137 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, B_mem_50611.mem,
                                           mem_50633.mem, 0, tmp_offs_50905 * 4,
                                           sizze_47137 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_50905 += sizze_47137;
    if (y_50511 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_50630.mem,
                                           mem_50633.mem, 0, tmp_offs_50905 * 4,
                                           y_50511 * sizeof(float), 0, NULL,
                                           NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    tmp_offs_50905 += y_50511;
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    
    int32_t convop_x_50635 = num_threads_48930 * per_chunk_50514;
    int64_t binop_x_50636 = sext_i32_i64(convop_x_50635);
    int64_t bytes_50634 = 4 * binop_x_50636;
    struct memblock_device mem_50637;
    
    mem_50637.references = NULL;
    memblock_alloc_device(ctx, &mem_50637, bytes_50634, "mem_50637");
    
    int call_ret_51340 = futrts_map_transpose_opencl_f32(ctx, mem_50637, 0,
                                                         mem_50633, 0, 1,
                                                         per_chunk_50514,
                                                         num_threads_48930,
                                                         num_threads_48930 *
                                                         per_chunk_50514,
                                                         num_threads_48930 *
                                                         per_chunk_50514);
    
    assert(call_ret_51340 == 0);
    memblock_unref_device(ctx, &mem_50633, "mem_50633");
    
    int32_t convop_x_50662 = sizze_47136 * num_groups_48929;
    int64_t binop_x_50663 = sext_i32_i64(convop_x_50662);
    int64_t bytes_50661 = 4 * binop_x_50663;
    struct memblock_device mem_50664;
    
    mem_50664.references = NULL;
    memblock_alloc_device(ctx, &mem_50664, bytes_50661, "mem_50664");
    
    int32_t convop_x_50652 = sizze_47136 * group_sizze_48922;
    int64_t binop_x_50653 = sext_i32_i64(convop_x_50652);
    int64_t bytes_50651 = 4 * binop_x_50653;
    int64_t num_threads64_50713 = sext_i32_i64(num_threads_48930);
    int64_t total_sizze_50714 = bytes_50612 * num_threads64_50713;
    struct memblock_device mem_50645;
    
    mem_50645.references = NULL;
    memblock_alloc_device(ctx, &mem_50645, total_sizze_50714, "mem_50645");
    
    int64_t total_sizze_50715 = bytes_50612 * num_threads64_50713;
    struct memblock_device mem_50648;
    
    mem_50648.references = NULL;
    memblock_alloc_device(ctx, &mem_50648, total_sizze_50715, "mem_50648");
    
    struct memblock_local mem_50654;
    
    mem_50654.references = NULL;
    
    int64_t total_sizze_50716 = bytes_50612 * num_threads64_50713;
    struct memblock_device mem_50657;
    
    mem_50657.references = NULL;
    memblock_alloc_device(ctx, &mem_50657, total_sizze_50716, "mem_50657");
    
    int64_t total_sizze_50717 = bytes_50612 * num_threads64_50713;
    struct memblock_device mem_50660;
    
    mem_50660.references = NULL;
    memblock_alloc_device(ctx, &mem_50660, total_sizze_50717, "mem_50660");
    
    int64_t total_sizze_50718 = bytes_50612 * num_threads64_50713;
    struct memblock_device double_buffer_mem_50693;
    
    double_buffer_mem_50693.references = NULL;
    memblock_alloc_device(ctx, &double_buffer_mem_50693, total_sizze_50718,
                          "double_buffer_mem_50693");
    if (ctx->debugging)
        fprintf(stderr, "%s: %d\n", "input size", (int) sizze_47135);
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 0,
                                  bytes_50651, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 1,
                                  sizeof(sizze_47135), &sizze_47135));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 2,
                                  sizeof(sizze_47136), &sizze_47136));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 3,
                                  sizeof(num_threads_48930),
                                  &num_threads_48930));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 4,
                                  sizeof(per_thread_elements_48933),
                                  &per_thread_elements_48933));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 5,
                                  sizeof(per_chunk_50514), &per_chunk_50514));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 6,
                                  sizeof(mem_50614.mem), &mem_50614.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 7,
                                  sizeof(binop_x_50624), &binop_x_50624));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 8,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 9,
                                  sizeof(mem_50637.mem), &mem_50637.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 10,
                                  sizeof(mem_50645.mem), &mem_50645.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 11,
                                  sizeof(mem_50648.mem), &mem_50648.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 12,
                                  sizeof(mem_50657.mem), &mem_50657.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 13,
                                  sizeof(mem_50660.mem), &mem_50660.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 14,
                                  sizeof(mem_50664.mem), &mem_50664.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->chunked_reduce_kernel_48938, 15,
                                  sizeof(double_buffer_mem_50693.mem),
                                  &double_buffer_mem_50693.mem));
    if (1 * (num_groups_48929 * group_sizze_48922) != 0) {
        const size_t global_work_sizze_51341[1] = {num_groups_48929 *
                     group_sizze_48922};
        const size_t local_work_sizze_51345[1] = {group_sizze_48922};
        int64_t time_start_51342 = 0, time_end_51343 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "chunked_reduce_kernel_48938");
            fprintf(stderr, "%zu", global_work_sizze_51341[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51345[0]);
            fprintf(stderr, "].\n");
            time_start_51342 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->chunked_reduce_kernel_48938,
                                              1, NULL, global_work_sizze_51341,
                                              local_work_sizze_51345, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51343 = get_wall_time();
            
            long time_diff_51344 = time_end_51343 - time_start_51342;
            
            ctx->chunked_reduce_kernel_48938_total_runtime += time_diff_51344;
            ctx->chunked_reduce_kernel_48938_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n",
                    "chunked_reduce_kernel_48938", time_diff_51344);
        }
    }
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &mem_50637, "mem_50637");
    memblock_unref_device(ctx, &mem_50645, "mem_50645");
    memblock_unref_device(ctx, &mem_50648, "mem_50648");
    memblock_unref_local(ctx, &mem_50654, "mem_50654");
    memblock_unref_device(ctx, &mem_50657, "mem_50657");
    memblock_unref_device(ctx, &mem_50660, "mem_50660");
    memblock_unref_device(ctx, &double_buffer_mem_50693,
                          "double_buffer_mem_50693");
    
    struct memblock_device mem_50683;
    
    mem_50683.references = NULL;
    memblock_alloc_device(ctx, &mem_50683, bytes_50612, "mem_50683");
    
    int32_t convop_x_50672 = sizze_47136 * max_num_groups_48924;
    int64_t binop_x_50673 = sext_i32_i64(convop_x_50672);
    int64_t bytes_50671 = 4 * binop_x_50673;
    int64_t num_threads64_50719 = sext_i32_i64(max_num_groups_48924);
    struct memblock_local mem_50674;
    
    mem_50674.references = NULL;
    
    int64_t total_sizze_50720 = bytes_50612 * num_threads64_50719;
    struct memblock_device mem_50677;
    
    mem_50677.references = NULL;
    memblock_alloc_device(ctx, &mem_50677, total_sizze_50720, "mem_50677");
    
    int64_t total_sizze_50721 = bytes_50612 * num_threads64_50719;
    struct memblock_device mem_50680;
    
    mem_50680.references = NULL;
    memblock_alloc_device(ctx, &mem_50680, total_sizze_50721, "mem_50680");
    
    int64_t total_sizze_50722 = bytes_50612 * num_threads64_50719;
    struct memblock_device mem_50690;
    
    mem_50690.references = NULL;
    memblock_alloc_device(ctx, &mem_50690, total_sizze_50722, "mem_50690");
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 0, bytes_50671,
                                  NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 1,
                                  sizeof(sizze_47136), &sizze_47136));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 2,
                                  sizeof(num_groups_48929), &num_groups_48929));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 3,
                                  sizeof(mem_50614.mem), &mem_50614.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 4,
                                  sizeof(mem_50664.mem), &mem_50664.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 5,
                                  sizeof(mem_50677.mem), &mem_50677.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 6,
                                  sizeof(mem_50680.mem), &mem_50680.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 7,
                                  sizeof(mem_50683.mem), &mem_50683.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->reduce_kernel_48993, 8,
                                  sizeof(mem_50690.mem), &mem_50690.mem));
    if (1 * max_num_groups_48924 != 0) {
        const size_t global_work_sizze_51346[1] = {max_num_groups_48924};
        const size_t local_work_sizze_51350[1] = {max_num_groups_48924};
        int64_t time_start_51347 = 0, time_end_51348 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "reduce_kernel_48993");
            fprintf(stderr, "%zu", global_work_sizze_51346[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51350[0]);
            fprintf(stderr, "].\n");
            time_start_51347 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->reduce_kernel_48993, 1, NULL,
                                              global_work_sizze_51346,
                                              local_work_sizze_51350, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51348 = get_wall_time();
            
            long time_diff_51349 = time_end_51348 - time_start_51347;
            
            ctx->reduce_kernel_48993_total_runtime += time_diff_51349;
            ctx->reduce_kernel_48993_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "reduce_kernel_48993",
                    time_diff_51349);
        }
    }
    memblock_unref_device(ctx, &mem_50614, "mem_50614");
    memblock_unref_device(ctx, &mem_50664, "mem_50664");
    memblock_unref_local(ctx, &mem_50674, "mem_50674");
    memblock_unref_device(ctx, &mem_50677, "mem_50677");
    memblock_unref_device(ctx, &mem_50680, "mem_50680");
    memblock_unref_device(ctx, &mem_50690, "mem_50690");
    
    struct memblock_device mem_50686;
    
    mem_50686.references = NULL;
    memblock_alloc_device(ctx, &mem_50686, bytes_50612, "mem_50686");
    if (sizze_47136 * sizeof(float) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(ctx->opencl.queue, mem_50683.mem,
                                           mem_50686.mem, 0, 0, sizze_47136 *
                                           sizeof(float), 0, NULL, NULL));
        if (ctx->debugging)
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
    }
    memblock_unref_device(ctx, &mem_50683, "mem_50683");
    out_arrsizze_50899 = sizze_47136;
    out_memsizze_50898 = bytes_50612;
    memblock_set_device(ctx, &out_mem_50897, &mem_50686, "mem_50686");
    *out_out_memsizze_51331 = out_memsizze_50898;
    (*out_mem_p_51332).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51332, &out_mem_50897,
                        "out_mem_50897");
    *out_out_arrsizze_51333 = out_arrsizze_50899;
    memblock_unref_device(ctx, &mem_50686, "mem_50686");
    memblock_unref_device(ctx, &mem_50690, "mem_50690");
    memblock_unref_device(ctx, &mem_50680, "mem_50680");
    memblock_unref_device(ctx, &mem_50677, "mem_50677");
    memblock_unref_local(ctx, &mem_50674, "mem_50674");
    memblock_unref_device(ctx, &mem_50683, "mem_50683");
    memblock_unref_device(ctx, &double_buffer_mem_50693,
                          "double_buffer_mem_50693");
    memblock_unref_device(ctx, &mem_50660, "mem_50660");
    memblock_unref_device(ctx, &mem_50657, "mem_50657");
    memblock_unref_local(ctx, &mem_50654, "mem_50654");
    memblock_unref_device(ctx, &mem_50648, "mem_50648");
    memblock_unref_device(ctx, &mem_50645, "mem_50645");
    memblock_unref_device(ctx, &mem_50664, "mem_50664");
    memblock_unref_device(ctx, &mem_50637, "mem_50637");
    memblock_unref_device(ctx, &mem_50633, "mem_50633");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50614, "mem_50614");
    memblock_unref_device(ctx, &out_mem_50897, "out_mem_50897");
    return 0;
}
static int futrts_t1_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51351,
                        struct memblock_device *out_mem_p_51352,
                        int32_t *out_out_arrsizze_51353,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611, int32_t sizze_47160,
                        int32_t sizze_47161, int32_t sizze_47162)
{
    int64_t out_memsizze_50944;
    struct memblock_device out_mem_50943;
    
    out_mem_50943.references = NULL;
    
    int32_t out_arrsizze_50945;
    bool dim_zzero_47165 = 0 == sizze_47162;
    bool dim_zzero_47166 = 0 == sizze_47161;
    bool both_empty_47167 = dim_zzero_47165 && dim_zzero_47166;
    bool dim_match_47168 = sizze_47161 == sizze_47162;
    bool empty_or_match_47169 = both_empty_47167 || dim_match_47168;
    bool empty_or_match_cert_47170;
    
    if (!empty_or_match_47169) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:41:1-42:47",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50943, "out_mem_50943");
        return 1;
    }
    
    int32_t group_sizze_49015;
    
    group_sizze_49015 = ctx->sizes.group_sizze_49014;
    
    int32_t y_49016 = group_sizze_49015 - 1;
    int32_t x_49017 = sizze_47160 + y_49016;
    int32_t num_groups_49018 = squot32(x_49017, group_sizze_49015);
    int32_t num_threads_49019 = group_sizze_49015 * num_groups_49018;
    int32_t convop_x_50613 = sizze_47160 * sizze_47161;
    int64_t binop_x_50614 = sext_i32_i64(convop_x_50613);
    int64_t bytes_50612 = 4 * binop_x_50614;
    struct memblock_device mem_50615;
    
    mem_50615.references = NULL;
    memblock_alloc_device(ctx, &mem_50615, bytes_50612, "mem_50615");
    
    int call_ret_51354 = futrts_map_transpose_opencl_f32(ctx, mem_50615, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47161,
                                                         sizze_47160,
                                                         sizze_47160 *
                                                         sizze_47161,
                                                         sizze_47160 *
                                                         sizze_47161);
    
    assert(call_ret_51354 == 0);
    
    int64_t binop_x_50617 = sext_i32_i64(sizze_47160);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50616, "mem_50618");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49020, 0, sizeof(sizze_47160),
                                  &sizze_47160));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49020, 1, sizeof(sizze_47161),
                                  &sizze_47161));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49020, 2,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49020, 3,
                                  sizeof(mem_50615.mem), &mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49020, 4,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    if (1 * (num_groups_49018 * group_sizze_49015) != 0) {
        const size_t global_work_sizze_51355[1] = {num_groups_49018 *
                     group_sizze_49015};
        const size_t local_work_sizze_51359[1] = {group_sizze_49015};
        int64_t time_start_51356 = 0, time_end_51357 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49020");
            fprintf(stderr, "%zu", global_work_sizze_51355[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51359[0]);
            fprintf(stderr, "].\n");
            time_start_51356 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49020, 1, NULL,
                                              global_work_sizze_51355,
                                              local_work_sizze_51359, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51357 = get_wall_time();
            
            long time_diff_51358 = time_end_51357 - time_start_51356;
            
            ctx->map_kernel_49020_total_runtime += time_diff_51358;
            ctx->map_kernel_49020_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49020",
                    time_diff_51358);
        }
    }
    memblock_unref_device(ctx, &mem_50615, "mem_50615");
    out_arrsizze_50945 = sizze_47160;
    out_memsizze_50944 = bytes_50616;
    memblock_set_device(ctx, &out_mem_50943, &mem_50618, "mem_50618");
    *out_out_memsizze_51351 = out_memsizze_50944;
    (*out_mem_p_51352).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51352, &out_mem_50943,
                        "out_mem_50943");
    *out_out_arrsizze_51353 = out_arrsizze_50945;
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50615, "mem_50615");
    memblock_unref_device(ctx, &out_mem_50943, "out_mem_50943");
    return 0;
}
static int futrts_t2_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51360,
                        struct memblock_device *out_mem_p_51361,
                        int32_t *out_out_arrsizze_51362,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47181,
                        int32_t sizze_47182, int32_t sizze_47183,
                        int32_t sizze_47184)
{
    int64_t out_memsizze_50951;
    struct memblock_device out_mem_50950;
    
    out_mem_50950.references = NULL;
    
    int32_t out_arrsizze_50952;
    bool dim_zzero_47188 = 0 == sizze_47183;
    bool dim_zzero_47189 = 0 == sizze_47182;
    bool both_empty_47190 = dim_zzero_47188 && dim_zzero_47189;
    bool dim_match_47191 = sizze_47182 == sizze_47183;
    bool empty_or_match_47192 = both_empty_47190 || dim_match_47191;
    bool empty_or_match_cert_47193;
    
    if (!empty_or_match_47192) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:44:1-45:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50950, "out_mem_50950");
        return 1;
    }
    
    bool dim_zzero_47194 = 0 == sizze_47184;
    bool dim_zzero_47195 = 0 == sizze_47181;
    bool both_empty_47196 = dim_zzero_47194 && dim_zzero_47195;
    bool dim_match_47197 = sizze_47181 == sizze_47184;
    bool empty_or_match_47198 = both_empty_47196 || dim_match_47197;
    bool empty_or_match_cert_47199;
    
    if (!empty_or_match_47198) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:44:1-45:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50950, "out_mem_50950");
        return 1;
    }
    
    int32_t group_sizze_49055;
    
    group_sizze_49055 = ctx->sizes.group_sizze_49054;
    
    int32_t y_49056 = group_sizze_49055 - 1;
    int32_t x_49057 = sizze_47181 + y_49056;
    int32_t num_groups_49058 = squot32(x_49057, group_sizze_49055);
    int32_t num_threads_49059 = group_sizze_49055 * num_groups_49058;
    int32_t convop_x_50615 = sizze_47181 * sizze_47182;
    int64_t binop_x_50616 = sext_i32_i64(convop_x_50615);
    int64_t bytes_50614 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50614, "mem_50617");
    
    int call_ret_51363 = futrts_map_transpose_opencl_f32(ctx, mem_50617, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47182,
                                                         sizze_47181,
                                                         sizze_47181 *
                                                         sizze_47182,
                                                         sizze_47181 *
                                                         sizze_47182);
    
    assert(call_ret_51363 == 0);
    
    int64_t binop_x_50619 = sext_i32_i64(sizze_47181);
    int64_t bytes_50618 = 4 * binop_x_50619;
    struct memblock_device mem_50620;
    
    mem_50620.references = NULL;
    memblock_alloc_device(ctx, &mem_50620, bytes_50618, "mem_50620");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 0, sizeof(sizze_47181),
                                  &sizze_47181));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 1, sizeof(sizze_47182),
                                  &sizze_47182));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 2,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 4,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49060, 5,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    if (1 * (num_groups_49058 * group_sizze_49055) != 0) {
        const size_t global_work_sizze_51364[1] = {num_groups_49058 *
                     group_sizze_49055};
        const size_t local_work_sizze_51368[1] = {group_sizze_49055};
        int64_t time_start_51365 = 0, time_end_51366 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49060");
            fprintf(stderr, "%zu", global_work_sizze_51364[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51368[0]);
            fprintf(stderr, "].\n");
            time_start_51365 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49060, 1, NULL,
                                              global_work_sizze_51364,
                                              local_work_sizze_51368, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51366 = get_wall_time();
            
            long time_diff_51367 = time_end_51366 - time_start_51365;
            
            ctx->map_kernel_49060_total_runtime += time_diff_51367;
            ctx->map_kernel_49060_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49060",
                    time_diff_51367);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    out_arrsizze_50952 = sizze_47181;
    out_memsizze_50951 = bytes_50618;
    memblock_set_device(ctx, &out_mem_50950, &mem_50620, "mem_50620");
    *out_out_memsizze_51360 = out_memsizze_50951;
    (*out_mem_p_51361).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51361, &out_mem_50950,
                        "out_mem_50950");
    *out_out_arrsizze_51362 = out_arrsizze_50952;
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &out_mem_50950, "out_mem_50950");
    return 0;
}
static int futrts_t3_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51369,
                        struct memblock_device *out_mem_p_51370,
                        int32_t *out_out_arrsizze_51371,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47213,
                        int32_t sizze_47214, int32_t sizze_47215,
                        int32_t sizze_47216, int32_t sizze_47217)
{
    int64_t out_memsizze_50958;
    struct memblock_device out_mem_50957;
    
    out_mem_50957.references = NULL;
    
    int32_t out_arrsizze_50959;
    bool dim_zzero_47221 = 0 == sizze_47215;
    bool dim_zzero_47222 = 0 == sizze_47216;
    bool old_empty_47223 = dim_zzero_47221 || dim_zzero_47222;
    bool dim_zzero_47224 = 0 == sizze_47213;
    bool dim_zzero_47225 = 0 == sizze_47214;
    bool new_empty_47226 = dim_zzero_47224 || dim_zzero_47225;
    bool both_empty_47227 = old_empty_47223 && new_empty_47226;
    bool dim_match_47228 = sizze_47213 == sizze_47215;
    bool dim_match_47229 = sizze_47214 == sizze_47216;
    bool match_47230 = dim_match_47228 && dim_match_47229;
    bool empty_or_match_47231 = both_empty_47227 || match_47230;
    bool empty_or_match_cert_47232;
    
    if (!empty_or_match_47231) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:48:1-49:67",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50957, "out_mem_50957");
        return 1;
    }
    
    bool dim_zzero_47234 = 0 == sizze_47217;
    bool both_empty_47235 = dim_zzero_47225 && dim_zzero_47234;
    bool dim_match_47236 = sizze_47214 == sizze_47217;
    bool empty_or_match_47237 = both_empty_47235 || dim_match_47236;
    bool empty_or_match_cert_47238;
    
    if (!empty_or_match_47237) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:48:1-49:67",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50957, "out_mem_50957");
        return 1;
    }
    
    int32_t group_sizze_49099;
    
    group_sizze_49099 = ctx->sizes.group_sizze_49098;
    
    int32_t y_49100 = group_sizze_49099 - 1;
    int32_t x_49101 = sizze_47213 + y_49100;
    int32_t num_groups_49102 = squot32(x_49101, group_sizze_49099);
    int32_t num_threads_49103 = group_sizze_49099 * num_groups_49102;
    int32_t convop_x_50615 = sizze_47213 * sizze_47214;
    int64_t binop_x_50616 = sext_i32_i64(convop_x_50615);
    int64_t bytes_50614 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50614, "mem_50617");
    
    int call_ret_51372 = futrts_map_transpose_opencl_f32(ctx, mem_50617, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47214,
                                                         sizze_47213,
                                                         sizze_47213 *
                                                         sizze_47214,
                                                         sizze_47213 *
                                                         sizze_47214);
    
    assert(call_ret_51372 == 0);
    
    int32_t convop_x_50619 = sizze_47215 * sizze_47216;
    int64_t binop_x_50620 = sext_i32_i64(convop_x_50619);
    int64_t bytes_50618 = 4 * binop_x_50620;
    struct memblock_device mem_50621;
    
    mem_50621.references = NULL;
    memblock_alloc_device(ctx, &mem_50621, bytes_50618, "mem_50621");
    
    int call_ret_51373 = futrts_map_transpose_opencl_f32(ctx, mem_50621, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47216,
                                                         sizze_47215,
                                                         sizze_47215 *
                                                         sizze_47216,
                                                         sizze_47215 *
                                                         sizze_47216);
    
    assert(call_ret_51373 == 0);
    
    int64_t binop_x_50623 = sext_i32_i64(sizze_47213);
    int64_t bytes_50622 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50622, "mem_50624");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 0, sizeof(sizze_47213),
                                  &sizze_47213));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 1, sizeof(sizze_47214),
                                  &sizze_47214));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 2, sizeof(sizze_47215),
                                  &sizze_47215));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 4,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 5,
                                  sizeof(mem_50621.mem), &mem_50621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49104, 6,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (num_groups_49102 * group_sizze_49099) != 0) {
        const size_t global_work_sizze_51374[1] = {num_groups_49102 *
                     group_sizze_49099};
        const size_t local_work_sizze_51378[1] = {group_sizze_49099};
        int64_t time_start_51375 = 0, time_end_51376 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49104");
            fprintf(stderr, "%zu", global_work_sizze_51374[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51378[0]);
            fprintf(stderr, "].\n");
            time_start_51375 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49104, 1, NULL,
                                              global_work_sizze_51374,
                                              local_work_sizze_51378, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51376 = get_wall_time();
            
            long time_diff_51377 = time_end_51376 - time_start_51375;
            
            ctx->map_kernel_49104_total_runtime += time_diff_51377;
            ctx->map_kernel_49104_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49104",
                    time_diff_51377);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    out_arrsizze_50959 = sizze_47213;
    out_memsizze_50958 = bytes_50622;
    memblock_set_device(ctx, &out_mem_50957, &mem_50624, "mem_50624");
    *out_out_memsizze_51369 = out_memsizze_50958;
    (*out_mem_p_51370).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51370, &out_mem_50957,
                        "out_mem_50957");
    *out_out_arrsizze_51371 = out_arrsizze_50959;
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50621, "mem_50621");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &out_mem_50957, "out_mem_50957");
    return 0;
}
static int futrts_t4_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51379,
                        struct memblock_device *out_mem_p_51380,
                        int32_t *out_out_arrsizze_51381,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47252,
                        int32_t sizze_47253, int32_t sizze_47254,
                        int32_t sizze_47255, int32_t sizze_47256,
                        int32_t sizze_47257)
{
    int64_t out_memsizze_50965;
    struct memblock_device out_mem_50964;
    
    out_mem_50964.references = NULL;
    
    int32_t out_arrsizze_50966;
    bool dim_zzero_47262 = 0 == sizze_47254;
    bool dim_zzero_47263 = 0 == sizze_47255;
    bool old_empty_47264 = dim_zzero_47262 || dim_zzero_47263;
    bool dim_zzero_47265 = 0 == sizze_47252;
    bool dim_zzero_47266 = 0 == sizze_47253;
    bool new_empty_47267 = dim_zzero_47265 || dim_zzero_47266;
    bool both_empty_47268 = old_empty_47264 && new_empty_47267;
    bool dim_match_47269 = sizze_47252 == sizze_47254;
    bool dim_match_47270 = sizze_47253 == sizze_47255;
    bool match_47271 = dim_match_47269 && dim_match_47270;
    bool empty_or_match_47272 = both_empty_47268 || match_47271;
    bool empty_or_match_cert_47273;
    
    if (!empty_or_match_47272) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:51:1-56:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50964, "out_mem_50964");
        return 1;
    }
    
    bool dim_zzero_47275 = 0 == sizze_47256;
    bool both_empty_47276 = dim_zzero_47266 && dim_zzero_47275;
    bool dim_match_47277 = sizze_47253 == sizze_47256;
    bool empty_or_match_47278 = both_empty_47276 || dim_match_47277;
    bool empty_or_match_cert_47279;
    
    if (!empty_or_match_47278) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:51:1-56:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50964, "out_mem_50964");
        return 1;
    }
    
    bool dim_zzero_47281 = 0 == sizze_47257;
    bool both_empty_47282 = dim_zzero_47266 && dim_zzero_47281;
    bool dim_match_47283 = sizze_47253 == sizze_47257;
    bool empty_or_match_47284 = both_empty_47282 || dim_match_47283;
    bool empty_or_match_cert_47285;
    
    if (!empty_or_match_47284) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:51:1-56:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50964, "out_mem_50964");
        return 1;
    }
    
    int32_t group_sizze_49133;
    
    group_sizze_49133 = ctx->sizes.group_sizze_49132;
    
    int32_t y_49134 = group_sizze_49133 - 1;
    int32_t x_49135 = sizze_47253 + y_49134;
    int32_t num_groups_49136 = squot32(x_49135, group_sizze_49133);
    int32_t num_threads_49137 = group_sizze_49133 * num_groups_49136;
    int64_t binop_x_50617 = sext_i32_i64(sizze_47253);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50616, "mem_50618");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49138, 0, sizeof(sizze_47253),
                                  &sizze_47253));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49138, 1,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49138, 2,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49138, 3,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    if (1 * (num_groups_49136 * group_sizze_49133) != 0) {
        const size_t global_work_sizze_51382[1] = {num_groups_49136 *
                     group_sizze_49133};
        const size_t local_work_sizze_51386[1] = {group_sizze_49133};
        int64_t time_start_51383 = 0, time_end_51384 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49138");
            fprintf(stderr, "%zu", global_work_sizze_51382[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51386[0]);
            fprintf(stderr, "].\n");
            time_start_51383 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49138, 1, NULL,
                                              global_work_sizze_51382,
                                              local_work_sizze_51386, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51384 = get_wall_time();
            
            long time_diff_51385 = time_end_51384 - time_start_51383;
            
            ctx->map_kernel_49138_total_runtime += time_diff_51385;
            ctx->map_kernel_49138_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49138",
                    time_diff_51385);
        }
    }
    
    int32_t group_sizze_49159;
    
    group_sizze_49159 = ctx->sizes.group_sizze_49158;
    
    int32_t y_49160 = group_sizze_49159 - 1;
    int32_t x_49161 = sizze_47252 + y_49160;
    int32_t num_groups_49162 = squot32(x_49161, group_sizze_49159);
    int32_t num_threads_49163 = group_sizze_49159 * num_groups_49162;
    int32_t convop_x_50620 = sizze_47252 * sizze_47253;
    int64_t binop_x_50621 = sext_i32_i64(convop_x_50620);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50619, "mem_50622");
    
    int call_ret_51387 = futrts_map_transpose_opencl_f32(ctx, mem_50622, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47253,
                                                         sizze_47252,
                                                         sizze_47252 *
                                                         sizze_47253,
                                                         sizze_47252 *
                                                         sizze_47253);
    
    assert(call_ret_51387 == 0);
    
    int32_t convop_x_50624 = sizze_47254 * sizze_47255;
    int64_t binop_x_50625 = sext_i32_i64(convop_x_50624);
    int64_t bytes_50623 = 4 * binop_x_50625;
    struct memblock_device mem_50626;
    
    mem_50626.references = NULL;
    memblock_alloc_device(ctx, &mem_50626, bytes_50623, "mem_50626");
    
    int call_ret_51388 = futrts_map_transpose_opencl_f32(ctx, mem_50626, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47255,
                                                         sizze_47254,
                                                         sizze_47254 *
                                                         sizze_47255,
                                                         sizze_47254 *
                                                         sizze_47255);
    
    assert(call_ret_51388 == 0);
    
    int64_t binop_x_50628 = sext_i32_i64(sizze_47252);
    int64_t bytes_50627 = 4 * binop_x_50628;
    struct memblock_device mem_50629;
    
    mem_50629.references = NULL;
    memblock_alloc_device(ctx, &mem_50629, bytes_50627, "mem_50629");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 0, sizeof(sizze_47252),
                                  &sizze_47252));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 1, sizeof(sizze_47253),
                                  &sizze_47253));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 2, sizeof(sizze_47254),
                                  &sizze_47254));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 3,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 4,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 5,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49164, 6,
                                  sizeof(mem_50629.mem), &mem_50629.mem));
    if (1 * (num_groups_49162 * group_sizze_49159) != 0) {
        const size_t global_work_sizze_51389[1] = {num_groups_49162 *
                     group_sizze_49159};
        const size_t local_work_sizze_51393[1] = {group_sizze_49159};
        int64_t time_start_51390 = 0, time_end_51391 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49164");
            fprintf(stderr, "%zu", global_work_sizze_51389[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51393[0]);
            fprintf(stderr, "].\n");
            time_start_51390 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49164, 1, NULL,
                                              global_work_sizze_51389,
                                              local_work_sizze_51393, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51391 = get_wall_time();
            
            long time_diff_51392 = time_end_51391 - time_start_51390;
            
            ctx->map_kernel_49164_total_runtime += time_diff_51392;
            ctx->map_kernel_49164_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49164",
                    time_diff_51392);
        }
    }
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    out_arrsizze_50966 = sizze_47252;
    out_memsizze_50965 = bytes_50627;
    memblock_set_device(ctx, &out_mem_50964, &mem_50629, "mem_50629");
    *out_out_memsizze_51379 = out_memsizze_50965;
    (*out_mem_p_51380).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51380, &out_mem_50964,
                        "out_mem_50964");
    *out_out_arrsizze_51381 = out_arrsizze_50966;
    memblock_unref_device(ctx, &mem_50629, "mem_50629");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &out_mem_50964, "out_mem_50964");
    return 0;
}
static int futrts_t5_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51394,
                        struct memblock_device *out_mem_p_51395,
                        int32_t *out_out_arrsizze_51396,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47303,
                        int32_t sizze_47304, int32_t sizze_47305,
                        int32_t sizze_47306, int32_t sizze_47307,
                        int32_t sizze_47308, float a_47309, float b_47311,
                        float c_47313, float d_47315)
{
    int64_t out_memsizze_50975;
    struct memblock_device out_mem_50974;
    
    out_mem_50974.references = NULL;
    
    int32_t out_arrsizze_50976;
    bool dim_zzero_47317 = 0 == sizze_47305;
    bool dim_zzero_47318 = 0 == sizze_47306;
    bool old_empty_47319 = dim_zzero_47317 || dim_zzero_47318;
    bool dim_zzero_47320 = 0 == sizze_47303;
    bool dim_zzero_47321 = 0 == sizze_47304;
    bool new_empty_47322 = dim_zzero_47320 || dim_zzero_47321;
    bool both_empty_47323 = old_empty_47319 && new_empty_47322;
    bool dim_match_47324 = sizze_47303 == sizze_47305;
    bool dim_match_47325 = sizze_47304 == sizze_47306;
    bool match_47326 = dim_match_47324 && dim_match_47325;
    bool empty_or_match_47327 = both_empty_47323 || match_47326;
    bool empty_or_match_cert_47328;
    
    if (!empty_or_match_47327) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:58:1-63:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50974, "out_mem_50974");
        return 1;
    }
    
    bool dim_zzero_47330 = 0 == sizze_47307;
    bool both_empty_47331 = dim_zzero_47321 && dim_zzero_47330;
    bool dim_match_47332 = sizze_47304 == sizze_47307;
    bool empty_or_match_47333 = both_empty_47331 || dim_match_47332;
    bool empty_or_match_cert_47334;
    
    if (!empty_or_match_47333) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:58:1-63:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50974, "out_mem_50974");
        return 1;
    }
    
    bool dim_zzero_47335 = 0 == sizze_47308;
    bool both_empty_47336 = dim_zzero_47321 && dim_zzero_47335;
    bool dim_match_47337 = sizze_47304 == sizze_47308;
    bool empty_or_match_47338 = both_empty_47336 || dim_match_47337;
    bool empty_or_match_cert_47339;
    
    if (!empty_or_match_47338) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:58:1-63:12",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50974, "out_mem_50974");
        return 1;
    }
    
    int32_t group_sizze_49193;
    
    group_sizze_49193 = ctx->sizes.group_sizze_49192;
    
    int32_t y_49194 = group_sizze_49193 - 1;
    int32_t x_49195 = sizze_47304 + y_49194;
    int32_t num_groups_49196 = squot32(x_49195, group_sizze_49193);
    int32_t num_threads_49197 = group_sizze_49193 * num_groups_49196;
    int64_t binop_x_50617 = sext_i32_i64(sizze_47304);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50616, "mem_50618");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 0, sizeof(sizze_47304),
                                  &sizze_47304));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 1, sizeof(c_47313),
                                  &c_47313));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 2, sizeof(d_47315),
                                  &d_47315));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 4,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49198, 5,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    if (1 * (num_groups_49196 * group_sizze_49193) != 0) {
        const size_t global_work_sizze_51397[1] = {num_groups_49196 *
                     group_sizze_49193};
        const size_t local_work_sizze_51401[1] = {group_sizze_49193};
        int64_t time_start_51398 = 0, time_end_51399 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49198");
            fprintf(stderr, "%zu", global_work_sizze_51397[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51401[0]);
            fprintf(stderr, "].\n");
            time_start_51398 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49198, 1, NULL,
                                              global_work_sizze_51397,
                                              local_work_sizze_51401, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51399 = get_wall_time();
            
            long time_diff_51400 = time_end_51399 - time_start_51398;
            
            ctx->map_kernel_49198_total_runtime += time_diff_51400;
            ctx->map_kernel_49198_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49198",
                    time_diff_51400);
        }
    }
    
    int32_t group_sizze_49221;
    
    group_sizze_49221 = ctx->sizes.group_sizze_49220;
    
    int32_t y_49222 = group_sizze_49221 - 1;
    int32_t x_49223 = sizze_47303 + y_49222;
    int32_t num_groups_49224 = squot32(x_49223, group_sizze_49221);
    int32_t num_threads_49225 = group_sizze_49221 * num_groups_49224;
    int32_t convop_x_50620 = sizze_47303 * sizze_47304;
    int64_t binop_x_50621 = sext_i32_i64(convop_x_50620);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50619, "mem_50622");
    
    int call_ret_51402 = futrts_map_transpose_opencl_f32(ctx, mem_50622, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47304,
                                                         sizze_47303,
                                                         sizze_47303 *
                                                         sizze_47304,
                                                         sizze_47303 *
                                                         sizze_47304);
    
    assert(call_ret_51402 == 0);
    
    int32_t convop_x_50624 = sizze_47305 * sizze_47306;
    int64_t binop_x_50625 = sext_i32_i64(convop_x_50624);
    int64_t bytes_50623 = 4 * binop_x_50625;
    struct memblock_device mem_50626;
    
    mem_50626.references = NULL;
    memblock_alloc_device(ctx, &mem_50626, bytes_50623, "mem_50626");
    
    int call_ret_51403 = futrts_map_transpose_opencl_f32(ctx, mem_50626, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47306,
                                                         sizze_47305,
                                                         sizze_47305 *
                                                         sizze_47306,
                                                         sizze_47305 *
                                                         sizze_47306);
    
    assert(call_ret_51403 == 0);
    
    int64_t binop_x_50628 = sext_i32_i64(sizze_47303);
    int64_t bytes_50627 = 4 * binop_x_50628;
    struct memblock_device mem_50629;
    
    mem_50629.references = NULL;
    memblock_alloc_device(ctx, &mem_50629, bytes_50627, "mem_50629");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 0, sizeof(sizze_47303),
                                  &sizze_47303));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 1, sizeof(sizze_47304),
                                  &sizze_47304));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 2, sizeof(sizze_47305),
                                  &sizze_47305));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 3, sizeof(a_47309),
                                  &a_47309));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 4, sizeof(b_47311),
                                  &b_47311));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 5,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 6,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 7,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49226, 8,
                                  sizeof(mem_50629.mem), &mem_50629.mem));
    if (1 * (num_groups_49224 * group_sizze_49221) != 0) {
        const size_t global_work_sizze_51404[1] = {num_groups_49224 *
                     group_sizze_49221};
        const size_t local_work_sizze_51408[1] = {group_sizze_49221};
        int64_t time_start_51405 = 0, time_end_51406 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49226");
            fprintf(stderr, "%zu", global_work_sizze_51404[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51408[0]);
            fprintf(stderr, "].\n");
            time_start_51405 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49226, 1, NULL,
                                              global_work_sizze_51404,
                                              local_work_sizze_51408, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51406 = get_wall_time();
            
            long time_diff_51407 = time_end_51406 - time_start_51405;
            
            ctx->map_kernel_49226_total_runtime += time_diff_51407;
            ctx->map_kernel_49226_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49226",
                    time_diff_51407);
        }
    }
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    out_arrsizze_50976 = sizze_47303;
    out_memsizze_50975 = bytes_50627;
    memblock_set_device(ctx, &out_mem_50974, &mem_50629, "mem_50629");
    *out_out_memsizze_51394 = out_memsizze_50975;
    (*out_mem_p_51395).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51395, &out_mem_50974,
                        "out_mem_50974");
    *out_out_arrsizze_51396 = out_arrsizze_50976;
    memblock_unref_device(ctx, &mem_50629, "mem_50629");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &out_mem_50974, "out_mem_50974");
    return 0;
}
static int futrts_t6_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51409,
                        struct memblock_device *out_mem_p_51410,
                        int32_t *out_out_arrsizze_51411,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47362,
                        int32_t sizze_47363, int32_t sizze_47364,
                        int32_t sizze_47365)
{
    int64_t out_memsizze_50985;
    struct memblock_device out_mem_50984;
    
    out_mem_50984.references = NULL;
    
    int32_t out_arrsizze_50986;
    bool dim_zzero_47370 = 0 == sizze_47364;
    bool dim_zzero_47371 = 0 == sizze_47362;
    bool both_empty_47372 = dim_zzero_47370 && dim_zzero_47371;
    bool dim_match_47373 = sizze_47362 == sizze_47364;
    bool empty_or_match_47374 = both_empty_47372 || dim_match_47373;
    bool empty_or_match_cert_47375;
    
    if (!empty_or_match_47374) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:65:1-66:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50984, "out_mem_50984");
        return 1;
    }
    
    bool dim_zzero_47377 = 0 == sizze_47365;
    bool dim_zzero_47378 = 0 == sizze_47363;
    bool both_empty_47379 = dim_zzero_47377 && dim_zzero_47378;
    bool dim_match_47380 = sizze_47363 == sizze_47365;
    bool empty_or_match_47381 = both_empty_47379 || dim_match_47380;
    bool empty_or_match_cert_47382;
    
    if (!empty_or_match_47381) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:65:1-66:64",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50984, "out_mem_50984");
        return 1;
    }
    
    int32_t group_sizze_49268;
    
    group_sizze_49268 = ctx->sizes.group_sizze_49267;
    
    int32_t y_49269 = group_sizze_49268 - 1;
    int32_t x_49270 = sizze_47362 + y_49269;
    int32_t num_groups_49271 = squot32(x_49270, group_sizze_49268);
    int32_t num_threads_49272 = group_sizze_49268 * num_groups_49271;
    int64_t binop_x_50623 = sext_i32_i64(sizze_47362);
    int64_t bytes_50622 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50622, "mem_50624");
    
    int64_t binop_x_50617 = sext_i32_i64(group_sizze_49268);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_local mem_50618;
    
    mem_50618.references = NULL;
    
    struct memblock_local mem_50621;
    
    mem_50621.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 0, bytes_50616, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 1, bytes_50616, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 2, sizeof(sizze_47362),
                                  &sizze_47362));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 3, sizeof(sizze_47363),
                                  &sizze_47363));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 7,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49273, 8,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (num_groups_49271 * group_sizze_49268) != 0) {
        const size_t global_work_sizze_51412[1] = {num_groups_49271 *
                     group_sizze_49268};
        const size_t local_work_sizze_51416[1] = {group_sizze_49268};
        int64_t time_start_51413 = 0, time_end_51414 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49273");
            fprintf(stderr, "%zu", global_work_sizze_51412[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51416[0]);
            fprintf(stderr, "].\n");
            time_start_51413 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49273, 1, NULL,
                                              global_work_sizze_51412,
                                              local_work_sizze_51416, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51414 = get_wall_time();
            
            long time_diff_51415 = time_end_51414 - time_start_51413;
            
            ctx->map_kernel_49273_total_runtime += time_diff_51415;
            ctx->map_kernel_49273_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49273",
                    time_diff_51415);
        }
    }
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_local(ctx, &mem_50621, "mem_50621");
    out_arrsizze_50986 = sizze_47362;
    out_memsizze_50985 = bytes_50622;
    memblock_set_device(ctx, &out_mem_50984, &mem_50624, "mem_50624");
    *out_out_memsizze_51409 = out_memsizze_50985;
    (*out_mem_p_51410).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51410, &out_mem_50984,
                        "out_mem_50984");
    *out_out_arrsizze_51411 = out_arrsizze_50986;
    memblock_unref_local(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &out_mem_50984, "out_mem_50984");
    return 0;
}
static int futrts_t7_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51417,
                        struct memblock_device *out_mem_p_51418,
                        int32_t *out_out_arrsizze_51419,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t C_mem_sizze_50610,
                        struct memblock_device C_mem_50611,
                        int64_t D_mem_sizze_50612,
                        struct memblock_device D_mem_50613, int32_t sizze_47396,
                        int32_t sizze_47397, int32_t sizze_47398,
                        int32_t sizze_47399, int32_t sizze_47400)
{
    int64_t out_memsizze_50996;
    struct memblock_device out_mem_50995;
    
    out_mem_50995.references = NULL;
    
    int32_t out_arrsizze_50997;
    bool dim_zzero_47404 = 0 == sizze_47398;
    bool dim_zzero_47405 = 0 == sizze_47399;
    bool old_empty_47406 = dim_zzero_47404 || dim_zzero_47405;
    bool dim_zzero_47407 = 0 == sizze_47397;
    bool new_empty_47408 = dim_zzero_47405 || dim_zzero_47407;
    bool both_empty_47409 = old_empty_47406 && new_empty_47408;
    bool dim_match_47410 = sizze_47397 == sizze_47398;
    bool empty_or_match_47411 = both_empty_47409 || dim_match_47410;
    bool empty_or_match_cert_47412;
    
    if (!empty_or_match_47411) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:68:1-71:9",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50995, "out_mem_50995");
        return 1;
    }
    
    bool dim_zzero_47414 = 0 == sizze_47400;
    bool both_empty_47415 = dim_zzero_47405 && dim_zzero_47414;
    bool dim_match_47416 = sizze_47399 == sizze_47400;
    bool empty_or_match_47417 = both_empty_47415 || dim_match_47416;
    bool empty_or_match_cert_47418;
    
    if (!empty_or_match_47417) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:68:1-71:9",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_50995, "out_mem_50995");
        return 1;
    }
    
    int32_t group_sizze_49311;
    
    group_sizze_49311 = ctx->sizes.group_sizze_49310;
    
    int32_t y_49312 = group_sizze_49311 - 1;
    int32_t x_49313 = sizze_47397 + y_49312;
    int32_t num_groups_49314 = squot32(x_49313, group_sizze_49311);
    int32_t num_threads_49315 = group_sizze_49311 * num_groups_49314;
    int32_t convop_x_50615 = sizze_47398 * sizze_47399;
    int64_t binop_x_50616 = sext_i32_i64(convop_x_50615);
    int64_t bytes_50614 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50614, "mem_50617");
    
    int call_ret_51420 = futrts_map_transpose_opencl_f32(ctx, mem_50617, 0,
                                                         C_mem_50611, 0, 1,
                                                         sizze_47399,
                                                         sizze_47398,
                                                         sizze_47398 *
                                                         sizze_47399,
                                                         sizze_47398 *
                                                         sizze_47399);
    
    assert(call_ret_51420 == 0);
    
    int64_t binop_x_50619 = sext_i32_i64(sizze_47397);
    int64_t bytes_50618 = 4 * binop_x_50619;
    struct memblock_device mem_50620;
    
    mem_50620.references = NULL;
    memblock_alloc_device(ctx, &mem_50620, bytes_50618, "mem_50620");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 0, sizeof(sizze_47397),
                                  &sizze_47397));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 1, sizeof(sizze_47398),
                                  &sizze_47398));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 2, sizeof(sizze_47399),
                                  &sizze_47399));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 3,
                                  sizeof(D_mem_50613.mem), &D_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 4,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49316, 5,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    if (1 * (num_groups_49314 * group_sizze_49311) != 0) {
        const size_t global_work_sizze_51421[1] = {num_groups_49314 *
                     group_sizze_49311};
        const size_t local_work_sizze_51425[1] = {group_sizze_49311};
        int64_t time_start_51422 = 0, time_end_51423 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49316");
            fprintf(stderr, "%zu", global_work_sizze_51421[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51425[0]);
            fprintf(stderr, "].\n");
            time_start_51422 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49316, 1, NULL,
                                              global_work_sizze_51421,
                                              local_work_sizze_51425, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51423 = get_wall_time();
            
            long time_diff_51424 = time_end_51423 - time_start_51422;
            
            ctx->map_kernel_49316_total_runtime += time_diff_51424;
            ctx->map_kernel_49316_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49316",
                    time_diff_51424);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    
    int32_t group_sizze_49351;
    
    group_sizze_49351 = ctx->sizes.group_sizze_49350;
    
    int32_t y_49352 = group_sizze_49351 - 1;
    int32_t x_49353 = sizze_47396 + y_49352;
    int32_t num_groups_49354 = squot32(x_49353, group_sizze_49351);
    int32_t num_threads_49355 = group_sizze_49351 * num_groups_49354;
    int32_t convop_x_50622 = sizze_47396 * sizze_47397;
    int64_t binop_x_50623 = sext_i32_i64(convop_x_50622);
    int64_t bytes_50621 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50621, "mem_50624");
    
    int call_ret_51426 = futrts_map_transpose_opencl_f32(ctx, mem_50624, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47397,
                                                         sizze_47396,
                                                         sizze_47396 *
                                                         sizze_47397,
                                                         sizze_47396 *
                                                         sizze_47397);
    
    assert(call_ret_51426 == 0);
    
    int64_t binop_x_50626 = sext_i32_i64(sizze_47396);
    int64_t bytes_50625 = 4 * binop_x_50626;
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, bytes_50625, "mem_50627");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49356, 0, sizeof(sizze_47396),
                                  &sizze_47396));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49356, 1, sizeof(sizze_47397),
                                  &sizze_47397));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49356, 2,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49356, 3,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49356, 4,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    if (1 * (num_groups_49354 * group_sizze_49351) != 0) {
        const size_t global_work_sizze_51427[1] = {num_groups_49354 *
                     group_sizze_49351};
        const size_t local_work_sizze_51431[1] = {group_sizze_49351};
        int64_t time_start_51428 = 0, time_end_51429 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49356");
            fprintf(stderr, "%zu", global_work_sizze_51427[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51431[0]);
            fprintf(stderr, "].\n");
            time_start_51428 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49356, 1, NULL,
                                              global_work_sizze_51427,
                                              local_work_sizze_51431, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51429 = get_wall_time();
            
            long time_diff_51430 = time_end_51429 - time_start_51428;
            
            ctx->map_kernel_49356_total_runtime += time_diff_51430;
            ctx->map_kernel_49356_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49356",
                    time_diff_51430);
        }
    }
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    out_arrsizze_50997 = sizze_47396;
    out_memsizze_50996 = bytes_50625;
    memblock_set_device(ctx, &out_mem_50995, &mem_50627, "mem_50627");
    *out_out_memsizze_51417 = out_memsizze_50996;
    (*out_mem_p_51418).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51418, &out_mem_50995,
                        "out_mem_50995");
    *out_out_arrsizze_51419 = out_arrsizze_50997;
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &out_mem_50995, "out_mem_50995");
    return 0;
}
static int futrts_t9_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51432,
                        struct memblock_device *out_mem_p_51433,
                        int32_t *out_out_arrsizze_51434,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47438,
                        int32_t sizze_47439, int32_t sizze_47440,
                        int32_t sizze_47441, int32_t sizze_47442,
                        int32_t sizze_47443)
{
    int64_t out_memsizze_51007;
    struct memblock_device out_mem_51006;
    
    out_mem_51006.references = NULL;
    
    int32_t out_arrsizze_51008;
    bool dim_zzero_47448 = 0 == sizze_47440;
    bool dim_zzero_47449 = 0 == sizze_47441;
    bool old_empty_47450 = dim_zzero_47448 || dim_zzero_47449;
    bool dim_zzero_47451 = 0 == sizze_47439;
    bool new_empty_47452 = dim_zzero_47449 || dim_zzero_47451;
    bool both_empty_47453 = old_empty_47450 && new_empty_47452;
    bool dim_match_47454 = sizze_47439 == sizze_47440;
    bool empty_or_match_47455 = both_empty_47453 || dim_match_47454;
    bool empty_or_match_cert_47456;
    
    if (!empty_or_match_47455) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:78:1-80:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51006, "out_mem_51006");
        return 1;
    }
    
    bool dim_zzero_47458 = 0 == sizze_47442;
    bool both_empty_47459 = dim_zzero_47449 && dim_zzero_47458;
    bool dim_match_47460 = sizze_47441 == sizze_47442;
    bool empty_or_match_47461 = both_empty_47459 || dim_match_47460;
    bool empty_or_match_cert_47462;
    
    if (!empty_or_match_47461) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:78:1-80:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51006, "out_mem_51006");
        return 1;
    }
    
    bool dim_zzero_47464 = 0 == sizze_47443;
    bool both_empty_47465 = dim_zzero_47449 && dim_zzero_47464;
    bool dim_match_47466 = sizze_47441 == sizze_47443;
    bool empty_or_match_47467 = both_empty_47465 || dim_match_47466;
    bool empty_or_match_cert_47468;
    
    if (!empty_or_match_47467) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:78:1-80:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51006, "out_mem_51006");
        return 1;
    }
    
    int32_t group_sizze_49380;
    
    group_sizze_49380 = ctx->sizes.group_sizze_49379;
    
    int32_t y_49381 = group_sizze_49380 - 1;
    int32_t x_49382 = sizze_47441 + y_49381;
    int32_t num_groups_49383 = squot32(x_49382, group_sizze_49380);
    int32_t num_threads_49384 = group_sizze_49380 * num_groups_49383;
    int64_t binop_x_50617 = sext_i32_i64(sizze_47441);
    int64_t bytes_50616 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50616, "mem_50618");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49385, 0, sizeof(sizze_47441),
                                  &sizze_47441));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49385, 1,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49385, 2,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49385, 3,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    if (1 * (num_groups_49383 * group_sizze_49380) != 0) {
        const size_t global_work_sizze_51435[1] = {num_groups_49383 *
                     group_sizze_49380};
        const size_t local_work_sizze_51439[1] = {group_sizze_49380};
        int64_t time_start_51436 = 0, time_end_51437 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49385");
            fprintf(stderr, "%zu", global_work_sizze_51435[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51439[0]);
            fprintf(stderr, "].\n");
            time_start_51436 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49385, 1, NULL,
                                              global_work_sizze_51435,
                                              local_work_sizze_51439, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51437 = get_wall_time();
            
            long time_diff_51438 = time_end_51437 - time_start_51436;
            
            ctx->map_kernel_49385_total_runtime += time_diff_51438;
            ctx->map_kernel_49385_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49385",
                    time_diff_51438);
        }
    }
    
    int32_t group_sizze_49415;
    
    group_sizze_49415 = ctx->sizes.group_sizze_49414;
    
    int32_t y_49416 = group_sizze_49415 - 1;
    int32_t x_49417 = sizze_47438 + y_49416;
    int32_t num_groups_49418 = squot32(x_49417, group_sizze_49415);
    int32_t num_threads_49419 = group_sizze_49415 * num_groups_49418;
    int32_t convop_x_50620 = sizze_47438 * sizze_47439;
    int64_t binop_x_50621 = sext_i32_i64(convop_x_50620);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50619, "mem_50622");
    
    int call_ret_51440 = futrts_map_transpose_opencl_f32(ctx, mem_50622, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47439,
                                                         sizze_47438,
                                                         sizze_47438 *
                                                         sizze_47439,
                                                         sizze_47438 *
                                                         sizze_47439);
    
    assert(call_ret_51440 == 0);
    
    int64_t binop_x_50624 = sext_i32_i64(sizze_47438);
    int64_t bytes_50623 = 4 * binop_x_50624;
    struct memblock_device mem_50625;
    
    mem_50625.references = NULL;
    memblock_alloc_device(ctx, &mem_50625, bytes_50623, "mem_50625");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 0, sizeof(sizze_47438),
                                  &sizze_47438));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 1, sizeof(sizze_47439),
                                  &sizze_47439));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 2, sizeof(sizze_47441),
                                  &sizze_47441));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 3,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 4,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 5,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49420, 6,
                                  sizeof(mem_50625.mem), &mem_50625.mem));
    if (1 * (num_groups_49418 * group_sizze_49415) != 0) {
        const size_t global_work_sizze_51441[1] = {num_groups_49418 *
                     group_sizze_49415};
        const size_t local_work_sizze_51445[1] = {group_sizze_49415};
        int64_t time_start_51442 = 0, time_end_51443 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49420");
            fprintf(stderr, "%zu", global_work_sizze_51441[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51445[0]);
            fprintf(stderr, "].\n");
            time_start_51442 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49420, 1, NULL,
                                              global_work_sizze_51441,
                                              local_work_sizze_51445, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51443 = get_wall_time();
            
            long time_diff_51444 = time_end_51443 - time_start_51442;
            
            ctx->map_kernel_49420_total_runtime += time_diff_51444;
            ctx->map_kernel_49420_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49420",
                    time_diff_51444);
        }
    }
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    out_arrsizze_51008 = sizze_47438;
    out_memsizze_51007 = bytes_50623;
    memblock_set_device(ctx, &out_mem_51006, &mem_50625, "mem_50625");
    *out_out_memsizze_51432 = out_memsizze_51007;
    (*out_mem_p_51433).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51433, &out_mem_51006,
                        "out_mem_51006");
    *out_out_arrsizze_51434 = out_arrsizze_51008;
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &out_mem_51006, "out_mem_51006");
    return 0;
}
static int futrts_t10_2d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51446,
                         struct memblock_device *out_mem_p_51447,
                         int32_t *out_out_arrsizze_51448,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int64_t D_mem_sizze_50614,
                         struct memblock_device D_mem_50615,
                         int32_t sizze_47491, int32_t sizze_47492,
                         int32_t sizze_47493, int32_t sizze_47494,
                         int32_t sizze_47495, int32_t sizze_47496,
                         int32_t sizze_47497)
{
    int64_t out_memsizze_51018;
    struct memblock_device out_mem_51017;
    
    out_mem_51017.references = NULL;
    
    int32_t out_arrsizze_51019;
    bool dim_zzero_47502 = 0 == sizze_47493;
    bool dim_zzero_47503 = 0 == sizze_47494;
    bool old_empty_47504 = dim_zzero_47502 || dim_zzero_47503;
    bool dim_zzero_47505 = 0 == sizze_47492;
    bool new_empty_47506 = dim_zzero_47503 || dim_zzero_47505;
    bool both_empty_47507 = old_empty_47504 && new_empty_47506;
    bool dim_match_47508 = sizze_47492 == sizze_47493;
    bool empty_or_match_47509 = both_empty_47507 || dim_match_47508;
    bool empty_or_match_cert_47510;
    
    if (!empty_or_match_47509) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:82:1-85:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51017, "out_mem_51017");
        return 1;
    }
    
    bool dim_zzero_47512 = 0 == sizze_47495;
    bool dim_zzero_47513 = 0 == sizze_47496;
    bool old_empty_47514 = dim_zzero_47512 || dim_zzero_47513;
    bool both_empty_47515 = new_empty_47506 && old_empty_47514;
    bool dim_match_47516 = sizze_47494 == sizze_47495;
    bool dim_match_47517 = sizze_47492 == sizze_47496;
    bool match_47518 = dim_match_47516 && dim_match_47517;
    bool empty_or_match_47519 = both_empty_47515 || match_47518;
    bool empty_or_match_cert_47520;
    
    if (!empty_or_match_47519) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:82:1-85:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51017, "out_mem_51017");
        return 1;
    }
    
    bool dim_zzero_47522 = 0 == sizze_47497;
    bool both_empty_47523 = dim_zzero_47505 && dim_zzero_47522;
    bool dim_match_47524 = sizze_47492 == sizze_47497;
    bool empty_or_match_47525 = both_empty_47523 || dim_match_47524;
    bool empty_or_match_cert_47526;
    
    if (!empty_or_match_47525) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:82:1-85:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51017, "out_mem_51017");
        return 1;
    }
    
    int32_t group_sizze_49473;
    
    group_sizze_49473 = ctx->sizes.group_sizze_49472;
    
    int32_t y_49474 = group_sizze_49473 - 1;
    int32_t x_49475 = sizze_47494 + y_49474;
    int32_t num_groups_49476 = squot32(x_49475, group_sizze_49473);
    int32_t num_threads_49477 = group_sizze_49473 * num_groups_49476;
    int32_t convop_x_50617 = sizze_47495 * sizze_47496;
    int64_t binop_x_50618 = sext_i32_i64(convop_x_50617);
    int64_t bytes_50616 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50616, "mem_50619");
    
    int call_ret_51449 = futrts_map_transpose_opencl_f32(ctx, mem_50619, 0,
                                                         C_mem_50613, 0, 1,
                                                         sizze_47496,
                                                         sizze_47495,
                                                         sizze_47495 *
                                                         sizze_47496,
                                                         sizze_47495 *
                                                         sizze_47496);
    
    assert(call_ret_51449 == 0);
    
    int64_t binop_x_50621 = sext_i32_i64(sizze_47494);
    int64_t bytes_50620 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50620, "mem_50622");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 0, sizeof(sizze_47492),
                                  &sizze_47492));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 1, sizeof(sizze_47494),
                                  &sizze_47494));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 2, sizeof(sizze_47495),
                                  &sizze_47495));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 3,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 4,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49478, 5,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    if (1 * (num_groups_49476 * group_sizze_49473) != 0) {
        const size_t global_work_sizze_51450[1] = {num_groups_49476 *
                     group_sizze_49473};
        const size_t local_work_sizze_51454[1] = {group_sizze_49473};
        int64_t time_start_51451 = 0, time_end_51452 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49478");
            fprintf(stderr, "%zu", global_work_sizze_51450[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51454[0]);
            fprintf(stderr, "].\n");
            time_start_51451 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49478, 1, NULL,
                                              global_work_sizze_51450,
                                              local_work_sizze_51454, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51452 = get_wall_time();
            
            long time_diff_51453 = time_end_51452 - time_start_51451;
            
            ctx->map_kernel_49478_total_runtime += time_diff_51453;
            ctx->map_kernel_49478_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49478",
                    time_diff_51453);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    
    int32_t group_sizze_49524;
    
    group_sizze_49524 = ctx->sizes.group_sizze_49523;
    
    int32_t y_49525 = group_sizze_49524 - 1;
    int32_t x_49526 = sizze_47491 + y_49525;
    int32_t num_groups_49527 = squot32(x_49526, group_sizze_49524);
    int32_t num_threads_49528 = group_sizze_49524 * num_groups_49527;
    int32_t convop_x_50624 = sizze_47491 * sizze_47492;
    int64_t binop_x_50625 = sext_i32_i64(convop_x_50624);
    int64_t bytes_50623 = 4 * binop_x_50625;
    struct memblock_device mem_50626;
    
    mem_50626.references = NULL;
    memblock_alloc_device(ctx, &mem_50626, bytes_50623, "mem_50626");
    
    int call_ret_51455 = futrts_map_transpose_opencl_f32(ctx, mem_50626, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47492,
                                                         sizze_47491,
                                                         sizze_47491 *
                                                         sizze_47492,
                                                         sizze_47491 *
                                                         sizze_47492);
    
    assert(call_ret_51455 == 0);
    
    int64_t binop_x_50628 = sext_i32_i64(sizze_47491);
    int64_t bytes_50627 = 4 * binop_x_50628;
    struct memblock_device mem_50629;
    
    mem_50629.references = NULL;
    memblock_alloc_device(ctx, &mem_50629, bytes_50627, "mem_50629");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 0, sizeof(sizze_47491),
                                  &sizze_47491));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 1, sizeof(sizze_47492),
                                  &sizze_47492));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 2, sizeof(sizze_47494),
                                  &sizze_47494));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 3,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 4,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 5,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49529, 6,
                                  sizeof(mem_50629.mem), &mem_50629.mem));
    if (1 * (num_groups_49527 * group_sizze_49524) != 0) {
        const size_t global_work_sizze_51456[1] = {num_groups_49527 *
                     group_sizze_49524};
        const size_t local_work_sizze_51460[1] = {group_sizze_49524};
        int64_t time_start_51457 = 0, time_end_51458 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49529");
            fprintf(stderr, "%zu", global_work_sizze_51456[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51460[0]);
            fprintf(stderr, "].\n");
            time_start_51457 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49529, 1, NULL,
                                              global_work_sizze_51456,
                                              local_work_sizze_51460, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51458 = get_wall_time();
            
            long time_diff_51459 = time_end_51458 - time_start_51457;
            
            ctx->map_kernel_49529_total_runtime += time_diff_51459;
            ctx->map_kernel_49529_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49529",
                    time_diff_51459);
        }
    }
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    out_arrsizze_51019 = sizze_47491;
    out_memsizze_51018 = bytes_50627;
    memblock_set_device(ctx, &out_mem_51017, &mem_50629, "mem_50629");
    *out_out_memsizze_51446 = out_memsizze_51018;
    (*out_mem_p_51447).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51447, &out_mem_51017,
                        "out_mem_51017");
    *out_out_arrsizze_51448 = out_arrsizze_51019;
    memblock_unref_device(ctx, &mem_50629, "mem_50629");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &out_mem_51017, "out_mem_51017");
    return 0;
}
static int futrts_t9_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51461,
                        struct memblock_device *out_mem_p_51462,
                        int32_t *out_out_arrsizze_51463,
                        int32_t *out_out_arrsizze_51464,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47554,
                        int32_t sizze_47555, int32_t sizze_47556,
                        int32_t sizze_47557, int32_t sizze_47558,
                        int32_t sizze_47559)
{
    int64_t out_memsizze_51030;
    struct memblock_device out_mem_51029;
    
    out_mem_51029.references = NULL;
    
    int32_t out_arrsizze_51031;
    int32_t out_arrsizze_51032;
    bool dim_zzero_47563 = 0 == sizze_47557;
    bool dim_zzero_47564 = 0 == sizze_47558;
    bool old_empty_47565 = dim_zzero_47563 || dim_zzero_47564;
    bool dim_zzero_47566 = 0 == sizze_47556;
    bool new_empty_47567 = dim_zzero_47564 || dim_zzero_47566;
    bool both_empty_47568 = old_empty_47565 && new_empty_47567;
    bool dim_match_47569 = sizze_47556 == sizze_47557;
    bool empty_or_match_47570 = both_empty_47568 || dim_match_47569;
    bool empty_or_match_cert_47571;
    
    if (!empty_or_match_47570) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:137:1-139:65",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51029, "out_mem_51029");
        return 1;
    }
    
    bool dim_zzero_47573 = 0 == sizze_47559;
    bool both_empty_47574 = dim_zzero_47564 && dim_zzero_47573;
    bool dim_match_47575 = sizze_47558 == sizze_47559;
    bool empty_or_match_47576 = both_empty_47574 || dim_match_47575;
    bool empty_or_match_cert_47577;
    
    if (!empty_or_match_47576) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:137:1-139:65",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51029, "out_mem_51029");
        return 1;
    }
    
    int32_t group_sizze_49582;
    
    group_sizze_49582 = ctx->sizes.group_sizze_49581;
    
    int32_t y_49583 = group_sizze_49582 - 1;
    int32_t x_49584 = sizze_47556 + y_49583;
    int32_t num_groups_49585 = squot32(x_49584, group_sizze_49582);
    int32_t num_threads_49586 = group_sizze_49582 * num_groups_49585;
    int32_t convop_x_50615 = sizze_47557 * sizze_47558;
    int64_t binop_x_50616 = sext_i32_i64(convop_x_50615);
    int64_t bytes_50614 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50614, "mem_50617");
    
    int call_ret_51465 = futrts_map_transpose_opencl_f32(ctx, mem_50617, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47558,
                                                         sizze_47557,
                                                         sizze_47557 *
                                                         sizze_47558,
                                                         sizze_47557 *
                                                         sizze_47558);
    
    assert(call_ret_51465 == 0);
    
    int64_t binop_x_50619 = sext_i32_i64(sizze_47556);
    int64_t bytes_50618 = 4 * binop_x_50619;
    struct memblock_device mem_50620;
    
    mem_50620.references = NULL;
    memblock_alloc_device(ctx, &mem_50620, bytes_50618, "mem_50620");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 0, sizeof(sizze_47556),
                                  &sizze_47556));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 1, sizeof(sizze_47557),
                                  &sizze_47557));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 2, sizeof(sizze_47558),
                                  &sizze_47558));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 4,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49587, 5,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    if (1 * (num_groups_49585 * group_sizze_49582) != 0) {
        const size_t global_work_sizze_51466[1] = {num_groups_49585 *
                     group_sizze_49582};
        const size_t local_work_sizze_51470[1] = {group_sizze_49582};
        int64_t time_start_51467 = 0, time_end_51468 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49587");
            fprintf(stderr, "%zu", global_work_sizze_51466[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51470[0]);
            fprintf(stderr, "].\n");
            time_start_51467 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49587, 1, NULL,
                                              global_work_sizze_51466,
                                              local_work_sizze_51470, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51468 = get_wall_time();
            
            long time_diff_51469 = time_end_51468 - time_start_51467;
            
            ctx->map_kernel_49587_total_runtime += time_diff_51469;
            ctx->map_kernel_49587_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49587",
                    time_diff_51469);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    
    int32_t nesting_sizze_49622 = sizze_47554 * sizze_47555;
    int32_t group_sizze_49624;
    
    group_sizze_49624 = ctx->sizes.group_sizze_49623;
    
    int32_t y_49625 = group_sizze_49624 - 1;
    int32_t x_49626 = nesting_sizze_49622 + y_49625;
    int32_t num_groups_49627 = squot32(x_49626, group_sizze_49624);
    int32_t num_threads_49628 = group_sizze_49624 * num_groups_49627;
    int32_t binop_x_50622 = sizze_47554 * sizze_47556;
    int32_t convop_x_50623 = sizze_47555 * binop_x_50622;
    int64_t binop_x_50624 = sext_i32_i64(convop_x_50623);
    int64_t bytes_50621 = 4 * binop_x_50624;
    struct memblock_device mem_50625;
    
    mem_50625.references = NULL;
    memblock_alloc_device(ctx, &mem_50625, bytes_50621, "mem_50625");
    
    int call_ret_51471 = futrts_map_transpose_opencl_f32(ctx, mem_50625, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47556,
                                                         sizze_47554 *
                                                         sizze_47555,
                                                         sizze_47554 *
                                                         sizze_47555 *
                                                         sizze_47556,
                                                         sizze_47554 *
                                                         sizze_47555 *
                                                         sizze_47556);
    
    assert(call_ret_51471 == 0);
    
    int64_t binop_x_50628 = sext_i32_i64(nesting_sizze_49622);
    int64_t bytes_50626 = 4 * binop_x_50628;
    struct memblock_device mem_50629;
    
    mem_50629.references = NULL;
    memblock_alloc_device(ctx, &mem_50629, bytes_50626, "mem_50629");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 0, sizeof(sizze_47554),
                                  &sizze_47554));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 1, sizeof(sizze_47555),
                                  &sizze_47555));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 2, sizeof(sizze_47556),
                                  &sizze_47556));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 3,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 4,
                                  sizeof(mem_50625.mem), &mem_50625.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49629, 5,
                                  sizeof(mem_50629.mem), &mem_50629.mem));
    if (1 * (num_groups_49627 * group_sizze_49624) != 0) {
        const size_t global_work_sizze_51472[1] = {num_groups_49627 *
                     group_sizze_49624};
        const size_t local_work_sizze_51476[1] = {group_sizze_49624};
        int64_t time_start_51473 = 0, time_end_51474 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49629");
            fprintf(stderr, "%zu", global_work_sizze_51472[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51476[0]);
            fprintf(stderr, "].\n");
            time_start_51473 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49629, 1, NULL,
                                              global_work_sizze_51472,
                                              local_work_sizze_51476, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51474 = get_wall_time();
            
            long time_diff_51475 = time_end_51474 - time_start_51473;
            
            ctx->map_kernel_49629_total_runtime += time_diff_51475;
            ctx->map_kernel_49629_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49629",
                    time_diff_51475);
        }
    }
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    out_arrsizze_51031 = sizze_47554;
    out_arrsizze_51032 = sizze_47555;
    out_memsizze_51030 = bytes_50626;
    memblock_set_device(ctx, &out_mem_51029, &mem_50629, "mem_50629");
    *out_out_memsizze_51461 = out_memsizze_51030;
    (*out_mem_p_51462).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51462, &out_mem_51029,
                        "out_mem_51029");
    *out_out_arrsizze_51463 = out_arrsizze_51031;
    *out_out_arrsizze_51464 = out_arrsizze_51032;
    memblock_unref_device(ctx, &mem_50629, "mem_50629");
    memblock_unref_device(ctx, &mem_50625, "mem_50625");
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &out_mem_51029, "out_mem_51029");
    return 0;
}
static int futrts_t8_2d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51477,
                        struct memblock_device *out_mem_p_51478,
                        int32_t *out_out_arrsizze_51479,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47599,
                        int32_t sizze_47600, int32_t sizze_47601,
                        int32_t sizze_47602, int32_t sizze_47603,
                        int32_t sizze_47604, int32_t sizze_47605)
{
    int64_t out_memsizze_51042;
    struct memblock_device out_mem_51041;
    
    out_mem_51041.references = NULL;
    
    int32_t out_arrsizze_51043;
    bool dim_zzero_47610 = 0 == sizze_47601;
    bool dim_zzero_47611 = 0 == sizze_47602;
    bool old_empty_47612 = dim_zzero_47610 || dim_zzero_47611;
    bool dim_zzero_47613 = 0 == sizze_47599;
    bool dim_zzero_47614 = 0 == sizze_47600;
    bool new_empty_47615 = dim_zzero_47613 || dim_zzero_47614;
    bool both_empty_47616 = old_empty_47612 && new_empty_47615;
    bool dim_match_47617 = sizze_47599 == sizze_47601;
    bool dim_match_47618 = sizze_47600 == sizze_47602;
    bool match_47619 = dim_match_47617 && dim_match_47618;
    bool empty_or_match_47620 = both_empty_47616 || match_47619;
    bool empty_or_match_cert_47621;
    
    if (!empty_or_match_47620) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:73:1-76:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51041, "out_mem_51041");
        return 1;
    }
    
    bool dim_zzero_47623 = 0 == sizze_47603;
    bool dim_zzero_47624 = 0 == sizze_47604;
    bool old_empty_47625 = dim_zzero_47623 || dim_zzero_47624;
    bool new_empty_47626 = dim_zzero_47614 || dim_zzero_47624;
    bool both_empty_47627 = old_empty_47625 && new_empty_47626;
    bool dim_match_47628 = sizze_47600 == sizze_47603;
    bool empty_or_match_47629 = both_empty_47627 || dim_match_47628;
    bool empty_or_match_cert_47630;
    
    if (!empty_or_match_47629) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:73:1-76:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51041, "out_mem_51041");
        return 1;
    }
    
    bool dim_zzero_47632 = 0 == sizze_47605;
    bool both_empty_47633 = dim_zzero_47624 && dim_zzero_47632;
    bool dim_match_47634 = sizze_47604 == sizze_47605;
    bool empty_or_match_47635 = both_empty_47633 || dim_match_47634;
    bool empty_or_match_cert_47636;
    
    if (!empty_or_match_47635) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:73:1-76:11",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51041, "out_mem_51041");
        return 1;
    }
    
    int32_t group_sizze_49664;
    
    group_sizze_49664 = ctx->sizes.group_sizze_49663;
    
    int32_t y_49665 = group_sizze_49664 - 1;
    int32_t x_49666 = sizze_47600 + y_49665;
    int32_t num_groups_49667 = squot32(x_49666, group_sizze_49664);
    int32_t num_threads_49668 = group_sizze_49664 * num_groups_49667;
    int32_t convop_x_50617 = sizze_47603 * sizze_47604;
    int64_t binop_x_50618 = sext_i32_i64(convop_x_50617);
    int64_t bytes_50616 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50616, "mem_50619");
    
    int call_ret_51480 = futrts_map_transpose_opencl_f32(ctx, mem_50619, 0,
                                                         C_mem_50613, 0, 1,
                                                         sizze_47604,
                                                         sizze_47603,
                                                         sizze_47603 *
                                                         sizze_47604,
                                                         sizze_47603 *
                                                         sizze_47604);
    
    assert(call_ret_51480 == 0);
    
    int64_t binop_x_50621 = sext_i32_i64(sizze_47600);
    int64_t bytes_50620 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50620, "mem_50622");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 0, sizeof(sizze_47600),
                                  &sizze_47600));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 1, sizeof(sizze_47603),
                                  &sizze_47603));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 2, sizeof(sizze_47604),
                                  &sizze_47604));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 3,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 4,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49669, 5,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    if (1 * (num_groups_49667 * group_sizze_49664) != 0) {
        const size_t global_work_sizze_51481[1] = {num_groups_49667 *
                     group_sizze_49664};
        const size_t local_work_sizze_51485[1] = {group_sizze_49664};
        int64_t time_start_51482 = 0, time_end_51483 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49669");
            fprintf(stderr, "%zu", global_work_sizze_51481[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51485[0]);
            fprintf(stderr, "].\n");
            time_start_51482 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49669, 1, NULL,
                                              global_work_sizze_51481,
                                              local_work_sizze_51485, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51483 = get_wall_time();
            
            long time_diff_51484 = time_end_51483 - time_start_51482;
            
            ctx->map_kernel_49669_total_runtime += time_diff_51484;
            ctx->map_kernel_49669_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49669",
                    time_diff_51484);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    
    int32_t group_sizze_49706;
    
    group_sizze_49706 = ctx->sizes.group_sizze_49705;
    
    int32_t y_49707 = group_sizze_49706 - 1;
    int32_t x_49708 = sizze_47599 + y_49707;
    int32_t num_groups_49709 = squot32(x_49708, group_sizze_49706);
    int32_t num_threads_49710 = group_sizze_49706 * num_groups_49709;
    int32_t convop_x_50624 = sizze_47599 * sizze_47600;
    int64_t binop_x_50625 = sext_i32_i64(convop_x_50624);
    int64_t bytes_50623 = 4 * binop_x_50625;
    struct memblock_device mem_50626;
    
    mem_50626.references = NULL;
    memblock_alloc_device(ctx, &mem_50626, bytes_50623, "mem_50626");
    
    int call_ret_51486 = futrts_map_transpose_opencl_f32(ctx, mem_50626, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47600,
                                                         sizze_47599,
                                                         sizze_47599 *
                                                         sizze_47600,
                                                         sizze_47599 *
                                                         sizze_47600);
    
    assert(call_ret_51486 == 0);
    
    int32_t convop_x_50628 = sizze_47601 * sizze_47602;
    int64_t binop_x_50629 = sext_i32_i64(convop_x_50628);
    int64_t bytes_50627 = 4 * binop_x_50629;
    struct memblock_device mem_50630;
    
    mem_50630.references = NULL;
    memblock_alloc_device(ctx, &mem_50630, bytes_50627, "mem_50630");
    
    int call_ret_51487 = futrts_map_transpose_opencl_f32(ctx, mem_50630, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47602,
                                                         sizze_47601,
                                                         sizze_47601 *
                                                         sizze_47602,
                                                         sizze_47601 *
                                                         sizze_47602);
    
    assert(call_ret_51487 == 0);
    
    int64_t binop_x_50632 = sext_i32_i64(sizze_47599);
    int64_t bytes_50631 = 4 * binop_x_50632;
    struct memblock_device mem_50633;
    
    mem_50633.references = NULL;
    memblock_alloc_device(ctx, &mem_50633, bytes_50631, "mem_50633");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 0, sizeof(sizze_47599),
                                  &sizze_47599));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 1, sizeof(sizze_47600),
                                  &sizze_47600));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 2, sizeof(sizze_47601),
                                  &sizze_47601));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 3,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 4,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 5,
                                  sizeof(mem_50630.mem), &mem_50630.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49711, 6,
                                  sizeof(mem_50633.mem), &mem_50633.mem));
    if (1 * (num_groups_49709 * group_sizze_49706) != 0) {
        const size_t global_work_sizze_51488[1] = {num_groups_49709 *
                     group_sizze_49706};
        const size_t local_work_sizze_51492[1] = {group_sizze_49706};
        int64_t time_start_51489 = 0, time_end_51490 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49711");
            fprintf(stderr, "%zu", global_work_sizze_51488[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51492[0]);
            fprintf(stderr, "].\n");
            time_start_51489 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49711, 1, NULL,
                                              global_work_sizze_51488,
                                              local_work_sizze_51492, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51490 = get_wall_time();
            
            long time_diff_51491 = time_end_51490 - time_start_51489;
            
            ctx->map_kernel_49711_total_runtime += time_diff_51491;
            ctx->map_kernel_49711_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49711",
                    time_diff_51491);
        }
    }
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    out_arrsizze_51043 = sizze_47599;
    out_memsizze_51042 = bytes_50631;
    memblock_set_device(ctx, &out_mem_51041, &mem_50633, "mem_50633");
    *out_out_memsizze_51477 = out_memsizze_51042;
    (*out_mem_p_51478).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51478, &out_mem_51041,
                        "out_mem_51041");
    *out_out_arrsizze_51479 = out_arrsizze_51043;
    memblock_unref_device(ctx, &mem_50633, "mem_50633");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &out_mem_51041, "out_mem_51041");
    return 0;
}
static int futrts_t1_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51493,
                        struct memblock_device *out_mem_p_51494,
                        int32_t *out_out_arrsizze_51495,
                        int32_t *out_out_arrsizze_51496,
                        int32_t *out_out_arrsizze_51497,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611, int32_t sizze_47659,
                        int32_t sizze_47660, int32_t sizze_47661,
                        int32_t sizze_47662, int32_t sizze_47663)
{
    int64_t out_memsizze_51053;
    struct memblock_device out_mem_51052;
    
    out_mem_51052.references = NULL;
    
    int32_t out_arrsizze_51054;
    int32_t out_arrsizze_51055;
    int32_t out_arrsizze_51056;
    bool dim_zzero_47666 = 0 == sizze_47662;
    bool dim_zzero_47667 = 0 == sizze_47663;
    bool old_empty_47668 = dim_zzero_47666 || dim_zzero_47667;
    bool dim_zzero_47669 = 0 == sizze_47661;
    bool new_empty_47670 = dim_zzero_47667 || dim_zzero_47669;
    bool both_empty_47671 = old_empty_47668 && new_empty_47670;
    bool dim_match_47672 = sizze_47661 == sizze_47662;
    bool empty_or_match_47673 = both_empty_47671 || dim_match_47672;
    bool empty_or_match_cert_47674;
    
    if (!empty_or_match_47673) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:104:1-105:75",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51052, "out_mem_51052");
        return 1;
    }
    
    int32_t tile_sizze_50528;
    
    tile_sizze_50528 = ctx->sizes.tile_sizze_50527;
    
    int32_t tiled_group_sizze_50529 = tile_sizze_50528 * tile_sizze_50528;
    int32_t y_50536 = tile_sizze_50528 - 1;
    int32_t x_50537 = sizze_47660 + y_50536;
    int32_t groups_in_dim_50538 = squot32(x_50537, tile_sizze_50528);
    int32_t x_50540 = sizze_47663 + y_50536;
    int32_t groups_in_dim_50541 = squot32(x_50540, tile_sizze_50528);
    int32_t y_50543 = groups_in_dim_50538 * groups_in_dim_50541;
    int32_t num_groups_50544 = sizze_47659 * y_50543;
    int32_t num_threads_50545 = tiled_group_sizze_50529 * num_groups_50544;
    int32_t binop_x_50621 = sizze_47659 * sizze_47660;
    int32_t convop_x_50622 = sizze_47663 * binop_x_50621;
    int64_t binop_x_50623 = sext_i32_i64(convop_x_50622);
    int64_t bytes_50620 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50620, "mem_50624");
    
    int64_t binop_x_50614 = sext_i32_i64(tiled_group_sizze_50529);
    int64_t bytes_50612 = 4 * binop_x_50614;
    struct memblock_local mem_50615;
    
    mem_50615.references = NULL;
    
    struct memblock_local mem_50619;
    
    mem_50619.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 0, bytes_50612, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 1, bytes_50612, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 2, sizeof(sizze_47659),
                                  &sizze_47659));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 3, sizeof(sizze_47660),
                                  &sizze_47660));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 4, sizeof(sizze_47661),
                                  &sizze_47661));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 5, sizeof(sizze_47663),
                                  &sizze_47663));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 6,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 7,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49760, 8,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (num_groups_50544 * tiled_group_sizze_50529) != 0) {
        const size_t global_work_sizze_51498[1] = {num_groups_50544 *
                     tiled_group_sizze_50529};
        const size_t local_work_sizze_51502[1] = {tiled_group_sizze_50529};
        int64_t time_start_51499 = 0, time_end_51500 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49760");
            fprintf(stderr, "%zu", global_work_sizze_51498[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51502[0]);
            fprintf(stderr, "].\n");
            time_start_51499 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49760, 1, NULL,
                                              global_work_sizze_51498,
                                              local_work_sizze_51502, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51500 = get_wall_time();
            
            long time_diff_51501 = time_end_51500 - time_start_51499;
            
            ctx->map_kernel_49760_total_runtime += time_diff_51501;
            ctx->map_kernel_49760_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49760",
                    time_diff_51501);
        }
    }
    memblock_unref_local(ctx, &mem_50615, "mem_50615");
    memblock_unref_local(ctx, &mem_50619, "mem_50619");
    out_arrsizze_51054 = sizze_47659;
    out_arrsizze_51055 = sizze_47660;
    out_arrsizze_51056 = sizze_47663;
    out_memsizze_51053 = bytes_50620;
    memblock_set_device(ctx, &out_mem_51052, &mem_50624, "mem_50624");
    *out_out_memsizze_51493 = out_memsizze_51053;
    (*out_mem_p_51494).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51494, &out_mem_51052,
                        "out_mem_51052");
    *out_out_arrsizze_51495 = out_arrsizze_51054;
    *out_out_arrsizze_51496 = out_arrsizze_51055;
    *out_out_arrsizze_51497 = out_arrsizze_51056;
    memblock_unref_local(ctx, &mem_50619, "mem_50619");
    memblock_unref_local(ctx, &mem_50615, "mem_50615");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &out_mem_51052, "out_mem_51052");
    return 0;
}
static int futrts_t2_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51503,
                        struct memblock_device *out_mem_p_51504,
                        int32_t *out_out_arrsizze_51505,
                        int32_t *out_out_arrsizze_51506,
                        int32_t *out_out_arrsizze_51507,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47690,
                        int32_t sizze_47691, int32_t sizze_47692,
                        int32_t sizze_47693, int32_t sizze_47694)
{
    int64_t out_memsizze_51066;
    struct memblock_device out_mem_51065;
    
    out_mem_51065.references = NULL;
    
    int32_t out_arrsizze_51067;
    int32_t out_arrsizze_51068;
    int32_t out_arrsizze_51069;
    bool dim_zzero_47698 = 0 == sizze_47693;
    bool dim_zzero_47699 = 0 == sizze_47692;
    bool both_empty_47700 = dim_zzero_47698 && dim_zzero_47699;
    bool dim_match_47701 = sizze_47692 == sizze_47693;
    bool empty_or_match_47702 = both_empty_47700 || dim_match_47701;
    bool empty_or_match_cert_47703;
    
    if (!empty_or_match_47702) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:107:1-109:79",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51065, "out_mem_51065");
        return 1;
    }
    
    int32_t nesting_sizze_49785 = sizze_47692 * sizze_47694;
    int32_t group_sizze_49787;
    
    group_sizze_49787 = ctx->sizes.group_sizze_49786;
    
    int32_t y_49788 = group_sizze_49787 - 1;
    int32_t x_49789 = nesting_sizze_49785 + y_49788;
    int32_t num_groups_49790 = squot32(x_49789, group_sizze_49787);
    int32_t num_threads_49791 = group_sizze_49787 * num_groups_49790;
    int64_t binop_x_50616 = sext_i32_i64(nesting_sizze_49785);
    int64_t bytes_50614 = 4 * binop_x_50616;
    struct memblock_device mem_50617;
    
    mem_50617.references = NULL;
    memblock_alloc_device(ctx, &mem_50617, bytes_50614, "mem_50617");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49792, 0, sizeof(sizze_47692),
                                  &sizze_47692));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49792, 1, sizeof(sizze_47694),
                                  &sizze_47694));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49792, 2,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49792, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49792, 4,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    if (1 * (num_groups_49790 * group_sizze_49787) != 0) {
        const size_t global_work_sizze_51508[1] = {num_groups_49790 *
                     group_sizze_49787};
        const size_t local_work_sizze_51512[1] = {group_sizze_49787};
        int64_t time_start_51509 = 0, time_end_51510 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49792");
            fprintf(stderr, "%zu", global_work_sizze_51508[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51512[0]);
            fprintf(stderr, "].\n");
            time_start_51509 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49792, 1, NULL,
                                              global_work_sizze_51508,
                                              local_work_sizze_51512, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51510 = get_wall_time();
            
            long time_diff_51511 = time_end_51510 - time_start_51509;
            
            ctx->map_kernel_49792_total_runtime += time_diff_51511;
            ctx->map_kernel_49792_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49792",
                    time_diff_51511);
        }
    }
    
    int32_t tile_sizze_50528;
    
    tile_sizze_50528 = ctx->sizes.tile_sizze_50527;
    
    int32_t tiled_group_sizze_50529 = tile_sizze_50528 * tile_sizze_50528;
    int32_t y_50536 = tile_sizze_50528 - 1;
    int32_t x_50537 = sizze_47691 + y_50536;
    int32_t groups_in_dim_50538 = squot32(x_50537, tile_sizze_50528);
    int32_t x_50540 = sizze_47694 + y_50536;
    int32_t groups_in_dim_50541 = squot32(x_50540, tile_sizze_50528);
    int32_t y_50543 = groups_in_dim_50538 * groups_in_dim_50541;
    int32_t num_groups_50544 = sizze_47690 * y_50543;
    int32_t num_threads_50545 = tiled_group_sizze_50529 * num_groups_50544;
    int32_t binop_x_50627 = sizze_47690 * sizze_47691;
    int32_t convop_x_50628 = sizze_47694 * binop_x_50627;
    int64_t binop_x_50629 = sext_i32_i64(convop_x_50628);
    int64_t bytes_50626 = 4 * binop_x_50629;
    struct memblock_device mem_50630;
    
    mem_50630.references = NULL;
    memblock_alloc_device(ctx, &mem_50630, bytes_50626, "mem_50630");
    
    int64_t binop_x_50620 = sext_i32_i64(tiled_group_sizze_50529);
    int64_t bytes_50618 = 4 * binop_x_50620;
    struct memblock_local mem_50621;
    
    mem_50621.references = NULL;
    
    struct memblock_local mem_50625;
    
    mem_50625.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 0, bytes_50618, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 1, bytes_50618, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 2, sizeof(sizze_47690),
                                  &sizze_47690));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 3, sizeof(sizze_47691),
                                  &sizze_47691));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 4, sizeof(sizze_47692),
                                  &sizze_47692));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 5, sizeof(sizze_47694),
                                  &sizze_47694));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 6,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 7,
                                  sizeof(mem_50617.mem), &mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49820, 8,
                                  sizeof(mem_50630.mem), &mem_50630.mem));
    if (1 * (num_groups_50544 * tiled_group_sizze_50529) != 0) {
        const size_t global_work_sizze_51513[1] = {num_groups_50544 *
                     tiled_group_sizze_50529};
        const size_t local_work_sizze_51517[1] = {tiled_group_sizze_50529};
        int64_t time_start_51514 = 0, time_end_51515 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49820");
            fprintf(stderr, "%zu", global_work_sizze_51513[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51517[0]);
            fprintf(stderr, "].\n");
            time_start_51514 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49820, 1, NULL,
                                              global_work_sizze_51513,
                                              local_work_sizze_51517, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51515 = get_wall_time();
            
            long time_diff_51516 = time_end_51515 - time_start_51514;
            
            ctx->map_kernel_49820_total_runtime += time_diff_51516;
            ctx->map_kernel_49820_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49820",
                    time_diff_51516);
        }
    }
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_local(ctx, &mem_50621, "mem_50621");
    memblock_unref_local(ctx, &mem_50625, "mem_50625");
    out_arrsizze_51067 = sizze_47690;
    out_arrsizze_51068 = sizze_47691;
    out_arrsizze_51069 = sizze_47694;
    out_memsizze_51066 = bytes_50626;
    memblock_set_device(ctx, &out_mem_51065, &mem_50630, "mem_50630");
    *out_out_memsizze_51503 = out_memsizze_51066;
    (*out_mem_p_51504).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51504, &out_mem_51065,
                        "out_mem_51065");
    *out_out_arrsizze_51505 = out_arrsizze_51067;
    *out_out_arrsizze_51506 = out_arrsizze_51068;
    *out_out_arrsizze_51507 = out_arrsizze_51069;
    memblock_unref_local(ctx, &mem_50625, "mem_50625");
    memblock_unref_local(ctx, &mem_50621, "mem_50621");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    memblock_unref_device(ctx, &mem_50617, "mem_50617");
    memblock_unref_device(ctx, &out_mem_51065, "out_mem_51065");
    return 0;
}
static int futrts_t10_3d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51518,
                         struct memblock_device *out_mem_p_51519,
                         int32_t *out_out_arrsizze_51520,
                         int32_t *out_out_arrsizze_51521,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int32_t sizze_47724, int32_t sizze_47725,
                         int32_t sizze_47726, int32_t sizze_47727,
                         int32_t sizze_47728, int32_t sizze_47729,
                         int32_t sizze_47730)
{
    int64_t out_memsizze_51082;
    struct memblock_device out_mem_51081;
    
    out_mem_51081.references = NULL;
    
    int32_t out_arrsizze_51083;
    int32_t out_arrsizze_51084;
    bool dim_zzero_47734 = 0 == sizze_47727;
    bool dim_zzero_47735 = 0 == sizze_47728;
    bool dim_zzero_47736 = 0 == sizze_47729;
    bool y_47737 = dim_zzero_47735 || dim_zzero_47736;
    bool old_empty_47738 = dim_zzero_47734 || y_47737;
    bool dim_zzero_47739 = 0 == sizze_47724;
    bool dim_zzero_47740 = 0 == sizze_47726;
    bool y_47741 = dim_zzero_47736 || dim_zzero_47740;
    bool new_empty_47742 = dim_zzero_47739 || y_47741;
    bool both_empty_47743 = old_empty_47738 && new_empty_47742;
    bool dim_match_47744 = sizze_47724 == sizze_47727;
    bool dim_match_47745 = sizze_47726 == sizze_47728;
    bool match_47746 = dim_match_47744 && dim_match_47745;
    bool empty_or_match_47747 = both_empty_47743 || match_47746;
    bool empty_or_match_cert_47748;
    
    if (!empty_or_match_47747) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:141:1-143:72",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51081, "out_mem_51081");
        return 1;
    }
    
    bool dim_zzero_47749 = 0 == sizze_47730;
    bool both_empty_47750 = dim_zzero_47736 && dim_zzero_47749;
    bool dim_match_47751 = sizze_47729 == sizze_47730;
    bool empty_or_match_47752 = both_empty_47750 || dim_match_47751;
    bool empty_or_match_cert_47753;
    
    if (!empty_or_match_47752) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:141:1-143:72",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51081, "out_mem_51081");
        return 1;
    }
    
    int32_t nesting_sizze_49900 = sizze_47724 * sizze_47726;
    int32_t group_sizze_49902;
    
    group_sizze_49902 = ctx->sizes.group_sizze_49901;
    
    int32_t y_49903 = group_sizze_49902 - 1;
    int32_t x_49904 = nesting_sizze_49900 + y_49903;
    int32_t num_groups_49905 = squot32(x_49904, group_sizze_49902);
    int32_t num_threads_49906 = group_sizze_49902 * num_groups_49905;
    int32_t binop_x_50615 = sizze_47727 * sizze_47729;
    int32_t convop_x_50616 = sizze_47728 * binop_x_50615;
    int64_t binop_x_50617 = sext_i32_i64(convop_x_50616);
    int64_t bytes_50614 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50614, "mem_50618");
    
    int call_ret_51522 = futrts_map_transpose_opencl_f32(ctx, mem_50618, 0,
                                                         B_mem_50611, 0, 1,
                                                         sizze_47729,
                                                         sizze_47727 *
                                                         sizze_47728,
                                                         sizze_47727 *
                                                         sizze_47728 *
                                                         sizze_47729,
                                                         sizze_47727 *
                                                         sizze_47728 *
                                                         sizze_47729);
    
    assert(call_ret_51522 == 0);
    
    int64_t binop_x_50621 = sext_i32_i64(nesting_sizze_49900);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50619, "mem_50622");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 0, sizeof(sizze_47724),
                                  &sizze_47724));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 1, sizeof(sizze_47726),
                                  &sizze_47726));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 2, sizeof(sizze_47727),
                                  &sizze_47727));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 3, sizeof(sizze_47728),
                                  &sizze_47728));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 4, sizeof(sizze_47729),
                                  &sizze_47729));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 5,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 6,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49907, 7,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    if (1 * (num_groups_49905 * group_sizze_49902) != 0) {
        const size_t global_work_sizze_51523[1] = {num_groups_49905 *
                     group_sizze_49902};
        const size_t local_work_sizze_51527[1] = {group_sizze_49902};
        int64_t time_start_51524 = 0, time_end_51525 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49907");
            fprintf(stderr, "%zu", global_work_sizze_51523[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51527[0]);
            fprintf(stderr, "].\n");
            time_start_51524 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49907, 1, NULL,
                                              global_work_sizze_51523,
                                              local_work_sizze_51527, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51525 = get_wall_time();
            
            long time_diff_51526 = time_end_51525 - time_start_51524;
            
            ctx->map_kernel_49907_total_runtime += time_diff_51526;
            ctx->map_kernel_49907_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49907",
                    time_diff_51526);
        }
    }
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    
    int32_t nesting_sizze_49857 = sizze_47724 * sizze_47725;
    int32_t group_sizze_49859;
    
    group_sizze_49859 = ctx->sizes.group_sizze_49858;
    
    int32_t y_49860 = group_sizze_49859 - 1;
    int32_t x_49861 = nesting_sizze_49857 + y_49860;
    int32_t num_groups_49862 = squot32(x_49861, group_sizze_49859);
    int32_t num_threads_49863 = group_sizze_49859 * num_groups_49862;
    int32_t convop_x_50625 = sizze_47725 * nesting_sizze_49900;
    int64_t binop_x_50626 = sext_i32_i64(convop_x_50625);
    int64_t bytes_50623 = 4 * binop_x_50626;
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, bytes_50623, "mem_50627");
    
    int call_ret_51528 = futrts_map_transpose_opencl_f32(ctx, mem_50627, 0,
                                                         A_mem_50609, 0, 1,
                                                         sizze_47726,
                                                         sizze_47724 *
                                                         sizze_47725,
                                                         sizze_47724 *
                                                         sizze_47725 *
                                                         sizze_47726,
                                                         sizze_47724 *
                                                         sizze_47725 *
                                                         sizze_47726);
    
    assert(call_ret_51528 == 0);
    
    int64_t binop_x_50630 = sext_i32_i64(nesting_sizze_49857);
    int64_t bytes_50628 = 4 * binop_x_50630;
    struct memblock_device mem_50631;
    
    mem_50631.references = NULL;
    memblock_alloc_device(ctx, &mem_50631, bytes_50628, "mem_50631");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 0, sizeof(sizze_47724),
                                  &sizze_47724));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 1, sizeof(sizze_47725),
                                  &sizze_47725));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 2, sizeof(sizze_47726),
                                  &sizze_47726));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 3,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 4,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49864, 5,
                                  sizeof(mem_50631.mem), &mem_50631.mem));
    if (1 * (num_groups_49862 * group_sizze_49859) != 0) {
        const size_t global_work_sizze_51529[1] = {num_groups_49862 *
                     group_sizze_49859};
        const size_t local_work_sizze_51533[1] = {group_sizze_49859};
        int64_t time_start_51530 = 0, time_end_51531 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49864");
            fprintf(stderr, "%zu", global_work_sizze_51529[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51533[0]);
            fprintf(stderr, "].\n");
            time_start_51530 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49864, 1, NULL,
                                              global_work_sizze_51529,
                                              local_work_sizze_51533, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51531 = get_wall_time();
            
            long time_diff_51532 = time_end_51531 - time_start_51530;
            
            ctx->map_kernel_49864_total_runtime += time_diff_51532;
            ctx->map_kernel_49864_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49864",
                    time_diff_51532);
        }
    }
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    out_arrsizze_51083 = sizze_47724;
    out_arrsizze_51084 = sizze_47725;
    out_memsizze_51082 = bytes_50628;
    memblock_set_device(ctx, &out_mem_51081, &mem_50631, "mem_50631");
    *out_out_memsizze_51518 = out_memsizze_51082;
    (*out_mem_p_51519).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51519, &out_mem_51081,
                        "out_mem_51081");
    *out_out_arrsizze_51520 = out_arrsizze_51083;
    *out_out_arrsizze_51521 = out_arrsizze_51084;
    memblock_unref_device(ctx, &mem_50631, "mem_50631");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &out_mem_51081, "out_mem_51081");
    return 0;
}
static int futrts_t11_3d(struct futhark_context *ctx,
                         int64_t *out_out_memsizze_51534,
                         struct memblock_device *out_mem_p_51535,
                         int32_t *out_out_arrsizze_51536,
                         int32_t *out_out_arrsizze_51537,
                         int64_t A_mem_sizze_50608,
                         struct memblock_device A_mem_50609,
                         int64_t B_mem_sizze_50610,
                         struct memblock_device B_mem_50611,
                         int64_t C_mem_sizze_50612,
                         struct memblock_device C_mem_50613,
                         int64_t D_mem_sizze_50614,
                         struct memblock_device D_mem_50615,
                         int64_t E_mem_sizze_50616,
                         struct memblock_device E_mem_50617,
                         int32_t sizze_47777, int32_t sizze_47778,
                         int32_t sizze_47779, int32_t sizze_47780,
                         int32_t sizze_47781, int32_t sizze_47782,
                         int32_t sizze_47783)
{
    int64_t out_memsizze_51094;
    struct memblock_device out_mem_51093;
    
    out_mem_51093.references = NULL;
    
    int32_t out_arrsizze_51095;
    int32_t out_arrsizze_51096;
    bool dim_zzero_47789 = 0 == sizze_47780;
    bool dim_zzero_47790 = 0 == sizze_47781;
    bool dim_zzero_47791 = 0 == sizze_47782;
    bool y_47792 = dim_zzero_47790 || dim_zzero_47791;
    bool old_empty_47793 = dim_zzero_47789 || y_47792;
    bool dim_zzero_47794 = 0 == sizze_47777;
    bool dim_zzero_47795 = 0 == sizze_47779;
    bool y_47796 = dim_zzero_47791 || dim_zzero_47795;
    bool new_empty_47797 = dim_zzero_47794 || y_47796;
    bool both_empty_47798 = old_empty_47793 && new_empty_47797;
    bool dim_match_47799 = sizze_47777 == sizze_47780;
    bool dim_match_47800 = sizze_47779 == sizze_47781;
    bool match_47801 = dim_match_47799 && dim_match_47800;
    bool empty_or_match_47802 = both_empty_47798 || match_47801;
    bool empty_or_match_cert_47803;
    
    if (!empty_or_match_47802) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:145:1-148:76",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51093, "out_mem_51093");
        return 1;
    }
    
    bool dim_zzero_47804 = 0 == sizze_47783;
    bool both_empty_47805 = dim_zzero_47791 && dim_zzero_47804;
    bool dim_match_47806 = sizze_47782 == sizze_47783;
    bool empty_or_match_47807 = both_empty_47805 || dim_match_47806;
    bool empty_or_match_cert_47808;
    
    if (!empty_or_match_47807) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:145:1-148:76",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51093, "out_mem_51093");
        return 1;
    }
    
    int32_t nesting_sizze_49989 = sizze_47777 * sizze_47779;
    int32_t group_sizze_49991;
    
    group_sizze_49991 = ctx->sizes.group_sizze_49990;
    
    int32_t y_49992 = group_sizze_49991 - 1;
    int32_t x_49993 = nesting_sizze_49989 + y_49992;
    int32_t num_groups_49994 = squot32(x_49993, group_sizze_49991);
    int32_t num_threads_49995 = group_sizze_49991 * num_groups_49994;
    int32_t binop_x_50619 = sizze_47780 * sizze_47782;
    int32_t convop_x_50620 = sizze_47781 * binop_x_50619;
    int64_t binop_x_50621 = sext_i32_i64(convop_x_50620);
    int64_t bytes_50618 = 4 * binop_x_50621;
    struct memblock_device mem_50622;
    
    mem_50622.references = NULL;
    memblock_alloc_device(ctx, &mem_50622, bytes_50618, "mem_50622");
    
    int call_ret_51538 = futrts_map_transpose_opencl_f32(ctx, mem_50622, 0,
                                                         D_mem_50615, 0, 1,
                                                         sizze_47782,
                                                         sizze_47780 *
                                                         sizze_47781,
                                                         sizze_47780 *
                                                         sizze_47781 *
                                                         sizze_47782,
                                                         sizze_47780 *
                                                         sizze_47781 *
                                                         sizze_47782);
    
    assert(call_ret_51538 == 0);
    
    int64_t binop_x_50625 = sext_i32_i64(nesting_sizze_49989);
    int64_t bytes_50623 = 4 * binop_x_50625;
    struct memblock_device mem_50626;
    
    mem_50626.references = NULL;
    memblock_alloc_device(ctx, &mem_50626, bytes_50623, "mem_50626");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 0, sizeof(sizze_47777),
                                  &sizze_47777));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 1, sizeof(sizze_47779),
                                  &sizze_47779));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 2, sizeof(sizze_47780),
                                  &sizze_47780));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 3, sizeof(sizze_47781),
                                  &sizze_47781));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 4, sizeof(sizze_47782),
                                  &sizze_47782));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 5,
                                  sizeof(E_mem_50617.mem), &E_mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 6,
                                  sizeof(mem_50622.mem), &mem_50622.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49996, 7,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    if (1 * (num_groups_49994 * group_sizze_49991) != 0) {
        const size_t global_work_sizze_51539[1] = {num_groups_49994 *
                     group_sizze_49991};
        const size_t local_work_sizze_51543[1] = {group_sizze_49991};
        int64_t time_start_51540 = 0, time_end_51541 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49996");
            fprintf(stderr, "%zu", global_work_sizze_51539[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51543[0]);
            fprintf(stderr, "].\n");
            time_start_51540 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49996, 1, NULL,
                                              global_work_sizze_51539,
                                              local_work_sizze_51543, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51541 = get_wall_time();
            
            long time_diff_51542 = time_end_51541 - time_start_51540;
            
            ctx->map_kernel_49996_total_runtime += time_diff_51542;
            ctx->map_kernel_49996_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49996",
                    time_diff_51542);
        }
    }
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    
    int32_t nesting_sizze_49943 = sizze_47777 * sizze_47778;
    int32_t group_sizze_49945;
    
    group_sizze_49945 = ctx->sizes.group_sizze_49944;
    
    int32_t y_49946 = group_sizze_49945 - 1;
    int32_t x_49947 = nesting_sizze_49943 + y_49946;
    int32_t num_groups_49948 = squot32(x_49947, group_sizze_49945);
    int32_t num_threads_49949 = group_sizze_49945 * num_groups_49948;
    int64_t binop_x_50635 = sext_i32_i64(nesting_sizze_49943);
    int64_t bytes_50633 = 4 * binop_x_50635;
    struct memblock_device mem_50636;
    
    mem_50636.references = NULL;
    memblock_alloc_device(ctx, &mem_50636, bytes_50633, "mem_50636");
    
    int64_t binop_x_50628 = sext_i32_i64(group_sizze_49945);
    int64_t bytes_50627 = 4 * binop_x_50628;
    struct memblock_local mem_50629;
    
    mem_50629.references = NULL;
    
    struct memblock_local mem_50632;
    
    mem_50632.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 0, bytes_50627, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 1, bytes_50627, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 2, sizeof(sizze_47777),
                                  &sizze_47777));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 3, sizeof(sizze_47778),
                                  &sizze_47778));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 4, sizeof(sizze_47779),
                                  &sizze_47779));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 5,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 6,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 7,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 8,
                                  sizeof(mem_50626.mem), &mem_50626.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_49950, 9,
                                  sizeof(mem_50636.mem), &mem_50636.mem));
    if (1 * (num_groups_49948 * group_sizze_49945) != 0) {
        const size_t global_work_sizze_51544[1] = {num_groups_49948 *
                     group_sizze_49945};
        const size_t local_work_sizze_51548[1] = {group_sizze_49945};
        int64_t time_start_51545 = 0, time_end_51546 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_49950");
            fprintf(stderr, "%zu", global_work_sizze_51544[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51548[0]);
            fprintf(stderr, "].\n");
            time_start_51545 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_49950, 1, NULL,
                                              global_work_sizze_51544,
                                              local_work_sizze_51548, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51546 = get_wall_time();
            
            long time_diff_51547 = time_end_51546 - time_start_51545;
            
            ctx->map_kernel_49950_total_runtime += time_diff_51547;
            ctx->map_kernel_49950_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_49950",
                    time_diff_51547);
        }
    }
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_local(ctx, &mem_50629, "mem_50629");
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    out_arrsizze_51095 = sizze_47777;
    out_arrsizze_51096 = sizze_47778;
    out_memsizze_51094 = bytes_50633;
    memblock_set_device(ctx, &out_mem_51093, &mem_50636, "mem_50636");
    *out_out_memsizze_51534 = out_memsizze_51094;
    (*out_mem_p_51535).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51535, &out_mem_51093,
                        "out_mem_51093");
    *out_out_arrsizze_51536 = out_arrsizze_51095;
    *out_out_arrsizze_51537 = out_arrsizze_51096;
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    memblock_unref_local(ctx, &mem_50629, "mem_50629");
    memblock_unref_device(ctx, &mem_50636, "mem_50636");
    memblock_unref_device(ctx, &mem_50626, "mem_50626");
    memblock_unref_device(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &out_mem_51093, "out_mem_51093");
    return 0;
}
static int futrts_t3_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51549,
                        struct memblock_device *out_mem_p_51550,
                        int32_t *out_out_arrsizze_51551,
                        int32_t *out_out_arrsizze_51552,
                        int32_t *out_out_arrsizze_51553,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47834,
                        int32_t sizze_47835, int32_t sizze_47836,
                        int32_t sizze_47837, int32_t sizze_47838,
                        int32_t sizze_47839, int32_t sizze_47840)
{
    int64_t out_memsizze_51110;
    struct memblock_device out_mem_51109;
    
    out_mem_51109.references = NULL;
    
    int32_t out_arrsizze_51111;
    int32_t out_arrsizze_51112;
    int32_t out_arrsizze_51113;
    bool dim_zzero_47845 = 0 == sizze_47837;
    bool dim_zzero_47846 = 0 == sizze_47838;
    bool old_empty_47847 = dim_zzero_47845 || dim_zzero_47846;
    bool dim_zzero_47848 = 0 == sizze_47836;
    bool new_empty_47849 = dim_zzero_47846 || dim_zzero_47848;
    bool both_empty_47850 = old_empty_47847 && new_empty_47849;
    bool dim_match_47851 = sizze_47836 == sizze_47837;
    bool empty_or_match_47852 = both_empty_47850 || dim_match_47851;
    bool empty_or_match_cert_47853;
    
    if (!empty_or_match_47852) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:111:1-112:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51109, "out_mem_51109");
        return 1;
    }
    
    bool dim_zzero_47855 = 0 == sizze_47839;
    bool both_empty_47856 = dim_zzero_47848 && dim_zzero_47855;
    bool dim_match_47857 = sizze_47836 == sizze_47839;
    bool empty_or_match_47858 = both_empty_47856 || dim_match_47857;
    bool empty_or_match_cert_47859;
    
    if (!empty_or_match_47858) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:111:1-112:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51109, "out_mem_51109");
        return 1;
    }
    
    bool dim_zzero_47860 = 0 == sizze_47840;
    bool dim_zzero_47861 = 0 == sizze_47835;
    bool both_empty_47862 = dim_zzero_47860 && dim_zzero_47861;
    bool dim_match_47863 = sizze_47835 == sizze_47840;
    bool empty_or_match_47864 = both_empty_47862 || dim_match_47863;
    bool empty_or_match_cert_47865;
    
    if (!empty_or_match_47864) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:111:1-112:114",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51109, "out_mem_51109");
        return 1;
    }
    
    int32_t nesting_sizze_50034 = sizze_47835 * sizze_47838;
    int32_t nesting_sizze_50035 = sizze_47834 * nesting_sizze_50034;
    int32_t group_sizze_50037;
    
    group_sizze_50037 = ctx->sizes.group_sizze_50036;
    
    int32_t y_50038 = group_sizze_50037 - 1;
    int32_t x_50039 = nesting_sizze_50035 + y_50038;
    int32_t num_groups_50040 = squot32(x_50039, group_sizze_50037);
    int32_t num_threads_50041 = group_sizze_50037 * num_groups_50040;
    int32_t binop_x_50617 = sizze_47834 * sizze_47835;
    int32_t convop_x_50618 = sizze_47838 * binop_x_50617;
    int64_t binop_x_50619 = sext_i32_i64(convop_x_50618);
    int64_t bytes_50616 = 4 * binop_x_50619;
    struct memblock_device mem_50620;
    
    mem_50620.references = NULL;
    memblock_alloc_device(ctx, &mem_50620, bytes_50616, "mem_50620");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 0, sizeof(sizze_47834),
                                  &sizze_47834));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 1, sizeof(sizze_47835),
                                  &sizze_47835));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 2, sizeof(sizze_47836),
                                  &sizze_47836));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 3, sizeof(sizze_47838),
                                  &sizze_47838));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 4,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 5,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 7,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50042, 8,
                                  sizeof(mem_50620.mem), &mem_50620.mem));
    if (1 * (num_groups_50040 * group_sizze_50037) != 0) {
        const size_t global_work_sizze_51554[1] = {num_groups_50040 *
                     group_sizze_50037};
        const size_t local_work_sizze_51558[1] = {group_sizze_50037};
        int64_t time_start_51555 = 0, time_end_51556 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50042");
            fprintf(stderr, "%zu", global_work_sizze_51554[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51558[0]);
            fprintf(stderr, "].\n");
            time_start_51555 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50042, 1, NULL,
                                              global_work_sizze_51554,
                                              local_work_sizze_51558, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51556 = get_wall_time();
            
            long time_diff_51557 = time_end_51556 - time_start_51555;
            
            ctx->map_kernel_50042_total_runtime += time_diff_51557;
            ctx->map_kernel_50042_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50042",
                    time_diff_51557);
        }
    }
    out_arrsizze_51111 = sizze_47834;
    out_arrsizze_51112 = sizze_47835;
    out_arrsizze_51113 = sizze_47838;
    out_memsizze_51110 = bytes_50616;
    memblock_set_device(ctx, &out_mem_51109, &mem_50620, "mem_50620");
    *out_out_memsizze_51549 = out_memsizze_51110;
    (*out_mem_p_51550).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51550, &out_mem_51109,
                        "out_mem_51109");
    *out_out_arrsizze_51551 = out_arrsizze_51111;
    *out_out_arrsizze_51552 = out_arrsizze_51112;
    *out_out_arrsizze_51553 = out_arrsizze_51113;
    memblock_unref_device(ctx, &mem_50620, "mem_50620");
    memblock_unref_device(ctx, &out_mem_51109, "out_mem_51109");
    return 0;
}
static int futrts_t4_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51559,
                        struct memblock_device *out_mem_p_51560,
                        int32_t *out_out_arrsizze_51561,
                        int32_t *out_out_arrsizze_51562,
                        int32_t *out_out_arrsizze_51563,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_47886,
                        int32_t sizze_47887, int32_t sizze_47888,
                        int32_t sizze_47889, int32_t sizze_47890,
                        int32_t sizze_47891, int32_t sizze_47892,
                        int32_t sizze_47893)
{
    int64_t out_memsizze_51119;
    struct memblock_device out_mem_51118;
    
    out_mem_51118.references = NULL;
    
    int32_t out_arrsizze_51120;
    int32_t out_arrsizze_51121;
    int32_t out_arrsizze_51122;
    bool dim_zzero_47897 = 0 == sizze_47889;
    bool dim_zzero_47898 = 0 == sizze_47890;
    bool dim_zzero_47899 = 0 == sizze_47891;
    bool y_47900 = dim_zzero_47898 || dim_zzero_47899;
    bool old_empty_47901 = dim_zzero_47897 || y_47900;
    bool dim_zzero_47902 = 0 == sizze_47886;
    bool dim_zzero_47903 = 0 == sizze_47887;
    bool dim_zzero_47904 = 0 == sizze_47888;
    bool y_47905 = dim_zzero_47903 || dim_zzero_47904;
    bool new_empty_47906 = dim_zzero_47902 || y_47905;
    bool both_empty_47907 = old_empty_47901 && new_empty_47906;
    bool dim_match_47908 = sizze_47886 == sizze_47889;
    bool dim_match_47909 = sizze_47887 == sizze_47890;
    bool dim_match_47910 = sizze_47888 == sizze_47891;
    bool y_47911 = dim_match_47909 && dim_match_47910;
    bool match_47912 = dim_match_47908 && y_47911;
    bool empty_or_match_47913 = both_empty_47907 || match_47912;
    bool empty_or_match_cert_47914;
    
    if (!empty_or_match_47913) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:114:1-115:85",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51118, "out_mem_51118");
        return 1;
    }
    
    bool dim_zzero_47915 = 0 == sizze_47892;
    bool dim_zzero_47916 = 0 == sizze_47893;
    bool old_empty_47917 = dim_zzero_47915 || dim_zzero_47916;
    bool new_empty_47918 = dim_zzero_47904 || dim_zzero_47916;
    bool both_empty_47919 = old_empty_47917 && new_empty_47918;
    bool dim_match_47920 = sizze_47888 == sizze_47892;
    bool empty_or_match_47921 = both_empty_47919 || dim_match_47920;
    bool empty_or_match_cert_47922;
    
    if (!empty_or_match_47921) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:114:1-115:85",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51118, "out_mem_51118");
        return 1;
    }
    
    int32_t nesting_sizze_50121 = sizze_47887 * sizze_47888;
    int32_t nesting_sizze_50122 = sizze_47886 * nesting_sizze_50121;
    int32_t group_sizze_50124;
    
    group_sizze_50124 = ctx->sizes.group_sizze_50123;
    
    int32_t y_50125 = group_sizze_50124 - 1;
    int32_t x_50126 = nesting_sizze_50122 + y_50125;
    int32_t num_groups_50127 = squot32(x_50126, group_sizze_50124);
    int32_t num_threads_50128 = group_sizze_50124 * num_groups_50127;
    int32_t binop_x_50615 = sizze_47886 * sizze_47887;
    int32_t convop_x_50616 = sizze_47888 * binop_x_50615;
    int64_t binop_x_50617 = sext_i32_i64(convop_x_50616);
    int64_t bytes_50614 = 4 * binop_x_50617;
    struct memblock_device mem_50618;
    
    mem_50618.references = NULL;
    memblock_alloc_device(ctx, &mem_50618, bytes_50614, "mem_50618");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 0, sizeof(sizze_47886),
                                  &sizze_47886));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 1, sizeof(sizze_47887),
                                  &sizze_47887));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 2, sizeof(sizze_47888),
                                  &sizze_47888));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 3, sizeof(sizze_47890),
                                  &sizze_47890));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 4, sizeof(sizze_47891),
                                  &sizze_47891));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 5,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 6,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50129, 7,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    if (1 * (num_groups_50127 * group_sizze_50124) != 0) {
        const size_t global_work_sizze_51564[1] = {num_groups_50127 *
                     group_sizze_50124};
        const size_t local_work_sizze_51568[1] = {group_sizze_50124};
        int64_t time_start_51565 = 0, time_end_51566 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50129");
            fprintf(stderr, "%zu", global_work_sizze_51564[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51568[0]);
            fprintf(stderr, "].\n");
            time_start_51565 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50129, 1, NULL,
                                              global_work_sizze_51564,
                                              local_work_sizze_51568, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51566 = get_wall_time();
            
            long time_diff_51567 = time_end_51566 - time_start_51565;
            
            ctx->map_kernel_50129_total_runtime += time_diff_51567;
            ctx->map_kernel_50129_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50129",
                    time_diff_51567);
        }
    }
    
    int32_t tile_sizze_50528;
    
    tile_sizze_50528 = ctx->sizes.tile_sizze_50527;
    
    int32_t tiled_group_sizze_50529 = tile_sizze_50528 * tile_sizze_50528;
    int32_t y_50536 = tile_sizze_50528 - 1;
    int32_t x_50537 = sizze_47887 + y_50536;
    int32_t groups_in_dim_50538 = squot32(x_50537, tile_sizze_50528);
    int32_t x_50540 = sizze_47893 + y_50536;
    int32_t groups_in_dim_50541 = squot32(x_50540, tile_sizze_50528);
    int32_t y_50543 = groups_in_dim_50538 * groups_in_dim_50541;
    int32_t num_groups_50544 = sizze_47886 * y_50543;
    int32_t num_threads_50545 = tiled_group_sizze_50529 * num_groups_50544;
    int32_t convop_x_50629 = sizze_47893 * binop_x_50615;
    int64_t binop_x_50630 = sext_i32_i64(convop_x_50629);
    int64_t bytes_50627 = 4 * binop_x_50630;
    struct memblock_device mem_50631;
    
    mem_50631.references = NULL;
    memblock_alloc_device(ctx, &mem_50631, bytes_50627, "mem_50631");
    
    int64_t binop_x_50621 = sext_i32_i64(tiled_group_sizze_50529);
    int64_t bytes_50619 = 4 * binop_x_50621;
    struct memblock_local mem_50622;
    
    mem_50622.references = NULL;
    
    struct memblock_local mem_50626;
    
    mem_50626.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 0, bytes_50619, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 1, bytes_50619, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 2, sizeof(sizze_47886),
                                  &sizze_47886));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 3, sizeof(sizze_47887),
                                  &sizze_47887));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 4, sizeof(sizze_47888),
                                  &sizze_47888));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 5, sizeof(sizze_47893),
                                  &sizze_47893));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 6,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 7,
                                  sizeof(mem_50618.mem), &mem_50618.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50095, 8,
                                  sizeof(mem_50631.mem), &mem_50631.mem));
    if (1 * (num_groups_50544 * tiled_group_sizze_50529) != 0) {
        const size_t global_work_sizze_51569[1] = {num_groups_50544 *
                     tiled_group_sizze_50529};
        const size_t local_work_sizze_51573[1] = {tiled_group_sizze_50529};
        int64_t time_start_51570 = 0, time_end_51571 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50095");
            fprintf(stderr, "%zu", global_work_sizze_51569[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51573[0]);
            fprintf(stderr, "].\n");
            time_start_51570 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50095, 1, NULL,
                                              global_work_sizze_51569,
                                              local_work_sizze_51573, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51571 = get_wall_time();
            
            long time_diff_51572 = time_end_51571 - time_start_51570;
            
            ctx->map_kernel_50095_total_runtime += time_diff_51572;
            ctx->map_kernel_50095_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50095",
                    time_diff_51572);
        }
    }
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    memblock_unref_local(ctx, &mem_50626, "mem_50626");
    out_arrsizze_51120 = sizze_47886;
    out_arrsizze_51121 = sizze_47887;
    out_arrsizze_51122 = sizze_47893;
    out_memsizze_51119 = bytes_50627;
    memblock_set_device(ctx, &out_mem_51118, &mem_50631, "mem_50631");
    *out_out_memsizze_51559 = out_memsizze_51119;
    (*out_mem_p_51560).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51560, &out_mem_51118,
                        "out_mem_51118");
    *out_out_arrsizze_51561 = out_arrsizze_51120;
    *out_out_arrsizze_51562 = out_arrsizze_51121;
    *out_out_arrsizze_51563 = out_arrsizze_51122;
    memblock_unref_local(ctx, &mem_50626, "mem_50626");
    memblock_unref_local(ctx, &mem_50622, "mem_50622");
    memblock_unref_device(ctx, &mem_50631, "mem_50631");
    memblock_unref_device(ctx, &mem_50618, "mem_50618");
    memblock_unref_device(ctx, &out_mem_51118, "out_mem_51118");
    return 0;
}
static int futrts_t5_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51574,
                        struct memblock_device *out_mem_p_51575,
                        int32_t *out_out_arrsizze_51576,
                        int32_t *out_out_arrsizze_51577,
                        int32_t *out_out_arrsizze_51578,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_47945,
                        int32_t sizze_47946, int32_t sizze_47947,
                        int32_t sizze_47948, int32_t sizze_47949,
                        int32_t sizze_47950, int32_t sizze_47951,
                        int32_t sizze_47952, int32_t sizze_47953,
                        int32_t sizze_47954)
{
    int64_t out_memsizze_51135;
    struct memblock_device out_mem_51134;
    
    out_mem_51134.references = NULL;
    
    int32_t out_arrsizze_51136;
    int32_t out_arrsizze_51137;
    int32_t out_arrsizze_51138;
    bool dim_zzero_47959 = 0 == sizze_47948;
    bool dim_zzero_47960 = 0 == sizze_47949;
    bool dim_zzero_47961 = 0 == sizze_47950;
    bool y_47962 = dim_zzero_47960 || dim_zzero_47961;
    bool old_empty_47963 = dim_zzero_47959 || y_47962;
    bool dim_zzero_47964 = 0 == sizze_47945;
    bool dim_zzero_47965 = 0 == sizze_47946;
    bool dim_zzero_47966 = 0 == sizze_47947;
    bool y_47967 = dim_zzero_47965 || dim_zzero_47966;
    bool new_empty_47968 = dim_zzero_47964 || y_47967;
    bool both_empty_47969 = old_empty_47963 && new_empty_47968;
    bool dim_match_47970 = sizze_47945 == sizze_47948;
    bool dim_match_47971 = sizze_47946 == sizze_47949;
    bool dim_match_47972 = sizze_47947 == sizze_47950;
    bool y_47973 = dim_match_47971 && dim_match_47972;
    bool match_47974 = dim_match_47970 && y_47973;
    bool empty_or_match_47975 = both_empty_47969 || match_47974;
    bool empty_or_match_cert_47976;
    
    if (!empty_or_match_47975) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:117:1-118:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51134, "out_mem_51134");
        return 1;
    }
    
    bool dim_zzero_47977 = 0 == sizze_47951;
    bool dim_zzero_47978 = 0 == sizze_47952;
    bool old_empty_47979 = dim_zzero_47977 || dim_zzero_47978;
    bool new_empty_47980 = dim_zzero_47966 || dim_zzero_47978;
    bool both_empty_47981 = old_empty_47979 && new_empty_47980;
    bool dim_match_47982 = sizze_47947 == sizze_47951;
    bool empty_or_match_47983 = both_empty_47981 || dim_match_47982;
    bool empty_or_match_cert_47984;
    
    if (!empty_or_match_47983) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:117:1-118:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51134, "out_mem_51134");
        return 1;
    }
    
    bool dim_zzero_47986 = 0 == sizze_47953;
    bool dim_zzero_47987 = 0 == sizze_47954;
    bool old_empty_47988 = dim_zzero_47986 || dim_zzero_47987;
    bool both_empty_47989 = new_empty_47980 && old_empty_47988;
    bool dim_match_47990 = sizze_47947 == sizze_47953;
    bool dim_match_47991 = sizze_47952 == sizze_47954;
    bool match_47992 = dim_match_47990 && dim_match_47991;
    bool empty_or_match_47993 = both_empty_47989 || match_47992;
    bool empty_or_match_cert_47994;
    
    if (!empty_or_match_47993) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:117:1-118:95",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51134, "out_mem_51134");
        return 1;
    }
    
    int32_t nesting_sizze_50137 = sizze_47947 * sizze_47952;
    int32_t group_sizze_50139;
    
    group_sizze_50139 = ctx->sizes.group_sizze_50138;
    
    int32_t y_50140 = group_sizze_50139 - 1;
    int32_t x_50141 = nesting_sizze_50137 + y_50140;
    int32_t num_groups_50142 = squot32(x_50141, group_sizze_50139);
    int32_t num_threads_50143 = group_sizze_50139 * num_groups_50142;
    int64_t binop_x_50618 = sext_i32_i64(nesting_sizze_50137);
    int64_t bytes_50616 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50616, "mem_50619");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 0, sizeof(sizze_47947),
                                  &sizze_47947));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 1, sizeof(sizze_47952),
                                  &sizze_47952));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 2, sizeof(sizze_47954),
                                  &sizze_47954));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 3,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 4,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50144, 5,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    if (1 * (num_groups_50142 * group_sizze_50139) != 0) {
        const size_t global_work_sizze_51579[1] = {num_groups_50142 *
                     group_sizze_50139};
        const size_t local_work_sizze_51583[1] = {group_sizze_50139};
        int64_t time_start_51580 = 0, time_end_51581 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50144");
            fprintf(stderr, "%zu", global_work_sizze_51579[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51583[0]);
            fprintf(stderr, "].\n");
            time_start_51580 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50144, 1, NULL,
                                              global_work_sizze_51579,
                                              local_work_sizze_51583, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51581 = get_wall_time();
            
            long time_diff_51582 = time_end_51581 - time_start_51580;
            
            ctx->map_kernel_50144_total_runtime += time_diff_51582;
            ctx->map_kernel_50144_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50144",
                    time_diff_51582);
        }
    }
    
    int32_t nesting_sizze_50200 = sizze_47946 * sizze_47947;
    int32_t nesting_sizze_50201 = sizze_47945 * nesting_sizze_50200;
    int32_t group_sizze_50203;
    
    group_sizze_50203 = ctx->sizes.group_sizze_50202;
    
    int32_t y_50204 = group_sizze_50203 - 1;
    int32_t x_50205 = nesting_sizze_50201 + y_50204;
    int32_t num_groups_50206 = squot32(x_50205, group_sizze_50203);
    int32_t num_threads_50207 = group_sizze_50203 * num_groups_50206;
    int32_t binop_x_50621 = sizze_47945 * sizze_47946;
    int32_t convop_x_50622 = sizze_47947 * binop_x_50621;
    int64_t binop_x_50623 = sext_i32_i64(convop_x_50622);
    int64_t bytes_50620 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50620, "mem_50624");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 0, sizeof(sizze_47945),
                                  &sizze_47945));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 1, sizeof(sizze_47946),
                                  &sizze_47946));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 2, sizeof(sizze_47947),
                                  &sizze_47947));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 3, sizeof(sizze_47949),
                                  &sizze_47949));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 4, sizeof(sizze_47950),
                                  &sizze_47950));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 5,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 6,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50208, 7,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (num_groups_50206 * group_sizze_50203) != 0) {
        const size_t global_work_sizze_51584[1] = {num_groups_50206 *
                     group_sizze_50203};
        const size_t local_work_sizze_51588[1] = {group_sizze_50203};
        int64_t time_start_51585 = 0, time_end_51586 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50208");
            fprintf(stderr, "%zu", global_work_sizze_51584[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51588[0]);
            fprintf(stderr, "].\n");
            time_start_51585 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50208, 1, NULL,
                                              global_work_sizze_51584,
                                              local_work_sizze_51588, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51586 = get_wall_time();
            
            long time_diff_51587 = time_end_51586 - time_start_51585;
            
            ctx->map_kernel_50208_total_runtime += time_diff_51587;
            ctx->map_kernel_50208_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50208",
                    time_diff_51587);
        }
    }
    
    int32_t tile_sizze_50528;
    
    tile_sizze_50528 = ctx->sizes.tile_sizze_50527;
    
    int32_t tiled_group_sizze_50529 = tile_sizze_50528 * tile_sizze_50528;
    int32_t y_50536 = tile_sizze_50528 - 1;
    int32_t x_50537 = sizze_47946 + y_50536;
    int32_t groups_in_dim_50538 = squot32(x_50537, tile_sizze_50528);
    int32_t x_50540 = sizze_47952 + y_50536;
    int32_t groups_in_dim_50541 = squot32(x_50540, tile_sizze_50528);
    int32_t y_50543 = groups_in_dim_50538 * groups_in_dim_50541;
    int32_t num_groups_50544 = sizze_47945 * y_50543;
    int32_t num_threads_50545 = tiled_group_sizze_50529 * num_groups_50544;
    int32_t convop_x_50635 = sizze_47952 * binop_x_50621;
    int64_t binop_x_50636 = sext_i32_i64(convop_x_50635);
    int64_t bytes_50633 = 4 * binop_x_50636;
    struct memblock_device mem_50637;
    
    mem_50637.references = NULL;
    memblock_alloc_device(ctx, &mem_50637, bytes_50633, "mem_50637");
    
    int64_t binop_x_50627 = sext_i32_i64(tiled_group_sizze_50529);
    int64_t bytes_50625 = 4 * binop_x_50627;
    struct memblock_local mem_50628;
    
    mem_50628.references = NULL;
    
    struct memblock_local mem_50632;
    
    mem_50632.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 0, bytes_50625, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 1, bytes_50625, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 2, sizeof(sizze_47945),
                                  &sizze_47945));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 3, sizeof(sizze_47946),
                                  &sizze_47946));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 4, sizeof(sizze_47947),
                                  &sizze_47947));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 5, sizeof(sizze_47952),
                                  &sizze_47952));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 6,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 7,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50174, 8,
                                  sizeof(mem_50637.mem), &mem_50637.mem));
    if (1 * (num_groups_50544 * tiled_group_sizze_50529) != 0) {
        const size_t global_work_sizze_51589[1] = {num_groups_50544 *
                     tiled_group_sizze_50529};
        const size_t local_work_sizze_51593[1] = {tiled_group_sizze_50529};
        int64_t time_start_51590 = 0, time_end_51591 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50174");
            fprintf(stderr, "%zu", global_work_sizze_51589[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51593[0]);
            fprintf(stderr, "].\n");
            time_start_51590 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50174, 1, NULL,
                                              global_work_sizze_51589,
                                              local_work_sizze_51593, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51591 = get_wall_time();
            
            long time_diff_51592 = time_end_51591 - time_start_51590;
            
            ctx->map_kernel_50174_total_runtime += time_diff_51592;
            ctx->map_kernel_50174_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50174",
                    time_diff_51592);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_local(ctx, &mem_50628, "mem_50628");
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    out_arrsizze_51136 = sizze_47945;
    out_arrsizze_51137 = sizze_47946;
    out_arrsizze_51138 = sizze_47952;
    out_memsizze_51135 = bytes_50633;
    memblock_set_device(ctx, &out_mem_51134, &mem_50637, "mem_50637");
    *out_out_memsizze_51574 = out_memsizze_51135;
    (*out_mem_p_51575).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51575, &out_mem_51134,
                        "out_mem_51134");
    *out_out_arrsizze_51576 = out_arrsizze_51136;
    *out_out_arrsizze_51577 = out_arrsizze_51137;
    *out_out_arrsizze_51578 = out_arrsizze_51138;
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    memblock_unref_local(ctx, &mem_50628, "mem_50628");
    memblock_unref_device(ctx, &mem_50637, "mem_50637");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &out_mem_51134, "out_mem_51134");
    return 0;
}
static int futrts_t6_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51594,
                        struct memblock_device *out_mem_p_51595,
                        int32_t *out_out_arrsizze_51596,
                        int32_t *out_out_arrsizze_51597,
                        int32_t *out_out_arrsizze_51598,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615, int32_t sizze_48024,
                        int32_t sizze_48025, int32_t sizze_48026,
                        int32_t sizze_48027, int32_t sizze_48028,
                        int32_t sizze_48029, int32_t sizze_48030,
                        int32_t sizze_48031, int32_t sizze_48032,
                        int32_t sizze_48033, float a_48034, float b_48036,
                        float c_48038, float d_48040)
{
    int64_t out_memsizze_51154;
    struct memblock_device out_mem_51153;
    
    out_mem_51153.references = NULL;
    
    int32_t out_arrsizze_51155;
    int32_t out_arrsizze_51156;
    int32_t out_arrsizze_51157;
    bool dim_zzero_48042 = 0 == sizze_48027;
    bool dim_zzero_48043 = 0 == sizze_48028;
    bool dim_zzero_48044 = 0 == sizze_48029;
    bool y_48045 = dim_zzero_48043 || dim_zzero_48044;
    bool old_empty_48046 = dim_zzero_48042 || y_48045;
    bool dim_zzero_48047 = 0 == sizze_48024;
    bool dim_zzero_48048 = 0 == sizze_48025;
    bool dim_zzero_48049 = 0 == sizze_48026;
    bool y_48050 = dim_zzero_48048 || dim_zzero_48049;
    bool new_empty_48051 = dim_zzero_48047 || y_48050;
    bool both_empty_48052 = old_empty_48046 && new_empty_48051;
    bool dim_match_48053 = sizze_48024 == sizze_48027;
    bool dim_match_48054 = sizze_48025 == sizze_48028;
    bool dim_match_48055 = sizze_48026 == sizze_48029;
    bool y_48056 = dim_match_48054 && dim_match_48055;
    bool match_48057 = dim_match_48053 && y_48056;
    bool empty_or_match_48058 = both_empty_48052 || match_48057;
    bool empty_or_match_cert_48059;
    
    if (!empty_or_match_48058) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:120:1-123:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51153, "out_mem_51153");
        return 1;
    }
    
    bool dim_zzero_48060 = 0 == sizze_48030;
    bool dim_zzero_48061 = 0 == sizze_48031;
    bool old_empty_48062 = dim_zzero_48060 || dim_zzero_48061;
    bool new_empty_48063 = dim_zzero_48049 || dim_zzero_48061;
    bool both_empty_48064 = old_empty_48062 && new_empty_48063;
    bool dim_match_48065 = sizze_48026 == sizze_48030;
    bool empty_or_match_48066 = both_empty_48064 || dim_match_48065;
    bool empty_or_match_cert_48067;
    
    if (!empty_or_match_48066) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:120:1-123:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51153, "out_mem_51153");
        return 1;
    }
    
    bool dim_zzero_48068 = 0 == sizze_48032;
    bool dim_zzero_48069 = 0 == sizze_48033;
    bool old_empty_48070 = dim_zzero_48068 || dim_zzero_48069;
    bool both_empty_48071 = new_empty_48063 && old_empty_48070;
    bool dim_match_48072 = sizze_48026 == sizze_48032;
    bool dim_match_48073 = sizze_48031 == sizze_48033;
    bool match_48074 = dim_match_48072 && dim_match_48073;
    bool empty_or_match_48075 = both_empty_48071 || match_48074;
    bool empty_or_match_cert_48076;
    
    if (!empty_or_match_48075) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:120:1-123:81",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51153, "out_mem_51153");
        return 1;
    }
    
    int32_t nesting_sizze_50216 = sizze_48026 * sizze_48031;
    int32_t group_sizze_50218;
    
    group_sizze_50218 = ctx->sizes.group_sizze_50217;
    
    int32_t y_50219 = group_sizze_50218 - 1;
    int32_t x_50220 = nesting_sizze_50216 + y_50219;
    int32_t num_groups_50221 = squot32(x_50220, group_sizze_50218);
    int32_t num_threads_50222 = group_sizze_50218 * num_groups_50221;
    int64_t binop_x_50618 = sext_i32_i64(nesting_sizze_50216);
    int64_t bytes_50616 = 4 * binop_x_50618;
    struct memblock_device mem_50619;
    
    mem_50619.references = NULL;
    memblock_alloc_device(ctx, &mem_50619, bytes_50616, "mem_50619");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 0, sizeof(sizze_48026),
                                  &sizze_48026));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 1, sizeof(sizze_48031),
                                  &sizze_48031));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 2, sizeof(sizze_48033),
                                  &sizze_48033));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 3, sizeof(c_48038),
                                  &c_48038));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 4, sizeof(d_48040),
                                  &d_48040));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 5,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 6,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50223, 7,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    if (1 * (num_groups_50221 * group_sizze_50218) != 0) {
        const size_t global_work_sizze_51599[1] = {num_groups_50221 *
                     group_sizze_50218};
        const size_t local_work_sizze_51603[1] = {group_sizze_50218};
        int64_t time_start_51600 = 0, time_end_51601 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50223");
            fprintf(stderr, "%zu", global_work_sizze_51599[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51603[0]);
            fprintf(stderr, "].\n");
            time_start_51600 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50223, 1, NULL,
                                              global_work_sizze_51599,
                                              local_work_sizze_51603, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51601 = get_wall_time();
            
            long time_diff_51602 = time_end_51601 - time_start_51600;
            
            ctx->map_kernel_50223_total_runtime += time_diff_51602;
            ctx->map_kernel_50223_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50223",
                    time_diff_51602);
        }
    }
    
    int32_t nesting_sizze_50281 = sizze_48025 * sizze_48026;
    int32_t nesting_sizze_50282 = sizze_48024 * nesting_sizze_50281;
    int32_t group_sizze_50284;
    
    group_sizze_50284 = ctx->sizes.group_sizze_50283;
    
    int32_t y_50285 = group_sizze_50284 - 1;
    int32_t x_50286 = nesting_sizze_50282 + y_50285;
    int32_t num_groups_50287 = squot32(x_50286, group_sizze_50284);
    int32_t num_threads_50288 = group_sizze_50284 * num_groups_50287;
    int32_t binop_x_50621 = sizze_48024 * sizze_48025;
    int32_t convop_x_50622 = sizze_48026 * binop_x_50621;
    int64_t binop_x_50623 = sext_i32_i64(convop_x_50622);
    int64_t bytes_50620 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50620, "mem_50624");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 0, sizeof(sizze_48024),
                                  &sizze_48024));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 1, sizeof(sizze_48025),
                                  &sizze_48025));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 2, sizeof(sizze_48026),
                                  &sizze_48026));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 3, sizeof(sizze_48028),
                                  &sizze_48028));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 4, sizeof(sizze_48029),
                                  &sizze_48029));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 5, sizeof(a_48034),
                                  &a_48034));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 6, sizeof(b_48036),
                                  &b_48036));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 7,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 8,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50289, 9,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (num_groups_50287 * group_sizze_50284) != 0) {
        const size_t global_work_sizze_51604[1] = {num_groups_50287 *
                     group_sizze_50284};
        const size_t local_work_sizze_51608[1] = {group_sizze_50284};
        int64_t time_start_51605 = 0, time_end_51606 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50289");
            fprintf(stderr, "%zu", global_work_sizze_51604[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51608[0]);
            fprintf(stderr, "].\n");
            time_start_51605 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50289, 1, NULL,
                                              global_work_sizze_51604,
                                              local_work_sizze_51608, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51606 = get_wall_time();
            
            long time_diff_51607 = time_end_51606 - time_start_51605;
            
            ctx->map_kernel_50289_total_runtime += time_diff_51607;
            ctx->map_kernel_50289_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50289",
                    time_diff_51607);
        }
    }
    
    int32_t tile_sizze_50528;
    
    tile_sizze_50528 = ctx->sizes.tile_sizze_50527;
    
    int32_t tiled_group_sizze_50529 = tile_sizze_50528 * tile_sizze_50528;
    int32_t y_50536 = tile_sizze_50528 - 1;
    int32_t x_50537 = sizze_48025 + y_50536;
    int32_t groups_in_dim_50538 = squot32(x_50537, tile_sizze_50528);
    int32_t x_50540 = sizze_48031 + y_50536;
    int32_t groups_in_dim_50541 = squot32(x_50540, tile_sizze_50528);
    int32_t y_50543 = groups_in_dim_50538 * groups_in_dim_50541;
    int32_t num_groups_50544 = sizze_48024 * y_50543;
    int32_t num_threads_50545 = tiled_group_sizze_50529 * num_groups_50544;
    int32_t convop_x_50635 = sizze_48031 * binop_x_50621;
    int64_t binop_x_50636 = sext_i32_i64(convop_x_50635);
    int64_t bytes_50633 = 4 * binop_x_50636;
    struct memblock_device mem_50637;
    
    mem_50637.references = NULL;
    memblock_alloc_device(ctx, &mem_50637, bytes_50633, "mem_50637");
    
    int64_t binop_x_50627 = sext_i32_i64(tiled_group_sizze_50529);
    int64_t bytes_50625 = 4 * binop_x_50627;
    struct memblock_local mem_50628;
    
    mem_50628.references = NULL;
    
    struct memblock_local mem_50632;
    
    mem_50632.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 0, bytes_50625, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 1, bytes_50625, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 2, sizeof(sizze_48024),
                                  &sizze_48024));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 3, sizeof(sizze_48025),
                                  &sizze_48025));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 4, sizeof(sizze_48026),
                                  &sizze_48026));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 5, sizeof(sizze_48031),
                                  &sizze_48031));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 6,
                                  sizeof(mem_50619.mem), &mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 7,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50255, 8,
                                  sizeof(mem_50637.mem), &mem_50637.mem));
    if (1 * (num_groups_50544 * tiled_group_sizze_50529) != 0) {
        const size_t global_work_sizze_51609[1] = {num_groups_50544 *
                     tiled_group_sizze_50529};
        const size_t local_work_sizze_51613[1] = {tiled_group_sizze_50529};
        int64_t time_start_51610 = 0, time_end_51611 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50255");
            fprintf(stderr, "%zu", global_work_sizze_51609[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51613[0]);
            fprintf(stderr, "].\n");
            time_start_51610 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50255, 1, NULL,
                                              global_work_sizze_51609,
                                              local_work_sizze_51613, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51611 = get_wall_time();
            
            long time_diff_51612 = time_end_51611 - time_start_51610;
            
            ctx->map_kernel_50255_total_runtime += time_diff_51612;
            ctx->map_kernel_50255_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50255",
                    time_diff_51612);
        }
    }
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_local(ctx, &mem_50628, "mem_50628");
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    out_arrsizze_51155 = sizze_48024;
    out_arrsizze_51156 = sizze_48025;
    out_arrsizze_51157 = sizze_48031;
    out_memsizze_51154 = bytes_50633;
    memblock_set_device(ctx, &out_mem_51153, &mem_50637, "mem_50637");
    *out_out_memsizze_51594 = out_memsizze_51154;
    (*out_mem_p_51595).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51595, &out_mem_51153,
                        "out_mem_51153");
    *out_out_arrsizze_51596 = out_arrsizze_51155;
    *out_out_arrsizze_51597 = out_arrsizze_51156;
    *out_out_arrsizze_51598 = out_arrsizze_51157;
    memblock_unref_local(ctx, &mem_50632, "mem_50632");
    memblock_unref_local(ctx, &mem_50628, "mem_50628");
    memblock_unref_device(ctx, &mem_50637, "mem_50637");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &mem_50619, "mem_50619");
    memblock_unref_device(ctx, &out_mem_51153, "out_mem_51153");
    return 0;
}
static int futrts_t7_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51614,
                        struct memblock_device *out_mem_p_51615,
                        int32_t *out_out_arrsizze_51616,
                        int32_t *out_out_arrsizze_51617,
                        int32_t *out_out_arrsizze_51618,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613,
                        int64_t D_mem_sizze_50614,
                        struct memblock_device D_mem_50615,
                        int64_t E_mem_sizze_50616,
                        struct memblock_device E_mem_50617,
                        int64_t F_mem_sizze_50618,
                        struct memblock_device F_mem_50619,
                        int64_t G_mem_sizze_50620,
                        struct memblock_device G_mem_50621,
                        int64_t H_mem_sizze_50622,
                        struct memblock_device H_mem_50623, int32_t sizze_48111,
                        int32_t sizze_48112, int32_t sizze_48113,
                        int32_t sizze_48114, int32_t sizze_48115,
                        int32_t sizze_48116, int32_t sizze_48117,
                        int32_t sizze_48118, int32_t sizze_48119,
                        int32_t sizze_48120, int32_t sizze_48121,
                        int32_t sizze_48122, int32_t sizze_48123,
                        int32_t sizze_48124, float a_48125, float b_48127,
                        float c_48129, float d_48131, float e_48133,
                        float f_48135, float g_48137, float h_48139)
{
    int64_t out_memsizze_51173;
    struct memblock_device out_mem_51172;
    
    out_mem_51172.references = NULL;
    
    int32_t out_arrsizze_51174;
    int32_t out_arrsizze_51175;
    int32_t out_arrsizze_51176;
    bool dim_zzero_48141 = 0 == sizze_48114;
    bool dim_zzero_48142 = 0 == sizze_48115;
    bool dim_zzero_48143 = 0 == sizze_48116;
    bool y_48144 = dim_zzero_48142 || dim_zzero_48143;
    bool old_empty_48145 = dim_zzero_48141 || y_48144;
    bool dim_zzero_48146 = 0 == sizze_48111;
    bool dim_zzero_48147 = 0 == sizze_48112;
    bool dim_zzero_48148 = 0 == sizze_48113;
    bool y_48149 = dim_zzero_48147 || dim_zzero_48148;
    bool new_empty_48150 = dim_zzero_48146 || y_48149;
    bool both_empty_48151 = old_empty_48145 && new_empty_48150;
    bool dim_match_48152 = sizze_48111 == sizze_48114;
    bool dim_match_48153 = sizze_48112 == sizze_48115;
    bool dim_match_48154 = sizze_48113 == sizze_48116;
    bool y_48155 = dim_match_48153 && dim_match_48154;
    bool match_48156 = dim_match_48152 && y_48155;
    bool empty_or_match_48157 = both_empty_48151 || match_48156;
    bool empty_or_match_cert_48158;
    
    if (!empty_or_match_48157) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48159 = 0 == sizze_48117;
    bool dim_zzero_48160 = 0 == sizze_48118;
    bool old_empty_48161 = dim_zzero_48159 || dim_zzero_48160;
    bool new_empty_48162 = dim_zzero_48148 || dim_zzero_48160;
    bool both_empty_48163 = old_empty_48161 && new_empty_48162;
    bool dim_match_48164 = sizze_48113 == sizze_48117;
    bool empty_or_match_48165 = both_empty_48163 || dim_match_48164;
    bool empty_or_match_cert_48166;
    
    if (!empty_or_match_48165) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48167 = 0 == sizze_48119;
    bool dim_zzero_48168 = 0 == sizze_48120;
    bool old_empty_48169 = dim_zzero_48167 || dim_zzero_48168;
    bool both_empty_48170 = new_empty_48162 && old_empty_48169;
    bool dim_match_48171 = sizze_48113 == sizze_48119;
    bool dim_match_48172 = sizze_48118 == sizze_48120;
    bool match_48173 = dim_match_48171 && dim_match_48172;
    bool empty_or_match_48174 = both_empty_48170 || match_48173;
    bool empty_or_match_cert_48175;
    
    if (!empty_or_match_48174) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48176 = 0 == sizze_48121;
    bool both_empty_48177 = dim_zzero_48148 && dim_zzero_48176;
    bool dim_match_48178 = sizze_48113 == sizze_48121;
    bool empty_or_match_48179 = both_empty_48177 || dim_match_48178;
    bool empty_or_match_cert_48180;
    
    if (!empty_or_match_48179) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48181 = 0 == sizze_48122;
    bool both_empty_48182 = dim_zzero_48148 && dim_zzero_48181;
    bool dim_match_48183 = sizze_48113 == sizze_48122;
    bool empty_or_match_48184 = both_empty_48182 || dim_match_48183;
    bool empty_or_match_cert_48185;
    
    if (!empty_or_match_48184) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48186 = 0 == sizze_48123;
    bool both_empty_48187 = dim_zzero_48147 && dim_zzero_48186;
    bool dim_match_48188 = sizze_48112 == sizze_48123;
    bool empty_or_match_48189 = both_empty_48187 || dim_match_48188;
    bool empty_or_match_cert_48190;
    
    if (!empty_or_match_48189) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    bool dim_zzero_48191 = 0 == sizze_48124;
    bool both_empty_48192 = dim_zzero_48147 && dim_zzero_48191;
    bool dim_match_48193 = sizze_48112 == sizze_48124;
    bool empty_or_match_48194 = both_empty_48192 || dim_match_48193;
    bool empty_or_match_cert_48195;
    
    if (!empty_or_match_48194) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:125:1-130:126",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
        return 1;
    }
    
    int32_t nesting_sizze_50314 = sizze_48113 * sizze_48118;
    int32_t group_sizze_50316;
    
    group_sizze_50316 = ctx->sizes.group_sizze_50315;
    
    int32_t y_50317 = group_sizze_50316 - 1;
    int32_t x_50318 = nesting_sizze_50314 + y_50317;
    int32_t num_groups_50319 = squot32(x_50318, group_sizze_50316);
    int32_t num_threads_50320 = group_sizze_50316 * num_groups_50319;
    int64_t binop_x_50626 = sext_i32_i64(nesting_sizze_50314);
    int64_t bytes_50624 = 4 * binop_x_50626;
    struct memblock_device mem_50627;
    
    mem_50627.references = NULL;
    memblock_alloc_device(ctx, &mem_50627, bytes_50624, "mem_50627");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 0, sizeof(sizze_48113),
                                  &sizze_48113));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 1, sizeof(sizze_48118),
                                  &sizze_48118));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 2, sizeof(sizze_48120),
                                  &sizze_48120));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 3, sizeof(c_48129),
                                  &c_48129));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 4, sizeof(d_48131),
                                  &d_48131));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 5,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 6,
                                  sizeof(D_mem_50615.mem), &D_mem_50615.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50321, 7,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    if (1 * (num_groups_50319 * group_sizze_50316) != 0) {
        const size_t global_work_sizze_51619[1] = {num_groups_50319 *
                     group_sizze_50316};
        const size_t local_work_sizze_51623[1] = {group_sizze_50316};
        int64_t time_start_51620 = 0, time_end_51621 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50321");
            fprintf(stderr, "%zu", global_work_sizze_51619[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51623[0]);
            fprintf(stderr, "].\n");
            time_start_51620 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50321, 1, NULL,
                                              global_work_sizze_51619,
                                              local_work_sizze_51623, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51621 = get_wall_time();
            
            long time_diff_51622 = time_end_51621 - time_start_51620;
            
            ctx->map_kernel_50321_total_runtime += time_diff_51622;
            ctx->map_kernel_50321_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50321",
                    time_diff_51622);
        }
    }
    
    int32_t group_sizze_50299;
    
    group_sizze_50299 = ctx->sizes.group_sizze_50298;
    
    int32_t y_50300 = group_sizze_50299 - 1;
    int32_t x_50301 = sizze_48113 + y_50300;
    int32_t num_groups_50302 = squot32(x_50301, group_sizze_50299);
    int32_t num_threads_50303 = group_sizze_50299 * num_groups_50302;
    int64_t binop_x_50629 = sext_i32_i64(sizze_48113);
    int64_t bytes_50628 = 4 * binop_x_50629;
    struct memblock_device mem_50630;
    
    mem_50630.references = NULL;
    memblock_alloc_device(ctx, &mem_50630, bytes_50628, "mem_50630");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 0, sizeof(sizze_48113),
                                  &sizze_48113));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 1, sizeof(e_48133),
                                  &e_48133));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 2, sizeof(f_48135),
                                  &f_48135));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 3,
                                  sizeof(E_mem_50617.mem), &E_mem_50617.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 4,
                                  sizeof(F_mem_50619.mem), &F_mem_50619.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50304, 5,
                                  sizeof(mem_50630.mem), &mem_50630.mem));
    if (1 * (num_groups_50302 * group_sizze_50299) != 0) {
        const size_t global_work_sizze_51624[1] = {num_groups_50302 *
                     group_sizze_50299};
        const size_t local_work_sizze_51628[1] = {group_sizze_50299};
        int64_t time_start_51625 = 0, time_end_51626 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50304");
            fprintf(stderr, "%zu", global_work_sizze_51624[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51628[0]);
            fprintf(stderr, "].\n");
            time_start_51625 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50304, 1, NULL,
                                              global_work_sizze_51624,
                                              local_work_sizze_51628, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51626 = get_wall_time();
            
            long time_diff_51627 = time_end_51626 - time_start_51625;
            
            ctx->map_kernel_50304_total_runtime += time_diff_51627;
            ctx->map_kernel_50304_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50304",
                    time_diff_51627);
        }
    }
    
    int32_t group_sizze_50331;
    
    group_sizze_50331 = ctx->sizes.group_sizze_50330;
    
    int32_t y_50332 = group_sizze_50331 - 1;
    int32_t x_50333 = sizze_48112 + y_50332;
    int32_t num_groups_50334 = squot32(x_50333, group_sizze_50331);
    int32_t num_threads_50335 = group_sizze_50331 * num_groups_50334;
    int64_t binop_x_50632 = sext_i32_i64(sizze_48112);
    int64_t bytes_50631 = 4 * binop_x_50632;
    struct memblock_device mem_50633;
    
    mem_50633.references = NULL;
    memblock_alloc_device(ctx, &mem_50633, bytes_50631, "mem_50633");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 0, sizeof(sizze_48112),
                                  &sizze_48112));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 1, sizeof(g_48137),
                                  &g_48137));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 2, sizeof(h_48139),
                                  &h_48139));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 3,
                                  sizeof(G_mem_50621.mem), &G_mem_50621.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 4,
                                  sizeof(H_mem_50623.mem), &H_mem_50623.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50336, 5,
                                  sizeof(mem_50633.mem), &mem_50633.mem));
    if (1 * (num_groups_50334 * group_sizze_50331) != 0) {
        const size_t global_work_sizze_51629[1] = {num_groups_50334 *
                     group_sizze_50331};
        const size_t local_work_sizze_51633[1] = {group_sizze_50331};
        int64_t time_start_51630 = 0, time_end_51631 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50336");
            fprintf(stderr, "%zu", global_work_sizze_51629[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51633[0]);
            fprintf(stderr, "].\n");
            time_start_51630 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50336, 1, NULL,
                                              global_work_sizze_51629,
                                              local_work_sizze_51633, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51631 = get_wall_time();
            
            long time_diff_51632 = time_end_51631 - time_start_51630;
            
            ctx->map_kernel_50336_total_runtime += time_diff_51632;
            ctx->map_kernel_50336_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50336",
                    time_diff_51632);
        }
    }
    
    int32_t nesting_sizze_50402 = sizze_48112 * sizze_48113;
    int32_t nesting_sizze_50403 = sizze_48111 * nesting_sizze_50402;
    int32_t group_sizze_50405;
    
    group_sizze_50405 = ctx->sizes.group_sizze_50404;
    
    int32_t y_50406 = group_sizze_50405 - 1;
    int32_t x_50407 = nesting_sizze_50403 + y_50406;
    int32_t num_groups_50408 = squot32(x_50407, group_sizze_50405);
    int32_t num_threads_50409 = group_sizze_50405 * num_groups_50408;
    int32_t binop_x_50635 = sizze_48111 * sizze_48112;
    int32_t convop_x_50636 = sizze_48113 * binop_x_50635;
    int64_t binop_x_50637 = sext_i32_i64(convop_x_50636);
    int64_t bytes_50634 = 4 * binop_x_50637;
    struct memblock_device mem_50638;
    
    mem_50638.references = NULL;
    memblock_alloc_device(ctx, &mem_50638, bytes_50634, "mem_50638");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 0, sizeof(sizze_48111),
                                  &sizze_48111));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 1, sizeof(sizze_48112),
                                  &sizze_48112));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 2, sizeof(sizze_48113),
                                  &sizze_48113));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 3, sizeof(sizze_48115),
                                  &sizze_48115));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 4, sizeof(sizze_48116),
                                  &sizze_48116));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 5, sizeof(a_48125),
                                  &a_48125));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 6, sizeof(b_48127),
                                  &b_48127));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 7,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 8,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50410, 9,
                                  sizeof(mem_50638.mem), &mem_50638.mem));
    if (1 * (num_groups_50408 * group_sizze_50405) != 0) {
        const size_t global_work_sizze_51634[1] = {num_groups_50408 *
                     group_sizze_50405};
        const size_t local_work_sizze_51638[1] = {group_sizze_50405};
        int64_t time_start_51635 = 0, time_end_51636 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50410");
            fprintf(stderr, "%zu", global_work_sizze_51634[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51638[0]);
            fprintf(stderr, "].\n");
            time_start_51635 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50410, 1, NULL,
                                              global_work_sizze_51634,
                                              local_work_sizze_51638, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51636 = get_wall_time();
            
            long time_diff_51637 = time_end_51636 - time_start_51635;
            
            ctx->map_kernel_50410_total_runtime += time_diff_51637;
            ctx->map_kernel_50410_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50410",
                    time_diff_51637);
        }
    }
    
    int32_t nesting_sizze_50362 = sizze_48112 * sizze_48118;
    int32_t nesting_sizze_50363 = sizze_48111 * nesting_sizze_50362;
    int32_t group_sizze_50365;
    
    group_sizze_50365 = ctx->sizes.group_sizze_50364;
    
    int32_t y_50366 = group_sizze_50365 - 1;
    int32_t x_50367 = nesting_sizze_50363 + y_50366;
    int32_t num_groups_50368 = squot32(x_50367, group_sizze_50365);
    int32_t num_threads_50369 = group_sizze_50365 * num_groups_50368;
    int32_t convop_x_50641 = sizze_48118 * binop_x_50635;
    int64_t binop_x_50642 = sext_i32_i64(convop_x_50641);
    int64_t bytes_50639 = 4 * binop_x_50642;
    struct memblock_device mem_50643;
    
    mem_50643.references = NULL;
    memblock_alloc_device(ctx, &mem_50643, bytes_50639, "mem_50643");
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 0, sizeof(sizze_48111),
                                  &sizze_48111));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 1, sizeof(sizze_48112),
                                  &sizze_48112));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 2, sizeof(sizze_48113),
                                  &sizze_48113));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 3, sizeof(sizze_48118),
                                  &sizze_48118));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 4,
                                  sizeof(mem_50627.mem), &mem_50627.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 5,
                                  sizeof(mem_50630.mem), &mem_50630.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 6,
                                  sizeof(mem_50633.mem), &mem_50633.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 7,
                                  sizeof(mem_50638.mem), &mem_50638.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50370, 8,
                                  sizeof(mem_50643.mem), &mem_50643.mem));
    if (1 * (num_groups_50368 * group_sizze_50365) != 0) {
        const size_t global_work_sizze_51639[1] = {num_groups_50368 *
                     group_sizze_50365};
        const size_t local_work_sizze_51643[1] = {group_sizze_50365};
        int64_t time_start_51640 = 0, time_end_51641 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50370");
            fprintf(stderr, "%zu", global_work_sizze_51639[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51643[0]);
            fprintf(stderr, "].\n");
            time_start_51640 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50370, 1, NULL,
                                              global_work_sizze_51639,
                                              local_work_sizze_51643, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51641 = get_wall_time();
            
            long time_diff_51642 = time_end_51641 - time_start_51640;
            
            ctx->map_kernel_50370_total_runtime += time_diff_51642;
            ctx->map_kernel_50370_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50370",
                    time_diff_51642);
        }
    }
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    memblock_unref_device(ctx, &mem_50633, "mem_50633");
    memblock_unref_device(ctx, &mem_50638, "mem_50638");
    out_arrsizze_51174 = sizze_48111;
    out_arrsizze_51175 = sizze_48112;
    out_arrsizze_51176 = sizze_48118;
    out_memsizze_51173 = bytes_50639;
    memblock_set_device(ctx, &out_mem_51172, &mem_50643, "mem_50643");
    *out_out_memsizze_51614 = out_memsizze_51173;
    (*out_mem_p_51615).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51615, &out_mem_51172,
                        "out_mem_51172");
    *out_out_arrsizze_51616 = out_arrsizze_51174;
    *out_out_arrsizze_51617 = out_arrsizze_51175;
    *out_out_arrsizze_51618 = out_arrsizze_51176;
    memblock_unref_device(ctx, &mem_50643, "mem_50643");
    memblock_unref_device(ctx, &mem_50638, "mem_50638");
    memblock_unref_device(ctx, &mem_50633, "mem_50633");
    memblock_unref_device(ctx, &mem_50630, "mem_50630");
    memblock_unref_device(ctx, &mem_50627, "mem_50627");
    memblock_unref_device(ctx, &out_mem_51172, "out_mem_51172");
    return 0;
}
static int futrts_t8_3d(struct futhark_context *ctx,
                        int64_t *out_out_memsizze_51644,
                        struct memblock_device *out_mem_p_51645,
                        int32_t *out_out_arrsizze_51646,
                        int32_t *out_out_arrsizze_51647,
                        int32_t *out_out_arrsizze_51648,
                        int64_t A_mem_sizze_50608,
                        struct memblock_device A_mem_50609,
                        int64_t B_mem_sizze_50610,
                        struct memblock_device B_mem_50611,
                        int64_t C_mem_sizze_50612,
                        struct memblock_device C_mem_50613, int32_t sizze_48250,
                        int32_t sizze_48251, int32_t sizze_48252,
                        int32_t sizze_48253, int32_t sizze_48254,
                        int32_t sizze_48255, int32_t sizze_48256)
{
    int64_t out_memsizze_51194;
    struct memblock_device out_mem_51193;
    
    out_mem_51193.references = NULL;
    
    int32_t out_arrsizze_51195;
    int32_t out_arrsizze_51196;
    int32_t out_arrsizze_51197;
    bool dim_zzero_48260 = 0 == sizze_48253;
    bool dim_zzero_48261 = 0 == sizze_48254;
    bool old_empty_48262 = dim_zzero_48260 || dim_zzero_48261;
    bool dim_zzero_48263 = 0 == sizze_48252;
    bool new_empty_48264 = dim_zzero_48261 || dim_zzero_48263;
    bool both_empty_48265 = old_empty_48262 && new_empty_48264;
    bool dim_match_48266 = sizze_48252 == sizze_48253;
    bool empty_or_match_48267 = both_empty_48265 || dim_match_48266;
    bool empty_or_match_cert_48268;
    
    if (!empty_or_match_48267) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:132:1-135:41",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51193, "out_mem_51193");
        return 1;
    }
    
    bool dim_zzero_48270 = 0 == sizze_48255;
    bool dim_zzero_48271 = 0 == sizze_48256;
    bool old_empty_48272 = dim_zzero_48270 || dim_zzero_48271;
    bool dim_zzero_48273 = 0 == sizze_48251;
    bool new_empty_48274 = dim_zzero_48271 || dim_zzero_48273;
    bool both_empty_48275 = old_empty_48272 && new_empty_48274;
    bool dim_match_48276 = sizze_48251 == sizze_48255;
    bool empty_or_match_48277 = both_empty_48275 || dim_match_48276;
    bool empty_or_match_cert_48278;
    
    if (!empty_or_match_48277) {
        ctx->error = msgprintf("Error at %s:\n%s\n",
                               "futhark_test.fut:132:1-135:41",
                               "function arguments of wrong shape");
        memblock_unref_device(ctx, &out_mem_51193, "out_mem_51193");
        return 1;
    }
    
    int32_t group_sizze_50446;
    
    group_sizze_50446 = ctx->sizes.group_sizze_50445;
    
    int32_t inner_ldim_50533 = smin32(sizze_48256, group_sizze_50446);
    int32_t y_50535 = sizze_48254 * sizze_48256;
    int32_t threads_necessary_50536 = sizze_48250 * y_50535;
    int32_t y_50537 = inner_ldim_50533 - 1;
    int32_t x_50538 = threads_necessary_50536 + y_50537;
    int32_t groups_necessary_50539 = squot32(x_50538, inner_ldim_50533);
    int32_t num_threads_50540 = inner_ldim_50533 * groups_necessary_50539;
    int32_t binop_x_50621 = sizze_48250 * sizze_48254;
    int32_t convop_x_50622 = sizze_48256 * binop_x_50621;
    int64_t binop_x_50623 = sext_i32_i64(convop_x_50622);
    int64_t bytes_50620 = 4 * binop_x_50623;
    struct memblock_device mem_50624;
    
    mem_50624.references = NULL;
    memblock_alloc_device(ctx, &mem_50624, bytes_50620, "mem_50624");
    
    int64_t binop_x_50615 = sext_i32_i64(group_sizze_50446);
    int64_t bytes_50614 = 4 * binop_x_50615;
    struct memblock_local mem_50616;
    
    mem_50616.references = NULL;
    
    struct memblock_local mem_50619;
    
    mem_50619.references = NULL;
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 0, bytes_50614, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 1, bytes_50614, NULL));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 2, sizeof(sizze_48250),
                                  &sizze_48250));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 3, sizeof(sizze_48251),
                                  &sizze_48251));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 4, sizeof(sizze_48252),
                                  &sizze_48252));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 5, sizeof(sizze_48254),
                                  &sizze_48254));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 6, sizeof(sizze_48256),
                                  &sizze_48256));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 7,
                                  sizeof(inner_ldim_50533), &inner_ldim_50533));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 8,
                                  sizeof(A_mem_50609.mem), &A_mem_50609.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 9,
                                  sizeof(B_mem_50611.mem), &B_mem_50611.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 10,
                                  sizeof(C_mem_50613.mem), &C_mem_50613.mem));
    OPENCL_SUCCEED(clSetKernelArg(ctx->map_kernel_50451, 11,
                                  sizeof(mem_50624.mem), &mem_50624.mem));
    if (1 * (groups_necessary_50539 * inner_ldim_50533) != 0) {
        const size_t global_work_sizze_51649[1] = {groups_necessary_50539 *
                     inner_ldim_50533};
        const size_t local_work_sizze_51653[1] = {inner_ldim_50533};
        int64_t time_start_51650 = 0, time_end_51651 = 0;
        
        if (ctx->debugging) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_50451");
            fprintf(stderr, "%zu", global_work_sizze_51649[0]);
            fprintf(stderr, "] and local work size [");
            fprintf(stderr, "%zu", local_work_sizze_51653[0]);
            fprintf(stderr, "].\n");
            time_start_51650 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(ctx->opencl.queue,
                                              ctx->map_kernel_50451, 1, NULL,
                                              global_work_sizze_51649,
                                              local_work_sizze_51653, 0, NULL,
                                              NULL));
        if (ctx->debugging) {
            OPENCL_SUCCEED(clFinish(ctx->opencl.queue));
            time_end_51651 = get_wall_time();
            
            long time_diff_51652 = time_end_51651 - time_start_51650;
            
            ctx->map_kernel_50451_total_runtime += time_diff_51652;
            ctx->map_kernel_50451_runs++;
            fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_50451",
                    time_diff_51652);
        }
    }
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    memblock_unref_local(ctx, &mem_50619, "mem_50619");
    out_arrsizze_51195 = sizze_48250;
    out_arrsizze_51196 = sizze_48254;
    out_arrsizze_51197 = sizze_48256;
    out_memsizze_51194 = bytes_50620;
    memblock_set_device(ctx, &out_mem_51193, &mem_50624, "mem_50624");
    *out_out_memsizze_51644 = out_memsizze_51194;
    (*out_mem_p_51645).references = NULL;
    memblock_set_device(ctx, &*out_mem_p_51645, &out_mem_51193,
                        "out_mem_51193");
    *out_out_arrsizze_51646 = out_arrsizze_51195;
    *out_out_arrsizze_51647 = out_arrsizze_51196;
    *out_out_arrsizze_51648 = out_arrsizze_51197;
    memblock_unref_local(ctx, &mem_50619, "mem_50619");
    memblock_unref_local(ctx, &mem_50616, "mem_50616");
    memblock_unref_device(ctx, &mem_50624, "mem_50624");
    memblock_unref_device(ctx, &out_mem_51193, "out_mem_51193");
    return 0;
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
int futhark_entry_sum(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int32_t sizze_46897;
    float scalar_out_50763;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_46897 = in0->shape[0];
    
    int ret = futrts_sum(ctx, &scalar_out_50763, A_mem_sizze_50608, A_mem_50609,
                         sizze_46897);
    
    if (ret == 0) {
        *out0 = scalar_out_50763;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_prod(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const float in0, const
                       struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int32_t sizze_46904;
    float c_46905;
    int64_t out_memsizze_50780;
    struct memblock_device out_mem_50779;
    
    out_mem_50779.references = NULL;
    
    int32_t out_arrsizze_50781;
    
    lock_lock(&ctx->lock);
    c_46905 = in0;
    A_mem_50609 = in1->mem;
    A_mem_sizze_50608 = in1->mem.size;
    sizze_46904 = in1->shape[0];
    
    int ret = futrts_prod(ctx, &out_memsizze_50780, &out_mem_50779,
                          &out_arrsizze_50781, A_mem_sizze_50608, A_mem_50609,
                          sizze_46904, c_46905);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50779;
        (*out0)->shape[0] = out_arrsizze_50781;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int32_t sizze_46910;
    int32_t sizze_46911;
    float scalar_out_50785;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_46910 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_46911 = in1->shape[0];
    
    int ret = futrts_dot(ctx, &scalar_out_50785, A_mem_sizze_50608, A_mem_50609,
                         B_mem_sizze_50610, B_mem_50611, sizze_46910,
                         sizze_46911);
    
    if (ret == 0) {
        *out0 = scalar_out_50785;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot2(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_46928;
    int32_t sizze_46929;
    int32_t sizze_46930;
    float scalar_out_50801;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_46928 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_46929 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_46930 = in2->shape[0];
    
    int ret = futrts_dot2(ctx, &scalar_out_50801, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, sizze_46928,
                          sizze_46929, sizze_46930);
    
    if (ret == 0) {
        *out0 = scalar_out_50801;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot3(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_46956;
    int32_t sizze_46957;
    int32_t sizze_46958;
    float scalar_out_50817;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_46956 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_46957 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_46958 = in2->shape[0];
    
    int ret = futrts_dot3(ctx, &scalar_out_50817, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, sizze_46956,
                          sizze_46957, sizze_46958);
    
    if (ret == 0) {
        *out0 = scalar_out_50817;
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
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_46985;
    int32_t sizze_46986;
    int32_t sizze_46987;
    int32_t sizze_46988;
    float scalar_out_50833;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_46985 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_46986 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_46987 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_46988 = in3->shape[0];
    
    int ret = futrts_dot4(ctx, &scalar_out_50833, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, D_mem_sizze_50614,
                          D_mem_50615, sizze_46985, sizze_46986, sizze_46987,
                          sizze_46988);
    
    if (ret == 0) {
        *out0 = scalar_out_50833;
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
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47023;
    int32_t sizze_47024;
    int32_t sizze_47025;
    int32_t sizze_47026;
    float a_47027;
    float b_47029;
    float c_47031;
    float d_47033;
    float scalar_out_50849;
    
    lock_lock(&ctx->lock);
    a_47027 = in0;
    A_mem_50609 = in1->mem;
    A_mem_sizze_50608 = in1->mem.size;
    sizze_47023 = in1->shape[0];
    b_47029 = in2;
    B_mem_50611 = in3->mem;
    B_mem_sizze_50610 = in3->mem.size;
    sizze_47024 = in3->shape[0];
    c_47031 = in4;
    C_mem_50613 = in5->mem;
    C_mem_sizze_50612 = in5->mem.size;
    sizze_47025 = in5->shape[0];
    d_47033 = in6;
    D_mem_50615 = in7->mem;
    D_mem_sizze_50614 = in7->mem.size;
    sizze_47026 = in7->shape[0];
    
    int ret = futrts_dot5(ctx, &scalar_out_50849, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, D_mem_sizze_50614,
                          D_mem_50615, sizze_47023, sizze_47024, sizze_47025,
                          sizze_47026, a_47027, b_47029, c_47031, d_47033);
    
    if (ret == 0) {
        *out0 = scalar_out_50849;
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
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47069;
    int32_t sizze_47070;
    int32_t sizze_47071;
    int32_t sizze_47072;
    float scalar_out_50865;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47069 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47070 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47071 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47072 = in3->shape[0];
    
    int ret = futrts_dot6(ctx, &scalar_out_50865, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, D_mem_sizze_50614,
                          D_mem_50615, sizze_47069, sizze_47070, sizze_47071,
                          sizze_47072);
    
    if (ret == 0) {
        *out0 = scalar_out_50865;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47107;
    int32_t sizze_47108;
    int32_t sizze_47109;
    float scalar_out_50881;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47107 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47108 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47109 = in2->shape[0];
    
    int ret = futrts_dot1(ctx, &scalar_out_50881, A_mem_sizze_50608,
                          A_mem_50609, B_mem_sizze_50610, B_mem_50611,
                          C_mem_sizze_50612, C_mem_50613, sizze_47107,
                          sizze_47108, sizze_47109);
    
    if (ret == 0) {
        *out0 = scalar_out_50881;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t1b_2d(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int32_t sizze_47135;
    int32_t sizze_47136;
    int32_t sizze_47137;
    int64_t out_memsizze_50898;
    struct memblock_device out_mem_50897;
    
    out_mem_50897.references = NULL;
    
    int32_t out_arrsizze_50899;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47135 = in0->shape[0];
    sizze_47136 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47137 = in1->shape[0];
    
    int ret = futrts_t1b_2d(ctx, &out_memsizze_50898, &out_mem_50897,
                            &out_arrsizze_50899, A_mem_sizze_50608, A_mem_50609,
                            B_mem_sizze_50610, B_mem_50611, sizze_47135,
                            sizze_47136, sizze_47137);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50897;
        (*out0)->shape[0] = out_arrsizze_50899;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t1_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_1d *in1)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int32_t sizze_47160;
    int32_t sizze_47161;
    int32_t sizze_47162;
    int64_t out_memsizze_50944;
    struct memblock_device out_mem_50943;
    
    out_mem_50943.references = NULL;
    
    int32_t out_arrsizze_50945;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47160 = in0->shape[0];
    sizze_47161 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47162 = in1->shape[0];
    
    int ret = futrts_t1_2d(ctx, &out_memsizze_50944, &out_mem_50943,
                           &out_arrsizze_50945, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, sizze_47160,
                           sizze_47161, sizze_47162);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50943;
        (*out0)->shape[0] = out_arrsizze_50945;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t2_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47181;
    int32_t sizze_47182;
    int32_t sizze_47183;
    int32_t sizze_47184;
    int64_t out_memsizze_50951;
    struct memblock_device out_mem_50950;
    
    out_mem_50950.references = NULL;
    
    int32_t out_arrsizze_50952;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47181 = in0->shape[0];
    sizze_47182 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47183 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47184 = in2->shape[0];
    
    int ret = futrts_t2_2d(ctx, &out_memsizze_50951, &out_mem_50950,
                           &out_arrsizze_50952, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, sizze_47181, sizze_47182, sizze_47183,
                           sizze_47184);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50950;
        (*out0)->shape[0] = out_arrsizze_50952;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t3_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47213;
    int32_t sizze_47214;
    int32_t sizze_47215;
    int32_t sizze_47216;
    int32_t sizze_47217;
    int64_t out_memsizze_50958;
    struct memblock_device out_mem_50957;
    
    out_mem_50957.references = NULL;
    
    int32_t out_arrsizze_50959;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47213 = in0->shape[0];
    sizze_47214 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47215 = in1->shape[0];
    sizze_47216 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47217 = in2->shape[0];
    
    int ret = futrts_t3_2d(ctx, &out_memsizze_50958, &out_mem_50957,
                           &out_arrsizze_50959, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, sizze_47213, sizze_47214, sizze_47215,
                           sizze_47216, sizze_47217);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50957;
        (*out0)->shape[0] = out_arrsizze_50959;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t4_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47252;
    int32_t sizze_47253;
    int32_t sizze_47254;
    int32_t sizze_47255;
    int32_t sizze_47256;
    int32_t sizze_47257;
    int64_t out_memsizze_50965;
    struct memblock_device out_mem_50964;
    
    out_mem_50964.references = NULL;
    
    int32_t out_arrsizze_50966;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47252 = in0->shape[0];
    sizze_47253 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47254 = in1->shape[0];
    sizze_47255 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47256 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47257 = in3->shape[0];
    
    int ret = futrts_t4_2d(ctx, &out_memsizze_50965, &out_mem_50964,
                           &out_arrsizze_50966, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47252, sizze_47253, sizze_47254, sizze_47255,
                           sizze_47256, sizze_47257);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50964;
        (*out0)->shape[0] = out_arrsizze_50966;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t5_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const float in0, const
                        struct futhark_f32_2d *in1, const float in2, const
                        struct futhark_f32_2d *in3, const float in4, const
                        struct futhark_f32_1d *in5, const float in6, const
                        struct futhark_f32_1d *in7)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47303;
    int32_t sizze_47304;
    int32_t sizze_47305;
    int32_t sizze_47306;
    int32_t sizze_47307;
    int32_t sizze_47308;
    float a_47309;
    float b_47311;
    float c_47313;
    float d_47315;
    int64_t out_memsizze_50975;
    struct memblock_device out_mem_50974;
    
    out_mem_50974.references = NULL;
    
    int32_t out_arrsizze_50976;
    
    lock_lock(&ctx->lock);
    a_47309 = in0;
    A_mem_50609 = in1->mem;
    A_mem_sizze_50608 = in1->mem.size;
    sizze_47303 = in1->shape[0];
    sizze_47304 = in1->shape[1];
    b_47311 = in2;
    B_mem_50611 = in3->mem;
    B_mem_sizze_50610 = in3->mem.size;
    sizze_47305 = in3->shape[0];
    sizze_47306 = in3->shape[1];
    c_47313 = in4;
    C_mem_50613 = in5->mem;
    C_mem_sizze_50612 = in5->mem.size;
    sizze_47307 = in5->shape[0];
    d_47315 = in6;
    D_mem_50615 = in7->mem;
    D_mem_sizze_50614 = in7->mem.size;
    sizze_47308 = in7->shape[0];
    
    int ret = futrts_t5_2d(ctx, &out_memsizze_50975, &out_mem_50974,
                           &out_arrsizze_50976, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47303, sizze_47304, sizze_47305, sizze_47306,
                           sizze_47307, sizze_47308, a_47309, b_47311, c_47313,
                           d_47315);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50974;
        (*out0)->shape[0] = out_arrsizze_50976;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t6_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_1d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47362;
    int32_t sizze_47363;
    int32_t sizze_47364;
    int32_t sizze_47365;
    int64_t out_memsizze_50985;
    struct memblock_device out_mem_50984;
    
    out_mem_50984.references = NULL;
    
    int32_t out_arrsizze_50986;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47362 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47363 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47364 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47365 = in3->shape[0];
    
    int ret = futrts_t6_2d(ctx, &out_memsizze_50985, &out_mem_50984,
                           &out_arrsizze_50986, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47362, sizze_47363, sizze_47364, sizze_47365);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50984;
        (*out0)->shape[0] = out_arrsizze_50986;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t7_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t C_mem_sizze_50610;
    struct memblock_device C_mem_50611;
    
    C_mem_50611.references = NULL;
    
    int64_t D_mem_sizze_50612;
    struct memblock_device D_mem_50613;
    
    D_mem_50613.references = NULL;
    
    int32_t sizze_47396;
    int32_t sizze_47397;
    int32_t sizze_47398;
    int32_t sizze_47399;
    int32_t sizze_47400;
    int64_t out_memsizze_50996;
    struct memblock_device out_mem_50995;
    
    out_mem_50995.references = NULL;
    
    int32_t out_arrsizze_50997;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47396 = in0->shape[0];
    sizze_47397 = in0->shape[1];
    C_mem_50611 = in1->mem;
    C_mem_sizze_50610 = in1->mem.size;
    sizze_47398 = in1->shape[0];
    sizze_47399 = in1->shape[1];
    D_mem_50613 = in2->mem;
    D_mem_sizze_50612 = in2->mem.size;
    sizze_47400 = in2->shape[0];
    
    int ret = futrts_t7_2d(ctx, &out_memsizze_50996, &out_mem_50995,
                           &out_arrsizze_50997, A_mem_sizze_50608, A_mem_50609,
                           C_mem_sizze_50610, C_mem_50611, D_mem_sizze_50612,
                           D_mem_50613, sizze_47396, sizze_47397, sizze_47398,
                           sizze_47399, sizze_47400);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_50995;
        (*out0)->shape[0] = out_arrsizze_50997;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t9_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47438;
    int32_t sizze_47439;
    int32_t sizze_47440;
    int32_t sizze_47441;
    int32_t sizze_47442;
    int32_t sizze_47443;
    int64_t out_memsizze_51007;
    struct memblock_device out_mem_51006;
    
    out_mem_51006.references = NULL;
    
    int32_t out_arrsizze_51008;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47438 = in0->shape[0];
    sizze_47439 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47440 = in1->shape[0];
    sizze_47441 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47442 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47443 = in3->shape[0];
    
    int ret = futrts_t9_2d(ctx, &out_memsizze_51007, &out_mem_51006,
                           &out_arrsizze_51008, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47438, sizze_47439, sizze_47440, sizze_47441,
                           sizze_47442, sizze_47443);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_51006;
        (*out0)->shape[0] = out_arrsizze_51008;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t10_2d(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_2d *in1, const
                         struct futhark_f32_2d *in2, const
                         struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47491;
    int32_t sizze_47492;
    int32_t sizze_47493;
    int32_t sizze_47494;
    int32_t sizze_47495;
    int32_t sizze_47496;
    int32_t sizze_47497;
    int64_t out_memsizze_51018;
    struct memblock_device out_mem_51017;
    
    out_mem_51017.references = NULL;
    
    int32_t out_arrsizze_51019;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47491 = in0->shape[0];
    sizze_47492 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47493 = in1->shape[0];
    sizze_47494 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47495 = in2->shape[0];
    sizze_47496 = in2->shape[1];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47497 = in3->shape[0];
    
    int ret = futrts_t10_2d(ctx, &out_memsizze_51018, &out_mem_51017,
                            &out_arrsizze_51019, A_mem_sizze_50608, A_mem_50609,
                            B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                            C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                            sizze_47491, sizze_47492, sizze_47493, sizze_47494,
                            sizze_47495, sizze_47496, sizze_47497);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_51017;
        (*out0)->shape[0] = out_arrsizze_51019;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t9_3d(struct futhark_context *ctx,
                        struct futhark_f32_2d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47554;
    int32_t sizze_47555;
    int32_t sizze_47556;
    int32_t sizze_47557;
    int32_t sizze_47558;
    int32_t sizze_47559;
    int64_t out_memsizze_51030;
    struct memblock_device out_mem_51029;
    
    out_mem_51029.references = NULL;
    
    int32_t out_arrsizze_51031;
    int32_t out_arrsizze_51032;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47554 = in0->shape[0];
    sizze_47555 = in0->shape[1];
    sizze_47556 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47557 = in1->shape[0];
    sizze_47558 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47559 = in2->shape[0];
    
    int ret = futrts_t9_3d(ctx, &out_memsizze_51030, &out_mem_51029,
                           &out_arrsizze_51031, &out_arrsizze_51032,
                           A_mem_sizze_50608, A_mem_50609, B_mem_sizze_50610,
                           B_mem_50611, C_mem_sizze_50612, C_mem_50613,
                           sizze_47554, sizze_47555, sizze_47556, sizze_47557,
                           sizze_47558, sizze_47559);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_51029;
        (*out0)->shape[0] = out_arrsizze_51031;
        (*out0)->shape[1] = out_arrsizze_51032;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t8_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_2d *in2, const
                        struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47599;
    int32_t sizze_47600;
    int32_t sizze_47601;
    int32_t sizze_47602;
    int32_t sizze_47603;
    int32_t sizze_47604;
    int32_t sizze_47605;
    int64_t out_memsizze_51042;
    struct memblock_device out_mem_51041;
    
    out_mem_51041.references = NULL;
    
    int32_t out_arrsizze_51043;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47599 = in0->shape[0];
    sizze_47600 = in0->shape[1];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47601 = in1->shape[0];
    sizze_47602 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47603 = in2->shape[0];
    sizze_47604 = in2->shape[1];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47605 = in3->shape[0];
    
    int ret = futrts_t8_2d(ctx, &out_memsizze_51042, &out_mem_51041,
                           &out_arrsizze_51043, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47599, sizze_47600, sizze_47601, sizze_47602,
                           sizze_47603, sizze_47604, sizze_47605);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_1d))) != NULL);
        (*out0)->mem = out_mem_51041;
        (*out0)->shape[0] = out_arrsizze_51043;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t1_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int32_t sizze_47659;
    int32_t sizze_47660;
    int32_t sizze_47661;
    int32_t sizze_47662;
    int32_t sizze_47663;
    int64_t out_memsizze_51053;
    struct memblock_device out_mem_51052;
    
    out_mem_51052.references = NULL;
    
    int32_t out_arrsizze_51054;
    int32_t out_arrsizze_51055;
    int32_t out_arrsizze_51056;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47659 = in0->shape[0];
    sizze_47660 = in0->shape[1];
    sizze_47661 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47662 = in1->shape[0];
    sizze_47663 = in1->shape[1];
    
    int ret = futrts_t1_3d(ctx, &out_memsizze_51053, &out_mem_51052,
                           &out_arrsizze_51054, &out_arrsizze_51055,
                           &out_arrsizze_51056, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, sizze_47659,
                           sizze_47660, sizze_47661, sizze_47662, sizze_47663);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51052;
        (*out0)->shape[0] = out_arrsizze_51054;
        (*out0)->shape[1] = out_arrsizze_51055;
        (*out0)->shape[2] = out_arrsizze_51056;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t2_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47690;
    int32_t sizze_47691;
    int32_t sizze_47692;
    int32_t sizze_47693;
    int32_t sizze_47694;
    int64_t out_memsizze_51066;
    struct memblock_device out_mem_51065;
    
    out_mem_51065.references = NULL;
    
    int32_t out_arrsizze_51067;
    int32_t out_arrsizze_51068;
    int32_t out_arrsizze_51069;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47690 = in0->shape[0];
    sizze_47691 = in0->shape[1];
    sizze_47692 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47693 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47694 = in2->shape[0];
    
    int ret = futrts_t2_3d(ctx, &out_memsizze_51066, &out_mem_51065,
                           &out_arrsizze_51067, &out_arrsizze_51068,
                           &out_arrsizze_51069, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, sizze_47690, sizze_47691, sizze_47692,
                           sizze_47693, sizze_47694);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51065;
        (*out0)->shape[0] = out_arrsizze_51067;
        (*out0)->shape[1] = out_arrsizze_51068;
        (*out0)->shape[2] = out_arrsizze_51069;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t10_3d(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_3d *in0, const
                         struct futhark_f32_3d *in1, const
                         struct futhark_f32_1d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47724;
    int32_t sizze_47725;
    int32_t sizze_47726;
    int32_t sizze_47727;
    int32_t sizze_47728;
    int32_t sizze_47729;
    int32_t sizze_47730;
    int64_t out_memsizze_51082;
    struct memblock_device out_mem_51081;
    
    out_mem_51081.references = NULL;
    
    int32_t out_arrsizze_51083;
    int32_t out_arrsizze_51084;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47724 = in0->shape[0];
    sizze_47725 = in0->shape[1];
    sizze_47726 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47727 = in1->shape[0];
    sizze_47728 = in1->shape[1];
    sizze_47729 = in1->shape[2];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47730 = in2->shape[0];
    
    int ret = futrts_t10_3d(ctx, &out_memsizze_51082, &out_mem_51081,
                            &out_arrsizze_51083, &out_arrsizze_51084,
                            A_mem_sizze_50608, A_mem_50609, B_mem_sizze_50610,
                            B_mem_50611, C_mem_sizze_50612, C_mem_50613,
                            sizze_47724, sizze_47725, sizze_47726, sizze_47727,
                            sizze_47728, sizze_47729, sizze_47730);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_51081;
        (*out0)->shape[0] = out_arrsizze_51083;
        (*out0)->shape[1] = out_arrsizze_51084;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t11_3d(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_1d *in0, const
                         struct futhark_f32_1d *in1, const
                         struct futhark_f32_1d *in2, const
                         struct futhark_f32_3d *in3, const
                         struct futhark_f32_1d *in4)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int64_t E_mem_sizze_50616;
    struct memblock_device E_mem_50617;
    
    E_mem_50617.references = NULL;
    
    int32_t sizze_47777;
    int32_t sizze_47778;
    int32_t sizze_47779;
    int32_t sizze_47780;
    int32_t sizze_47781;
    int32_t sizze_47782;
    int32_t sizze_47783;
    int64_t out_memsizze_51094;
    struct memblock_device out_mem_51093;
    
    out_mem_51093.references = NULL;
    
    int32_t out_arrsizze_51095;
    int32_t out_arrsizze_51096;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47777 = in0->shape[0];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47778 = in1->shape[0];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47779 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47780 = in3->shape[0];
    sizze_47781 = in3->shape[1];
    sizze_47782 = in3->shape[2];
    E_mem_50617 = in4->mem;
    E_mem_sizze_50616 = in4->mem.size;
    sizze_47783 = in4->shape[0];
    
    int ret = futrts_t11_3d(ctx, &out_memsizze_51094, &out_mem_51093,
                            &out_arrsizze_51095, &out_arrsizze_51096,
                            A_mem_sizze_50608, A_mem_50609, B_mem_sizze_50610,
                            B_mem_50611, C_mem_sizze_50612, C_mem_50613,
                            D_mem_sizze_50614, D_mem_50615, E_mem_sizze_50616,
                            E_mem_50617, sizze_47777, sizze_47778, sizze_47779,
                            sizze_47780, sizze_47781, sizze_47782, sizze_47783);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_2d))) != NULL);
        (*out0)->mem = out_mem_51093;
        (*out0)->shape[0] = out_arrsizze_51095;
        (*out0)->shape[1] = out_arrsizze_51096;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t3_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47834;
    int32_t sizze_47835;
    int32_t sizze_47836;
    int32_t sizze_47837;
    int32_t sizze_47838;
    int32_t sizze_47839;
    int32_t sizze_47840;
    int64_t out_memsizze_51110;
    struct memblock_device out_mem_51109;
    
    out_mem_51109.references = NULL;
    
    int32_t out_arrsizze_51111;
    int32_t out_arrsizze_51112;
    int32_t out_arrsizze_51113;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47834 = in0->shape[0];
    sizze_47835 = in0->shape[1];
    sizze_47836 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47837 = in1->shape[0];
    sizze_47838 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47839 = in2->shape[0];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47840 = in3->shape[0];
    
    int ret = futrts_t3_3d(ctx, &out_memsizze_51110, &out_mem_51109,
                           &out_arrsizze_51111, &out_arrsizze_51112,
                           &out_arrsizze_51113, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47834, sizze_47835, sizze_47836, sizze_47837,
                           sizze_47838, sizze_47839, sizze_47840);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51109;
        (*out0)->shape[0] = out_arrsizze_51111;
        (*out0)->shape[1] = out_arrsizze_51112;
        (*out0)->shape[2] = out_arrsizze_51113;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t4_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_3d *in1, const
                        struct futhark_f32_2d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_47886;
    int32_t sizze_47887;
    int32_t sizze_47888;
    int32_t sizze_47889;
    int32_t sizze_47890;
    int32_t sizze_47891;
    int32_t sizze_47892;
    int32_t sizze_47893;
    int64_t out_memsizze_51119;
    struct memblock_device out_mem_51118;
    
    out_mem_51118.references = NULL;
    
    int32_t out_arrsizze_51120;
    int32_t out_arrsizze_51121;
    int32_t out_arrsizze_51122;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47886 = in0->shape[0];
    sizze_47887 = in0->shape[1];
    sizze_47888 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47889 = in1->shape[0];
    sizze_47890 = in1->shape[1];
    sizze_47891 = in1->shape[2];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47892 = in2->shape[0];
    sizze_47893 = in2->shape[1];
    
    int ret = futrts_t4_3d(ctx, &out_memsizze_51119, &out_mem_51118,
                           &out_arrsizze_51120, &out_arrsizze_51121,
                           &out_arrsizze_51122, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, sizze_47886, sizze_47887, sizze_47888,
                           sizze_47889, sizze_47890, sizze_47891, sizze_47892,
                           sizze_47893);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51118;
        (*out0)->shape[0] = out_arrsizze_51120;
        (*out0)->shape[1] = out_arrsizze_51121;
        (*out0)->shape[2] = out_arrsizze_51122;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t5_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_3d *in1, const
                        struct futhark_f32_2d *in2, const
                        struct futhark_f32_2d *in3)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_47945;
    int32_t sizze_47946;
    int32_t sizze_47947;
    int32_t sizze_47948;
    int32_t sizze_47949;
    int32_t sizze_47950;
    int32_t sizze_47951;
    int32_t sizze_47952;
    int32_t sizze_47953;
    int32_t sizze_47954;
    int64_t out_memsizze_51135;
    struct memblock_device out_mem_51134;
    
    out_mem_51134.references = NULL;
    
    int32_t out_arrsizze_51136;
    int32_t out_arrsizze_51137;
    int32_t out_arrsizze_51138;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_47945 = in0->shape[0];
    sizze_47946 = in0->shape[1];
    sizze_47947 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_47948 = in1->shape[0];
    sizze_47949 = in1->shape[1];
    sizze_47950 = in1->shape[2];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_47951 = in2->shape[0];
    sizze_47952 = in2->shape[1];
    D_mem_50615 = in3->mem;
    D_mem_sizze_50614 = in3->mem.size;
    sizze_47953 = in3->shape[0];
    sizze_47954 = in3->shape[1];
    
    int ret = futrts_t5_3d(ctx, &out_memsizze_51135, &out_mem_51134,
                           &out_arrsizze_51136, &out_arrsizze_51137,
                           &out_arrsizze_51138, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_47945, sizze_47946, sizze_47947, sizze_47948,
                           sizze_47949, sizze_47950, sizze_47951, sizze_47952,
                           sizze_47953, sizze_47954);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51134;
        (*out0)->shape[0] = out_arrsizze_51136;
        (*out0)->shape[1] = out_arrsizze_51137;
        (*out0)->shape[2] = out_arrsizze_51138;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t6_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const float in0, const
                        struct futhark_f32_3d *in1, const float in2, const
                        struct futhark_f32_3d *in3, const float in4, const
                        struct futhark_f32_2d *in5, const float in6, const
                        struct futhark_f32_2d *in7)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int32_t sizze_48024;
    int32_t sizze_48025;
    int32_t sizze_48026;
    int32_t sizze_48027;
    int32_t sizze_48028;
    int32_t sizze_48029;
    int32_t sizze_48030;
    int32_t sizze_48031;
    int32_t sizze_48032;
    int32_t sizze_48033;
    float a_48034;
    float b_48036;
    float c_48038;
    float d_48040;
    int64_t out_memsizze_51154;
    struct memblock_device out_mem_51153;
    
    out_mem_51153.references = NULL;
    
    int32_t out_arrsizze_51155;
    int32_t out_arrsizze_51156;
    int32_t out_arrsizze_51157;
    
    lock_lock(&ctx->lock);
    a_48034 = in0;
    A_mem_50609 = in1->mem;
    A_mem_sizze_50608 = in1->mem.size;
    sizze_48024 = in1->shape[0];
    sizze_48025 = in1->shape[1];
    sizze_48026 = in1->shape[2];
    b_48036 = in2;
    B_mem_50611 = in3->mem;
    B_mem_sizze_50610 = in3->mem.size;
    sizze_48027 = in3->shape[0];
    sizze_48028 = in3->shape[1];
    sizze_48029 = in3->shape[2];
    c_48038 = in4;
    C_mem_50613 = in5->mem;
    C_mem_sizze_50612 = in5->mem.size;
    sizze_48030 = in5->shape[0];
    sizze_48031 = in5->shape[1];
    d_48040 = in6;
    D_mem_50615 = in7->mem;
    D_mem_sizze_50614 = in7->mem.size;
    sizze_48032 = in7->shape[0];
    sizze_48033 = in7->shape[1];
    
    int ret = futrts_t6_3d(ctx, &out_memsizze_51154, &out_mem_51153,
                           &out_arrsizze_51155, &out_arrsizze_51156,
                           &out_arrsizze_51157, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           sizze_48024, sizze_48025, sizze_48026, sizze_48027,
                           sizze_48028, sizze_48029, sizze_48030, sizze_48031,
                           sizze_48032, sizze_48033, a_48034, b_48036, c_48038,
                           d_48040);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51153;
        (*out0)->shape[0] = out_arrsizze_51155;
        (*out0)->shape[1] = out_arrsizze_51156;
        (*out0)->shape[2] = out_arrsizze_51157;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t7_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const float in0, const
                        struct futhark_f32_3d *in1, const float in2, const
                        struct futhark_f32_3d *in3, const float in4, const
                        struct futhark_f32_2d *in5, const float in6, const
                        struct futhark_f32_2d *in7, const float in8, const
                        struct futhark_f32_1d *in9, const float in10, const
                        struct futhark_f32_1d *in11, const float in12, const
                        struct futhark_f32_1d *in13, const float in14, const
                        struct futhark_f32_1d *in15)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int64_t D_mem_sizze_50614;
    struct memblock_device D_mem_50615;
    
    D_mem_50615.references = NULL;
    
    int64_t E_mem_sizze_50616;
    struct memblock_device E_mem_50617;
    
    E_mem_50617.references = NULL;
    
    int64_t F_mem_sizze_50618;
    struct memblock_device F_mem_50619;
    
    F_mem_50619.references = NULL;
    
    int64_t G_mem_sizze_50620;
    struct memblock_device G_mem_50621;
    
    G_mem_50621.references = NULL;
    
    int64_t H_mem_sizze_50622;
    struct memblock_device H_mem_50623;
    
    H_mem_50623.references = NULL;
    
    int32_t sizze_48111;
    int32_t sizze_48112;
    int32_t sizze_48113;
    int32_t sizze_48114;
    int32_t sizze_48115;
    int32_t sizze_48116;
    int32_t sizze_48117;
    int32_t sizze_48118;
    int32_t sizze_48119;
    int32_t sizze_48120;
    int32_t sizze_48121;
    int32_t sizze_48122;
    int32_t sizze_48123;
    int32_t sizze_48124;
    float a_48125;
    float b_48127;
    float c_48129;
    float d_48131;
    float e_48133;
    float f_48135;
    float g_48137;
    float h_48139;
    int64_t out_memsizze_51173;
    struct memblock_device out_mem_51172;
    
    out_mem_51172.references = NULL;
    
    int32_t out_arrsizze_51174;
    int32_t out_arrsizze_51175;
    int32_t out_arrsizze_51176;
    
    lock_lock(&ctx->lock);
    a_48125 = in0;
    A_mem_50609 = in1->mem;
    A_mem_sizze_50608 = in1->mem.size;
    sizze_48111 = in1->shape[0];
    sizze_48112 = in1->shape[1];
    sizze_48113 = in1->shape[2];
    b_48127 = in2;
    B_mem_50611 = in3->mem;
    B_mem_sizze_50610 = in3->mem.size;
    sizze_48114 = in3->shape[0];
    sizze_48115 = in3->shape[1];
    sizze_48116 = in3->shape[2];
    c_48129 = in4;
    C_mem_50613 = in5->mem;
    C_mem_sizze_50612 = in5->mem.size;
    sizze_48117 = in5->shape[0];
    sizze_48118 = in5->shape[1];
    d_48131 = in6;
    D_mem_50615 = in7->mem;
    D_mem_sizze_50614 = in7->mem.size;
    sizze_48119 = in7->shape[0];
    sizze_48120 = in7->shape[1];
    e_48133 = in8;
    E_mem_50617 = in9->mem;
    E_mem_sizze_50616 = in9->mem.size;
    sizze_48121 = in9->shape[0];
    f_48135 = in10;
    F_mem_50619 = in11->mem;
    F_mem_sizze_50618 = in11->mem.size;
    sizze_48122 = in11->shape[0];
    g_48137 = in12;
    G_mem_50621 = in13->mem;
    G_mem_sizze_50620 = in13->mem.size;
    sizze_48123 = in13->shape[0];
    h_48139 = in14;
    H_mem_50623 = in15->mem;
    H_mem_sizze_50622 = in15->mem.size;
    sizze_48124 = in15->shape[0];
    
    int ret = futrts_t7_3d(ctx, &out_memsizze_51173, &out_mem_51172,
                           &out_arrsizze_51174, &out_arrsizze_51175,
                           &out_arrsizze_51176, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, D_mem_sizze_50614, D_mem_50615,
                           E_mem_sizze_50616, E_mem_50617, F_mem_sizze_50618,
                           F_mem_50619, G_mem_sizze_50620, G_mem_50621,
                           H_mem_sizze_50622, H_mem_50623, sizze_48111,
                           sizze_48112, sizze_48113, sizze_48114, sizze_48115,
                           sizze_48116, sizze_48117, sizze_48118, sizze_48119,
                           sizze_48120, sizze_48121, sizze_48122, sizze_48123,
                           sizze_48124, a_48125, b_48127, c_48129, d_48131,
                           e_48133, f_48135, g_48137, h_48139);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51172;
        (*out0)->shape[0] = out_arrsizze_51174;
        (*out0)->shape[1] = out_arrsizze_51175;
        (*out0)->shape[2] = out_arrsizze_51176;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
int futhark_entry_t8_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_2d *in2)
{
    int64_t A_mem_sizze_50608;
    struct memblock_device A_mem_50609;
    
    A_mem_50609.references = NULL;
    
    int64_t B_mem_sizze_50610;
    struct memblock_device B_mem_50611;
    
    B_mem_50611.references = NULL;
    
    int64_t C_mem_sizze_50612;
    struct memblock_device C_mem_50613;
    
    C_mem_50613.references = NULL;
    
    int32_t sizze_48250;
    int32_t sizze_48251;
    int32_t sizze_48252;
    int32_t sizze_48253;
    int32_t sizze_48254;
    int32_t sizze_48255;
    int32_t sizze_48256;
    int64_t out_memsizze_51194;
    struct memblock_device out_mem_51193;
    
    out_mem_51193.references = NULL;
    
    int32_t out_arrsizze_51195;
    int32_t out_arrsizze_51196;
    int32_t out_arrsizze_51197;
    
    lock_lock(&ctx->lock);
    A_mem_50609 = in0->mem;
    A_mem_sizze_50608 = in0->mem.size;
    sizze_48250 = in0->shape[0];
    sizze_48251 = in0->shape[1];
    sizze_48252 = in0->shape[2];
    B_mem_50611 = in1->mem;
    B_mem_sizze_50610 = in1->mem.size;
    sizze_48253 = in1->shape[0];
    sizze_48254 = in1->shape[1];
    C_mem_50613 = in2->mem;
    C_mem_sizze_50612 = in2->mem.size;
    sizze_48255 = in2->shape[0];
    sizze_48256 = in2->shape[1];
    
    int ret = futrts_t8_3d(ctx, &out_memsizze_51194, &out_mem_51193,
                           &out_arrsizze_51195, &out_arrsizze_51196,
                           &out_arrsizze_51197, A_mem_sizze_50608, A_mem_50609,
                           B_mem_sizze_50610, B_mem_50611, C_mem_sizze_50612,
                           C_mem_50613, sizze_48250, sizze_48251, sizze_48252,
                           sizze_48253, sizze_48254, sizze_48255, sizze_48256);
    
    if (ret == 0) {
        assert((*out0 = malloc(sizeof(struct futhark_f32_3d))) != NULL);
        (*out0)->mem = out_mem_51193;
        (*out0)->shape[0] = out_arrsizze_51195;
        (*out0)->shape[1] = out_arrsizze_51196;
        (*out0)->shape[2] = out_arrsizze_51197;
    }
    lock_unlock(&ctx->lock);
    return ret;
}
