/*
 * Headers
*/

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>


/*
 * Initialisation
*/

struct futhark_context_config ;
struct futhark_context_config *futhark_context_config_new();
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
struct futhark_context ;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);

/*
 * Arrays
*/

struct f32_1d ;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx,
                                          float *data, int dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx,
                                              char *data, int offset, int dim0);
int futhark_free_f32_1d(struct futhark_context *ctx,
                        struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx,
                          struct futhark_f32_1d *arr, float *data);
char *futhark_values_raw_f32_1d(struct futhark_context *ctx,
                                struct futhark_f32_1d *arr);
int64_t *futhark_shape_f32_1d(struct futhark_context *ctx,
                              struct futhark_f32_1d *arr);

/*
 * Opaque values
*/


/*
 * Entry points
*/

int futhark_entry_BaselineSum(struct futhark_context *ctx, float *out0, const
                              struct futhark_f32_1d *in0);
int futhark_entry_BaselineProd(struct futhark_context *ctx,
                               struct futhark_f32_1d **out0, const float in0,
                               const struct futhark_f32_1d *in1);
int futhark_entry_Dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1);
int futhark_entry_Dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_Dot2(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_Dot3(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3);
int futhark_entry_Dot4(struct futhark_context *ctx, float *out0, const
                       float in0, const struct futhark_f32_1d *in1, const
                       float in2, const struct futhark_f32_1d *in3, const
                       float in4, const struct futhark_f32_1d *in5, const
                       float in6, const struct futhark_f32_1d *in7);
int futhark_entry_Dot5(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_Dot6(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2, const
                       struct futhark_f32_1d *in3);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
