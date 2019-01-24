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

int futhark_entry_sum(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0);
int futhark_entry_prod(struct futhark_context *ctx,
                       struct futhark_f32_1d **out0, const float in0, const
                       struct futhark_f32_1d *in1);
int futhark_entry_dot(struct futhark_context *ctx, float *out0, const
                      struct futhark_f32_1d *in0, const
                      struct futhark_f32_1d *in1);
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
int futhark_entry_dot1(struct futhark_context *ctx, float *out0, const
                       struct futhark_f32_1d *in0, const
                       struct futhark_f32_1d *in1, const
                       struct futhark_f32_1d *in2);
int futhark_entry_t1b_2d(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_1d *in1);
int futhark_entry_t1_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_1d *in1);
int futhark_entry_t2_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2);
int futhark_entry_t3_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2);
int futhark_entry_t4_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3);
int futhark_entry_t5_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const float in0, const
                        struct futhark_f32_2d *in1, const float in2, const
                        struct futhark_f32_2d *in3, const float in4, const
                        struct futhark_f32_1d *in5, const float in6, const
                        struct futhark_f32_1d *in7);
int futhark_entry_t6_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_1d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3);
int futhark_entry_t7_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2);
int futhark_entry_t9_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3);
int futhark_entry_t10_2d(struct futhark_context *ctx,
                         struct futhark_f32_1d **out0, const
                         struct futhark_f32_2d *in0, const
                         struct futhark_f32_2d *in1, const
                         struct futhark_f32_2d *in2, const
                         struct futhark_f32_1d *in3);
int futhark_entry_t9_3d(struct futhark_context *ctx,
                        struct futhark_f32_2d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2);
int futhark_entry_t8_2d(struct futhark_context *ctx,
                        struct futhark_f32_1d **out0, const
                        struct futhark_f32_2d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_2d *in2, const
                        struct futhark_f32_1d *in3);
int futhark_entry_t1_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1);
int futhark_entry_t2_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_1d *in1, const
                        struct futhark_f32_1d *in2);
int futhark_entry_t10_3d(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_3d *in0, const
                         struct futhark_f32_3d *in1, const
                         struct futhark_f32_1d *in2);
int futhark_entry_t11_3d(struct futhark_context *ctx,
                         struct futhark_f32_2d **out0, const
                         struct futhark_f32_1d *in0, const
                         struct futhark_f32_1d *in1, const
                         struct futhark_f32_1d *in2, const
                         struct futhark_f32_3d *in3, const
                         struct futhark_f32_1d *in4);
int futhark_entry_t3_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_1d *in2, const
                        struct futhark_f32_1d *in3);
int futhark_entry_t4_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_3d *in1, const
                        struct futhark_f32_2d *in2);
int futhark_entry_t5_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_3d *in1, const
                        struct futhark_f32_2d *in2, const
                        struct futhark_f32_2d *in3);
int futhark_entry_t6_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const float in0, const
                        struct futhark_f32_3d *in1, const float in2, const
                        struct futhark_f32_3d *in3, const float in4, const
                        struct futhark_f32_2d *in5, const float in6, const
                        struct futhark_f32_2d *in7);
int futhark_entry_t7_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const float in0, const
                        struct futhark_f32_3d *in1, const float in2, const
                        struct futhark_f32_3d *in3, const float in4, const
                        struct futhark_f32_2d *in5, const float in6, const
                        struct futhark_f32_2d *in7, const float in8, const
                        struct futhark_f32_1d *in9, const float in10, const
                        struct futhark_f32_1d *in11, const float in12, const
                        struct futhark_f32_1d *in13, const float in14, const
                        struct futhark_f32_1d *in15);
int futhark_entry_t8_3d(struct futhark_context *ctx,
                        struct futhark_f32_3d **out0, const
                        struct futhark_f32_3d *in0, const
                        struct futhark_f32_2d *in1, const
                        struct futhark_f32_2d *in2);

/*
 * Miscellaneous
*/

void futhark_debugging_report(struct futhark_context *ctx);
