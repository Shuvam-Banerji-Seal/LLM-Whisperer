#ifndef LLM_PLUGIN_INTERFACE_H
#define LLM_PLUGIN_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct plugin_info_s {
    const char* name;
    int version;
} plugin_info_t;

int plugin_init(void);
const plugin_info_t* plugin_get_info(void);
int plugin_infer(const float* input, int input_len, float* output, int output_len);
void plugin_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif  // LLM_PLUGIN_INTERFACE_H
