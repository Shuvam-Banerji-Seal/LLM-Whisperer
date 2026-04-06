#include "plugin_interface.h"

static plugin_info_t g_info = {"llm_plugin_stub", 1};

int plugin_init(void) {
    return 0;
}

const plugin_info_t* plugin_get_info(void) {
    return &g_info;
}

int plugin_infer(const float* input, int input_len, float* output, int output_len) {
    int n = input_len < output_len ? input_len : output_len;
    for (int i = 0; i < n; ++i) {
        output[i] = input[i] * 2.0f;
    }
    return n;
}

void plugin_shutdown(void) {
}
