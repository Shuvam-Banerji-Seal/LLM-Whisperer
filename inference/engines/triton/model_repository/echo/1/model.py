import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.input_name = "TEXT"
        self.output_name = "ECHO_TEXT"

    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, self.input_name)
            input_data = in_tensor.as_numpy()
            output_data = np.array(input_data, dtype=object)
            out_tensor = pb_utils.Tensor(self.output_name, output_data)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        return
