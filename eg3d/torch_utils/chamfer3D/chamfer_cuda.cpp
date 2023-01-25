#include <torch/torch.h>
#include <vector>

///TMP
//#include "common.h"
/// NOT TMP
	

int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2);


int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2);




int chamfer_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2) {
    return chamfer_cuda_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
}


int chamfer_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, 
					  at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2) {

    return chamfer_cuda_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &chamfer_forward, "chamfer forward (CUDA)");
  m.def("backward", &chamfer_backward, "chamfer backward (CUDA)");
}