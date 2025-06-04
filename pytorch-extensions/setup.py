from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="custom_softmax_cuda",
    ext_modules=[
        CUDAExtension(
            "custom_softmax_cuda",
            ["safe_softmax.cu", "online_softmax.cu", "binding.cpp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
