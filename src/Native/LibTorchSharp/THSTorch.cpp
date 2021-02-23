// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#include "THSTorch.h"

#include "torch/torch.h"
#include "torch/cuda.h"

//#include <ATen/cuda/CUDAContext.h>
//#include <c10/cuda/CUDACachingAllocator.h>

void THSTorch_manual_seed(const int64_t seed)
{
    torch::manual_seed(seed);
}

int THSTorchCuda_is_available()
{
    return torch::cuda::is_available();
}

int THSTorchCuda_cudnn_is_available()
{
    return torch::cuda::cudnn_is_available();
}

int THSTorchCuda_device_count()
{
    return (int)torch::cuda::device_count();
}
//
//c10::cuda::CUDACachingAllocator::DeviceStats THSTorchCudaMemory_getDeviceStats(int device)
//{
//    //at::cuda::getDeviceProperties(device);
//    const c10::cuda::CUDACachingAllocator::DeviceStats stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
//    return stats;
//}
//
//void THSTorchCudaMemory_resetPeakStats(int device)
//{
//    c10::cuda::CUDACachingAllocator::resetPeakStats(device);
//}
//
//int64_t THSTorchCudaMemory_getAllocatedBytes(int device)
//{
//    const c10::cuda::CUDACachingAllocator::DeviceStats stats = THSTorchCudaMemory_getDeviceStats(device);
//    return stats.allocated_bytes[0].current;
//}
//
//int64_t THSTorchCudaMemory_getMaxAllocatedBytes(int device)
//{
//    const c10::cuda::CUDACachingAllocator::DeviceStats stats = THSTorchCudaMemory_getDeviceStats(device);
//    return stats.allocated_bytes[0].peak;
//}

const char * THSTorch_get_and_reset_last_err()
{
    char *tmp = torch_last_err;
    torch_last_err = nullptr;
    return tmp;
}

Scalar THSTorch_int8_to_scalar(int8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_uint8_to_scalar(uint8_t value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int16_to_scalar(short value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int32_to_scalar(int value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_int64_to_scalar(long value)
{
    return new torch::Scalar(int64_t(value));
}

Scalar THSTorch_float32_to_scalar(float value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float64_to_scalar(double value)
{
    return new torch::Scalar(value);
}

Scalar THSTorch_float16_to_scalar(float value)
{
    return new torch::Scalar((c10::Half)value);
}

Scalar THSTorch_bfloat16_to_scalar(float value)
{
    return new torch::Scalar((c10::BFloat16)value);
}

Scalar THSTorch_bool_to_scalar(bool value)
{
    return new torch::Scalar(value);
}

int8_t THSTorch_scalar_to_int8(Scalar value)
{
    return value->toChar();
}

uint8_t THSTorch_scalar_to_uint8(Scalar value)
{
    return value->toByte();
}

int16_t THSTorch_scalar_to_int16(Scalar value)
{
    return value->toShort();
}

int32_t THSTorch_scalar_to_int32(Scalar value)
{
    return value->toInt();
}

int64_t THSTorch_scalar_to_int64(Scalar value)
{
    return value->toLong();
}

float THSTorch_scalar_to_float32(Scalar value)
{
    return value->toFloat();
}

double THSTorch_scalar_to_float64(Scalar value)
{
    return value->toDouble();
}

bool THSTorch_scalar_to_bool(Scalar value)
{
    return value->toBool();
}

void THSTorch_dispose_scalar(Scalar scalar)
{
    delete scalar;
}
