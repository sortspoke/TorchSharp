// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
#pragma once

#include "../Stdafx.h"

#include "Utils.h"

// API.

// Sets manually the seed.
EXPORT_API(void) THSTorch_manual_seed(const int64_t seed);

EXPORT_API(int) THSTorchCuda_is_available();
EXPORT_API(int) THSTorchCuda_cudnn_is_available();
EXPORT_API(int) THSTorchCuda_device_count();

//EXPORT_API(void) THSTorchCudaMemory_resetPeakStats(int device);
//EXPORT_API(int64_t) THSTorchCudaMemory_getAllocatedBytes(int device);
//EXPORT_API(int64_t) THSTorchCudaMemory_getMaxAllocatedBytes(int device);
//EXPORT_API(void) THSTorch_synchronize();

// Returns the latest error. This is thread-local.
EXPORT_API(const char *) THSTorch_get_and_reset_last_err();

EXPORT_API(Scalar) THSTorch_int8_to_scalar(int8_t value);
EXPORT_API(Scalar) THSTorch_uint8_to_scalar(uint8_t value);
EXPORT_API(Scalar) THSTorch_int16_to_scalar(short value);
EXPORT_API(Scalar) THSTorch_int32_to_scalar(int value);
EXPORT_API(Scalar) THSTorch_int64_to_scalar(long value);
EXPORT_API(Scalar) THSTorch_float32_to_scalar(float value);
EXPORT_API(Scalar) THSTorch_float64_to_scalar(double value);
EXPORT_API(Scalar) THSTorch_bool_to_scalar(bool value);
EXPORT_API(Scalar) THSTorch_float16_to_scalar(float value);
EXPORT_API(Scalar) THSTorch_bfloat16_to_scalar(float value);

EXPORT_API(int8_t) THSTorch_scalar_to_int8(Scalar value);
EXPORT_API(uint8_t) THSTorch_scalar_to_uint8(Scalar value);
EXPORT_API(int16_t) THSTorch_scalar_to_int16(Scalar value);
EXPORT_API(int32_t) THSTorch_scalar_to_int32(Scalar value);
EXPORT_API(int64_t) THSTorch_scalar_to_int64(Scalar value);
EXPORT_API(float) THSTorch_scalar_to_float32(Scalar value);
EXPORT_API(double) THSTorch_scalar_to_float64(Scalar value);
EXPORT_API(bool) THSTorch_scalar_to_bool(Scalar value);

// Dispose the scalar.
EXPORT_API(void) THSTorch_dispose_scalar(Scalar scalar);
