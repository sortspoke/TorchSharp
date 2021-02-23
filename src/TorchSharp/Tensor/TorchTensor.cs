// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

#nullable enable
namespace TorchSharp.Tensor
{
    public sealed class TorchTensor : IDisposable
    {
        internal IntPtr handle;

        internal TorchTensor(IntPtr handle)
        {
            this.handle = handle;
        }

        public override bool Equals(object? obj)
        {
            return (obj is TorchTensor) && this.Equal((obj as TorchTensor)!);

        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        /// <summary>
        ///   Finalize the tensor. Releases the tensor and its associated data.
        /// </summary>
        ~TorchTensor() => Dispose(false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        [DllImport("LibTorchSharp")]
        extern static void THSTensor_dispose(IntPtr handle);

        /// <summary>
        ///   Implements the .NET Dispose pattern.
        /// </summary>
        void Dispose(bool disposing)
        {
            if (handle != IntPtr.Zero) {
                THSTensor_dispose(handle);
                handle = IntPtr.Zero;
                //Console.WriteLine($"Disposing {this.ToString()} {DeviceString}");
            }
        }

        public IntPtr Handle => handle;

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_ndimension(IntPtr handle);

        /// <summary>
        ///  Returns the number of dimensions for this tensor
        /// </summary>
        public long Dimensions => THSTensor_ndimension(handle);

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_element_size(IntPtr handle);

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_numel(IntPtr handle);

        /// <summary>
        ///  Get the number of elements in the tensor.
        /// </summary>
        public long NumberOfElements => THSTensor_numel(handle);

        /// <summary>
        ///  Get the size of each element in the tensor.
        /// </summary>
        public long ElementSize => THSTensor_element_size(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_data(IntPtr handle);

        /// <summary>
        ///  Returns a pointer to the unmanaged data managed by this tensor.
        /// </summary>
        public Span<T> Data<T>()
        {
            if (NumberOfElements > int.MaxValue) {
                throw new ArgumentException("Span only supports up to int.MaxValue elements.");
            }
            if (DeviceType != DeviceType.CPU) throw new InvalidOperationException("Can only get a data pointer when tensor is on CPU");

            unsafe {
                var res = THSTensor_data(handle);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                // NOTE: there is no safety here.
                return new Span<T>((void*)res, (int)NumberOfElements);
            }
        }

        public T DataItem<T>()
        {
            if (NumberOfElements != 1) throw new ArgumentException("Number of elements in the tensor must be 1");

            return Data<T>()[0];
        }

        public double ReadCpuDouble(long i) => Data<double>()[(int)i];
        public float ReadCpuSingle(long i) => Data<float>()[(int)i];
        public int ReadCpuInt32(long i) => Data<int>()[(int)i];
        public long ReadCpuInt64(long i) => Data<long>()[(int)i];
        public byte ReadCpuByte(long i) => Data<byte>()[(int)i];
        public sbyte ReadCpuSByte(long i) => Data<sbyte>()[(int)i];
        public short ReadCpuInt16(long i) => Data<short>()[(int)i];
        public bool ReadCpuBool(long i) => Data<bool>()[(int)i];

        [DllImport("LibTorchSharp")]
        private static extern float THSTensor_data_idx_float16(IntPtr handle, long i);

        public float ReadCpuFloat16(long i)
        {
            if (i >= NumberOfElements) {
                throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
            }
            return THSTensor_data_idx_float16(handle, i);
        }

        [DllImport("LibTorchSharp")]
        private static extern float THSTensor_data_idx_bfloat16(IntPtr handle, long i);

        public float ReadCpuBFloat16(long i)
        {
            if (i >= NumberOfElements) {
                throw new IndexOutOfRangeException("The index is greater than the number of elements in the tensor");
            }
            return THSTensor_data_idx_bfloat16(handle, i);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_item(IntPtr handle);

        public TorchScalar ToScalar()
        {
            var res = THSTensor_item(Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchScalar(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fill_(IntPtr handle, IntPtr value);

        public TorchTensor FillInPlace(TorchScalar value)
        {
            var res = THSTensor_fill_(handle, value.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get1(IntPtr handle, long i1);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set1(IntPtr handle, long i1, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1] {
            get {
                var res = THSTensor_get1(handle, i1);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set {
                THSTensor_set1(handle, i1, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get2(IntPtr handle, long i1, long i2);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set2(IntPtr handle, long i1, long i2, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2] {
            get {
                var res = THSTensor_get2(handle, i1, i2);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
            set {
                THSTensor_set2(handle, i1, i2, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get3(IntPtr handle, long i1, long i2, long i3);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set3(IntPtr handle, long i1, long i2, long i3, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3] {
            get {
                var res = THSTensor_get3(handle, i1, i2, i3);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set3(handle, i1, i2, i3, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get4(IntPtr handle, long i1, long i2, long i3, long i4);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set4(IntPtr handle, long i1, long i2, long i3, long i4, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4] {
            get {
                var res = THSTensor_get4(handle, i1, i2, i3, i4);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set4(handle, i1, i2, i3, i4, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get5(IntPtr handle, long i1, long i2, long i3, long i4, long i5);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set5(IntPtr handle, long i1, long i2, long i3, long i4, long i5, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5] {
            get {
                var res = THSTensor_get5(handle, i1, i2, i3, i4, i5);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set5(handle, i1, i2, i3, i4, i5, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_get6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set6(IntPtr handle, long i1, long i2, long i3, long i4, long i5, long i6, IntPtr value);

        [IndexerName("TensorItems")]
        public TorchTensor this[long i1, long i2, long i3, long i4, long i5, long i6] {
            get {
                var res = THSTensor_get6(handle, i1, i2, i3, i4, i5, i6);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
            set {
                THSTensor_set6(handle, i1, i2, i3, i4, i5, i6, value.ToScalar().Handle);
                Torch.CheckForErrors();
            }
        }
        [DllImport("LibTorchSharp")]
        private static extern sbyte THSTensor_type(IntPtr handle);

        public ScalarType Type => (ScalarType)THSTensor_type(handle);

        [DllImport("LibTorchSharp")]
        [return: MarshalAs(UnmanagedType.LPStr)]
        private static extern string THSTensor_device_str(IntPtr handle);

        public string DeviceString {
            get {
                var res = THSTensor_device_str(handle);
                if (res == null)
                    Torch.CheckForErrors();
                return res!;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern int THSTensor_device_index(IntPtr handle);

        public int DeviceIndex {
            get {
                var res = THSTensor_device_index(handle);
                Torch.CheckForErrors();
                return res;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern int THSTensor_device_type(IntPtr handle);

        public DeviceType DeviceType {
            get {
                var res = THSTensor_device_type(handle);
                Torch.CheckForErrors();
                return (DeviceType)res;
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_is_sparse(IntPtr handle);

        public bool IsSparse {
            get {
                var res = THSTensor_is_sparse(handle);
                Torch.CheckForErrors();
                return res;
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_type(IntPtr handle, sbyte scalar_type);

        public TorchTensor ToType(ScalarType type)
        {
            var res = THSTensor_to_type(handle, (sbyte)type);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_load([MarshalAs(UnmanagedType.LPStr)] string location);

        public static TorchTensor Load(string location)
        {
            var res = THSTensor_load(location);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_save(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

        public void Save(string location)
        {
            THSTensor_save(handle, location);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_loadInto(IntPtr tensor, [MarshalAs(UnmanagedType.LPStr)] string location);

        public void LoadInto(string location)
        {
            THSTensor_loadInto(handle, location);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_requires_grad(IntPtr handle);

        public bool IsGradRequired => THSTensor_requires_grad(handle);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_set_requires_grad(IntPtr handle, bool requires_grad);

        public TorchTensor RequiresGrad(bool requiresGrad)
        {
            var res = THSTensor_set_requires_grad(handle, requiresGrad);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cpu(IntPtr handle);

        public TorchTensor Cpu()
        {
            var res = THSTensor_cpu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cuda(IntPtr handle);

        public TorchTensor Cuda()
        {
            Torch.InitializeDeviceType(DeviceType.CUDA);
            var res = THSTensor_cuda(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_device(IntPtr handle, int device_type, int device_index);

        public TorchTensor ToDevice(DeviceType deviceType, int deviceIndex = 0)
        {
            Torch.InitializeDeviceType(deviceType);
            var res = THSTensor_to_device(handle, (int)deviceType, deviceIndex);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_size(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the size of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorDimension(int dim)
        {
            var res = THSTensor_size(handle, dim);
            Torch.CheckForErrors();
            return res;
        }

        /// <summary>
        /// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
        /// </summary>
        /// <remarks>
        ///     An array of size 0 is used for constants, an array of size 1 is used
        ///     for single-dimension arrays, where the dimension is the value of the
        ///     first element.   And so on.
        /// </remarks>
        public long[] Shape {
            get {
                var dims = new long[Dimensions];
                for (var i = 0; i < dims.Length; i++)
                    dims[i] = GetTensorDimension(i);

                return dims;
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_indices(IntPtr handle);

        public TorchTensor SparseIndices {
            get {
                var res = THSTensor_indices(handle);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_values(IntPtr handle);

        public TorchTensor SparseValues {
            get {
                var res = THSTensor_values(handle);
                if (res == IntPtr.Zero)
                    Torch.CheckForErrors();
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern long THSTensor_stride(IntPtr handle, long dimension);

        /// <summary>
        ///  Retrieves the stride of the specified dimension in the tensor.
        /// </summary>
        public long GetTensorStride(int dim)
        {
            var res = THSTensor_stride(handle, dim);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_backward(IntPtr handle);

        public void Backward()
        {
            THSTensor_backward(handle);
            Torch.CheckForErrors();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_dense(IntPtr handle);

        public TorchTensor ToDense()
        {
            var res = THSTensor_to_dense(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_to_sparse(IntPtr handle);

        public TorchTensor ToSparse()
        {
            var res = THSTensor_to_sparse(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_clone(IntPtr handle);

        public TorchTensor Clone()
        {
            var res = THSTensor_clone(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_contiguous(IntPtr handle);

        public TorchTensor Contiguous()
        {
            var res = THSTensor_contiguous(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_grad(IntPtr handle);

        public TorchTensor Grad()
        {
            var res = THSTensor_grad(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_index_select(IntPtr tensor, long dimension, IntPtr index);

        public TorchTensor IndexSelect(long dimension, TorchTensor index)
        {
            var res = THSTensor_index_select(handle, dimension, index.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_index_fill(IntPtr tensor, long dimension, IntPtr index, IntPtr value);

        public TorchTensor IndexFill(long dimension, TorchTensor index, TorchScalar value)
        {
            var res = THSTensor_index_fill(handle, dimension, index.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_reshape(IntPtr tensor, IntPtr shape, int length);

        public TorchTensor Reshape(params long[] shape)
        {
            unsafe {
                fixed (long* pshape = shape) {
                    var res = THSTensor_reshape(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_squeeze(IntPtr tensor, long dimension);

        public TorchTensor Squeeze(long dimension)
        {
            var res = THSTensor_squeeze(handle, dimension);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_t(IntPtr tensor);

        public TorchTensor T()
        {
            var res = THSTensor_t(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_transpose(IntPtr tensor, long dim1, long dim2);

        public TorchTensor Transpose(long dimension1, long dimension2)
        {
            var res = THSTensor_transpose(handle, dimension1, dimension2);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_transpose_(IntPtr tensor, long dim1, long dim2);

        public TorchTensor TransposeInPlace(long dimension1, long dimension2)
        {
            return new TorchTensor(THSTensor_transpose_(handle, dimension1, dimension2));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_view(IntPtr tensor, IntPtr shape, int length);

        public TorchTensor View(params long[] shape)
        {
            unsafe {
                fixed (long* pshape = shape) {
                    var res = THSTensor_view(handle, (IntPtr)pshape, shape.Length);
                    if (res == IntPtr.Zero)
                        Torch.CheckForErrors();
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add(IntPtr tensor, IntPtr trg, IntPtr alpha);

        public TorchTensor Add(TorchTensor target)
        {
            return Add(target, 1);
        }

        public TorchTensor Add(TorchTensor target, TorchScalar alpha)
        {
            var res = THSTensor_add(handle, target.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_scalar(IntPtr tensor, IntPtr trg, IntPtr alpha);

        public TorchTensor Add(TorchScalar scalar)
        {
            return Add(scalar, 1);
        }
        public TorchTensor Add(TorchScalar scalar, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_scalar(handle, scalar.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(TorchTensor target)
        {
            return AddInPlace(target, 1);
        }

        public TorchTensor AddInPlace(TorchTensor target, TorchScalar alpha)
        {
            return new TorchTensor(THSTensor_add_(handle, target.Handle, alpha.Handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_add_scalar_(IntPtr tensor, IntPtr trg, IntPtr alpha);

        public TorchTensor AddInPlace(TorchScalar scalar)
        {
            return AddInPlace(scalar, 1);
        }

        public TorchTensor AddInPlace(TorchScalar scalar, TorchScalar alpha)
        {
            var res = THSTensor_add_scalar_(handle, scalar.Handle, alpha.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addbmm(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public TorchTensor Addbmm(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addbmm_(IntPtr mat, IntPtr batch1, IntPtr batch2, float beta, float alpha);

        public TorchTensor AddbmmInPlace(TorchTensor batch1, TorchTensor batch2, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_addbmm_(handle, batch1.Handle, batch2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcdiv(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor Addcdiv(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcdiv_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor AddcdivInPlace(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcdiv_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcmul(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor Addcmul(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addcmul_(IntPtr tensor, IntPtr tensor1, IntPtr tensor2, IntPtr value);

        public TorchTensor AddcmulInPlace(TorchTensor tensor1, TorchTensor tensor2, TorchScalar value)
        {
            var res = THSTensor_addcmul_(handle, tensor1.Handle, tensor2.Handle, value.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmm(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public TorchTensor Addmm(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmm_(IntPtr mat, IntPtr mat1, IntPtr mat2, float beta, float alpha);

        public TorchTensor AddmmInPlace(TorchTensor mat1, TorchTensor mat2, float beta, float alpha)
        {
            var res = THSTensor_addmm_(handle, mat1.Handle, mat2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmv(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        public TorchTensor Addmv(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv(handle, mat1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_addmv_(IntPtr mat, IntPtr mat1, IntPtr vec2, float beta, float alpha);

        public TorchTensor AddmvInPlace(TorchTensor mat1, TorchTensor vec2, float beta, float alpha)
        {
            var res = THSTensor_addmv_(handle, mat1.Handle, vec2.Handle, beta, alpha);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_all(IntPtr tensor);

        public TorchTensor All()
        {
            var res = THSTensor_all(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_all_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        public TorchTensor All(long dimension, bool keepDim = false)
        {
            var res = THSTensor_all_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_any(IntPtr tensor);

        public TorchTensor Any()
        {
            var res = THSTensor_any(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_any_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        public TorchTensor Any(long dimension, bool keepDim = false)
        {
            var res = THSTensor_any_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmax(IntPtr tensor);

        public TorchTensor Argmax()
        {
            var res = THSTensor_argmax(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmax_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        public TorchTensor Argmax(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmax_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmin(IntPtr tensor);

        public TorchTensor Argmin()
        {
            var res = THSTensor_argmin(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_argmin_along_dimension(IntPtr tensor, long dimension, bool keep_dim);

        public TorchTensor Argmin(long dimension, bool keepDim = false)
        {
            var res = THSTensor_argmin_along_dimension(handle, dimension, keepDim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos(IntPtr tensor);

        public TorchTensor Cos()
        {
            var res = THSTensor_cos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cos_(IntPtr tensor);

        public TorchTensor CosInPlace()
        {
            var res = THSTensor_cos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin(IntPtr tensor);

        public TorchTensor Sin()
        {
            var res = THSTensor_sin(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sin_(IntPtr tensor);

        public TorchTensor SinInPlace()
        {
            var res = THSTensor_sin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan(IntPtr tensor);

        public TorchTensor Tan()
        {
            var res = THSTensor_tan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tan_(IntPtr tensor);

        public TorchTensor TanInPlace()
        {
            var res = THSTensor_tan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_angle(IntPtr tensor);

        public TorchTensor Angle()
        {
            return new TorchTensor(THSTensor_angle(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin(IntPtr tensor);

        public TorchTensor Asin()
        {
            return new TorchTensor(THSTensor_asin(handle));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_asin_(IntPtr tensor);

        public TorchTensor AsinInPlace()
        {
            var res = THSTensor_asin_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos(IntPtr tensor);

        public TorchTensor Acos()
        {
            var res = THSTensor_acos(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_acos_(IntPtr tensor);

        public TorchTensor AcosInPlace()
        {
            var res = THSTensor_acos_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan(IntPtr tensor);

        public TorchTensor Atan()
        {
            var res = THSTensor_atan(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan_(IntPtr tensor);

        public TorchTensor AtanInPlace()
        {
            var res = THSTensor_atan_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan2(IntPtr tensor, IntPtr other);

        public TorchTensor Atan2(TorchTensor other)
        {
            var res = THSTensor_atan2(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_atan2_(IntPtr tensor, IntPtr other);

        public TorchTensor Atan2InPlace(TorchTensor other)
        {
            var res = THSTensor_atan2_(handle, other.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh(IntPtr tensor);

        public TorchTensor Sinh()
        {
            var res = THSTensor_sinh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sinh_(IntPtr tensor);

        public TorchTensor SinhInPlace()
        {
            var res = THSTensor_sinh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh(IntPtr tensor);

        public TorchTensor Cosh()
        {
            var res = THSTensor_cosh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cosh_(IntPtr tensor);

        public TorchTensor CoshInPlace()
        {
            var res = THSTensor_cosh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh(IntPtr tensor);

        public TorchTensor Tanh()
        {
            var res = THSTensor_tanh(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_tanh_(IntPtr tensor);

        public TorchTensor TanhInPlace()
        {
            var res = THSTensor_tanh_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor(IntPtr tensor);

        public TorchTensor Floor()
        {
            var res = THSTensor_floor(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_floor_(IntPtr tensor);

        public TorchTensor FloorInPlace()
        {
            var res = THSTensor_floor_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_digamma(IntPtr tensor);

        public TorchTensor Digamma()
        {
            var res = THSTensor_digamma(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_digamma_(IntPtr tensor);

        public TorchTensor DigammaInPlace()
        {
            var res = THSTensor_digamma_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lgamma(IntPtr tensor);

        public TorchTensor Lgamma()
        {
            var res = THSTensor_lgamma(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lgamma_(IntPtr tensor);

        public TorchTensor LgammaInPlace()
        {
            var res = THSTensor_lgamma_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mvlgamma(IntPtr tensor, long p);

        public TorchTensor Mvlgamma(long p)
        {
            var res = THSTensor_mvlgamma(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mvlgamma_(IntPtr tensor, long p);

        public TorchTensor MvlgammaInPlace(long p)
        {
            var res = THSTensor_mvlgamma_(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_polygamma(IntPtr tensor, long p);

        public TorchTensor Polygamma(long p)
        {
            var res = THSTensor_polygamma(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_polygamma_(IntPtr tensor, long p);

        public TorchTensor PolygammaInPlace(long p)
        {
            var res = THSTensor_polygamma_(handle, p);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil(IntPtr tensor);

        public TorchTensor Ceil()
        {
            var res = THSTensor_ceil(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ceil_(IntPtr tensor);

        public TorchTensor CeilInPlace()
        {
            var res = THSTensor_ceil_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign(IntPtr tensor);

        public TorchTensor Sign()
        {
            var res = THSTensor_sign(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sign_(IntPtr tensor);

        public TorchTensor SignInPlace()
        {
            var res = THSTensor_sign_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_softplus(IntPtr tensor);

        public TorchTensor Softplus()
        {
            var res = THSTensor_softplus(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu(IntPtr tensor);

        public TorchTensor Relu()
        {
            var res = THSTensor_relu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu_(IntPtr tensor);

        public TorchTensor ReluInPlace()
        {
            var res = THSTensor_relu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu6(IntPtr tensor);

        public TorchTensor Relu6()
        {
            var res = THSTensor_relu6(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_relu6_(IntPtr tensor);

        public TorchTensor Relu6InPlace()
        {
            var res = THSTensor_relu6_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_celu(IntPtr tensor);

        public TorchTensor Celu()
        {
            var res = THSTensor_celu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_celu_(IntPtr tensor);

        public TorchTensor CeluInPlace()
        {
            var res = THSTensor_celu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_elu(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor Elu(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_elu_(IntPtr tensor, IntPtr alpha, IntPtr scale, IntPtr input_scale);

        public TorchTensor EluInPlace(TorchScalar alpha, TorchScalar scale, TorchScalar input_scale)
        {
            var res = THSTensor_elu_(handle, alpha.Handle, scale.Handle, input_scale.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gelu(IntPtr tensor);

        public TorchTensor Gelu()
        {
            var res = THSTensor_gelu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardsigmoid(IntPtr tensor);

        public TorchTensor Hardsigmoid()
        {
            var res = THSTensor_hardsigmoid(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardsigmoid_(IntPtr tensor);

        public TorchTensor HardsigmoidInPlace()
        {
            var res = THSTensor_hardsigmoid_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardswish(IntPtr tensor);

        public TorchTensor Hardswish()
        {
            var res = THSTensor_hardswish(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardswish_(IntPtr tensor);

        public TorchTensor HardswishInPlace()
        {
            var res = THSTensor_hardswish_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardtanh(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor Hardtanh(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_hardtanh_(IntPtr tensor, IntPtr min, IntPtr max);

        public TorchTensor HardtanhInPlace(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_hardtanh_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_leaky_relu(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor LeakyRelu(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_leaky_relu_(IntPtr tensor, IntPtr negative_slope);

        public TorchTensor LeakyReluInPlace(TorchScalar negative_slope)
        {
            var res = THSTensor_leaky_relu_(handle, negative_slope.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round(IntPtr tensor);

        public TorchTensor Round()
        {
            var res = THSTensor_round(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_round_(IntPtr tensor);

        public TorchTensor RoundInPlace()
        {
            var res = THSTensor_round_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_selu(IntPtr tensor);

        public TorchTensor Selu()
        {
            var res = THSTensor_selu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_selu_(IntPtr tensor);

        public TorchTensor SeluInPlace()
        {
            var res = THSTensor_selu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_silu(IntPtr tensor);

        public TorchTensor Silu()
        {
            var res = THSTensor_silu(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_silu_(IntPtr tensor);

        public TorchTensor SiluInPlace()
        {
            var res = THSTensor_silu_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs(IntPtr tensor);

        public TorchTensor Abs()
        {
            var res = THSTensor_abs(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_abs_(IntPtr tensor);

        public TorchTensor AbsInPlace()
        {
            var res = THSTensor_abs_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_sigmoid(IntPtr tensor);

        public TorchTensor LogSigmoid()
        {
            var res = THSTensor_log_sigmoid(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logcumsumexp(IntPtr tensor, long dim);

        public TorchTensor LogCumulativeSumExp(long dim)
        {
            var res = THSTensor_logcumsumexp(handle, dim);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10(IntPtr tensor);

        public TorchTensor Log10()
        {
            var res = THSTensor_log10(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log10_(IntPtr tensor);

        public TorchTensor Log10InPlace()
        {
            var res = THSTensor_log10_(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lerp(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor Lerp(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lerp_(IntPtr tensor, IntPtr end, IntPtr weight);

        public TorchTensor LerpInPlace(TorchTensor end, TorchTensor weight)
        {
            var res = THSTensor_lerp_(handle, end.Handle, weight.Handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log1p(IntPtr tensor);

        public TorchTensor Log1p()
        {
            var res = THSTensor_log1p(handle);
            if (res == IntPtr.Zero)
                Torch.CheckForErrors();
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log1p_(IntPtr tensor);

        public TorchTensor Log1pInPlace()
        {
            var res = THSTensor_log1p_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt(IntPtr tensor);

        public TorchTensor Sqrt()
        {
            var res = THSTensor_sqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sqrt_(IntPtr tensor);

        public TorchTensor SqrtInPlace()
        {
            var res = THSTensor_sqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_rsqrt(IntPtr tensor);

        public TorchTensor Rsqrt()
        {
            var res = THSTensor_rsqrt(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_rsqrt_(IntPtr tensor);

        public TorchTensor RsqrtInPlace()
        {
            var res = THSTensor_rsqrt_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        public static TorchTensor operator -(TorchTensor tensor)
        {
            return tensor.Neg();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg(IntPtr tensor);

        public TorchTensor Neg()
        {
            var res = THSTensor_neg(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_neg_(IntPtr tensor);

        public TorchTensor NegInPlace()
        {
            var res = THSTensor_neg_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_baddbmm(IntPtr batch1, IntPtr batch2, IntPtr mat, float beta,
            float alpha);

        public TorchTensor Baddbmm(TorchTensor batch2, TorchTensor mat, float beta = 1, float alpha = 1)
        {
            var res = THSTensor_baddbmm(handle, batch2.Handle, mat.Handle, beta, alpha);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bmm(IntPtr batch1, IntPtr batch2);

        public TorchTensor Bmm(TorchTensor batch2)
        {
            var res = THSTensor_bmm(handle, batch2.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bincount(IntPtr tensor, IntPtr weights, long minlength);

        /// <summary>
        /// Count the frequency of each value in an array of non-negative ints.
        /// </summary>
        public TorchTensor Bincount(TorchTensor? weights, long minlength = 0)
        {
            var weightsHandle = (weights is null ? IntPtr.Zero : weights.Handle);
            var res = THSTensor_bincount(handle, weightsHandle, minlength);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_and(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseAnd(TorchTensor other)
        {
            var res = THSTensor_bitwise_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_and_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseAndInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_not(IntPtr tensor);

        public TorchTensor BitwiseNot()
        {
            var res = THSTensor_bitwise_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_not_(IntPtr tensor);

        public TorchTensor BitwiseNotInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_or(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseOr(TorchTensor other)
        {
            var res = THSTensor_bitwise_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_or_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseOrInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_xor(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseXor(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bitwise_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor BitwiseXorInPlace(TorchTensor other)
        {
            var res = THSTensor_bitwise_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_and(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalAnd(TorchTensor other)
        {
            var res = THSTensor_logical_and(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_and_(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalAndInPlace(TorchTensor other)
        {
            var res = THSTensor_logical_and_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_not(IntPtr tensor);

        public TorchTensor LogicalNot()
        {
            var res = THSTensor_logical_not(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_not_(IntPtr tensor);

        public TorchTensor LogicalNotInPlace(TorchTensor other)
        {
            var res = THSTensor_logical_not_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_or(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalOr(TorchTensor other)
        {
            var res = THSTensor_logical_or(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_or_(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalOrInPlace(TorchTensor other)
        {
            var res = THSTensor_logical_or_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_xor(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalXor(TorchTensor other)
        {
            var res = THSTensor_logical_xor(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_logical_xor_(IntPtr tensor, IntPtr other);

        public TorchTensor LogicalXorInPlace(TorchTensor other)
        {
            var res = THSTensor_logical_xor_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky(IntPtr input, bool upper);

        public TorchTensor Cholesky(bool upper = false)
        {
            var res = THSTensor_cholesky(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky_inverse(IntPtr input, bool upper);

        public TorchTensor CholeskyInverse(bool upper = false)
        {
            var res = THSTensor_cholesky_inverse(handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cholesky_solve(IntPtr input, IntPtr input2, bool upper);

        public TorchTensor CholeskySolve(TorchTensor input2, bool upper = false)
        {
            var res = THSTensor_cholesky_solve(handle, input2.Handle, upper);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor Clamp(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_(IntPtr input, IntPtr min, IntPtr max);

        public TorchTensor ClampInPlace(TorchScalar min, TorchScalar max)
        {
            var res = THSTensor_clamp_(handle, min.Handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_max(IntPtr input, IntPtr max);

        public TorchTensor ClampMax(TorchScalar max)
        {
            var res = THSTensor_clamp_max(handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_max_(IntPtr input, IntPtr max);

        public TorchTensor ClampMaxInPlace(TorchScalar max)
        {
            var res = THSTensor_clamp_max_(handle, max.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_min(IntPtr input, IntPtr min);

        public TorchTensor ClampMin(TorchScalar min)
        {
            var res = THSTensor_clamp_min(handle, min.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_clamp_min_(IntPtr input, IntPtr min);

        public TorchTensor ClampMinInPlace(TorchScalar min)
        {
            var res = THSTensor_clamp_min_(handle, min.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cross(IntPtr input, IntPtr other, long dim);

        /// <summary>
        /// Returns the cross product of vectors in dimension dim of input and other.
        /// input and other must have the same size, and the size of their dim dimension should be 3.
        /// </summary>
        public TorchTensor Cross(TorchScalar other, long dim)
        {
            var res = THSTensor_cross(handle, other.Handle, dim);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_cummax(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public (TorchTensor values, TorchTensor indexes) CumulativeMax(long dimension)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_cummax(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_cummin(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public (TorchTensor values, TorchTensor indexes) CumulativeMin(long dimension)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_cummin(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cumsum(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor CumulativeSum(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumsum(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cumprod(IntPtr tensor, long dimension, bool has_type, sbyte scalar_type);

        public TorchTensor CumulativeProd(long dimension, ScalarType? type = null)
        {
            var res = THSTensor_cumprod(handle, dimension, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div(IntPtr tensor, IntPtr trg);

        public TorchTensor Div(TorchTensor target)
        {
            var res = THSTensor_div(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Div(TorchScalar target)
        {
            var res = THSTensor_div_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_(IntPtr tensor, IntPtr trg);

        public TorchTensor DivInPlace(TorchTensor target)
        {
            var res = THSTensor_div_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_div_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor DivInPlace(TorchScalar target)
        {
            var res = THSTensor_div_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf(IntPtr tensor);

        public TorchTensor Erf()
        {
            var res = THSTensor_erf(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erf_(IntPtr tensor);

        public TorchTensor ErfInPlace()
        {
            var res = THSTensor_erf_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erfc(IntPtr tensor);

        public TorchTensor Erfc()
        {
            var res = THSTensor_erfc(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erfc_(IntPtr tensor);

        public TorchTensor ErfcInPlace()
        {
            var res = THSTensor_erfc_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erfinv(IntPtr tensor);

        public TorchTensor Erfinv()
        {
            var res = THSTensor_erfinv(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_erfinv_(IntPtr tensor);

        public TorchTensor ErfinvInPlace()
        {
            var res = THSTensor_erfinv_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq(IntPtr tensor, IntPtr trg);

        public TorchTensor Eq(TorchTensor target)
        {
            var res = THSTensor_eq(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_(IntPtr tensor, IntPtr trg);

        public TorchTensor EqInPlace(TorchTensor target)
        {
            var res = THSTensor_eq_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Eq(TorchScalar target)
        {
            var res = THSTensor_eq_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_eq_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor EqInPlace(TorchScalar target)
        {
            var res = THSTensor_eq_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_equal(IntPtr tensor, IntPtr trg);

        public bool Equal(TorchTensor target)
        {
            var res = THSTensor_equal(handle, target.Handle);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern bool THSTensor_allclose(IntPtr tensor, IntPtr trg, double rtol, double atol, bool equal_nan);

        public bool AllClose(TorchTensor target, double rtol = 1e-05, double atol = 1e-08, bool equal_nan = false)
        {
            var res = THSTensor_allclose(handle, target.Handle, rtol, atol, equal_nan);
            Torch.CheckForErrors();
            return res;
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exp(IntPtr tensor);

        public TorchTensor Exp()
        {
            var res = THSTensor_exp(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exp_(IntPtr tensor);

        public TorchTensor ExpInPlace()
        {
            var res = THSTensor_exp_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_expm1(IntPtr tensor);

        public TorchTensor ExpMinusOne()
        {
            var res = THSTensor_expm1(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_expm1_(IntPtr tensor);

        public TorchTensor ExpMinusOneInPlace()
        {
            var res = THSTensor_expm1_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fft(IntPtr tensor, long dim, bool normalized);

        public TorchTensor fft(long dim, bool normalized = false)
        {
            var res = THSTensor_fft(handle, dim, normalized);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ifft(IntPtr tensor, long signal_ndim, bool normalized);

        public TorchTensor ifft(long signal_ndim, bool normalized = false)
        {
            var res = THSTensor_ifft(handle, signal_ndim, normalized);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_irfft(IntPtr tensor, long signal_ndim, bool normalized, bool onesided, IntPtr signal_sizes, int signal_sizes_length);

        public TorchTensor irfft(long signal_ndim, bool normalized = false, bool onesided = true, long[]? signal_sizes = null)
        {
            unsafe {
                fixed (long* psignal_sizes = signal_sizes) {
                    var res =
                        signal_sizes == null
                        ? THSTensor_irfft(handle, signal_ndim, normalized, onesided, IntPtr.Zero, 0)
                        : THSTensor_irfft(handle, signal_ndim, normalized, onesided, (IntPtr)psignal_sizes, signal_sizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_rfft(IntPtr tensor, long signal_ndim, bool normalized, bool onesided);

        public TorchTensor rfft(long signal_ndim, bool normalized = false, bool onesided = true)
        {
            var res = THSTensor_rfft(handle, signal_ndim, normalized, onesided);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_frac(IntPtr tensor);

        public TorchTensor Frac()
        {
            var res = THSTensor_frac(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_frac_(IntPtr tensor);

        public TorchTensor FracInPlace()
        {
            var res = THSTensor_frac_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gcd(IntPtr tensor, IntPtr other);

        public TorchTensor Gcd(TorchTensor other)
        {
            var res = THSTensor_gcd(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gcd_(IntPtr tensor, IntPtr other);

        public TorchTensor GcdInPlace(TorchTensor other)
        {
            var res = THSTensor_gcd_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge(IntPtr tensor, IntPtr trg);

        public TorchTensor Ge(TorchTensor target)
        {
            var res = THSTensor_ge(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_(IntPtr tensor, IntPtr trg);

        public TorchTensor GeInPlace(TorchTensor target)
        {
            var res = THSTensor_ge_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Ge(TorchScalar target)
        {
            var res = THSTensor_ge_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ge_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor GeInPlace(TorchScalar target)
        {
            var res = THSTensor_ge_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt(IntPtr tensor, IntPtr trg);

        public TorchTensor Gt(TorchTensor target)
        {
            var res = THSTensor_gt(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_(IntPtr tensor, IntPtr trg);

        public TorchTensor GtInPlace(TorchTensor target)
        {
            var res = THSTensor_gt_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Gt(TorchScalar target)
        {
            var res = THSTensor_gt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_gt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor GtInPlace(TorchScalar target)
        {
            var res = THSTensor_gt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lcm(IntPtr tensor, IntPtr other);

        public TorchTensor Lcm(TorchTensor other)
        {
            var res = THSTensor_lcm(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lcm_(IntPtr tensor, IntPtr other);

        public TorchTensor LcmInPlace(TorchTensor other)
        {
            var res = THSTensor_lcm_(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le(IntPtr tensor, IntPtr trg);

        public TorchTensor Le(TorchTensor target)
        {
            var res = THSTensor_le(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_(IntPtr tensor, IntPtr trg);

        public TorchTensor LeInPlace(TorchTensor target)
        {
            var res = THSTensor_le_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Le(TorchScalar target)
        {
            var res = THSTensor_le_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_le_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor LeInPlace(TorchScalar target)
        {
            var res = THSTensor_le_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log(IntPtr tensor);

        public TorchTensor Log()
        {
            var res = THSTensor_log(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_(IntPtr tensor);

        public TorchTensor LogInPlace()
        {
            var res = THSTensor_log_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt(IntPtr tensor, IntPtr trg);

        public TorchTensor Lt(TorchTensor target)
        {
            var res = THSTensor_lt(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_(IntPtr tensor, IntPtr trg);

        public TorchTensor LtInPlace(TorchTensor target)
        {
            var res = THSTensor_lt_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Lt(TorchScalar target)
        {
            var res = THSTensor_lt_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_lt_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor LtInPlace(TorchScalar target)
        {
            var res = THSTensor_lt_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_matmul(IntPtr tensor, IntPtr target);

        public TorchTensor MatMul(TorchTensor target)
        {
            var res = THSTensor_matmul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_topk(IntPtr tensor, AllocatePinnedArray allocator, int k,
            long dimension, bool largest, bool sorted);

        public (TorchTensor values, TorchTensor indexes) TopK(int k, int dimension = -1, bool largest = true, bool sorted = true)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_topk(handle, pa.CreateArray, k, dimension, largest, sorted);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_unbind(IntPtr tensor, AllocatePinnedArray allocator, long dimension);

        public TorchTensor[] Unbind(int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_unbind(handle, pa.CreateArray, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }


        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_split_with_size(IntPtr tensor, AllocatePinnedArray allocator, long size, long dimension);

        public TorchTensor[] SplitWithSize(long size, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_split_with_size(handle, pa.CreateArray, size, dimension);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_split_with_sizes(IntPtr tensor, AllocatePinnedArray allocator, IntPtr psizes, int length, long dimension);

        public TorchTensor[] SplitWithSizes(long[] sizes, int dimension = 0)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* psizes = sizes) {
                        THSTensor_split_with_sizes(handle, pa.CreateArray, (IntPtr)psizes, sizes.Length, dimension);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }

            return ptrArray.Select(x => new TorchTensor(x)).ToArray();
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max(IntPtr tensor);

        public TorchTensor Max()
        {
            var res = THSTensor_max(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor Max(TorchTensor other)
        {
            var res = THSTensor_max_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) Max(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_max_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mean(IntPtr tensor);

        public TorchTensor Mean()
        {
            var res = THSTensor_mean(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mean_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        public TorchTensor Mean(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_mean_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_median(IntPtr tensor);

        public TorchTensor Median()
        {
            var res = THSTensor_median(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_min(IntPtr tensor);

        public TorchTensor Min()
        {
            var res = THSTensor_min(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_min_elementwise(IntPtr tensor, IntPtr other);

        public TorchTensor Min(TorchTensor other)
        {
            var res = THSTensor_min_elementwise(handle, other.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_min_along_dimension(IntPtr tensor, AllocatePinnedArray allocator, long dimension,
            bool keep_dim);

        public (TorchTensor values, TorchTensor indexes) Min(long dimension, bool keepDim = false)
        {
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                THSTensor_min_along_dimension(handle, pa.CreateArray, dimension, keepDim);
                Torch.CheckForErrors();
                ptrArray = pa.Array;
            }

            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mm(IntPtr tensor, IntPtr target);

        public TorchTensor Mm(TorchTensor target)
        {
            var res = THSTensor_mm(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul(IntPtr tensor, IntPtr target);

        public TorchTensor Mul(TorchTensor target)
        {
            var res = THSTensor_mul(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor Mul(TorchScalar scalar)
        {
            var res = THSTensor_mul_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_(IntPtr tensor, IntPtr target);

        public TorchTensor MulInPlace(TorchTensor target)
        {
            var res = THSTensor_mul_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_mul_scalar_(IntPtr tensor, IntPtr target);

        public TorchTensor MulInPlace(TorchScalar target)
        {
            var res = THSTensor_mul_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne(IntPtr tensor, IntPtr trg);

        public TorchTensor Ne(TorchTensor target)
        {
            var res = THSTensor_ne(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_(IntPtr tensor, IntPtr trg);

        public TorchTensor NeInPlace(TorchTensor target)
        {
            var res = THSTensor_ne_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Ne(TorchScalar target)
        {
            var res = THSTensor_ne_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_ne_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor NeInPlace(TorchScalar target)
        {
            var res = THSTensor_ne_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_dist(IntPtr tensor, IntPtr other, float p);

        public TorchTensor Dist(TorchTensor other, float p = 2.0f)
        {
            var res = THSTensor_dist(handle, other.Handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm(IntPtr tensor, float p);

        public TorchTensor Norm(float p = 2.0f)
        {
            var res = THSTensor_norm(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_norm_along_dimension(IntPtr tensor, int dimension, bool keepdim, float p);

        public TorchTensor Norm(int dimension, bool keepdim = false, float p = 2.0f)
        {
            var res = THSTensor_norm_along_dimension(handle, dimension, keepdim, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow(IntPtr tensor, IntPtr exponent);

        public TorchTensor Pow(TorchTensor exponent)
        {
            var res = THSTensor_pow(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_(IntPtr tensor, IntPtr exponent);

        public TorchTensor PowInPlace(TorchTensor exponent)
        {
            var res = THSTensor_pow_(handle, exponent.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor Pow(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_pow_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor PowInPlace(TorchScalar scalar)
        {
            var res = THSTensor_pow_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_prelu(IntPtr tensor, IntPtr trg);

        public TorchTensor Prelu(TorchTensor target)
        {
            var res = THSTensor_prelu(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder(IntPtr tensor, IntPtr trg);

        public TorchTensor Remainder(TorchTensor target)
        {
            var res = THSTensor_remainder(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_(IntPtr tensor, IntPtr trg);

        public TorchTensor RemainderInPlace(TorchTensor target)
        {
            var res = THSTensor_remainder_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor Remainder(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_remainder_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor RemainderInPlace(TorchScalar scalar)
        {
            var res = THSTensor_remainder_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fmod(IntPtr tensor, IntPtr trg);

        public TorchTensor Fmod(TorchTensor target)
        {
            var res = THSTensor_fmod(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fmod_(IntPtr tensor, IntPtr trg);

        public TorchTensor FmodInPlace(TorchTensor target)
        {
            var res = THSTensor_fmod_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fmod_scalar(IntPtr tensor, IntPtr scalar);

        public TorchTensor Fmod(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_fmod_scalar_(IntPtr tensor, IntPtr scalar);

        public TorchTensor FmodInPlace(TorchScalar scalar)
        {
            var res = THSTensor_fmod_scalar_(handle, scalar.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_renorm(IntPtr tensor, float p, long dim, float maxnorm);

        public TorchTensor Renorm(TorchScalar scalar, float p, long dim, float maxnorm)
        {
            var res = THSTensor_renorm(handle, p, dim, maxnorm);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sigmoid(IntPtr tensor);

        public TorchTensor Sigmoid()
        {
            var res = THSTensor_sigmoid(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sigmoid_(IntPtr tensor);

        public TorchTensor SigmoidInPlace()
        {
            var res = THSTensor_sigmoid_(handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub(IntPtr tensor, IntPtr trg);

        public TorchTensor Sub(TorchTensor target)
        {
            var res = THSTensor_sub(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_scalar(IntPtr tensor, IntPtr trg);

        public TorchTensor Sub(TorchScalar target)
        {
            var res = THSTensor_sub_scalar(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_(IntPtr tensor, IntPtr trg);

        public TorchTensor SubInPlace(TorchTensor target)
        {
            var res = THSTensor_sub_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sub_scalar_(IntPtr tensor, IntPtr trg);

        public TorchTensor SubInPlace(TorchScalar target)
        {
            var res = THSTensor_sub_scalar_(handle, target.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum(IntPtr tensor, bool has_type, sbyte scalar_type);

        /// <summary>
        /// Returns the sum of all elements in the :attr:`input` tensor.
        /// </summary>
        public TorchTensor Sum(ScalarType? type = null)
        {
            var res = THSTensor_sum(handle, type.HasValue, (sbyte)type.GetValueOrDefault());
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_sum_along_dimensions(IntPtr tensor, IntPtr dimensions, int length, bool keepdim, bool has_type, sbyte scalar_type);

        /// <summary>
        ///  Returns the sum of each row of the input tensor in the given dimensions.
        /// </summary>
        public TorchTensor Sum(long[] dimensions, bool keepDimension = false, ScalarType? type = null)
        {
            unsafe {
                fixed (long* pdims = dimensions) {
                    var res = THSTensor_sum_along_dimensions(handle, (IntPtr)pdims, dimensions.Length, keepDimension, type.HasValue, (sbyte)type.GetValueOrDefault());
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bernoulli(IntPtr tensor, double p);

        public TorchTensor Bernoulli(double p)
        {
            var res = THSTensor_bernoulli(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_bernoulli_(IntPtr tensor, double p);

        public TorchTensor BernoulliInPlace(double p)
        {
            var res = THSTensor_bernoulli_(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_cauchy_(IntPtr tensor, double median, double sigma);

        public TorchTensor CauchyInPlace(double median = 0.0, double sigma = 1.0)
        {
            var res = THSTensor_cauchy_(handle, median, sigma);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_exponential_(IntPtr tensor, double lambd);

        public TorchTensor ExponentialInPlace(double lambd = 1.0)
        {
            var res = THSTensor_exponential_(handle, lambd);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_geometric_(IntPtr tensor, double p);

        public TorchTensor GeometricInPlace(double p)
        {
            var res = THSTensor_geometric_(handle, p);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_log_normal_(IntPtr tensor, double mean, double std);

        public TorchTensor LogNormalInPlace(double mean = 1.0, double std = 2.0)
        {
            var res = THSTensor_log_normal_(handle, mean, std);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_normal_(IntPtr tensor, double mean, double std);

        public TorchTensor NormalInPlace(double mean = 0.0, double std = 1.0)
        {
            var res = THSTensor_normal_(handle, mean, std);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_uniform_(IntPtr tensor, double from, double to);

        public TorchTensor UniformInPlace(double from = 0.0, double to = 1.0)
        {
            var res = THSTensor_uniform_(handle, from, to);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }
        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_multinomial(IntPtr tensor, double num_samples, bool replacement);

        public TorchTensor Multinomial(double num_samples, bool replacement = false)
        {
            var res = THSTensor_multinomial(handle, num_samples, replacement);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_expand(IntPtr tensor, IntPtr psizes, int length, bool isImplicit);

        /// <summary>
        ///  Returns a new view of the tensor with singleton dimensions expanded to a larger size.
        /// </summary>
        public TorchTensor Expand(long[] sizes, bool isImplicit = false)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_expand(handle, (IntPtr)psizes, sizes.Length, isImplicit);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randn_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public TorchTensor RandomNInPlace(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_randn_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_rand_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a uniform distribution in [0, 1).
        /// </summary>
        public TorchTensor RandInPlace(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_rand_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randint_out(long high, IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with random values taken from a normal distribution with mean 0 and variance 1.
        /// </summary>
        public TorchTensor RandomIntegersInPlace(long high, long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_randint_out(high, (IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }
        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_randperm_out(long n, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be a 1-D tensor of size [n] with a random permutation of [0, n).
        /// </summary>
        public TorchTensor RandomPermutationInPlace(long n)
        {
            var res = THSTensor_randperm_out(n, handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_arange_out(IntPtr start, IntPtr strp, IntPtr step, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to be filled with with values from interval [start, end) and
		/// common difference step, starting from start.
        /// </summary>
        public TorchTensor ArangeInPlace(TorchScalar start, TorchScalar stop, TorchScalar step)
        {
            var res = THSTensor_arange_out(start.Handle, stop.Handle, step.Handle, handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_permute(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 1
        /// </summary>
        public TorchTensor Permute(long[] permutation)
        {
            unsafe {
                fixed (long* pPermutation = permutation) {
                    var res = THSTensor_permute(handle, (IntPtr)pPermutation, permutation.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_ones_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 1
        /// </summary>
        public TorchTensor OnesInPlace(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_ones_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_zeros_out(IntPtr psizes, int length, IntPtr tensorOut);

        /// <summary>
        ///  Mutates the tensor to have the given size with all values set to 0
        /// </summary>
        public TorchTensor ZerosInPlace(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_zeros_out((IntPtr)psizes, sizes.Length, handle);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_scatter(IntPtr tensor, long dimension, IntPtr index, IntPtr source);

        /// <summary>
        ///  Writes all values from the tensor src into self at the indices specified in the index tensor. For each
        ///  value in src, its output index is specified by its index in src for dimension != dim and by the #
        ///  corresponding value in index for dimension = dim.
        /// </summary>
        public TorchTensor Scatter(long dimension, TorchTensor index, TorchTensor src)
        {
            var res = THSTensor_scatter(handle, dimension, index.Handle, src.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_gather(IntPtr tensor, long dimension, IntPtr index);

        /// <summary>
        /// Gathers values along an axis specified by dim.
        /// </summary>
        public TorchTensor Gather(long dimension, TorchTensor index)
        {
            var res = THSTensor_gather(handle, dimension, index.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_flip(IntPtr tensor, IntPtr psizes, int length);

        /// <summary>
        ///  Reverse the order of a n-D tensor along given axis in dims.
        /// </summary>
        public TorchTensor Flip(long[] sizes)
        {
            unsafe {
                fixed (long* psizes = sizes) {
                    var res = THSTensor_flip(handle, (IntPtr)psizes, sizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_narrow(IntPtr tensor, long dimension, long start, long length);

        /// <summary>
        ///  Returns a new tensor that is a narrowed version of the input along one dimension. The
        /// dimension is input from start to start + length. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Narrow(long dimension, long start, long length)
        {
            var res = THSTensor_narrow(handle, dimension, start, length);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_slice(IntPtr tensor, long dimension, long start, long length, long step);

        /// <summary>
        ///  Returns a new tensor that is a sliced version of the input along one dimension. The
        /// dimension is input from start to finish-1. The
        /// returned tensor and the input tensor share the same underlying storage.
        /// </summary>
        public TorchTensor Slice(long dimension, long start, long finish, long step)
        {
            var res = THSTensor_slice(handle, dimension, start, finish, step);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_unsqueeze(IntPtr tensor, long dimension);

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv1D(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddingArray = new long[] { padding ?? 0 };
            var dilationArray = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = paddingArray, pdilation = dilationArray) {
                    var res =
                        THSTensor_conv1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddingArray.Length,
                            (IntPtr)pdilation, dilationArray.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }


        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv2D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_conv2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor Conv3D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            padding = (padding == null) ? new long[] { 0 } : padding;
            dilation = (dilation == null) ? new long[] { 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_conv3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose1d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose1D(TorchTensor weight, TorchTensor? bias = null,
            long? stride = null,
            long? padding = null,
            long? outputPadding = null,
            long? dilation = null,
            long groups = 1)
        {
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var outputPaddings = new long[] { outputPadding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = paddings, poutputPadding = outputPaddings, pdilation = dilations) {
                    var res =
                        THSTensor_conv_transpose1d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)poutputPadding, outputPaddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose2d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose2D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation) {
                    var res =
                        THSTensor_conv_transpose2d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_conv_transpose3d(IntPtr input, IntPtr weight, IntPtr bias,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr outputPadding, int outputPaddingLength,
                IntPtr dilation, int dilationLength,
                long groups);

        public TorchTensor ConvTranspose3D(TorchTensor weight, TorchTensor? bias = null,
            long[]? strides = null,
            long[]? padding = null,
            long[]? outputPadding = null,
            long[]? dilation = null,
            long groups = 1)
        {
            strides = (strides == null) ? new long[] { 1, 1, 1 } : strides;
            padding = (padding == null) ? new long[] { 0, 0, 0 } : padding;
            outputPadding = (outputPadding == null) ? new long[] { 0, 0, 0 } : outputPadding;
            dilation = (dilation == null) ? new long[] { 1, 1, 1 } : dilation;
            var biasHandle = (bias is null ? IntPtr.Zero : bias.Handle);
            unsafe {
                fixed (long* pstrides = strides, ppadding = padding, poutputPadding = outputPadding, pdilation = dilation) {
                    var res =
                        THSTensor_conv_transpose3d(handle, weight.Handle, biasHandle,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)poutputPadding, outputPadding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            groups);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool1D(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                    var res =
                        THSTensor_max_pool1d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max_pool1d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool1DWithIndices(long kernelSize, long? stride = null,
            long? padding = null, long? dilation = null, bool ceil_mode = false)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            var dilations = new long[] { dilation ?? 1 };
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings, pdilation = dilations) {
                        THSTensor_max_pool1d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            (IntPtr)pdilation, dilations.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool2D(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_max_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max_pool2d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool2DWithIndices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool2d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_max_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public TorchTensor MaxPool3D(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            unsafe {
                fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                    var res =
                        THSTensor_max_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern void THSTensor_max_pool3d_with_indices(IntPtr input, AllocatePinnedArray allocator,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                IntPtr dilation, int dilationLength,
                bool ceil_mode);

        public (TorchTensor output, TorchTensor indices) MaxPool3DWithIndices(long[] kernelSize, long[]? strides = null,
            long[]? padding = null, long[]? dilation = null, bool ceil_mode = false)
        {
            strides = strides ?? kernelSize.Select(x => 1L).ToArray();
            padding = padding ?? kernelSize.Select(x => 0L).ToArray();
            dilation = dilation ?? kernelSize.Select(x => 1L).ToArray();
            IntPtr[] ptrArray;

            using (var pa = new PinnedArray<IntPtr>()) {
                unsafe {
                    fixed (long* pkernelSize = kernelSize, pstrides = strides, ppadding = padding, pdilation = dilation) {
                        THSTensor_max_pool3d_with_indices(handle,
                            pa.CreateArray,
                            (IntPtr)pkernelSize, kernelSize.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, padding.Length,
                            (IntPtr)pdilation, dilation.Length,
                            ceil_mode);
                        Torch.CheckForErrors();
                    }
                }
                ptrArray = pa.Array;
            }
            return (new TorchTensor(ptrArray[0]), new TorchTensor(ptrArray[1]));
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxunpool2d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength);

        public TorchTensor MaxUnpool2D(TorchTensor indices, long[] outputSize)
        {
            unsafe {
                fixed (long* poutputSize = outputSize) {
                    var res = THSTensor_maxunpool2d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_maxunpool3d(IntPtr input, IntPtr indices, IntPtr outputSize, int outputSizeLength, IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength);

        public TorchTensor MaxUnpool3D(TorchTensor indices, long[] outputSize, long[] strides, long[] padding)
        {
            unsafe {
                fixed (long* poutputSize = outputSize, pstrides = strides, ppadding = padding) {
                    var res = THSTensor_maxunpool3d(handle, indices.Handle,
                        (IntPtr)poutputSize, outputSize.Length,
                        (IntPtr)pstrides, strides.Length,
                        (IntPtr)ppadding, padding.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_avg_pool1d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor AvgPool1D(long kernelSize, long? stride = null,
            long? padding = null, bool ceil_mode = false, bool count_include_pad = true)
        {
            var kernelSizes = new long[] { kernelSize };
            var strides = new long[] { stride ?? 1 };
            var paddings = new long[] { padding ?? 0 };
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool1d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_avg_pool2d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor AvgPool2D(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_avg_pool3d(IntPtr input,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad);

        public TorchTensor AvgPool3D(long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d(handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_avg_pool2d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor AvgPool2DBackward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool2d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput,
                IntPtr kernelSize, int kernelSizeLength,
                IntPtr strides, int stridesLength,
                IntPtr padding, int paddingLength,
                bool ceil_mode,
                bool count_include_pad,
                long divisorOverride);

        public TorchTensor AvgPool3DBackward(TorchTensor originalInput,
            long[] kernelSizes,
            long[]? strides = null,
            long[]? paddings = null,
            bool ceil_mode = false,
            bool count_include_pad = true,
            long divisorOverride = 0)
        {
            strides = (strides == null) ? new long[] { 1 } : strides;
            paddings = (paddings == null) ? new long[] { 0 } : paddings;
            unsafe {
                fixed (long* pkernelSize = kernelSizes, pstrides = strides, ppadding = paddings) {
                    var res =
                        THSTensor_avg_pool3d_backward(handle, originalInput.Handle,
                            (IntPtr)pkernelSize, kernelSizes.Length,
                            (IntPtr)pstrides, strides.Length,
                            (IntPtr)ppadding, paddings.Length,
                            ceil_mode,
                            count_include_pad,
                            divisorOverride);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_adaptive_avg_pool1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        public TorchTensor AdaptiveAvgPool1D(long outputSize)
        {
            var outputSizes = new long[] { outputSize };
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool1d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_adaptive_avg_pool2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength);

        public TorchTensor AdaptiveAvgPool2D(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool2d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_adaptive_avg_pool3d(IntPtr input, IntPtr outputSize, int outputSizeLength);

        public TorchTensor AdaptiveAvgPool3D(long[] outputSizes)
        {
            unsafe {
                fixed (long* poutputSize = outputSizes) {
                    var res =
                        THSTensor_adaptive_avg_pool3d(handle, (IntPtr)poutputSize, outputSizes.Length);
                    if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                    return new TorchTensor(res);
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_adaptive_avg_pool3d_backward(IntPtr gradOutput, IntPtr originalInput);

        public TorchTensor AdaptiveAvgPool3DBackward(TorchTensor originalInput)
        {
            var res = THSTensor_adaptive_avg_pool3d_backward(handle, originalInput.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest1d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest1D(long? outputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest1d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest1DBackward(long? outputSize, long inputSize, double? scaleFactor)
        {
            var outputSizes = outputSize.HasValue ? new long[] { outputSize.Value } : null;
            var outputSizesLength = outputSize.HasValue ? 1 : 0;
            var inputSizes = new long[] { inputSize };
            var scaleFactors = scaleFactor.HasValue ? new double[] { scaleFactor.Value } : null;
            var scaleFactorsLength = scaleFactor.HasValue ? 1 : 0;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest1d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest2d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest2D(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest2d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest2DBackward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest2d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest3d_backward(IntPtr grad_output,
                IntPtr outputSize, int outputSizeLength,
                IntPtr inputSize, int inputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest3DBackward(long[] inputSizes, long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes, pinputSizes = inputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d_backward(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pinputSizes, inputSizes.Length,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }

        [DllImport("LibTorchSharp")]
        private static extern IntPtr THSTensor_upsample_nearest3d(IntPtr input,
                IntPtr outputSize, int outputSizeLength,
                IntPtr scaleFactors, int scaleFactorsLength);

        public TorchTensor UpSampleNearest3D(long[]? outputSizes = null, double[]? scaleFactors = null)
        {
            var outputSizesLength = outputSizes == null ? 0 : outputSizes.Length;
            var scaleFactorsLength = scaleFactors == null ? 0 : scaleFactors.Length;
            unsafe {
                fixed (long* poutputSizes = outputSizes) {
                    fixed (double* pscaleFactors = scaleFactors) {
                        var res =
                            THSTensor_upsample_nearest3d(handle,
                                (IntPtr)poutputSizes, outputSizesLength,
                                (IntPtr)pscaleFactors, scaleFactorsLength);
                        if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                        return new TorchTensor(res);
                    }
                }
            }
        }


        /// <summary>
        ///  Returns a new tensor with a dimension of size one inserted at the specified position.
        ///  The returned tensor shares the same underlying data with this tensor.
        /// </summary>

        public TorchTensor Unsqueeze(long dimension)
        {
            var res = THSTensor_unsqueeze(handle, dimension);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor(res);
        }

        // Operators overloading

        public static TorchTensor operator ==(TorchTensor left, TorchTensor right)
        {
            return left.Eq(right);
        }

        public static TorchTensor operator ==(TorchTensor left, TorchScalar right)
        {
            return left.Eq(right);
        }

        public static TorchTensor operator ==(TorchScalar left, TorchTensor right)
        {
            return right.Eq(left);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchTensor right)
        {
            return left.Ne(right);
        }

        public static TorchTensor operator !=(TorchTensor left, TorchScalar right)
        {
            return left.Ne(right);
        }

        public static TorchTensor operator !=(TorchScalar left, TorchTensor right)
        {
            return right.Ne(left);
        }

        public static TorchTensor operator +(TorchTensor left, TorchTensor right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(TorchTensor left, TorchScalar right)
        {
            return left.Add(right);
        }

        public static TorchTensor operator +(TorchScalar left, TorchTensor right)
        {
            return right.Add(left);
        }

        public static TorchTensor operator *(TorchTensor left, TorchTensor right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(TorchTensor left, TorchScalar right)
        {
            return left.Mul(right);
        }

        public static TorchTensor operator *(TorchScalar left, TorchTensor right)
        {
            return right.Mul(left);
        }

        public static TorchTensor operator -(TorchTensor left, TorchTensor right)
        {
            return left.Sub(right);
        }

        public static TorchTensor operator -(TorchTensor left, TorchScalar right)
        {
            return left.Sub(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchTensor right)
        {
            return left.Div(right);
        }

        public static TorchTensor operator /(TorchTensor left, TorchScalar right)
        {
            return left.Div(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchTensor right)
        {
            return left.Remainder(right);
        }

        public static TorchTensor operator %(TorchTensor left, TorchScalar right)
        {
            return left.Remainder(right);
        }

        public static TorchTensor operator <(TorchTensor left, TorchTensor right)
        {
            return left.Lt(right);
        }

        public static TorchTensor operator <(TorchTensor left, TorchScalar right)
        {
            return left.Lt(right);
        }

        public static TorchTensor operator <(TorchScalar left, TorchTensor right)
        {
            return right.Gt(left);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchTensor right)
        {
            return left.Le(right);
        }

        public static TorchTensor operator <=(TorchTensor left, TorchScalar right)
        {
            return left.Le(right);
        }

        public static TorchTensor operator <=(TorchScalar left, TorchTensor right)
        {
            return right.Ge(left);
        }

        public static TorchTensor operator >(TorchTensor left, TorchTensor right)
        {
            return left.Gt(right);
        }

        public static TorchTensor operator >(TorchTensor left, TorchScalar right)
        {
            return left.Gt(right);
        }

        public static TorchTensor operator >(TorchScalar left, TorchTensor right)
        {
            return right.Lt(left);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchTensor right)
        {
            return left.Ge(right);
        }

        public static TorchTensor operator >=(TorchTensor left, TorchScalar right)
        {
            return left.Ge(right);
        }

        public static TorchTensor operator >=(TorchScalar left, TorchTensor right)
        {
            return right.Le(left);
        }

        /// <summary>
        ///   Get a string representation of the tensor.
        /// </summary>
        public override string ToString()
        {
            if (Handle == IntPtr.Zero) return "";

            var n = Dimensions;
            if (n == 0)
                return "[]";

            var sb = new StringBuilder("[");
            for (var i = 0; i < n; i++) {
                sb.Append(GetTensorDimension(i));
                if (i + 1 < n)
                    sb.Append("x");
            }

            sb.Append("]");
            sb.Append($", device = {DeviceString}");

            if (Type == ScalarType.Float32) {
                if (n == 1) {
                    sb.AppendLine();
                    sb.Append("[");
                    for (int j = 0; j < Math.Min(10, GetTensorDimension(0)); j++) {
                        sb.AppendFormat($"{this[j].ToSingle():0.##},");
                    }
                    sb.Length = sb.Length - 1;
                    sb.Append("]");
                } else if (n == 2) {
                    for (var i = 0; i < Math.Min(10, GetTensorDimension(0)); i++) {
                        sb.AppendLine();
                        sb.Append("[");
                        for (int j = 0; j < Math.Min(10, GetTensorDimension(1)); j++) {
                            sb.AppendFormat($"{this[i, j].ToSingle():0.##},");
                        }
                        sb.Length = sb.Length - 1;
                        sb.Append("]");
                    }
                }
            }


            return sb.ToString();
        }

        public static explicit operator float(TorchTensor value) => value.ToSingle();
        public static explicit operator double(TorchTensor value) => value.ToDouble();
        public static explicit operator sbyte(TorchTensor value) => value.ToSByte();
        public static explicit operator byte(TorchTensor value) => value.ToByte();
        public static explicit operator short(TorchTensor value) => value.ToInt16();
        public static explicit operator int(TorchTensor value) => value.ToInt32();
        public static explicit operator long(TorchTensor value) => value.ToInt64();
        public static explicit operator bool(TorchTensor value) => value.ToBoolean();
    }

    public enum ScalarType : sbyte
    {
        Byte = 0,
        Int8 = 1,
        Int16 = 2,
        Int32 = 3,
        Int64 = 4,
        Float16 = 5,
        Float32 = 6,
        Float64 = 7,
        //ComplexFloat16 = 8,
        ComplexFloat32 = 9,
        ComplexFloat64 = 10,
        Bool = 11,
        //QInt8 = 12,
        //QUInt8 = 13,
        //QUInt32 = 14,
        BFloat16 = 15
    }

    public static class TensorExtensionMethods
    {
        public static TorchTensor ToTorchTensor<T>(this T[] rawArray, long[] dimensions, bool doCopy = false, bool requiresGrad = false)
        {
            var array = doCopy ? (T[])rawArray.Clone() : rawArray;

            switch (true) {
            case bool _ when typeof(T) == typeof(byte): {
                    return ByteTensor.From(array as byte[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(sbyte): {
                    return Int8Tensor.From(array as sbyte[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(short): {
                    return Int16Tensor.From(array as short[], dimensions, requiresGrad); ;
                }
            case bool _ when typeof(T) == typeof(int): {
                    return Int32Tensor.From(array as int[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(long): {
                    return Int64Tensor.From(array as long[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(double): {
                    return Float64Tensor.From(array as double[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(float): {
                    return Float32Tensor.From(array as float[], dimensions, requiresGrad);
                }
            case bool _ when typeof(T) == typeof(bool): {
                    return BoolTensor.From(array as bool[], dimensions, requiresGrad);
                }
            //case bool _ when typeof(T) == typeof(System.Numerics.Complex):
            //    {
            //        return ComplexFloat64Tensor.From(array as System.Numerics.Complex[], dimensions, requiresGrad);
            //    }
            default: throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
            }
        }

        public static TorchTensor ToTorchTensor<T>(this T scalar, DeviceType deviceType = DeviceType.CPU, int deviceIndex = 0, bool requiresGrad = false) where T : struct
        {
            if (requiresGrad && typeof(T) != typeof(float) && typeof(T) != typeof(double)) {
                throw new ArgumentException(nameof(requiresGrad), "Only floating point types support gradients.");
            }

            if (typeof(T) == typeof(byte))
                return ByteTensor.From((byte)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(sbyte))
                return Int8Tensor.From((sbyte)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(short))
                return Int16Tensor.From((short)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(int))
                return Int32Tensor.From((int)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(long))
                return Int64Tensor.From((long)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(double))
                return Float64Tensor.From((double)(object)scalar, deviceType, deviceIndex, requiresGrad);
            if (typeof(T) == typeof(float))
                return Float32Tensor.From((float)(object)scalar, deviceType, deviceIndex, requiresGrad);
            throw new NotImplementedException($"Creating tensor of type {typeof(T)} is not supported.");
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_cat(IntPtr tensor, int len, long dim);

        public static TorchTensor Cat(this TorchTensor[] tensors, long dimension)
        {
            if (tensors.Length == 0) {
                throw new ArgumentException(nameof(tensors));
            }
            if (tensors.Length == 1) {
                return tensors[0];
            }

            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                return new TorchTensor(THSTensor_cat(tensorsRef, parray.Array.Length, dimension));
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_stack(IntPtr tensor, int len, long dim);

        public static TorchTensor Stack(this TorchTensor[] tensors, long dimension)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_stack(tensorsRef, parray.Array.Length, dimension);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        [DllImport("LibTorchSharp")]
        extern static IntPtr THSTensor_einsum([MarshalAs(UnmanagedType.LPStr)] string location, IntPtr tensors, int len);

        public static TorchTensor Einsum(string equation, params TorchTensor[] tensors)
        {
            using (var parray = new PinnedArray<IntPtr>()) {
                IntPtr tensorsRef = parray.CreateArray(tensors.Select(p => p.Handle).ToArray());

                var res = THSTensor_einsum(equation, tensorsRef, parray.Array.Length);
                if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
                return new TorchTensor(res);
            }
        }

        public static float ToSingle(this TorchTensor value) => value.ToScalar().ToSingle();
        public static double ToDouble(this TorchTensor value) => value.ToScalar().ToDouble();
        public static sbyte ToSByte(this TorchTensor value) => value.ToScalar().ToSByte();
        public static byte ToByte(this TorchTensor value) => value.ToScalar().ToByte();
        public static short ToInt16(this TorchTensor value) => value.ToScalar().ToInt16();
        public static int ToInt32(this TorchTensor value) => value.ToScalar().ToInt32();
        public static long ToInt64(this TorchTensor value) => value.ToScalar().ToInt64();
        public static bool ToBoolean(this TorchTensor value) => value.ToScalar().ToBoolean();
    }
}