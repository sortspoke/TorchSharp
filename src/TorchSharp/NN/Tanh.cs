// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a Tanh module.
    /// </summary>
    public class Tanh : Module
    {
        internal Tanh (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Tanh_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Tanh_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Tanh).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Tanh_ctor (bool inplace, out IntPtr pBoxedModule);

        static public Tanh Tanh (bool inPlace = false)
        {
            var handle = THSNN_Tanh_ctor (inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Tanh (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Tanh (TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.Tanh (inPlace)) {
                return m.Forward (x);
            }
        }
    }

}
