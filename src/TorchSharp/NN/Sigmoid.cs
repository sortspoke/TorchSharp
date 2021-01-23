// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a Sigmoid module.
    /// </summary>
    public class Sigmoid : Module
    {
        internal Sigmoid (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_Sigmoid_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_Sigmoid_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Sigmoid).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_Sigmoid_ctor (bool inplace, out IntPtr pBoxedModule);

        static public Sigmoid Sigmoid (bool inPlace = false)
        {
            var handle = THSNN_Sigmoid_ctor (inPlace, out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Sigmoid (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Sigmoid (TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.Sigmoid (inPlace)) {
                return m.Forward (x);
            }
        }
    }

}
