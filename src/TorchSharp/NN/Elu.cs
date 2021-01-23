// Copyright (c) Microsoft Corporation and contributors.  All Rights Reserved.  See License.txt in the project root for license information.
using System;
using System.Runtime.InteropServices;
using TorchSharp.Tensor;

namespace TorchSharp.NN
{
    /// <summary>
    /// This class is used to represent a ELU module.
    /// </summary>
    public class Elu : Module
    {
        internal Elu (IntPtr handle, IntPtr boxedHandle) : base (handle, boxedHandle) { }

        [DllImport ("LibTorchSharp")]
        private static extern IntPtr THSNN_ELU_forward (Module.HType module, IntPtr tensor);

        public TorchTensor Forward (TorchTensor tensor)
        {
            var res = THSNN_ELU_forward (handle, tensor.Handle);
            if (res == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new TorchTensor (res);
        }

        public override string GetName ()
        {
            return typeof (Elu).Name;
        }
    }

    public static partial class Modules
    {
        [DllImport ("LibTorchSharp")]
        extern static IntPtr THSNN_ELU_ctor (out IntPtr pBoxedModule);

        static public Elu Elu (bool inPlace = false)
        {
            var handle = THSNN_ELU_ctor (out var boxedHandle);
            if (handle == IntPtr.Zero) { Torch.CheckForErrors(); }
            return new Elu (handle, boxedHandle);
        }
    }
    public static partial class Functions
    {
        static public TorchTensor Elu (TorchTensor x, bool inPlace = false)
        {
            using (var m = Modules.Elu (inPlace)) {
                return m.Forward (x);
            }
        }
    }

}
