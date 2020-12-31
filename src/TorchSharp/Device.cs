using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TorchSharp
{
    [StructLayout(LayoutKind.Sequential)]
    public struct Device
    {
        public Device(DeviceType device_type, int device_index = -1)
        {
            _type = (short)device_type;
            _index = (short)device_index;
        }

        private readonly short _type;
        private readonly short _index;

        public DeviceType Type => (DeviceType)_type;
        public int Index => _index;
    }
}
