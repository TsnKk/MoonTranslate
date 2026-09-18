import io
import struct
import unittest
from unittest.mock import patch,Mock
import numpy as np
import inpaint_gpu
import manga_inpaint


class FragmentedStream(io.BytesIO):
    def read(self,n):return super().read(min(n,3))


class GpuTests(unittest.TestCase):
    def test_packet_handles_short_pipe_reads(self):
        stream=io.BytesIO();inpaint_gpu.write_packet(stream,b'abcdefghijk')
        self.assertEqual(inpaint_gpu.read_packet(FragmentedStream(stream.getvalue())),b'abcdefghijk')

    def test_packet_rejects_truncated_and_oversized_data(self):
        with self.assertRaises(EOFError):inpaint_gpu.read_packet(io.BytesIO(b'\x04\x00'))
        with self.assertRaises(EOFError):inpaint_gpu.read_packet(io.BytesIO(struct.pack('<I',8)+b'x'))
        with self.assertRaises(ValueError):inpaint_gpu.read_packet(io.BytesIO(struct.pack('<I',inpaint_gpu.LIMIT+1)))

    def test_gpu_failure_retries_original_model_on_cpu(self):
        gpu=Mock(spec=inpaint_gpu.DmlSession);gpu.run.side_effect=RuntimeError('device lost')
        cpu=Mock();cpu.run.return_value=[np.zeros((1,3,512,512),np.float32)]
        image=np.full((50,60,3),123,np.uint8);mask=np.zeros((50,60),np.uint8);mask[10:20,10:20]=255
        with patch.object(manga_inpaint,'available',return_value=True),patch.object(manga_inpaint,'_session',gpu),patch.object(manga_inpaint,'cpu_session',return_value=cpu):
            result=manga_inpaint.inpaint(image,mask)
            self.assertIs(manga_inpaint._session,cpu)
        gpu.close.assert_called_once()
        np.testing.assert_array_equal(result[mask==0],image[mask==0])
        self.assertTrue(np.all(result[mask>0]==0))

    def test_missing_gpu_runtime_keeps_cpu_path(self):
        cpu=Mock()
        with patch.object(inpaint_gpu,'available',return_value=False),patch.object(manga_inpaint,'cpu_session',return_value=cpu):
            self.assertIs(manga_inpaint.create_session(),cpu)


if __name__=='__main__':unittest.main()
