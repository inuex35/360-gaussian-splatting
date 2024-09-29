import unittest
from unittest.mock import MagicMock, patch
from utils.camera_utils import loadCam, cameraList_from_camInfos, camera_to_JSON
from scene.cameras import Camera
import numpy as np
import torch

class TestCameraUtils(unittest.TestCase):

    @patch('utils.camera_utils.PILtoTorch')
    @patch('utils.camera_utils.torchvision.utils.save_image')
    def test_loadCam(self, mock_save_image, mock_PILtoTorch):
        # Mocking the arguments and cam_info
        args = MagicMock()
        args.resolution = 2
        args.model_path = '/mock/path'
        args.data_device = 'cpu'

        cam_info = MagicMock()
        cam_info.image.size = (800, 600)
        cam_info.mask = None
        cam_info.uid = 1
        cam_info.R = np.eye(3)
        cam_info.T = np.zeros(3)
        cam_info.FovX = 90
        cam_info.FovY = 60
        cam_info.image_name = 'test_image.png'
        cam_info.panorama = False

        mock_PILtoTorch.return_value = torch.rand(3, 400, 300)

        camera = loadCam(args, 0, cam_info, 1.0)

        self.assertIsInstance(camera, Camera)
        self.assertEqual(camera.image.shape, (3, 400, 300))
        self.assertEqual(camera.uid, 0)
        self.assertEqual(camera.data_device, 'cpu')

    @patch('utils.camera_utils.loadCam')
    @patch('utils.camera_utils.tqdm')
    def test_cameraList_from_camInfos(self, mock_tqdm, mock_loadCam):
        args = MagicMock()
        cam_infos = [MagicMock() for _ in range(5)]
        mock_loadCam.side_effect = [MagicMock() for _ in range(5)]
        mock_tqdm.return_value = enumerate(cam_infos)

        camera_list = cameraList_from_camInfos(cam_infos, 1.0, args)

        self.assertEqual(len(camera_list), 5)
        self.assertTrue(all(isinstance(cam, MagicMock) for cam in camera_list))

    def test_camera_to_JSON(self):
        camera = MagicMock()
        camera.R = np.eye(3)
        camera.T = np.zeros(3)
        camera.image_name = 'test_image.png'
        camera.width = 800
        camera.height = 600
        camera.FovX = 90
        camera.FovY = 60

        json_data = camera_to_JSON(1, camera)

        self.assertEqual(json_data['id'], 1)
        self.assertEqual(json_data['img_name'], 'test_image.png')
        self.assertEqual(json_data['width'], 800)
        self.assertEqual(json_data['height'], 600)
        self.assertEqual(json_data['position'], [0.0, 0.0, 0.0])
        self.assertEqual(json_data['rotation'], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(json_data['fx'], 400.0, places=1)
        self.assertAlmostEqual(json_data['fy'], 300.0, places=1)

if __name__ == '__main__':
    unittest.main()