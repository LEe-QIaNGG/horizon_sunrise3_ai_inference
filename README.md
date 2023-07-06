# horizon_sunrise3_ai_inference
地平线Horzion旭日sunrise3派开发板部署开源项目（[FERPlus](https://github.com/ebarsoum/FERPlus) / [a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
）
![总体流程](procedure.png)
***
## dependencies
before deployment:
  - pytorch 1.12.1
  - onnx 1.14.0

sunrise3:
  - Ubuntu 20.04 Desktop
  - balenaEtcher(flash OS image to USB drives)
  - openexplorer/ai_toolchain_centos_7.v1.13.6
  - openexplorer toolkit

    `wget -c ftp://vrftp.horizon.ai/Open_Explorer_gcc_9.3.0/2.2.3/horizon_xj3_open_explorer_v2.2.3_20220617.tar.gz`
***
## run 
Run utils.ipynb in local environment to
  - converting onnx model version
  - Preprocess input images required for runtime validation on the development board
  - Local verification of onnx running on the development board

To call the onnx model on the development board to verify and infer the images under the working path

  `python3 ./sunrise/inference.py`

Directly use binary files on the development board to infer images under the working path

  `python3 ./sunrise/ai_inference.py`

Based on MIPI camera on development board inference

  `python3 ./sunrise/mipi_camera.py`

Train your image captioning model according to [@sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning). Net.py is added to obtain an end-to-end model in order to be converted to an onnx model. Other files in ./image_caption have been modified accordingly.

If you don't want to train and convert a cntk model, there's ready-made onnx model here [ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus).
