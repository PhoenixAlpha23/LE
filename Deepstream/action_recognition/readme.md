# TAO 3D Action Recognition Integration

## Overview
This module implements spatio-temporal action recognition (e.g., "pocketing", "picking_up") for DeepStream. Because 3D CNNs require a sliding window of frames rather than single spatial frames, it uses `nvdspreprocess` to buffer 32-frame sequences before passing the tensor to a secondary `nvinfer` engine.

## Configuration Files
* **`config_preprocess.txt`**: Manages the temporal buffer. Configures the tensor sequence shape (`1;3;32;224;224`) and ensures the tracker output is batched into a 3D tensor.
* **`config_infer_action.txt`**: The secondary inference configuration. Bypasses standard frame input (`input-tensor-from-meta=1`) to consume the preprocessed 3D tensor using custom network type 100.

## Pipeline Architecture
```text
... -> nvtracker -> nvdspreprocess -> nvinfer (Action) -> nvosd -> ...
```

## Python Integration Steps
* initialise elements


```python
# Create preprocessor for sliding window
preprocess = Gst.ElementFactory.make("nvdspreprocess", "action-preprocessor")
preprocess.set_property("config-file", "config_preprocess.txt")

# Create secondary infer engine
action_infer = Gst.ElementFactory.make("nvinfer", "action-inference")
action_infer.set_property("config-file-path", "config_infer_action.txt")
```

## reference links
[TAO 3D app](https://docs.nvidia.com/deepstream/text/DS_3D_Action)

[TAO model card](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/actionrecognitionnet/-?_lr=1)

[yolo world](https://share.google/aimode/dCEuPgIVN0GAQ16WS)
