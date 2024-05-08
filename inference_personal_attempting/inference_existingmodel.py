from mmdet.apis import DetInferencer

# Initialize the DetInferencer with a model configuration and optionally a checkpoint
inferencer = DetInferencer(
    model='rtmdet_tiny_8xb32-300e_coco',
                           weights='rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth')

# Perform inference on an image and visualize the results
results = inferencer('inference_demo.jpg', show=True)

# from mmdet.apis import DetInferencer
#
# # Initialize the DetInferencer
# inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')
#
# # Perform inference
# inferencer('inference_demo.jpg', show=True)