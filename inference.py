import torch
from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead

# Define the function to change softmax to argmax
def change_softmax_to_argmax(input_tensor):
    _, indices = torch.max(input_tensor, dim=-1)
    return indices

# Initialize the model
model = NanoDetPlusHead()

# Load the model weights
model.load_state_dict(torch.load("path/to/model_weights.pth"))
model.eval()

# Define the input data
input_data = torch.rand(1, 3, 224, 224)

# Forward pass through the model
output = model(input_data)

# Replace softmax with argmax in the decode_bbox function
def decode_bbox(outputs,strides,conf_threshold=-2):
    scales=len(strides)
    bboxes=[]
    for i in range(scales):
        stride=strides[i]
        cls_map=np.transpose(np.squeeze(change_softmax_to_argmax(outputs[i*3+2]),axis=0),[2,0,1])
        reg_map=np.transpose(np.squeeze(outputs[i*3],axis=0),[2,0,1])
        conf_map=np.squeeze(outputs[i*3+1],axis=(0,-1))
        h=conf_map.shape[0]
        w=conf_map.shape[1]
        grid_centers=(np.stack((np.meshgrid(range(w),range(h))),2)+0.5)*stride
        valid_grids=np.argwhere(conf_map>=conf_threshold)
        for valid_grid in valid_grids:
            (x_grid,y_grid)=(valid_grid[0],valid_grid[1])
            grid_center=grid_centers[x_grid,y_grid,:]
            x_center=grid_center[0]+reg_map[0,x_grid,y_grid]
            y_center=grid_center[1]+reg_map[1,x_grid,y_grid]
            x_topleft=x_center-np.exp(reg_map[2,x_grid,y_grid])*stride/2
            y_topleft=y_center-np.exp(reg_map[3,x_grid,y_grid])*stride/2
            x_botright=x_center+np.exp(reg_map[2,x_grid,y_grid])*stride/2
            y_botright=y_center+np.exp(reg_map[3,x_grid,y_grid])*stride/2
            cls=cls_map[:,x_grid,y_grid]
            conf=conf_map[x_grid,y_grid]
            bboxes.append([x_topleft,y_topleft,x_botright,y_botright,conf,cls])
    return bboxes


# Instantiate the HB_ONNXRuntime class
onnx_runtime = HB_ONNXRuntime()

# Load the model
onnx_runtime.load_model("path/to/model.onnx")

# Get the detection results
detection_results = onnx_runtime.predict(input_data)

# Visualize the detection results
onnx_runtime.draw_bboxes(input_data, detection_results)
