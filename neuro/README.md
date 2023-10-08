#### Neural network

Image classification using a neural network. Works with 28x28 black-white images

You can change:
- amount of layers (minimum amount - 2: input and output layer)
- activation function for hidden layers
- separate activation function for output layer
- batch size
- learning norm
- amount of epoch
- amount of neurons for each layer

You can see two text inputs in central panel:
- `Input file` - file name with input image (28x28). Program will update classification **after changes appears in file**. For now you need to hover over program to see new classification result (due to some egui behaviour. Maybe will be fixed later)
- `Training data load/save path`:
    - if given path is a directory and you clicked `Load`: it is treated as a directory with training data. Each training image need to have name as `label_*`. Also directory should contain `train.json` with all labels in array. Index of label in array is equal to activated neuron for label.
    
    Example of `train.json` : `["auto","heli","plane","ship"]`

    - if given path is a file and you clicked `Load`: it is treated as saved trained neural network and it is loaded in app. You can't change loaded neural network.
    - if given path is a file and you clicked `Save`: it will save your neural network to file (JSON format).

`Drop` button just drops current neural network from app, so you can create new one.

(in the example gif - classification of types of vehicles)

![neuro.gif](/forReadme/neuro.gif)