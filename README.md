SHAPE DETECTOR CNN PROJECT
Here's a brief overview of what I have created!
*What It Does:*
The main function of this project is to detect what kind of shape a person has drawn (either a circle, square or triangle), typically on a piece of paper. On the OpenCV video feed, a box is shown surrounding the ROI field, which is captured and ran against a model that was trained on a public database of images from Kaggle (found in this repository). Following this, the most likely prediction is displayed on screen. This detection works not only for drawings, but typical high contrast examples of the aforementioned shapes. 

To elaborate, I have created a file called "model.py" which is responsible for reading user-chosen training parameters for a pytorch CNN, specifically batch size, epochs and learning rate per step. After this has been chosen, every database picture is resized to 64x64 and converted to grayscale, among other transformations like converting the files to tensor format and normalizing the pixel value range to 0 -> 1, from -1 -> +1. After this, the neural network is crafted using a set of sequential steps, where every step observes two times more feature maps than the previous step, at half the previous batch parse's resolution, creating a (B, 64, 8, 8) network in the end. Finally this is collapsed. 
- Throughout the training process, cross entropy loss is used as the loss function and Adam optimization is used
- After completion, a graph of how the errors and accuracy progressed over each epoch is displayed, where the trained model is ready for use by the OpenCV file: "detection.py"

*Fundamental Implementations:*
- PyTorch
- OpenCV
- Tkinter
- API / JSON File Management

*Why?*
This project was created so that I could understand the basics of machine learning and integrating computer vision. 
