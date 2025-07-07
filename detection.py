# imports
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import model

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
testModel = model.CNN().to(device)
testModel.load_state_dict(torch.load("shapeGuesser_cnn.pth", map_location=device))
testModel.eval()

# labels for classes
classes = ['circle', 'square', 'triangle']

# camera stuff, using external webcam
capture = cv2.VideoCapture(1)

print("press \'C\' to classify, \'ESC\' to exit.")

while True:
    ret, frame = capture.read()
    if not ret:
        break

    height, width, _ = frame.shape
    size = 224
    top_left_corner = (width // 2 - size // 2, height // 2 - size // 2)
    bottom_right_corner = (width // 2 + size // 2, height // 2 + size // 2)

    # where we will detect within: region of interest (roi)
    roi = frame[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
    cv2.rectangle(frame, top_left_corner, bottom_right_corner, (255, 255, 0), 2)
    cv2.putText(frame, "Draw your shape here and press \'C\'.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # show the frame
    cv2.imshow("Shape Guesser", frame)
    keyPress = cv2.waitKey(1)

    # if esc, quit. If 'C', guess
    if keyPress == 27:
        break
    elif keyPress == 99:

        # take the ROI and convert to a PIL image. REASON: PIL (formerly python image library, now 'Pillow') is what torch expects as an input. Why? because it is modifyable and you can use it with the PIL library, in any format. This allows transformations like the preprocessing to happen.
        # TL:DR, PIL images are expected by pyTorch, and easier to use/manipulate 

        # turn the roi to RGB to use with 'Image' library
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        # transform frame that was just captured with 'img'
        input_tensor = model.preprocess(pil_img).unsqueeze(0).to(device) # unsqueeze adds a batch number of 1 with '.unsqueeze(0)', so that the other code from 'model.py' can be used to run the detection model on our frame

        # predict the result
        with torch.no_grad():

            # get logits (prediction scores), and only take those (_, predicted)
            outputs = testModel(input_tensor)

            _, predicted = torch.max(outputs, 1) # max value across dimension 1 is the index of the class
            label = classes[predicted.item()]

            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted2 = torch.max(probabilities, 1)
            print(f"Prediction: {label}")
            print(f"Confidence (probability): {max_prob.item():.4f}")
            print(outputs)

            # Show prediction
            cv2.putText(frame, f"Predicted: {label}", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 225, 255), 2)
            cv2.imshow("Shape Guesser", frame)
            cv2.waitKey(1000)  # Wait 1/2s to show result

capture.release()
cv2.destroyAllWindows()