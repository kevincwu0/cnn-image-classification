# cnn-image-classification

Overview:
- What are Convolutional Neural Networks? -> End goals, features, examples, human brains vs ANN image recognition
- Step 1 - Convolution Operation -> feature detector, filters, features maps, different parameters, visual examples
- Step 1(b) - Rectified Linear Unit (ReLU) Layer - why linearity is not good, more non-linearity for image recognition
- Step 2  - Pooling, max pooling, mean pooling, sub pooling, and other approaches. Cool example (visual interactive tool)
- Step 3 - Flattening
- Step 4 - Full Connection - puts everything together, how it all works, final neurons classifies neurons
- Summary
- Extra - Softmax and Cross-Entropy

What are Convolutional Neural Networks?
- What our brain is looking for is features, we categorize and classify things in a certain way. 
- Process certain features and classifies them.
- Convolutional Neural Network Search Term > Artificial Neural Network Search Term
- CNNs -> Self-driving Cars recognize stop signs, tag people in images in Facebook
- Yann Lecun grandfather of CNN, Geoffrey Hinton's student, NYU Professor, Facebook, Mafia of Deep Learning
- How CNN works
  - Input Image
  - CNN
  - Output Label Image (Cheetah)
- After being trained, categorized images prior
  - Example -> Smiling face -> CNN -> Happy (probability0
  - Example -> Frowning -> CNN -> Sad (probability)
  - Sometimes we don't see enough features, it's all about features
  - How to recognize these features?
- B/W Image 2x2px 
  -> 2-D Array (Pixel 1, Pixel 2, Pixel 3, Pixel 4) 
  - 0 < pixel 255 value
  - Any black and white image has a digital form 
- Colored Image 2x2px 
  -> 3-D Array -> RGB layers (Red, Green, Blue), 0 < pixel value < 255
  - Red Channel, Blue Channel, Green Channel
  - 0 < pixel 255 value
- Smiling Face Example
  - 0 = white, 1 = black
- Step 1: Convolution
- Step 2: Max Pooling
- Step 3: Flattening
- Step 4: Full Connection
- Yann LeCun et al., 1998, Gradient-Based Learning Applied to Document Recognition http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
