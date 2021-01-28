# 2019-全国高校计算机能力挑战赛-人工智能算法赛

# Usage Scenario
1 Four-digit verification code
2 Fixed size image form input

# Review
Looking back on this competition more than a year later, I have to say that its difficulty is very suitable for novices to participate. Reading the source code, programming by myself, and then learning to reuse the code, taught me a lot step by step. When I encountered a problem which I didn’t understand, I would search the problem on the Internet, then tried to solve the problem. What's more, I also learnt how to optimize. In general, this competition gave me a good understanding of the basic knowledge of deep learning, and it also had a great influence on my participation in more competitions.

# Introduction
This competition requires verification code identification, the verification code is 4 digits, composed of numbers and letters. The difficulty lies in the presence of interference lines, salt and pepper noise and contrast in the picture (this is especially important for achieving a high ranking).

Since the position of the verification code in the picture is relatively fixed, I directly modified the final output channel of the convolutional layer and converted it to 4 channels to output four results. This effect performed well in the test.

# Installation
This program based on tensorflow1.x and keras, so you should install these libraries.

Then you should install all the libraries in requirements.txt.

# Run
Run "python main.py"

And you can control some configurations in config.json which contains "train" and "test" options

# Suggestions
In the competition, this program can easily achieve 96% accuracy on datasets. If you can augment the data and properly preprocess the image to solve the problem of contrast, you can easily achieve more than 98% accuracy.
