# Machine_Learning_methods_letter_recogniztion
Machine Learning on how to recognize a letter, methods including LDA, logistic regression, CART, Random Forest, along with bagging and boosting.


## Background

One of the most widespread applications of machine learning is to do optical character/letter
recognition, which is used in applications from physical sorting of mail at post offices, to reading
license plates at toll booths, to processing check deposits at ATMs. Optical character recognition
by now involves very well-tuned and sophisticated models. In this problem, we will build a simple
model that uses attributes of images of four letters in the Roman alphabet { A, B, P, and R } to
predict which letter a particular image corresponds to.<br/>
In this problem, we have four possible labels of each observation, namely whether the observation
is the letter A, B, P, or R. Hence this is a multi-class classification problem.


## Data
The file Letters.csv contains 3116 observations, each of which corresponds to a certain image
of one of the four letters A, B, P and R. The images came from 20 different fonts, which were
then randomly distorted to produce the final images; each such distorted image is represented as
a collection of pixels, and each pixel is either "on" or "off". For each such distorted image, we
have available certain attributes of the image in terms of these pixels, as well as which of the four
letters the image is. These features are described in the data. Data Cleaning process has been finished and exclude from this repo.
Feel free to look at my other ML projects if you are interested in data cleaning process.
