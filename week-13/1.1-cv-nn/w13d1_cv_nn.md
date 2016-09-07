---
title: Computer Vision + Neural Networks
duration: "1:25"
creator:
    name: Alexander Combs
    city: NYC
---

# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Computer Vision + Neural Networks

Week 13 | Lesson 1.1

### LEARNING OBJECTIVES
*After this lesson, you will be able to:*
- Explain the steps involved in computer vision
- Describe what a perceptron is and how it works
- Explain how deep learning models are constructed
- Build a system that utilizes 'deep features' to find similar images


### STUDENT PRE-WORK
*Before this lesson, you should already be able to:*
- Use Python
- Explain how SVMs work
- Describe how cosine similarity works

### INSTRUCTOR PREP
*Before this lesson, instructors will need to:*
- Generate a brief slide deck
- Ask guiding questions and review previously covered topics
- Prepare and tailor specific materials to your student's needs


### LESSON GUIDE
| TIMING  | TYPE  | TOPIC  |
|:-:|---|---|
| 5 min  | [Opening](#opening)  |  Opening |
| 5 min | [Introduction](#introduction) | Introduction |
| 15 min | [Demo/Guided Practice](#images) | Working with Images |
| 15 min | [Demo/Guided Practice](#similarity) | Finding Similar Images |
| 20 min | [Demo/Guided Practice](#neural_nets) | Neural Nets |
| 20 min | [Demo/Guided Practice](#image_sim) | Deep Feature Image Similarity |
| 5 min | [Conclusion](#conclusion) | Conclusion |

---
<a name="opening"></a>
## Opening (5 mins)

Today we're going to discuss computer vision, so let's start by taking a look at some common applications.

![Image & Facial Recognition](http://blogs-images.forbes.com/amitchowdhry/files/2014/03/DeepFace.jpg)

![Self-Driving Cars](http://spectrum.ieee.org/image/MjczNDMxNw)

[![Microsoft Kinect] (http://i.imgur.com/f3fmXIq.png)](https://youtu.be/ECnaCYnQBMQ)

![Self-Driving Cars](http://spectrum.ieee.org/image/MjczNDMxNw)


 <a name="intro">
## Introduction (5 Min)

Up until this point in the course, we have worked with either columns of numbers or with text. Today we are going to learn how to work with images. Though they may seem like they would be far more complicated to work with, they are just as simple as any other medium we've worked with. In fact, image recognition shares a great deal with natural language processing. 

So let's talk about what we might need to do to an image to use machine learning on it. 

#### Check: What are some challenges we might face working with images?

> 
- Converting to numerical representation
- Extracting features
- Higher level representation of base features

We'll now take a look at how we might handle these issues.

<a name="images">
## Working with Images (15 Min)

Our first problem is how to convert our image into a machine readable format. 

#### Check: When we did this with natural language processing, how did we do it? How might we do the same thing with images?

Since images are nothing more than a grid of pixels, we can represent the images as a matrix of the same size where each pixel corresponds to a cell in our matrix. For a greyscale image, the value would represent the intensity of the pixel. For example, typically a value 0 to 16 is used where 0 is pure white and 16 is solid black with the in-between values representing increasingly darker shades of gray. 

If we wanted to represent a color picture, we could use 3 features for each pixel. These would then be the RGB value for each.

Let's now look at the 'hello world' of image recognition, the MNIST digits data set. This is a large data set of handwritten digits. This data set is frequently used for image recognition tasks.

```python
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

digits = datasets.load_digits()

def display_img(img_no):
 fig, ax = plt.subplots()
 ax.set_xticklabels([])
 ax.set_yticklabels([])
 ax.matshow(digits.images[img_no], cmap = plt.cm.binary);
 
display_img(0)
```
This will generate the following image:

![Zero](http://imgur.com/6mJpcvq.png)

We can see the matrix representation with the following code:
```python
digits.images[0]
```
![Zero Matrix](http://i.imgur.com/G77LvUO.png)


#### Exercise: Try displaying an image from the MNIST data set and then visualizing it both as an image and also as a matrix.

<a name="similarity">
## Finding Similar Images (15 Min)

Now, let's imagine we'd like to find the image in the data set that is the closest to the image we're interested in.

#### Check:  How might we find the image that is the closest to our sample image?

> We can use cosine similarity


#### Exercise: Take the image you generated above and spend the next few minutes writing the code to find the image most similar to that one.


```python 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
X = digits.data
co_sim = cosine_similarity(X[100].reshape(1,-1), X)
cosf = pd.DataFrame(co_sim).T
cosf.columns = ['similarity']
cosf.sort_values('similarity', ascending=False).head()
```


If all went according to plan, you should see something like this:

Target:
![](http://i.imgur.com/lfKDTf9.png)

Nearest Match: 
![](http://i.imgur.com/c2W0YeY.png)

#### Exercise: What is the opposite of 0? Try to find the images with the least similarity to 0.

Note: While cosine similarity works well here, typically in machine vision tasks, something called a Chi-squared kernel is used vs. cosine similarity which is the preferred choice in language-based tasks. Why use one vs. the other? Simply because they have proven over time to work better in their respective domains.

<a name="keypoints">
## Keypoints (20 Min)

So far we've seen that we can use pixel-level features in our models, but surely this can't be enough for more advanced computer vision applications like object detection, right? Precisely. We need to incorporate features that capture a higher level abstraction of these pixel features alone.

This can be accomplished using something called keypoint detection. The idea here is that we identify the most salient points in an image. These points are then transformed so that they are invariant to scale and rotation. 

![Keypoints](http://i.stack.imgur.com/t3KIY.jpg)

After this step, the image is a collection of vectors for each keypoint descriptor in the image. Then next step is to generate a codebook from these vectors using something like k-means. This codebook forms a visual vocabulary much like we use a word vocabulary in NLP. Codewords are the centers of the clusters.

This can be illustrated with an example using textures. Here we have a number of textures. 

![](http://i.imgur.com/ytURhGQ.png)

Each can be represented as a repeating series of textural units.

![](http://i.imgur.com/nVT8Gss.png)

Each texture then is a historgram over these texutral units or textrons:

![](http://i.imgur.com/NoPplgr.png)

Finally, images can be classified using these features. Notice this process dispenses with spatial positioning of these features in the same manner that NLP does in bag-of-words processing. For this reason, this method is known as visual bag of words.

Typically the next step involves the use of SVMs. Until recently this was state-of-the-art. Today, however, **deep learning** has taken over the top stop for visual tasks.

<a name="neural_networks">
## Neural Networks (20 Min)

Deep learning is the most recent (and most advanced) incarnation of neural networks. To understand what a neural network is we need to first understand the basic building block of these networks: the perceptron. 

Perceptrons date back to the 1950s. They were an attempt to adopt the biological model of the brain to computers. There first application was, naturally, image recognition. 

Recalling your high school biology, neurons work in the following way.
- Each neuron is connected in a network to many other neurons
- These neurons both send and receive signals from connected neurons
- When a neuron receives a signal it can either fire or not depending on whether the incoming signal is above some threshold

A single perceptron then, like a neuron, can be thought of a decision-making unit. If the weight of the incoming signals is above the threshold, the perceptron fires, if not it doesn't. In this case firing equals outputting a value of 1 and not firing equals outputting a value of 0.

![](http://i.imgur.com/UTDz0Vw.png)

![](http://i.imgur.com/yjJSx39.png)

As a example, imagine you are considering taking a new job. There are several factors that go into the decision, things like the location, the pay, the work, the hip-ness of the office. Obviously, not all of these carry the same weight when making your decision. In the perceptron, each input then is modified by a weighting factor. If the incoming values are greater than the threshold the unit outputs a 1, if not, it outputs a 0.

Let's now see how learning can take place with a perceptron. We will train a perceptron with two inputs to learn an AND rule. That is we want the perceptron to output a 1 when both inputs are 1, and a 0 in all other cases.

Let's now begin our learning process. Our X1 input is equal to 1 and our X2 input is equal to -1. W1 is randomly set to 0.8, and W2 is randomly set to 0.2. As both must be positive for our output to be 1, then our threshold is any value greater than 1.

Therefore, we have W1*X1 + W2X2 = as 1 * 0.8 + -1 * 0.4 = 0.8 – 0.4 = 0.4. 

 Now, since we expect our output to be 0, we can say that we have an error of 0.4. We will now attempt to improve our model by pushing these errors back down to the inputs in order to update the weights.

To do this, we will evaluate each in turn using the following formula:

![](http://i.imgur.com/HIn24fw.png)

Here, wi is the weight of the ith input, t is the target outcome, and 0 is the actual outcome. Here, our target outcome is 0, and our actual outcome is 0.4. Ignore the n term for now. This is the learning rate. It determines how large or small our updates should be. For now, we will assume it is set to 1.

Let's look at X1 to update its weight. Therefore, we have 1 * (0 – 0.4) * 1, which equals -0.4. That is our w delta; therefore, updating equation 1, we have 0.8 – 0.4, which gives us our new weight for W1 as 0.4. Therefore, the weight for X1
has come down. What about the weight for X2? Let's take a look. That one is 1 * (0 – 0.4) * -1, which equals 0.4. Updating the weight, we have 0.2 + 0.4 = 0.6. Notice the weights are converging to parity which is what we would expect.

This ultra-simple model is the fundamental building block of today's deep learning algorithms. Many advancements have been applied over the years, such as changing the activation functions and adding hidden layers (layers between the input layers and output layers), but fundamentally, these perceptrons have been built upon to produce the deep learning neural networks used today.

![](http://cs231n.github.io/assets/nn1/neural_net.jpeg)
<a name="image sim"> 
## Imagine Similarity with Deep Features (20 Min)

Now that we've learned a bit about computer vision applications, feature extraction, and the building blocks of deep learning, let's use a pre-trained deep learning network to find the image most similar to ourselves in the CIFAR-10 image database.

We will first need to make sure GraphLab Create is installed. This is a platform that makes large-scale learning possible very simply. Fortunately for us, it is available free for a one year period for students (bootcamps included). 

Follow these directions to get it installed: https://turi.com/download/install-graphlab-create-command-line.html

Once that is completed, we can begin our code in our Jupyter notebook.

```python
import graphlab
graphlab.canvas.set_target('ipynb')
gl_img = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/coursera/deep_learning/image_train_data')
gl_img
```

This is output a DataFrame like the following:

![](http://i.imgur.com/k3hSsFq.png)

In this DataFrame, we have a number of columns describing the images, but we are primarily concerned with the deep_features column. These are feature that were learned from a deep learning neural network. They won't make any sense to you, but they are the learned features that discriminate from the various classes of images in the CIFAR-10 data set. We are going to use something called **transfer learning** to allow us to use these deep features for our own purposes. 

Because each layer of a network can be though of as a progressively more holistic representation of the trained object, we can 'cut off the head' of the network so to speak and use it to extract lower-level features to apply on our own images.

![](http://www.kdnuggets.com/wp-content/uploads/deep-learning-300x180.png)

We can see the images in the following way

```python
graphlab.image_analysis.resize(gl_img['image'][:5], 128,128).show()
```

![](http://i.imgur.com/B7NowQN.png)

Now let's extract the deep features of our own image in order to find the image most similar to our own. The will be our CIFAR-10 'spirit animal' !

```python
img = graphlab.Image('https://pbs.twimg.com/profile_images/627283912075702272/gsmHwYrT.jpg')
ppsf = graphlab.SArray([img])
ppsf = graphlab.image_analysis.resize(ppsf, 32,32)
ppsf.show()
```

```python
ppsf = graphlab.SFrame(ppsf).rename({'X1': 'image'})
ppsf
```

```python
extractor = graphlab.feature_engineering.DeepFeatureExtractor(features='image', model='auto')

extractor = extractor.fit(ppsf)

ppsf['deep_features'] = extractor.transform(ppsf)['deep_features.image']

ppsf
```

This gives us our deep features for our own image:

![](http://i.imgur.com/HzseeAr.png)

We now need to add that to our original DataFrame, and then find the most similar images.

```python
ppsf['label'] = 'me'
gl_img['id'].max()

ppsf['id'] = 50000
labels = ['id', 'image', 'label', 'deep_features']
part_train = gl_img[labels]
new_train = part_train.append(ppsf[labels])
new_train.tail()
```

This give us our new DataFrame:

![](http://i.imgur.com/Zo4ozh8.png)

Now, we'll use knn to find the closet matches:

```python
knn_model = graphlab.nearest_neighbors.create(new_train,features=['deep_features'], label='id')

me_test = new_train[-1:]
def reveal_my_twin(x):
    return gl_img.filter_by(x['reference_label'],'id')

spirit_animal = reveal_my_twin(knn_model.query(me_test))
graphlab.image_analysis.resize(spirit_animal['image'], 128,128).show()
```

![](http://i.imgur.com/BxVGU0f.png)

So there it is, our spirit animal!

<a name="conclusion">
## Conclusion (5 Min)


In this lesson, we learned about computer vision applications; some of the basic concepts in working with computer vision; the basics of neural networks, deep learning, and transfer learning; and finally we used deep features to find the image most similar to our own in the CIFAR-10 data set.













