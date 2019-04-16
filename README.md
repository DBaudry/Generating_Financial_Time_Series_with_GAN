# Generating_Financial_Time_Series_with_GAN

This project is part of the "Machine Learning for Finance" course conducted by Romuald Elie at ENSAE Paris. In this project we explored 
different Generative Adversarial Networks architectures in order to generate financial Time Series. For a complete description of the methodology please refer 
to our pdf report.

The lack of available data makes the training of Machine Learning algorithms sometimes difficult in Finance. Hence, there is a great interest 
in the development of Time Series generators as a data augmentation method. However, Financial Time Series exhibits different properties that 
are often difficult to reproduce with simple models. A good generator needs to be able both to get the properties of the general distribution of the data (Moments for instance),
but also to get its temporal properties (Autocorrelation structures). Along with the Neural Networks we implement several methods to check these properties.

The procedure for GAN training and evaluation is divided into the following steps:
* Generation of Training Samples (utils file)
* Implementation of Generator and Discriminator Neural Networks, following a given architecture (Lin_GAN, DCGAN, RGAN, CvGan,...)
* Implementation of a general training framework for all architectures (GAN file)
* Statistical tests to verify the adequacy of the properties of the generated samples with the ones of samples from the true distribution

We also provide some pre-trained models that can be loaded using functions from the utils module. An experimental report is also provided in a Notebook format.
