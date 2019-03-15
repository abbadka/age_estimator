# Age Estimator

This project is the capstone project for **Udacity Machine Learning Engineer Nanodegree**. The goal behind this project is to build a model that predicts the age of a person given a photo that includes the face of the person. This model is used to build a basic application which - for entertainment purposes - allows a person to upload a photo, and predicts the age of the person. The application also includes a sampling of photos from the dataset, and asks the user to estimate the age of the given photo, and see if they can beat the estimation of the developed model.

## Getting Started

The project has two portions, the Jupyter Notebook, which contains the actual processing and training for the model, and the sample flask application using the successful model. The Jupyter notebook depends on the following:

- keras: https://keras.io/#installation
- numpy: http://www.numpy.org/
- pandas: https://pandas.pydata.org/
- matplotlib: https://matplotlib.org/

The flask application has the following dependencies:

- Dlib Installation: http://dlib.net/compile.html
- OpenCV (cv2): https://pypi.org/project/opencv-python/
- keras: https://keras.io/#installation
- PIL: https://pillow.readthedocs.io/en/5.3.x/installation.html
- flask: http://flask.pocoo.org/docs/1.0/installation/

After installing the dependencies, the application can be run as follows:

```
cd age-estimator-app
python app.py
```

### Data Sources

The project uses the following datasets:

- **UTKFaces:** The UTKFaces dataset is available from this location: https://susanqq.github.io/UTKFace/

- **OUI-Adience:** The OUI-Adience dataset is available from this location: https://talhassner.github.io/home/projects/Adience/Adience-data.html

### License Claim

- For the datasets, please refer to the license details on their respective pages.
- For the remainder of the source code and study, it can be used for non-commercial purposes as long as you refer to this study.
