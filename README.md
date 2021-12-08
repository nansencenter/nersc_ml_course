# Course ML at NERSC
Internal ML course/practical demonstration intern to NERSC


## Link to the slides
- Session 1: Introduction, generalities on machine learning ([pdf](presentation/course-1/course-1.pdf))
- Session 2: Validation, overfitting, regularization ([pdf](presentation/course-2/course-2.pdf))
- Session 3: Random Forest, grid search ([pdf](presentation/course-3/nersc_ml_course_3.pdf))
- Session 4: Neural networks ([pdf](presentation/course-4/course-4.pdf))
- Session 5: Convolutional neural networks ([pdf](presentation/course-5/course-5.pdf))

## Practical demonstration

### Instruction for working on a cloud (recommended)
Run the tutorial in a cloud computing provider (require Google login):

- **Practice 1:** Introduction and linear regression [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p1_linear_regression.ipynb)
- **Practice 2:** Validation, overfitting, regularization [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p2_validation_and_regularization.ipynb)
- **Practice 3:** Random forests. Grid search. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p3_random_forest.ipynb)
- **Practice 4:** Neural networks. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p4_neural_networks.ipynb)
- **Practice 5:** Convolutional Neural Networks and Regularizations. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p5_cnn_regularization.ipynb)
- **HACKATHON** Data for hackathon. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nansencenter/nersc_ml_course/blob/main/notebooks/p6_hackathon_data.ipynb)

### Instructions for working locally

You can also run this notebook on your own (Linux/Windows/Mac) computer.
This is a bit snappier than running them online.

1. **Prerequisite**: Python>=3.7.
   If you're not a python expert:
   1a. Install Python via [Anaconda](https://www.anaconda.com/download).
   1b. Use the [Anaconda terminal](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda)
       to run the commands below.
   1c. (Optional) [Create & activate a new Python environment](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-environments).
       If the installation (below) fails, try doing step 1c first.

2. **Install**:
   Run these commands in the terminal (excluding the `$` sign):
   `$ git clone https://github.com/nansencenter/nersc_ml_course.git`
   `$ pip install -r nersc_ml_course/requirements.txt`

3. **Launch the Jupyter notebooks**:
   `$ jupyter-notebook`
   This will open up a page in your web browser that is a file navigator.  
   Enter the folder `nersc_ml_course/notebooks`, and click on the tutorial `notebook_name.ipynb`

<!-- markdownlint-disable-file heading-increment -->
