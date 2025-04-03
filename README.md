
# README - Deep AutoEncoder Based Recommender System

## Project Overview

This project implements a recommender system using both Singular Value Decomposition (SVD) and Deep AutoEncoders. The goal is to predict missing values in a sparse user-product rating matrix, allowing for personalized product recommendations. The project is based on the MovieLens dataset, which contains 100,000 ratings from 943 users on 1,682 movies.

## Files Included

- `SVD-Recommender.py`: Python script implementing SVD-based recommendation.
- `AutoEncoder_Recommender.ipynb`: Colab notebook for training the Deep AutoEncoder model.
- `data/`: Folder containing MovieLens dataset files (CSV format).
- `report.pdf`: A detailed report explaining implementation, methodology, and results.

## Installation and Setup

1. Ensure you have Python 3.x installed.
2. Install required libraries:
   ```sh
   pip install numpy pandas torch torchvision scikit-learn
   ```
3. Upload the dataset files to the `data/` folder.
4. Open `AutoEncoder_Recommender.ipynb` in Google Colab.

## Running the SVD Recommender

1. Run `SVD-Recommender.py`:
   ```sh
   python SVD-Recommender.py
   ```
2. The reconstructed user-product rating matrix will be printed, showing recommendations for users.

## Running the Deep AutoEncoder Model

1. Open `AutoEncoder_Recommender.ipynb` in Google Colab.
2. Uncomment and run the following lines for the first execution:
   ```python
   # prepare_train_validation_movielens_step1()
   # prepare_traintest_movielens_step2()
   ```
   This prepares training and validation datasets.
3. Train the AutoEncoder model using the provided training script.
4. The trained model will generate recommended ratings for users.
5. The highest output rating (excluding already rated items) will determine new recommendations.

## Model Details

### SVD Recommender

- Decomposes the user-product rating matrix using SVD.
- Uses top 3 eigenvalues for low-rank approximation.
- Reconstructs the matrix to fill missing ratings.

### Deep AutoEncoder Recommender

- Uses a deep neural network to predict missing ratings.
- Architecture:
  - Encoder: (9559, ReLU, 512, ReLU, 512, ReLU, 1024)
  - Decoder: (1024, ReLU, 512, ReLU, 512, ReLU, 9559)
- Uses Masked Mean Square Error (MMSE) loss function.
- Trains on the MovieLens dataset and outputs predicted ratings.

## Results and Evaluation

- The SVD approach provides a baseline for recommendation.
- The AutoEncoder model achieves improved prediction accuracy when trained on a large dataset.
- Final recommendations are generated based on the highest predicted ratings.

## Submission Guidelines

- Ensure all required files are included in a ZIP archive named:
  ```
  lastname_firstname_ID_Assignment_4.zip
  ```
- Submit the ZIP file as per the course requirements.

## Acknowledgments

- MovieLens dataset providers.
- Course instructors and teaching assistants.

---

For any issues or questions, please contact the instructor or refer to course materials.

